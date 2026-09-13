#!/usr/bin/env bash
# Shared sglang-server bootstrap for SAS eval, sourced by run_reasoning.sh and
# run_longbench.sh. It parses the common CONFIG env vars, starts ONE sglang
# server (in the sglang-blocksparse pixi env), waits until it is ready, and
# installs a cleanup trap. The sourcing script then runs its task-specific
# client + scorer against $BASE_URL, and the server is torn down on exit.
#
# Contract for the sourcing script:
#   * source this file AFTER `set -euo pipefail`
#   * call `sas_parse_config`  -> validates env, sets BACKEND/MODEL_NAME/BASE_URL/
#                                 GATES/BASE_MODEL/TOKENIZER_PATH/OUTPUT_ROOT/TS/...
#   * call `sas_start_server`  -> starts server + trap + waits for /health
#   * then use $BASE_URL / $MODEL_NAME / $TOKENIZER_PATH / $OUTPUT_ROOT / etc.
#
# Common CONFIG env vars (with defaults) — the knobs you normally touch:
#   GATES BUDGET TP DP MODE BASE_MODEL
#   PORT GPUS CONCURRENCY LIMIT MEM_FRAC EXTRA_SERVE_ARGS OUTPUT_ROOT
#   READY_TIMEOUT SGLANG_DIR SGLANG_ENV SKIP_ENV

# ---- common config knobs (env overridable) -----------------------------------
GATES="${GATES:-}"                     # AttnGates ckpt dir (seer mode; REQUIRED for seer)
BUDGET="${BUDGET:-2048}"               # seer decode token budget (per-query KV tokens); default 2048
TP="${TP:-1}"                          # tensor-parallel size
DP="${DP:-8}"                          # data-parallel size (replicas)
MODE="${MODE:-seer_4b}"                # seer_4b | dense_4b | seer_8b | ...
BASE_MODEL="${BASE_MODEL:-}"           # dense: path to base Qwen3 model (REQUIRED for dense)
PORT="${PORT:-30000}"                  # server port
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"        # CUDA_VISIBLE_DEVICES for the server
CONCURRENCY="${CONCURRENCY:-256}"      # in-flight client requests
LIMIT="${LIMIT:--1}"                   # limit #questions per task (debug; -1 = all)
READY_TIMEOUT="${READY_TIMEOUT:-1200}" # server readiness timeout (seconds)
MEM_FRAC="${MEM_FRAC:-}"               # --mem-fraction-static (empty -> sglang default; seer may need >0.8)
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"  # extra flags appended verbatim to sglang.launch_server
OUTPUT_ROOT="${OUTPUT_ROOT:-}"         # parent results dir (empty -> <script dir>/results)
SGLANG_DIR="${SGLANG_DIR:-}"           # sglang-blocksparse checkout (empty -> third_party/ submodule, else sibling)
SGLANG_ENV="${SGLANG_ENV:-}"           # legacy: explicit env script to source (empty -> auto)
SKIP_ENV="${SKIP_ENV:-0}"              # 1 = skip sglang env activation (env already active)
# YaRN context extension to 64k, applied at RUNTIME via --json-model-override-args
# (the on-disk gate config.json is NOT modified). Off by default; the agentic
# wrappers (bfcl/vitabench) turn it on since those benches need the long context.
YARN64K="${YARN64K:-0}"                # 1 = override rope_scaling=yarn factor=1.6, max_position=65536

# Resolve dirs relative to the sourcing script (set by caller before sourcing).
_SERVE_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # scripts/eval
SAS_DIR="$(cd "$_SERVE_COMMON_DIR/../.." && pwd)"                    # .../SAS
PIXI_MANIFEST="$SAS_DIR/pyproject.toml"

# ---- resolve the sglang-blocksparse checkout ---------------------------------
# Preference order: explicit SGLANG_DIR -> in-repo submodule -> sibling checkout.
sas_resolve_sglang_dir() {
  if [[ -n "${SGLANG_DIR:-}" ]]; then
    echo "$SGLANG_DIR"
  elif [[ -d "$SAS_DIR/third_party/sglang-blocksparse" ]]; then
    echo "$SAS_DIR/third_party/sglang-blocksparse"       # git submodule (preferred)
  else
    echo "$(cd "$SAS_DIR/.." && pwd)/sglang-blocksparse" # sibling fallback (legacy)
  fi
}

# ---- config validation + derived vars ----------------------------------------
sas_parse_config() {
  BACKEND="${MODE%%_*}"    # seer | dense
  SIZE="${MODE##*_}"       # 4b | 8b | ...  (label only)
  if [[ "$BACKEND" != "seer" && "$BACKEND" != "dense" ]]; then
    echo "ERROR: MODE='$MODE' — backend must be seer_<size> or dense_<size>." >&2
    exit 1
  fi
  MODEL_NAME="qwen3-${BACKEND}-${SIZE}"
  # Use 127.0.0.1 (not localhost): the sglang server binds IPv4 127.0.0.1, while
  # "localhost" may resolve to IPv6 ::1 first and httpx then fails to connect.
  BASE_URL="http://127.0.0.1:${PORT}"

  TS="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_ROOT="${OUTPUT_ROOT:-$_SERVE_COMMON_DIR/results}"
  BUDGET_TAG="${BUDGET:+-tb${BUDGET}}"   # tag runs so sweeps don't collide

  # GATES/BASE_MODEL validation. Resolve GATES to an absolute path and fail
  # loudly if it doesn't exist — the #1 cause of the server silently hanging.
  if [[ "$BACKEND" == "seer" ]]; then
    if [[ -z "${GATES:-}" ]]; then
      echo "ERROR: seer backend requires GATES=/path/to/AttnGates." >&2
      exit 1
    fi
    GATES_ABS="$(readlink -f "$GATES" 2>/dev/null || echo "$GATES")"
    if [[ ! -e "$GATES_ABS" ]]; then
      echo "ERROR: GATES path does not exist: '$GATES' (resolved: '$GATES_ABS')." >&2
      exit 1
    fi
    GATES="$GATES_ABS"
    TOKENIZER_PATH="$GATES"          # seer ships the tokenizer in the AttnGates dir
  else
    if [[ -z "${BASE_MODEL:-}" ]]; then
      echo "ERROR: dense backend requires BASE_MODEL=/path/to/Qwen3." >&2
      exit 1
    fi
    TOKENIZER_PATH="$BASE_MODEL"     # dense ships the tokenizer in the base model dir
  fi
}

# ---- serve function (runs in the sglang-blocksparse pixi env) ----------------
# Invoked in a backgrounded subshell below, so env activation stays isolated and
# never leaks into the client/scorer env.
_sas_serve() {
  # Activate the sglang-blocksparse pixi env (unless already active / skipped).
  # We activate via `pixi shell-hook` on the checkout's pyproject.toml directly —
  # no dependency on a personal myenv.sh. (A legacy SGLANG_ENV script, if given
  # or found, is still honored.)
  if [[ "$SKIP_ENV" != "1" ]] && ! python -c "import sglang" >/dev/null 2>&1; then
    local sg_dir sg_env
    sg_dir="$(sas_resolve_sglang_dir)"
    sg_env="${SGLANG_ENV:-}"
    if [[ -n "$sg_env" && -f "$sg_env" ]]; then
      echo ">>> activating sglang env (legacy script): $sg_env"
      # shellcheck disable=SC1090
      source "$sg_env"
    elif [[ -f "$sg_dir/pyproject.toml" ]] && command -v pixi >/dev/null 2>&1; then
      echo ">>> activating sglang pixi env: $sg_dir"
      # shellcheck disable=SC1090
      eval "$(pixi shell-hook --manifest-path "$sg_dir/pyproject.toml")"
    elif [[ -f "$sg_dir/myenv.sh" ]]; then
      echo ">>> activating sglang env (myenv.sh): $sg_dir/myenv.sh"
      # shellcheck disable=SC1090
      source "$sg_dir/myenv.sh"
    else
      echo "WARNING: could not activate sglang env at $sg_dir" >&2
      echo "         Expected $sg_dir/pyproject.toml (run 'pixi install' there)," >&2
      echo "         or set SGLANG_DIR=/path, SGLANG_ENV=/path/to/env.sh, or SKIP_ENV=1." >&2
    fi
  fi

  local model_path
  if [[ "$BACKEND" == "seer" ]]; then model_path="$GATES"; else model_path="$BASE_MODEL"; fi

  local -a args=(
    --model-path "$model_path"
    --served-model-name "$MODEL_NAME"
    --tool-call-parser qwen
    --trust-remote-code
    --port "$PORT"
  )
  [[ "$BACKEND" == "seer" ]] && args+=(--attention-backend seer_attn)
  [[ -n "${TP:-}" ]]       && args+=(--tp-size "$TP")
  [[ -n "${DP:-}" ]]       && args+=(--dp-size "$DP")
  [[ -n "${MEM_FRAC:-}" ]] && args+=(--mem-fraction-static "$MEM_FRAC")
  # YaRN 64k: override rope at load time; leaves the gate's config.json untouched.
  if [[ "$YARN64K" == "1" ]]; then
    args+=(--json-model-override-args \
      '{"max_position_embeddings": 65536, "rope_scaling": {"rope_type": "yarn", "factor": 1.6, "original_max_position_embeddings": 40960}}')
    echo ">>> yarn64k: rope override -> 65536 (factor 1.6)"
  fi
  # shellcheck disable=SC2206
  [[ -n "${EXTRA_SERVE_ARGS:-}" ]] && args+=(${EXTRA_SERVE_ARGS})

  echo ">>> $BACKEND serve: model=$model_path served=$MODEL_NAME port=$PORT tp=${TP:-<default>} dp=${DP:-<default>}"

  local -a env_prefix=()
  [[ -n "${GPUS:-}" ]] && env_prefix+=("CUDA_VISIBLE_DEVICES=$GPUS")
  if [[ "$BACKEND" == "seer" && -n "${BUDGET:-}" ]]; then
    # map our BUDGET knob to the env var the sglang seer_attn backend reads
    env_prefix+=("SGLANG_SEER_TOKEN_BUDGET=$BUDGET")
    echo ">>> budget=$BUDGET"
  fi

  env "${env_prefix[@]}" python -m sglang.launch_server "${args[@]}"
}

# ---- start server ONCE (background subshell) + trap + readiness --------------
sas_start_server() {
  mkdir -p "$OUTPUT_ROOT"
  SERVER_LOG="$OUTPUT_ROOT/server-${MODE}${BUDGET_TAG}-${TS}.log"
  echo ">>> starting sglang server ($MODE) on port $PORT, log -> $SERVER_LOG"
  _sas_serve > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!

  cleanup() {
    # run any extra cleanup the sourcing script registered (e.g. rm a rendered
    # secrets file), then tear down the server.
    for _f in "${SAS_CLEANUP_FILES[@]:-}"; do
      [[ -n "$_f" ]] && rm -f "$_f" 2>/dev/null || true
    done
    echo ">>> tearing down server (pgid of $SERVER_PID)"
    pkill -P "$SERVER_PID" 2>/dev/null || true
    kill "$SERVER_PID" 2>/dev/null || true
    pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
  }
  trap cleanup EXIT INT TERM

  echo ">>> waiting for server /health (timeout ${READY_TIMEOUT}s)"
  local deadline
  deadline=$(( $(date +%s) + READY_TIMEOUT ))
  until curl -sf "${BASE_URL}/health" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "ERROR: server process died during startup. Tail of $SERVER_LOG:" >&2
      tail -30 "$SERVER_LOG" >&2 || true
      exit 1
    fi
    if [[ $(date +%s) -ge $deadline ]]; then
      echo "ERROR: server not ready within ${READY_TIMEOUT}s. Tail of $SERVER_LOG:" >&2
      tail -30 "$SERVER_LOG" >&2 || true
      exit 1
    fi
    sleep 5
  done
  echo ">>> server ready."
}
