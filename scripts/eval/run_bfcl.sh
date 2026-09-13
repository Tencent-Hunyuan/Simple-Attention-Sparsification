#!/usr/bin/env bash
# BFCL (Berkeley Function-Call Leaderboard) eval driver via sglang.
#
# Starts an sglang server ONCE (see _serve.sh), then runs BFCL generate + evaluate
# against it over the OpenAI-compatible endpoint, and tears the server down.
#
# BFCL lives in the third_party/gorilla submodule (official repo). Two one-time
# prerequisites (see README) that this script only CHECKS, never mutates:
#   1. patch:  git -C third_party/gorilla apply scripts/eval/patches/bfcl.patch
#              (registers the qwen3-<be>-<size>-FC models BFCL serves)
#   2. venv:   python -m venv <BFCL>/.venv && <.venv>/bin/pip install -e <BFCL>
#              (creates .venv with the `bfcl` CLI)
#
# The SERVER runs in the sglang-blocksparse pixi env; BFCL runs in its own venv.
#
# Examples:
#   GATES=/path/to/AttnGates BUDGET=2048 TP=1 DP=8 bash run_bfcl.sh
#   CATEGORIES=simple_python NUM_THREADS=8 GATES=/path/to/AttnGates bash run_bfcl.sh   # smoke
#   MODE=dense_4b BASE_MODEL=/path/to/Qwen3-4B bash run_bfcl.sh
#
# Common server knobs (GATES/BUDGET/TP/DP/MODE/BASE_MODEL/PORT/GPUS/...) are
# documented in _serve.sh. BFCL-specific knobs below.

set -euo pipefail

# Agentic benches run long contexts -> enable YaRN 64k by default (runtime rope
# override; on-disk gate config is untouched). Set YARN64K=0 to disable.
export YARN64K="${YARN64K:-1}"

# ---- BFCL-specific config ----------------------------------------------------
CATEGORIES="${CATEGORIES:-single_turn,multi_turn}"   # BFCL test categories (comma-sep)
NUM_THREADS="${NUM_THREADS:-32}"                     # client concurrency
GEN_ONLY="${GEN_ONLY:-0}"                            # 1 = only generate, skip evaluate
EVAL_ONLY="${EVAL_ONLY:-0}"                           # 1 = only evaluate existing results
# Path to the BFCL package inside the gorilla submodule (override if relocated).
BFCL_DIR="${BFCL_DIR:-}"

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_serve.sh
source "$_SCRIPT_DIR/_serve.sh"

# ---- locate BFCL + one-time-setup checks -------------------------------------
GORILLA_DIR="$SAS_DIR/third_party/gorilla"
BFCL_DIR="${BFCL_DIR:-$GORILLA_DIR/berkeley-function-call-leaderboard}"
PATCH="$_SCRIPT_DIR/patches/bfcl.patch"

if [[ ! -d "$BFCL_DIR" ]]; then
  echo "ERROR: BFCL not found at $BFCL_DIR" >&2
  echo "       Run: git submodule update --init third_party/gorilla" >&2
  exit 1
fi

# Verify the qwen3 model-registration patch is applied (we never mutate the
# submodule ourselves — apply it once by hand, see README). If applying in
# reverse succeeds, the patch is already present.
if ! git -C "$GORILLA_DIR" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  echo "ERROR: BFCL model-registration patch not applied." >&2
  echo "       Run once: git -C third_party/gorilla apply $PATCH" >&2
  exit 1
fi

# ---- validate + start server -------------------------------------------------
sas_parse_config
MODEL="${MODEL_NAME}-FC"    # BFCL entry name = qwen3-<be>-<size>-FC (function-calling)
echo ">>> task=bfcl model=$MODEL categories=$CATEGORIES mode=$MODE budget=${BUDGET:-<default>} tp=${TP:-<default>} dp=${DP:-<default>}"
sas_start_server

# ---- activate BFCL venv ------------------------------------------------------
# venv lives inside the submodule (BFCL_VENV overrides; default .venv).
if ! command -v bfcl >/dev/null 2>&1; then
  VENV_ACT="${BFCL_VENV:-$BFCL_DIR/.venv}/bin/activate"
  if [[ -f "$VENV_ACT" ]]; then
    echo ">>> activating BFCL venv: $VENV_ACT"
    set +u; # shellcheck disable=SC1090
    source "$VENV_ACT"; set -u
  else
    echo "ERROR: BFCL venv not found at $VENV_ACT" >&2
    echo "       Create it once: python -m venv $BFCL_DIR/.venv && \\" >&2
    echo "                       source $BFCL_DIR/.venv/bin/activate && \\" >&2
    echo "                       pip install -e $BFCL_DIR" >&2
    exit 1
  fi
fi

# ---- point BFCL's OpenAI client at the local sglang server -------------------
# 127.0.0.1 (not localhost): httpx resolves localhost to IPv6 ::1 first, but
# sglang only binds IPv4. Numeric IP forces IPv4. Local server -> no proxy.
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY="dummy"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true

# ---- generate + evaluate -----------------------------------------------------
# Timestamped result/score dirs so sweeps don't collide; evaluate scans only this
# run's dir (whose single subdir is the registered model name).
RESULT_DIR="result/${MODE}${BUDGET_TAG}-${TS}"
SCORE_DIR="score/${MODE}${BUDGET_TAG}-${TS}"
cd "$BFCL_DIR"

echo ">>> BFCL endpoint=$OPENAI_BASE_URL result_dir=$RESULT_DIR score_dir=$SCORE_DIR"

if [[ "$EVAL_ONLY" != "1" ]]; then
  echo ">>> [bfcl] generate..."
  bfcl generate --model "$MODEL" --test-category "$CATEGORIES" \
    --num-threads "$NUM_THREADS" --result-dir "$RESULT_DIR" --skip-server-setup
fi

if [[ "$GEN_ONLY" != "1" ]]; then
  echo ">>> [bfcl] evaluate..."
  bfcl evaluate --model "$MODEL" --test-category "$CATEGORIES" \
    --result-dir "$RESULT_DIR" --score-dir "$SCORE_DIR"
fi

echo ""
echo "======== ALL DONE: bfcl $MODE budget=${BUDGET:-<default>} ========"
echo ">>> results: $BFCL_DIR/$RESULT_DIR/$MODEL/   scores: $BFCL_DIR/$SCORE_DIR/"
