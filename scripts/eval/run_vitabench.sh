#!/usr/bin/env bash
# VitaBench eval driver via sglang.
#
# Starts an sglang server ONCE (see _serve.sh) as the AGENT under test, then runs
# `vita run` against it, and tears the server down. The user-simulator and rubric
# evaluator are CLOUD models (DeepSeek); the local agent (127.0.0.1) stays in
# no_proxy so it never goes through a proxy. If your network needs a proxy for the
# outbound cloud calls, set PROXY_URL=http://<host>:<port>.
#
# VitaBench lives in the third_party/vitabench submodule (official repo). One-time
# prerequisites (see README) that this script only CHECKS, never mutates:
#   1. patch:  git -C third_party/vitabench apply scripts/eval/patches/vitabench.patch
#   2. venv:   python -m venv third_party/vitabench/.venv && <.venv>/bin/pip install -e third_party/vitabench
# And at run time you MUST provide DEEPSEEK_API_KEY (for the cloud user/evaluator).
#
# Examples:
#   DEEPSEEK_API_KEY=sk-... GATES=/path/to/AttnGates TP=1 DP=8 bash run_vitabench.sh
#   NUM_TASKS=2 DOMAIN=delivery DEEPSEEK_API_KEY=sk-... GATES=/path/to/AttnGates bash run_vitabench.sh   # smoke
#   PROXY_URL=http://<host>:<port> DEEPSEEK_API_KEY=sk-... GATES=/path/to/AttnGates bash run_vitabench.sh  # behind a proxy
#
# Common server knobs (GATES/BUDGET/TP/DP/MODE/BASE_MODEL/PORT/GPUS/...) are
# documented in _serve.sh. VitaBench-specific knobs below.

set -euo pipefail

# Agentic benches run long contexts -> enable YaRN 64k by default (runtime rope
# override; on-disk gate config is untouched). Set YARN64K=0 to disable.
export YARN64K="${YARN64K:-1}"

# ---- VitaBench-specific config -----------------------------------------------
DOMAIN="${DOMAIN:-delivery}"              # delivery | instore | ota | "[delivery,instore,ota]" (cross)
NUM_TASKS="${NUM_TASKS:-}"                # limit #tasks (empty = all; smoke: 2)
NUM_TRIALS="${NUM_TRIALS:-4}"            # trials per task
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"  # concurrent simulations
LANGUAGE="${LANGUAGE:-chinese}"          # chinese | english
USER_LLM="${USER_LLM:-deepseek-v4-pro}"     # user-simulator model (entry in models.yaml)
EVALUATOR_LLM="${EVALUATOR_LLM:-deepseek-v4-pro}"  # rubric evaluator model
# Optional HTTP(S) proxy for the CLOUD user-simulator / evaluator calls. Empty by
# default = connect directly; set it only if your network needs a proxy for
# outbound HTTPS. The local agent (127.0.0.1) always bypasses it via no_proxy.
PROXY_URL="${PROXY_URL:-}"
VITA_DIR="${VITA_DIR:-}"                  # vitabench checkout (override if relocated)

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_serve.sh
source "$_SCRIPT_DIR/_serve.sh"

# ---- require the cloud API key -----------------------------------------------
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "ERROR: DEEPSEEK_API_KEY is required (cloud user-simulator + evaluator)." >&2
  echo "       export DEEPSEEK_API_KEY=sk-... then re-run." >&2
  exit 1
fi

# ---- locate vitabench + one-time-setup checks --------------------------------
VITA_DIR="${VITA_DIR:-$SAS_DIR/third_party/vitabench}"
PATCH="$_SCRIPT_DIR/patches/vitabench.patch"

if [[ ! -d "$VITA_DIR" ]]; then
  echo "ERROR: vitabench not found at $VITA_DIR" >&2
  echo "       Run: git submodule update --init third_party/vitabench" >&2
  exit 1
fi
if ! git -C "$VITA_DIR" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  echo "ERROR: vitabench patch not applied." >&2
  echo "       Run once: git -C third_party/vitabench apply $PATCH" >&2
  exit 1
fi

# ---- validate + start server (local agent, no proxy) -------------------------
sas_parse_config
AGENT_MODEL="$MODEL_NAME"    # vita agent name = qwen3-<be>-<size> (matches served-model-name)
echo ">>> task=vitabench agent=$AGENT_MODEL domain=$DOMAIN mode=$MODE budget=${BUDGET:-<default>} tp=${TP:-<default>} dp=${DP:-<default>}"

# Rendered models.yaml holds the real DEEPSEEK_API_KEY -> write it to a PRIVATE
# temp file (600), never under results/, and register it for removal in the
# server-teardown trap (SAS_CLEANUP_FILES, honored by _serve.sh cleanup).
RENDERED_YAML="$(mktemp -t vita_models.XXXXXX.yaml)"
chmod 600 "$RENDERED_YAML"
declare -a SAS_CLEANUP_FILES=("$RENDERED_YAML")

sas_start_server

# ---- write models.yaml (agent=local sglang, user/evaluator=cloud DeepSeek) ---
# Inlined here (no separate template): bash expands $AGENT_MODEL/$PORT/$DEEPSEEK_API_KEY.
# Holds the real key -> the PRIVATE temp file (600) created above; removed on exit.
# 127.0.0.1 (not localhost): sglang binds IPv4 only.
cat > "$RENDERED_YAML" <<EOF
default:
  temperature: 0.0
  max_input_tokens: 32768
  headers:
    Content-Type: "application/json"

models:
  # agent under test -> local sglang server (name MUST match --served-model-name)
  - name: $AGENT_MODEL
    base_url: "http://127.0.0.1:${PORT}/v1/chat/completions"
    max_tokens: 8192
    max_input_tokens: 65536
    headers:
      Authorization: "Bearer dummy"
      Content-Type: "application/json"

  # user-simulator + evaluator -> cloud DeepSeek (one entry serves both roles)
  - name: deepseek-v4-pro
    base_url: "https://api.deepseek.com/chat/completions"
    max_tokens: 8192
    max_input_tokens: 65536
    thinking:
      type: "disabled"
    headers:
      Authorization: "Bearer ${DEEPSEEK_API_KEY}"
      Content-Type: "application/json"
EOF
export VITA_MODEL_CONFIG_PATH="$RENDERED_YAML"

# ---- activate vita venv ------------------------------------------------------
# venv lives inside the submodule (VITA_VENV overrides; default .venv).
if ! command -v vita >/dev/null 2>&1; then
  VENV_ACT="${VITA_VENV:-$VITA_DIR/.venv}/bin/activate"
  if [[ -f "$VENV_ACT" ]]; then
    echo ">>> activating vita venv: $VENV_ACT"
    set +u; # shellcheck disable=SC1090
    source "$VENV_ACT"; set -u
  else
    echo "ERROR: vita venv not found at $VENV_ACT" >&2
    echo "       Create it once: python -m venv $VITA_DIR/.venv && \\" >&2
    echo "                       source $VITA_DIR/.venv/bin/activate && \\" >&2
    echo "                       pip install -e $VITA_DIR" >&2
    exit 1
  fi
fi

# ---- proxy: cloud user/evaluator via proxy, local agent bypasses -------------
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="$no_proxy"
if [[ -n "${PROXY_URL:-}" ]]; then
  export http_proxy="$PROXY_URL" https_proxy="$PROXY_URL"
  export HTTP_PROXY="$PROXY_URL" HTTPS_PROXY="$PROXY_URL"
fi

# ---- run -----------------------------------------------------------------------
SAVE_TO="${MODE}${BUDGET_TAG}-${TS}_${DOMAIN//[^a-zA-Z0-9]/_}.json"
cd "$VITA_DIR"
echo ">>> vita run domain=$DOMAIN agent=$AGENT_MODEL user=$USER_LLM evaluator=$EVALUATOR_LLM"
echo ">>> config=$VITA_MODEL_CONFIG_PATH save_to=data/simulations/$SAVE_TO"

# auto-answer vita's "resume? (y/n)" prompt with yes
yes 2>/dev/null | vita run \
  --domain "$DOMAIN" \
  --agent-llm "$AGENT_MODEL" \
  --user-llm "$USER_LLM" \
  --evaluator-llm "$EVALUATOR_LLM" \
  --num-trials "$NUM_TRIALS" \
  ${NUM_TASKS:+--num-tasks "$NUM_TASKS"} \
  --max-concurrency "$MAX_CONCURRENCY" \
  --language "$LANGUAGE" \
  --save-to "$SAVE_TO"

echo ""
echo "======== ALL DONE: vitabench $MODE domain=$DOMAIN ========"
echo ">>> results: $VITA_DIR/data/simulations/$SAVE_TO"
echo ">>> browse:  vita view --file data/simulations/$SAVE_TO"
