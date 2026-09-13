#!/usr/bin/env bash
# LongBench-E eval driver via sglang.
#
# Starts an sglang server ONCE (see _serve.sh), then runs LongBench-E
# prediction (sas.eval.longbench.infer) + scoring (sas.eval.longbench.score)
# against that server, and tears the server down. Greedy decoding; metric is the
# official per-length-bucket LongBench-E score (averaged pass over datasets).
#
# The SERVER runs in the sglang-blocksparse pixi env; the CLIENT + scorer run in
# this repo's pixi env. Different envs on purpose.
#
# Examples:
#   GATES=/path/to/AttnGates TP=1 DP=8 bash run_longbench.sh
#   MODE=dense_4b BASE_MODEL=/path/to/Qwen3-4B TP=1 DP=8 bash run_longbench.sh
#   GATES=/path/to/AttnGates LIMIT=2 bash run_longbench.sh          # quick smoke test
#
# LongBench data is auto-downloaded from the HF Hub on first use and cached. On a
# network-restricted machine where the cache is ALREADY present, the per-dataset
# Hub lookup can hang until timeout; prepend HF_DATASETS_OFFLINE=1 to read
# straight from the cache instead:
#   HF_DATASETS_OFFLINE=1 GATES=/path/to/AttnGates bash run_longbench.sh
#
# Common server knobs (GATES/BUDGET/TP/DP/MODE/BASE_MODEL/PORT/GPUS/...) are
# documented in _serve.sh. LongBench-specific knobs below.

set -euo pipefail

# ---- longbench-specific config -----------------------------------------------
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-31500}"   # middle-truncation length (tokens)

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_serve.sh
source "$_SCRIPT_DIR/_serve.sh"

# ---- validate + start server -------------------------------------------------
sas_parse_config
echo ">>> task=longbench mode=$MODE model=$MODEL_NAME budget=${BUDGET:-<default>} tp=${TP:-<default>} dp=${DP:-<default>}"
sas_start_server

# ---- inference + scoring (repo pixi env) -------------------------------------
OUTPUT_DIR="$OUTPUT_ROOT/longbench-${MODE}${BUDGET_TAG}-${TS}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "======== TASK: longbench  (LongBench-E, greedy) ========"
echo ">>> output=$OUTPUT_DIR tokenizer=$TOKENIZER_PATH"

echo ">>> [longbench] inference..."
pixi run --manifest-path "$PIXI_MANIFEST" python -m sas.eval.longbench.infer \
  --output-dir "$OUTPUT_DIR" \
  --base-url "$BASE_URL" \
  --model "$MODEL_NAME" \
  --tokenizer "$TOKENIZER_PATH" \
  --max-prompt-length "$MAX_PROMPT_LENGTH" \
  --concurrency "$CONCURRENCY" \
  --limit "$LIMIT" \
  --e

echo ">>> [longbench] scoring..."
pixi run --manifest-path "$PIXI_MANIFEST" python -m sas.eval.longbench.score \
  --pred-dir "$OUTPUT_DIR" \
  --e

# ---- final summary -----------------------------------------------------------
echo ""
echo "======== ALL DONE: $MODE budget=${BUDGET:-<default>} ========"
if [[ -f "$OUTPUT_DIR/result.json" ]]; then
  avg=$(grep -m1 '"averaged"' "$OUTPUT_DIR/result.json" || true)
  echo "  LongBench-E averaged: ${avg}"
fi
echo ">>> results under $OUTPUT_DIR"
