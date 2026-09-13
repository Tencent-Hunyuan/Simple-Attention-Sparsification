#!/usr/bin/env bash
# Reasoning eval driver (aime24 | aime25 | gpqa | math) via sglang.
#
# Starts an sglang server ONCE (see _serve.sh), then for each task runs
# the async client (sas.eval.reasoning.infer) + scorer (sas.eval.reasoning.score)
# against that server, and tears the server down. Metric: averaged pass@1.
#
# The SERVER runs in the sglang-blocksparse pixi env; the CLIENT + scorer run in
# this repo's pixi env. Different envs on purpose.
#
# Examples:
#   TASK=aime24 GATES=/path/to/AttnGates BUDGET=2048 TP=1 DP=8 bash run_reasoning.sh
#   TASK=math,gpqa,aime24 GATES=/path/to/AttnGates TP=1 DP=8 bash run_reasoning.sh
#   TASK=math MODE=dense_4b BASE_MODEL=/path/to/Qwen3-4B bash run_reasoning.sh
#   TASK=aime24 LIMIT=2 SAMPLES=2 GATES=/path/to/AttnGates bash run_reasoning.sh
#
# Common server knobs (GATES/BUDGET/TP/DP/MODE/BASE_MODEL/PORT/GPUS/...) are
# documented in _serve.sh. Reasoning-specific knobs below.

set -euo pipefail

# ---- reasoning-specific config -----------------------------------------------
TASK="${TASK:-aime24}"                 # aime24 | aime25 | gpqa | math  (comma-separated for many)
SAMPLES="${SAMPLES:-}"                 # samples/question for ALL tasks (empty -> per-task default)
MAX_TOKENS="${MAX_TOKENS:-}"           # max gen tokens for ALL tasks   (empty -> 32768)
TEMPERATURE="${TEMPERATURE:-0.6}"      # sampling temperature
TOP_P="${TOP_P:-0.95}"                 # sampling top-p
MAX_TOKENS_DEFAULT=32768

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_serve.sh
source "$_SCRIPT_DIR/_serve.sh"

task_samples() {
  case "$1" in
    aime24|aime25) echo 64 ;;
    gpqa)          echo 16 ;;
    math)          echo 8  ;;
    *) echo "ERROR: unknown reasoning task '$1' (use aime24|aime25|gpqa|math)" >&2; return 1 ;;
  esac
}

# ---- validate + start server -------------------------------------------------
IFS=',' read -r -a TASK_ARR <<< "$TASK"
for t in "${TASK_ARR[@]}"; do task_samples "$t" >/dev/null; done  # validate up front

sas_parse_config
echo ">>> task=(${TASK_ARR[*]}) mode=$MODE model=$MODEL_NAME budget=${BUDGET:-<default>} tp=${TP:-<default>} dp=${DP:-<default>}"
sas_start_server
echo ">>> Running ${#TASK_ARR[@]} reasoning task(s): ${TASK_ARR[*]}"

# ---- per-task inference + scoring (repo pixi env) ----------------------------
declare -a SUMMARY_DIRS=()
for t in "${TASK_ARR[@]}"; do
  SAMPLES_T="${SAMPLES:-$(task_samples "$t")}"
  MAX_TOKENS_T="${MAX_TOKENS:-$MAX_TOKENS_DEFAULT}"
  OUTPUT_DIR="$OUTPUT_ROOT/${t}-${MODE}${BUDGET_TAG}-${TS}"
  mkdir -p "$OUTPUT_DIR"
  SUMMARY_DIRS+=("$OUTPUT_DIR")

  echo ""
  echo "======== TASK: $t  (samples=$SAMPLES_T max_tokens=$MAX_TOKENS_T) ========"
  echo ">>> output=$OUTPUT_DIR"

  echo ">>> [$t] inference..."
  pixi run --manifest-path "$PIXI_MANIFEST" python -m sas.eval.reasoning.infer \
    --data-name "$t" \
    --output-dir "$OUTPUT_DIR" \
    --base-url "$BASE_URL" \
    --model "$MODEL_NAME" \
    --samples "$SAMPLES_T" \
    --max-tokens "$MAX_TOKENS_T" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --concurrency "$CONCURRENCY" \
    --limit "$LIMIT"

  echo ">>> [$t] scoring..."
  pixi run --manifest-path "$PIXI_MANIFEST" python -m sas.eval.reasoning.score \
    --data-name "$t" \
    --output-dir "$OUTPUT_DIR" \
    --limit "$LIMIT"
done

# ---- final aggregate summary -------------------------------------------------
echo ""
echo "======== ALL DONE: $MODE budget=${BUDGET:-<default>} ========"
for d in "${SUMMARY_DIRS[@]}"; do
  if [[ -f "$d/summary.txt" ]]; then
    task=$(grep -m1 '^Task:' "$d/summary.txt" | awk '{print $2}')
    acc=$(grep -m1 '^Acc +/- SEM:' "$d/summary.txt" | cut -d: -f2-)
    printf '  %-10s %s\n' "$task" "${acc# }"
  fi
done
echo ">>> results under $OUTPUT_ROOT (per-task subdirs, tag ${MODE}${BUDGET_TAG}-${TS})"
