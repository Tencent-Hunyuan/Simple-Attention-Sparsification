# SAS: Simple Attention Sparsification via End-to-End Optimization of Context Ranking

SAS is a gated sparse-attention mechanism that learns, per query, which blocks to attend to — optimizing the context ranking **end-to-end**
with the language-modeling loss instead of distilling the original model's dense
attention.

## Setup
### Install via pixi

Get [pixi](https://pixi.sh) if you don't have it, then install the env + submodules:

```bash
curl -fsSL https://pixi.sh/install.sh | bash          # install pixi (skip if already present)
pixi install && git submodule update --init --recursive
```

## Training

```bash
export MODEL_PATH=/path/to/Qwen3-4B
export DATA_PATH=/path/to/openr1-math
bash scripts/train/simple_sparse_attention_Qwen3-4B.sh   # also: 8B / 14B
```

## Evaluation

Eval serves the model through the sglang-blocksparse, so build its env once first:

```bash
cd third_party/sglang-blocksparse && pixi install && cd -
```

> **Note.** `sglang-blocksparse` is our [SGLang](https://github.com/sgl-project/sglang)
> fork, published at [rayleizhu/sglang](https://github.com/rayleizhu/sglang) and fetched
> via the submodule above. Training does not need it; only the eval scripts do. If you
> already have a checkout elsewhere, point the scripts at it with
> `SGLANG_DIR=/path/to/sglang-blocksparse`.

Each script starts a sglang server once, runs the benchmark, and tears it down.
Point `GATES` at an exported `AttnGates` dir (or `MODE=dense_4b BASE_MODEL=...` for
the dense baseline); `BUDGET` is the seer decode token budget; `TP=1 DP=8` gives 8
full replicas.

### 1. Reasoning Task - MATH, GPQA-Diamond, AIME24/25

```bash
export GATES=/path/to/AttnGates    # or: export MODE=dense_4b BASE_MODEL=/path/to/Qwen3-4B
export BUDGET=2048
export TP=1 DP=8
export TASK=math,gpqa,aime24,aime25
bash scripts/eval/run_reasoning.sh
```

### 2. Long Context Understanding Task - LongBench-E

```bash
export GATES=/path/to/AttnGates    # or: export MODE=dense_4b BASE_MODEL=/path/to/Qwen3-4B
export BUDGET=2048
export TP=1 DP=8
bash scripts/eval/run_longbench.sh
```

### 3. Agentic Task - BFCL, VitaBench

Function-calling ([BFCL](https://github.com/ShishirPatil/gorilla)) and multi-domain
tool-use ([VitaBench](https://github.com/meituan-longcat/vitabench)). These are large
upstream benchmarks kept as submodules under `third_party/` and run in their **own
venvs**; SAS only starts the sglang server and drives their CLIs.

One-time setup (per benchmark): init the submodule, build its venv, apply our patch
(the patch registers the qwen3 models / adjusts the client — kept as a file, not
vendored source):

```bash
git submodule update --init third_party/gorilla third_party/vitabench

# BFCL: own venv (.venv) inside the submodule
# (soundfile is a missing transitive dep of qwen_agent — install it explicitly)
BFCL=third_party/gorilla/berkeley-function-call-leaderboard
python -m venv $BFCL/.venv && $BFCL/.venv/bin/pip install -e $BFCL soundfile
git -C third_party/gorilla apply "$PWD/scripts/eval/patches/bfcl.patch"

# VitaBench: own venv (.venv) inside the submodule
python -m venv third_party/vitabench/.venv && third_party/vitabench/.venv/bin/pip install -e third_party/vitabench
git -C third_party/vitabench apply "$PWD/scripts/eval/patches/vitabench.patch"
```

The run scripts auto-activate these venvs (override with `BFCL_VENV` / `VITA_VENV`).

**BFCL** (function-calling):

```bash
export GATES=/path/to/AttnGates    # or: export MODE=dense_4b BASE_MODEL=/path/to/Qwen3-4B
export BUDGET=2048
export TP=1 DP=8
export CATEGORIES=multi_turn
bash scripts/eval/run_bfcl.sh
```

**VitaBench** (multi-domain tool-use) 
+ This benchmark additionally needs a cloud user-simulator and evaluator (we use DeepSeek-V4-Pro for both), so export `DEEPSEEK_API_KEY`:

```bash
export GATES=/path/to/AttnGates    # or: export MODE=dense_14b BASE_MODEL=/path/to/Qwen3-14B
export BUDGET=2048
export TP=1 DP=8
export DEEPSEEK_API_KEY=sk-...
export DOMAIN=delivery,instore,ota
bash scripts/eval/run_vitabench.sh
```

## Acknowledgements

- [SeerAttention-R](https://github.com/microsoft/SeerAttention) — gate architecture, and grading logic.
- [VeOmni](https://github.com/ByteDance-Seed/VeOmni) — training backbone.
- [sglang-blocksparse](https://github.com/rayleizhu/sglang) — our [SGLang](https://github.com/sgl-project/sglang) fork providing the `seer_attn` block-sparse backend used for eval.
- [BFCL / Gorilla](https://github.com/ShishirPatil/gorilla) and [VitaBench](https://github.com/meituan-longcat/vitabench) — function-calling / agent tool-use benchmarks.
