#!/usr/bin/env python3
"""LongBench / LongBench-E prediction for SAS via the shared async sglang client.

Port of Focus's ``longbench_pred.py``, with the generation backend swapped from
in-process HF ``model.generate()`` (Accelerate data parallelism) to the shared
sglang HTTP client. Data parallelism is provided by the sglang server's
continuous batching + concurrent client requests — no Accelerate / multi-proc.

Prompt handling (matches SeerAttention / Focus):
  * Each dataset's raw prompt is built from ``dataset2prompt.json``.
  * Middle-truncation: prompts longer than ``max_length`` tokens keep the first
    and last halves (needs a tokenizer — loaded from --tokenizer, no model).
  * Chat template (server-side, "no-thinking"):
      - datasets NOT in NO_CHAT_TEMPLATE -> /v1/chat/completions with
        chat_template_kwargs={"enable_thinking": False}
      - datasets in NO_CHAT_TEMPLATE (trec/triviaqa/samsum/lsht/lcc/repobench-p)
        -> /v1/completions with the raw prompt (no template).
  * Greedy decoding: temperature=0. samsum additionally stops at "\n".

Output: one ``{dataset}.jsonl`` per dataset under the output dir, each line
``{pred, answers, all_classes, length}`` — consumed by longbench/score.py.

Usage (inside the SAS pixi env):
  python -m sas.eval.longbench.infer --output-dir results/longbench/... \
      --base-url http://127.0.0.1:30000 --model qwen3-seer-4b \
      --tokenizer /path/to/AttnGates --e
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from ..client import chat_completions, completions

# LongBench data is fetched by `datasets` from the HF Hub and cached under
# ~/.cache/huggingface — auto-downloaded on first use, reused from cache after.
# On a network-restricted machine where the cache is ALREADY present, the Hub
# lookup can hang until it times out on every dataset; pass HF_DATASETS_OFFLINE=1
# to skip the network and read straight from the cache (errors if not cached).

_CONFIG_DIR = Path(__file__).resolve().parent / "configs"

# Dataset lists (identical to SeerAttention / Focus)
DATASETS_E = [
    "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
    "gov_report", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]

DATASETS_ALL = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
    "hotpotqa", "2wikimqa", "musique", "dureader",
    "gov_report", "qmsum", "multi_news", "vcsum",
    "trec", "triviaqa", "samsum", "lsht",
    "passage_count", "passage_retrieval_en", "passage_retrieval_zh",
    "lcc", "repobench-p",
]

# Tasks where the chat template should NOT be applied (matching SeerAttention):
# these are few-shot / code-completion tasks fed as raw text.
NO_CHAT_TEMPLATE = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def build_prompt(tokenizer, json_obj: dict, prompt_format: str, max_length: int) -> str:
    """Format one sample's prompt and middle-truncate to ``max_length`` tokens.

    Middle-truncation keeps the first and last halves (matching SeerAttention):
    long-context tasks put the question at both ends, so eliding the middle
    preserves the instruction + query while dropping interior context.
    """
    prompt = prompt_format.format(**json_obj)
    tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized) > max_length:
        half = int(max_length / 2)
        prompt = (
            tokenizer.decode(tokenized[:half], skip_special_tokens=True)
            + tokenizer.decode(tokenized[-half:], skip_special_tokens=True)
        )
    return prompt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LongBench prediction via sglang (SAS).")
    p.add_argument("--output-dir", required=True, help="Output dir for {dataset}.jsonl files")
    p.add_argument("--base-url", default="http://127.0.0.1:30000",
                   help="sglang server base url (no trailing /v1)")
    p.add_argument("--model", required=True, help="served-model-name, e.g. qwen3-seer-4b")
    p.add_argument("--tokenizer", required=True,
                   help="path to tokenizer (seer: AttnGates dir; dense: base model)")
    p.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    p.add_argument("--max-prompt-length", type=int, default=31500,
                   help="Max prompt length in tokens for middle truncation")
    p.add_argument("--concurrency", type=int, default=256,
                   help="max in-flight requests to the server")
    p.add_argument("--limit", type=int, default=-1,
                   help="limit #samples per dataset (debug; -1 = all)")
    p.add_argument("--timeout", type=float, default=3600.0, help="per-request timeout (s)")
    p.add_argument("--max-retries", type=int, default=3)
    return p.parse_args()


async def predict_dataset(args, tokenizer, dataset, data, prompt_format, max_gen, max_length):
    """Build prompts for all samples in one dataset and generate via sglang.

    Returns a list of result dicts (same order as ``data``), one per sample.
    """
    prompts = [build_prompt(tokenizer, obj, prompt_format, max_length) for obj in data]

    # samsum tends to loop ("\nDialogue..."); stop at newline (matches Focus's
    # extra eos on samsum). No effect on other datasets.
    stop = ["\n"] if dataset == "samsum" else None

    if dataset in NO_CHAT_TEMPLATE:
        results = await completions(
            prompts, base_url=args.base_url, model=args.model, max_tokens=max_gen,
            temperature=0.0, top_p=1.0, concurrency=args.concurrency,
            timeout=args.timeout, max_retries=args.max_retries, stop=stop,
            desc=dataset,
        )
    else:
        messages_list = [[{"role": "user", "content": p}] for p in prompts]
        results = await chat_completions(
            messages_list, base_url=args.base_url, model=args.model, max_tokens=max_gen,
            temperature=0.0, top_p=1.0, concurrency=args.concurrency,
            timeout=args.timeout, max_retries=args.max_retries, stop=stop,
            chat_template_kwargs={"enable_thinking": False}, desc=dataset,
        )

    out = []
    for obj, (text, _n_tok) in zip(data, results):
        out.append({
            "pred": text,
            "answers": obj["answers"],
            "all_classes": obj["all_classes"],
            "length": obj["length"],
        })
    return out


async def run(args) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    max_length = args.max_prompt_length

    dataset2prompt = json.loads((_CONFIG_DIR / "dataset2prompt.json").read_text())
    dataset2maxlen = json.loads((_CONFIG_DIR / "dataset2maxlen.json").read_text())

    datasets = DATASETS_E if args.e else DATASETS_ALL
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[longbench] model={args.model} datasets={len(datasets)} "
          f"mode={'E' if args.e else 'full'} max_prompt_len={max_length}", flush=True)

    for dataset in datasets:
        cfg_name = f"{dataset}_e" if args.e else dataset
        data = load_dataset("THUDM/LongBench", cfg_name, split="test", trust_remote_code=True)
        data_all = [sample for sample in data]
        if args.limit > 0:
            data_all = data_all[: args.limit]

        print(f"\n======== {dataset} ({len(data_all)} samples, max_gen={dataset2maxlen[dataset]}) ========",
              flush=True)

        results = await predict_dataset(
            args, tokenizer, dataset, data_all,
            prompt_format=dataset2prompt[dataset],
            max_gen=dataset2maxlen[dataset],
            max_length=max_length,
        )

        out_path = os.path.join(args.output_dir, f"{dataset}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")
        print(f"  wrote {len(results)} predictions -> {out_path}", flush=True)

    print(f"\n[longbench] all predictions saved to: {args.output_dir}", flush=True)


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
