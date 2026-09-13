#!/usr/bin/env python3
"""Reasoning-task inference via the shared async sglang client.

Fires (num_questions x samples) chat-completion requests concurrently at an
already-running sglang server (started by run_eval.sh). The chat template is
applied server-side (we send `messages`, not a raw prompt).

Prompt construction:
  - math / aime24 / aime25 / olympiadbench: append the "reason step by step ...
    \\boxed{}" instruction to the user turn.
  - gpqa: no append (the instruction is already baked into the question text).

Output layout (consumed by score.py):
  <output-dir>/completions.jsonl   # one line per (question, sample):
                                   #   {"idx": int, "sample_id": int, "completion": str}
  <output-dir>/meta.json           # {"generate_lens": [...], "total_time": <minutes>, ...}

Usage (inside the SAS pixi env):
  python -m sas.eval.reasoning.infer --data-name aime24 --output-dir results/... \
      --base-url http://127.0.0.1:30000 --model qwen3-seer-4b --samples 64
"""
import argparse
import asyncio
import json
import os
import time

from ..client import chat_completions
from .utils.data_loader import load_data
from .utils.parser import parse_question

# Default reasoning data dir: <repo-root>/scripts/eval/data
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_DEFAULT_DATA_DIR = os.path.join(_REPO_ROOT, "scripts", "eval", "data")

# Tasks whose user turn needs the explicit "reason step by step ... \boxed{}"
# suffix (matches eval_vllm.py). gpqa already embeds its own instruction.
_BOXED_SUFFIX_TASKS = {"aime24", "aime25", "math", "olympiadbench"}
_BOXED_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def build_messages(example, data_name):
    """Build the OpenAI-style `messages` for one question (mirrors eval_vllm.py)."""
    question = parse_question(example, data_name)
    if data_name in _BOXED_SUFFIX_TASKS:
        content = question + _BOXED_SUFFIX
    else:
        content = question
    return [{"role": "user", "content": content}]


def parse_args():
    p = argparse.ArgumentParser(description="Async sglang client for reasoning eval.")
    p.add_argument("--data-name", required=True, help="aime24 | aime25 | math | gpqa | ...")
    p.add_argument("--data-dir", default=_DEFAULT_DATA_DIR)
    p.add_argument("--split", default="test")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:30000",
                   help="sglang server base url (no trailing /v1)")
    p.add_argument("--model", required=True, help="served-model-name, e.g. qwen3-seer-4b")
    p.add_argument("--samples", type=int, default=8, help="samples per question (pass@1 averaging)")
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--concurrency", type=int, default=256,
                   help="max in-flight requests to the server")
    p.add_argument("--limit", type=int, default=-1, help="limit number of questions (debug)")
    p.add_argument("--timeout", type=float, default=3600.0, help="per-request timeout (s)")
    p.add_argument("--max-retries", type=int, default=3)
    return p.parse_args()


async def run(args):
    examples = load_data(args.data_name, args.split, args.data_dir)
    if args.limit > 0:
        examples = examples[: args.limit]
    n_q = len(examples)
    print(f"[infer] task={args.data_name} questions={n_q} samples={args.samples} "
          f"model={args.model} -> {n_q * args.samples} requests", flush=True)

    # Flat list of messages + a parallel (idx, sample_id) list to reconstruct order.
    messages_list = []
    meta = []
    for idx, ex in enumerate(examples):
        messages = build_messages(ex, args.data_name)
        for sample_id in range(args.samples):
            messages_list.append(messages)
            meta.append((idx, sample_id))

    start = time.time()
    results = await chat_completions(
        messages_list,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        concurrency=args.concurrency,
        timeout=args.timeout,
        max_retries=args.max_retries,
        desc=args.data_name,
    )
    total_time = (time.time() - start) / 60.0

    # ---- write outputs ----
    os.makedirs(args.output_dir, exist_ok=True)
    completions_path = os.path.join(args.output_dir, "completions.jsonl")
    generate_lens = []
    with open(completions_path, "w") as f:
        for (idx, sample_id), (text, n_tok) in zip(meta, results):
            json.dump({"idx": idx, "sample_id": sample_id, "completion": text}, f)
            f.write("\n")
            if n_tok is not None:
                generate_lens.append(n_tok)

    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "data_name": args.data_name,
            "model": args.model,
            "num_questions": n_q,
            "samples": args.samples,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "generate_lens": generate_lens,
            "total_time": total_time,
        }, f, indent=2)

    n_failed = sum(1 for (t, _) in results if isinstance(t, str) and t.startswith("[REQUEST_FAILED"))
    print(f"[infer] done in {total_time:.1f} min, wrote {len(results)} completions "
          f"({n_failed} failed) -> {completions_path}", flush=True)


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
