#!/usr/bin/env python3
"""Score reasoning-task completions produced by reasoning/infer.py.

Reuses SeerAttention's grading logic verbatim (utils/parser + utils/grader):
  - parse_ground_truth(example)  -> gold answer string
  - extract_answer(completion)   -> predicted answer from the model's \\boxed{}
  - check_is_correct(pred, gt)   -> bool (math_equal under the hood)

Metric: **avg pass@1** — for each question, accuracy = (#correct samples / #samples);
then average over questions.

Input : <output-dir>/completions.jsonl  (lines: {idx, sample_id, completion})
        <output-dir>/meta.json          (generate_lens, total_time, ...)
Output: <output-dir>/summary.txt        (+ prints Acc)

Usage (inside the SAS pixi env):
  python -m sas.eval.reasoning.score --data-name aime24 --output-dir results/...
"""
import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutTimeout

from .utils.data_loader import load_data
from .utils.parser import parse_ground_truth, extract_answer
from .utils.grader import check_is_correct, choice_answer_clean

# Default reasoning data dir: <repo-root>/scripts/eval/data
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_DEFAULT_DATA_DIR = os.path.join(_REPO_ROOT, "scripts", "eval", "data")

# Multiple-choice tasks: judge by cleaned letter (A-E), NOT sympy math_equal.
# gpqa answers are single letters; running them through math_equal's
# multiprocessing-timeout path is both needlessly slow and prone to hanging
# (call_with_timeout's Queue.get() can block forever after terminate()).
_MC_TASKS = {"gpqa"}


def _judge_one(args_tuple):
    """Judge a single (pred_answer, gt, is_mc). Runs in a worker process.

    For math tasks we call check_is_correct with timeout=False here (the outer
    ProcessPoolExecutor provides the real per-item timeout), avoiding the fragile
    nested-multiprocessing timeout inside grader.call_with_timeout.
    """
    pred, gt, is_mc = args_tuple
    try:
        if is_mc:
            return choice_answer_clean(pred) == choice_answer_clean(gt)
        return check_is_correct(pred, gt, timeout=False)
    except Exception:
        return False


def parse_args():
    p = argparse.ArgumentParser(description="Score reasoning completions (avg pass@1).")
    p.add_argument("--data-name", required=True)
    p.add_argument("--data-dir", default=_DEFAULT_DATA_DIR)
    p.add_argument("--split", default="test")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()

    examples = load_data(args.data_name, args.split, args.data_dir)
    if args.limit > 0:
        examples = examples[: args.limit]

    completions_path = os.path.join(args.output_dir, "completions.jsonl")
    # group completions by question idx
    by_idx = defaultdict(list)
    with open(completions_path) as f:
        for line in f:
            item = json.loads(line)
            by_idx[item["idx"]].append(item["completion"])

    is_mc = args.data_name in _MC_TASKS

    # Build a flat job list: (question_idx, extracted_pred, gt, is_mc).
    # Extraction is cheap and done inline; the (possibly slow) equality check
    # runs in a process pool with a hard per-item timeout so one pathological
    # sympy parse can never hang the whole scorer.
    jobs = []          # list of (q_idx, pred, gt, is_mc)
    q_nsamples = {}    # q_idx -> number of samples
    for idx, ex in enumerate(examples):
        _, gt = parse_ground_truth(ex, args.data_name)
        comps = by_idx.get(idx, [])
        q_nsamples[idx] = len(comps)
        for c in comps:
            pred = extract_answer(c)
            jobs.append((idx, pred, gt, is_mc))

    # Judge all jobs. MC tasks are pure-python fast; math tasks may be slow, so
    # run in a pool with a per-item timeout (treat timeouts as incorrect).
    correct_per_q = defaultdict(int)
    PER_ITEM_TIMEOUT = 15.0  # seconds; math_equal on a single item should be well under this
    if is_mc:
        # fast path, no pool needed
        for (idx, pred, gt, _mc) in jobs:
            if _judge_one((pred, gt, True)):
                correct_per_q[idx] += 1
    else:
        with ProcessPoolExecutor(max_workers=min(32, (os.cpu_count() or 8))) as ex_pool:
            futs = {ex_pool.submit(_judge_one, (pred, gt, False)): idx
                    for (idx, pred, gt, _mc) in jobs}
            done = 0
            for fut in list(futs):
                idx = futs[fut]
                try:
                    ok = fut.result(timeout=PER_ITEM_TIMEOUT)
                except (FutTimeout, Exception):
                    ok = False
                if ok:
                    correct_per_q[idx] += 1
                done += 1
                if done % 200 == 0:
                    print(f"[score] judged {done}/{len(jobs)}", flush=True)

    per_q_acc = []
    n_samples_seen = []
    for idx in range(len(examples)):
        n = q_nsamples.get(idx, 0)
        n_samples_seen.append(n)
        per_q_acc.append((correct_per_q[idx] / n) if n else 0.0)

    acc = sum(per_q_acc) / len(per_q_acc) if per_q_acc else 0.0
    # Standard deviation and standard error of the mean over per-question pass@1.
    # SEM = std / sqrt(#questions); this is the error bar to report on the acc.
    n_q = len(per_q_acc)
    if n_q > 1:
        var = sum((a - acc) ** 2 for a in per_q_acc) / (n_q - 1)  # sample variance
        std = var ** 0.5
        sem = std / (n_q ** 0.5)
    else:
        std = 0.0
        sem = 0.0

    # generation-length / timing stats (best-effort, from meta.json)
    meta = {}
    meta_path = os.path.join(args.output_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    gen_lens = meta.get("generate_lens") or []
    avg_len = sum(gen_lens) / len(gen_lens) if gen_lens else 0.0
    max_len = max(gen_lens) if gen_lens else 0
    total_time = meta.get("total_time", 0.0)

    summary = (
        f"Task: {args.data_name}\n"
        f"Model: {meta.get('model', '?')}\n"
        f"Questions: {len(per_q_acc)}\n"
        f"Samples/question: {meta.get('samples', n_samples_seen[0] if n_samples_seen else '?')}\n"
        f"Acc (avg pass@1): {acc:.4f}\n"
        f"Std (over questions): {std:.4f}\n"
        f"SEM: {sem:.4f}\n"
        f"Acc +/- SEM: {acc*100:.2f} +/- {sem*100:.2f} (%)\n"
        f"Average generate length: {avg_len:.1f}\n"
        f"Max generate length: {max_len}\n"
        f"Total inference time (min): {total_time:.2f}\n"
    )
    print(summary)
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"[score] wrote {summary_path}")


if __name__ == "__main__":
    main()
