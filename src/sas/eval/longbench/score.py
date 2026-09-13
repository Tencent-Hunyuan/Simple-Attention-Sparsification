#!/usr/bin/env python3
"""LongBench / LongBench-E scoring for SAS.

Port of SeerAttention's ``eval.py`` (via Focus's ``longbench_eval.py``).
Single-process: reads JSONL prediction files produced by
``sas.eval.longbench.infer`` and scores them.

Usage (inside the SAS pixi env):
  python -m sas.eval.longbench.score --pred-dir results/longbench/... --e
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .metrics import (
    qa_f1_score,
    rouge_score,
    classification_score,
    retrieval_score,
    count_score,
    code_sim_score,
    rouge_zh_score,
    qa_f1_zh_score,
    retrieval_zh_score,
)

# ---------------------------------------------------------------------------
# Dataset → metric mapping (identical to SeerAttention's eval.py)
# ---------------------------------------------------------------------------
dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

# ---------------------------------------------------------------------------
# Official LongBench task taxonomy: 13 (LongBench-E) / 21 (full) tasks group
# into 6 categories.  Used to print a compact category-level summary on top of
# the per-dataset scores.
# ---------------------------------------------------------------------------
dataset2category = {
    # Single-Document QA
    "narrativeqa": "Single-Doc QA",
    "qasper": "Single-Doc QA",
    "multifieldqa_en": "Single-Doc QA",
    "multifieldqa_zh": "Single-Doc QA",
    # Multi-Document QA
    "hotpotqa": "Multi-Doc QA",
    "2wikimqa": "Multi-Doc QA",
    "musique": "Multi-Doc QA",
    "dureader": "Multi-Doc QA",
    # Summarization
    "gov_report": "Summarization",
    "qmsum": "Summarization",
    "multi_news": "Summarization",
    "vcsum": "Summarization",
    # Few-shot Learning
    "trec": "Few-shot",
    "triviaqa": "Few-shot",
    "samsum": "Few-shot",
    "lsht": "Few-shot",
    # Synthetic
    "passage_count": "Synthetic",
    "passage_retrieval_en": "Synthetic",
    "passage_retrieval_zh": "Synthetic",
    # Code
    "lcc": "Code",
    "repobench-p": "Code",
}

# Display order for the category summary table.
CATEGORY_ORDER = [
    "Single-Doc QA",
    "Multi-Doc QA",
    "Summarization",
    "Few-shot",
    "Synthetic",
    "Code",
]


# ---------------------------------------------------------------------------
# Scorer functions (exact port from SeerAttention)
# ---------------------------------------------------------------------------
def scorer_e(dataset: str, predictions: list, answers: list, lengths: list, all_classes: list) -> dict:
    """Per-length-bucket scoring for LongBench-E."""
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for prediction, ground_truths, length in zip(predictions, answers, lengths):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores:
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores


def scorer(dataset: str, predictions: list, answers: list, all_classes: list) -> float:
    """Overall scoring for non-E mode."""
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def calculate_averaged_scores(all_scores: dict) -> dict:
    """Average across all datasets (for LongBench-E)."""
    avg_scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for dataset, score in all_scores.items():
        if isinstance(score, dict):
            avg_scores["0-4k"].append(score.get("0-4k", 0))
            avg_scores["4-8k"].append(score.get("4-8k", 0))
            avg_scores["8k+"].append(score.get("8k+", 0))

    averaged_results = {}
    for key, values in avg_scores.items():
        if values:
            averaged_results[key] = round(np.mean(values), 2)
        else:
            averaged_results[key] = 0.0
    return averaged_results


def calculate_category_scores(all_scores: dict, is_e: bool) -> dict:
    """Group per-dataset scores into the 6 official LongBench categories.

    A category score is the unweighted mean of its member datasets' scores
    (per length bucket for LongBench-E, a single float otherwise), matching how
    ``calculate_averaged_scores`` averages across datasets.
    """
    if is_e:
        buckets = ["0-4k", "4-8k", "8k+"]
        grouped = {cat: {b: [] for b in buckets} for cat in CATEGORY_ORDER}
        for dataset, score in all_scores.items():
            cat = dataset2category.get(dataset)
            if cat is None or not isinstance(score, dict):
                continue
            for b in buckets:
                if b in score:
                    grouped[cat][b].append(score[b])
        category_scores = {}
        for cat in CATEGORY_ORDER:
            per_bucket = grouped[cat]
            if not any(per_bucket.values()):
                continue
            entry = {}
            all_vals = []
            for b in buckets:
                if per_bucket[b]:
                    entry[b] = round(float(np.mean(per_bucket[b])), 2)
                    all_vals.extend(per_bucket[b])
            entry["avg"] = round(float(np.mean(all_vals)), 2) if all_vals else 0.0
            category_scores[cat] = entry
        return category_scores

    grouped = {cat: [] for cat in CATEGORY_ORDER}
    for dataset, score in all_scores.items():
        cat = dataset2category.get(dataset)
        if cat is None or not isinstance(score, (int, float)):
            continue
        grouped[cat].append(score)
    return {
        cat: round(float(np.mean(vals)), 2)
        for cat, vals in grouped.items()
        if vals
    }


def print_category_summary(category_scores: dict, is_e: bool) -> None:
    """Pretty-print the category-level summary table."""
    if not category_scores:
        return
    print("\n  === By category ===")
    if is_e:
        header = f"  {'Category':<16}{'0-4k':>8}{'4-8k':>8}{'8k+':>8}{'avg':>8}"
        print(header)
        for cat in CATEGORY_ORDER:
            if cat not in category_scores:
                continue
            e = category_scores[cat]
            print(
                f"  {cat:<16}"
                f"{e.get('0-4k', 0):>8}{e.get('4-8k', 0):>8}"
                f"{e.get('8k+', 0):>8}{e.get('avg', 0):>8}"
            )
    else:
        for cat in CATEGORY_ORDER:
            if cat in category_scores:
                print(f"  {cat:<16}{category_scores[cat]:>8}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LongBench scoring (SAS).")
    parser.add_argument(
        "--pred-dir", type=str, required=True,
        help="Directory containing {dataset}.jsonl prediction files",
    )
    parser.add_argument("--e", action="store_true", help="Score using LongBench-E (per-length-bucket)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = args.pred_dir

    scores = {}
    all_files = os.listdir(pred_dir)
    print(f"Evaluating predictions in: {pred_dir}")
    print(f"Found files: {[f for f in all_files if f.endswith('.jsonl')]}")

    for filename in sorted(all_files):
        if not filename.endswith(".jsonl"):
            continue

        dataset = filename.split(".")[0]
        if dataset not in dataset2metric:
            print(f"  [SKIP] Unknown dataset: {dataset}")
            continue

        predictions, answers, lengths = [], [], []
        all_classes = []

        with open(os.path.join(pred_dir, filename), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])

        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score

        print(f"  {dataset}: {score}")

    # Compute averaged scores
    if args.e:
        averaged_results = calculate_averaged_scores(scores)
        category_scores = calculate_category_scores(scores, is_e=True)
        scores["averaged"] = averaged_results
        scores["categories"] = category_scores
        print(f"\n  Averaged: {averaged_results}")
        print_category_summary(category_scores, is_e=True)
    else:
        if scores:
            category_scores = calculate_category_scores(scores, is_e=False)
            avg = round(np.mean(list(scores.values())), 2)
            scores["averaged"] = avg
            scores["categories"] = category_scores
            print(f"\n  Averaged: {avg}")
            print_category_summary(category_scores, is_e=False)

    # Write result.json
    out_path = os.path.join(pred_dir, "result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
