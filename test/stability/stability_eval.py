#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stability_eval.py — Runs multiple independent evaluation rounds on a given answer file
to analyze the stability of the LLM judge model.

Features:
  - By default calls a remote scoring model (no need to start vLLM)
  - For each answer file, runs N rounds (default 5) of independent evaluation, scoring
    each round independently with the same rubric
  - Saves each round's result separately as JSON (format matches score_responses.py output,
    convenient for direct reuse)
  - Aggregates stability metrics and saves summary.json:
      * Per-round mean scores, overall mean and standard deviation
      * Between-round std (round_std) and between-round range (round_range)
      * Per-sample cross-round std, and the fraction of fully stable (std==0) samples
      * Rubric decision cross-round agreement rate (fraction of rubrics whose scores are
        identical across all 5 rounds)
      * Per-round mean scores for the cs / econ / physics categories

Dependencies: reuses the scoring pipeline from score_responses.py
(LLMClient / llm_score / _score_single_item).

Usage:
    python stability_eval.py                                        # solution + experiment, 5 rounds each
    python stability_eval.py --folders solution                     # solution only
    python stability_eval.py --rounds 5 --temperature 0.7           # custom rounds and sampling temperature
    python stability_eval.py --answer-name qwen3_4b_responses       # specify answer filename (without .json)
    python stability_eval.py --model DeepSeek-V4-Flash --num-workers 12
    python stability_eval.py --force                                 # ignore existing round results and force re-evaluation
"""

import argparse
import concurrent.futures
import json
import os
import sys
import threading
from collections import defaultdict

import numpy as np
import pandas as pd

from score_responses import (
    LLMClient,
    _score_single_item,
    load_answers,
    save_scores,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_URL = os.getenv("OPENSCIENCE_BASE_URL", "http://your-api/v1")
DEFAULT_API_KEY = os.getenv("OPENSCIENCE_API_KEY", "")
DEFAULT_MODEL = "DeepSeek-V4-Flash"
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "answer", "stablility")
CATEGORIES = ["cs", "econ", "physics"]

# Data path configuration (relative to the script directory)
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "data")


def score_round(score_client: LLMClient, items: list, answers: dict,
                temperature: float, num_workers: int, output_path: str) -> dict:
    """Run a full scoring pass over all items in items that already have answers,
    and save the result to output_path.

    Returns {idx_N: {"score": float, "individual_scores": [...]}}.
    """
    shared_results = {}
    lock = threading.Lock()
    total = len(items)
    indices = [
        idx for idx in range(total)
        if answers.get(f"idx_{idx}", {}).get("answer", "")
    ]
    print(f"  [Round] {len(indices)} items to score (temperature={temperature})", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(
                _score_single_item, score_client,
                items[idx], idx, total, answers,
                shared_results, lock, temperature
            ): idx
            for idx in indices
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            future.result()  # exceptions are caught internally and never raised

    save_scores(output_path, shared_results)
    return shared_results


def compute_metrics(rounds: list, cat_map: dict) -> dict:
    """Compute stability metrics from multiple rounds of scoring results.

    rounds: a list, each element is {idx_N: {"score": float, "individual_scores": [...]}}
    cat_map: {idx: category}, derived from the data_source column of the parquet
    """
    n_rounds = len(rounds)
    round_means = []
    round_means_by_cat = {c: [] for c in CATEGORIES}
    item_scores = defaultdict(dict)  # idx -> {round_idx: score}

    for r, res in enumerate(rounds):
        scores = [v["score"] for v in res.values()]
        round_means.append(float(np.mean(scores)) if scores else 0.0)

        cat_scores = {c: [] for c in CATEGORIES}
        for k, v in res.items():
            try:
                idx = int(k.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            item_scores[idx][r] = v["score"]
            c = cat_map.get(idx)
            if c in cat_scores:
                cat_scores[c].append(v["score"])
        for c in CATEGORIES:
            if cat_scores[c]:
                round_means_by_cat[c].append(float(np.mean(cat_scores[c])))

    # Overall statistics
    overall_scores = [v["score"] for res in rounds for v in res.values()]
    overall_mean = float(np.mean(overall_scores)) if overall_scores else None
    overall_std = float(np.std(overall_scores)) if overall_scores else None
    round_std = float(np.std(round_means)) if n_rounds >= 2 else 0.0
    round_range = float(max(round_means) - min(round_means)) if round_means else None

    # Per-sample cross-round stability (requires the sample to appear in at least 2 rounds)
    item_stds = []
    for idx, per_round in item_scores.items():
        vals = list(per_round.values())
        if len(vals) >= 2:
            item_stds.append(float(np.std(vals)))
    mean_item_std = float(np.mean(item_stds)) if item_stds else None
    pct_stable = float(np.mean([1.0 if s == 0 else 0.0 for s in item_stds])) if item_stds else None

    # Rubric decision cross-round agreement rate: fraction of (item, rubric_id) pairs
    # whose binary scores are fully identical across >= 2 rounds
    rubric_by_item = defaultdict(dict)  # idx -> {rubric_id: [scores across rounds]}
    for r, res in enumerate(rounds):
        for k, v in res.items():
            try:
                idx = int(k.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            for ind in v.get("individual_scores", []):
                rid = ind.get("rubric_id")
                if rid is None:
                    continue
                rubric_by_item[idx].setdefault(rid, []).append(ind.get("score"))

    n_pairs = 0
    n_agree = 0
    per_rubric_agree = defaultdict(lambda: {"n": 0, "agree": 0})
    for idx, rubrics_map in rubric_by_item.items():
        for rid, scores in rubrics_map.items():
            if len(scores) < 2:
                continue
            n_pairs += 1
            per_rubric_agree[rid]["n"] += 1
            if min(scores) == max(scores):
                n_agree += 1
                per_rubric_agree[rid]["agree"] += 1
    rubric_agreement_rate = (n_agree / n_pairs) if n_pairs else None

    # Overall mean score per category (across rounds)
    category_mean = {}
    for c in CATEGORIES:
        category_mean[c] = float(np.mean(round_means_by_cat[c])) if round_means_by_cat[c] else None

    return {
        "n_rounds": n_rounds,
        "round_means": round_means,
        "round_means_by_category": round_means_by_cat,
        "category_mean": category_mean,
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "round_std": round_std,
        "round_range": round_range,
        "mean_item_std": mean_item_std,
        "pct_stable_items": pct_stable,
        "rubric_agreement_rate": rubric_agreement_rate,
        "per_rubric_agreement": {str(k): dict(v) for k, v in per_rubric_agree.items()},
        "n_rubric_pairs": n_pairs,
    }


def print_summary(s: dict) -> None:
    """Print the stability summary."""
    print(f"\n{'=' * 90}")
    print(f"  Stability summary for dataset {s['folder']}")
    print(f"{'=' * 90}")
    print(f"  Per-round mean scores:      " + ", ".join(f"{m:.4f}" for m in s["round_means"]))
    print(f"  Overall mean:               {s['overall_mean']:.4f} ± {s['overall_std']:.4f}")
    print(f"  Between-round std:          {s['round_std']:.4f}")
    if s["round_range"] is not None:
        print(f"  Between-round range:        {s['round_range']:.4f}")
    if s.get("mean_item_std") is not None:
        print(f"  Per-sample cross-round mean std: {s['mean_item_std']:.4f}")
        print(f"  Fully stable sample ratio:        {s['pct_stable_items'] * 100:.1f}%")
    if s.get("rubric_agreement_rate") is not None:
        print(f"  Rubric decision agreement rate:   {s['rubric_agreement_rate'] * 100:.1f}%")
    cat_str = " | ".join(
        f"{c}={s['category_mean'][c]:.4f}"
        for c in CATEGORIES if s["category_mean"].get(c) is not None
    )
    if cat_str:
        print(f"  Per-category mean scores:  {cat_str}")
    print(f"  Per-round score files:     {', '.join(s['round_score_files'])}")
    print()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run multiple independent evaluation rounds on an answer file to analyze the stability of the LLM judge model"
    )
    p.add_argument("--folders", "-f", nargs="+", default=["solution", "experiment"],
                   choices=["solution", "experiment", "stability"],
                   help="Dataset folders to evaluate (default: solution experiment)")
    p.add_argument("--answer-name", type=str, default="qwen3_4b_responses",
                   help="Answer filename (without .json), located under answer/<folder>/")
    p.add_argument("--rounds", "-r", type=int, default=5,
                   help="Number of independent evaluation rounds (default 5)")
    p.add_argument("--temperature", "-t", type=float, default=0.7,
                   help="Sampling temperature for the judge model, used to introduce between-round randomness (default 0.7)")
    p.add_argument("--num-workers", "-w", type=int, default=8,
                   help="Number of concurrent scoring threads")
    p.add_argument("--max-items", "-m", type=int, default=0,
                   help="Max items to evaluate per folder; 0 means all")
    p.add_argument("--output-dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR,
                   help="Root directory for output results")
    p.add_argument("--flat-output", action="store_true",
                   help="Write results directly to --output-dir (without per-dataset subdirectories); useful when evaluating a single dataset split by model directory")
    p.add_argument("--force", action="store_true",
                   help="Force re-evaluation even if round results already exist")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help="Judge model name")
    p.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                   help="Judge API base URL")
    p.add_argument("--api-key", type=str, default=DEFAULT_API_KEY,
                   help="Judge API key")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.rounds < 1:
        print("[ERROR] --rounds must be >= 1", file=sys.stderr)
        sys.exit(1)

    score_client = LLMClient(api_key=args.api_key, base_url=args.base_url, model=args.model)
    os.makedirs(args.output_dir, exist_ok=True)
    all_summaries = {}

    for folder in args.folders:
        # Stability-test specific: only process stability.parquet
        if folder == "stability":
            parquet_path = os.path.join(SCRIPT_DIR, "data", "stability.parquet")
        else:
            parquet_path = os.path.join(DEFAULT_DATA_DIR, f"{folder}.parquet")
        answer_path = os.path.join(SCRIPT_DIR, "answer", folder, f"{args.answer_name}.json")
        out_folder = args.output_dir if args.flat_output else os.path.join(args.output_dir, folder)
        os.makedirs(out_folder, exist_ok=True)

        if not os.path.isfile(parquet_path):
            print(f"[ERROR] parquet not found: {parquet_path}", file=sys.stderr)
            continue
        if not os.path.isfile(answer_path):
            print(f"[ERROR] answer file not found: {answer_path}", file=sys.stderr)
            continue

        df = pd.read_parquet(parquet_path)
        if args.max_items and args.max_items > 0:
            df = df.iloc[:args.max_items]
        items = df.to_dict("records")
        cat_map = {idx: str(src) for idx, src in enumerate(df["data_source"].tolist())}
        answers = load_answers(answer_path)

        print(f"\n{'=' * 90}")
        print(f"  Dataset: {folder}")
        print(f"  Answer file: {answer_path}")
        print(f"  Items: {len(items)} | Rounds: {args.rounds} | Temperature: {args.temperature}")
        print(f"  Judge model: {args.model} @ {args.base_url}")
        print(f"  Output dir: {out_folder}")
        print(f"{'=' * 90}")

        rounds = []
        round_files = []
        for r in range(1, args.rounds + 1):
            out_path = os.path.join(out_folder, f"round_{r}_scores.json")
            round_files.append(os.path.basename(out_path))
            if os.path.isfile(out_path) and not args.force:
                with open(out_path, encoding="utf-8") as fh:
                    rounds.append(json.load(fh))
                print(f"  [Skip] round {r} result already exists: {out_path}", flush=True)
            else:
                print(f"\n>>> Starting round {r}/{args.rounds} evaluation...", flush=True)
                res = score_round(score_client, items, answers,
                                  args.temperature, args.num_workers, out_path)
                rounds.append(res)

        summary = compute_metrics(rounds, cat_map)
        summary["folder"] = folder
        summary["answer_file"] = answer_path
        summary["rounds"] = args.rounds
        summary["temperature"] = args.temperature
        summary["model"] = args.model
        summary["n_items"] = len(items)
        summary["n_scored_per_round"] = [len(res) for res in rounds]
        summary["round_score_files"] = round_files

        summary_path = os.path.join(out_folder, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        print(f"\n  Summary saved to: {summary_path}")
        print_summary(summary)

        all_summaries[folder] = summary

    if not args.flat_output:
        top_summary_path = os.path.join(args.output_dir, "summary.json")
        with open(top_summary_path, "w", encoding="utf-8") as fh:
            json.dump(all_summaries, fh, ensure_ascii=False, indent=2)
        print(f"\nCombined summary saved to: {top_summary_path}")


if __name__ == "__main__":
    main()
