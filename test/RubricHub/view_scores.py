#!/usr/bin/env python3
"""
View RubricHub scoring results, displaying the score statistics of each model as a table.

Usage:
    python view_scores.py                                    # view the scores_ds directory by default
    python view_scores.py /path/to/scores_dir                # specify a scoring directory
    python view_scores.py --detail                           # also show per-question scores
    python view_scores.py --sort-by score                    # sort by overall score (asc/desc)
"""

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from typing import Dict, List, Tuple


def load_score_file(path: str) -> Tuple[Dict, float, float, int, float, str]:
    """
    Load a single scoring file, returning:
    (items, total_raw, total_weight, num_items, overall_score, model_name)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_raw = 0.0
    total_weight = 0.0
    num_items = len(data)
    per_item_scores = []

    model_name = os.path.basename(path).replace("_scores.json", "")

    for item_key, item_data in data.items():
        raw = item_data.get("raw_score", 0)
        weight = item_data.get("total_weight", 1)
        total_raw += raw
        total_weight += weight
        per_item_scores.append((item_key, raw, weight))

    overall = total_raw / total_weight if total_weight > 0 else 0.0

    return data, total_raw, total_weight, num_items, overall, model_name, per_item_scores


def format_pct(value: float) -> str:
    """Format a percentage, keeping 2 decimal places."""
    return f"{value * 100:6.2f}%"


def print_summary_table(
    results: List[Tuple[str, float, float, int, float]],
    sort_by: str = "score",
    sort_desc: bool = False,
):
    """Print the summary table."""
    # Sorting
    if sort_by == "score":
        results.sort(key=lambda r: r[3], reverse=not sort_desc)
    elif sort_by == "name":
        results.sort(key=lambda r: r[0], reverse=sort_desc)
    elif sort_by == "samples":
        results.sort(key=lambda r: r[1], reverse=not sort_desc)

    if not results:
        print("No scoring files found.")
        return

    # Table header
    header = f"{'Model':<40} {'Samples':>6} {'Overall':>10}"
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("  RubricHub scoring results summary")
    print("=" * len(header))
    print(header)
    print(sep)

    for name, num_items, total_raw, total_weight, overall in results:
        print(f"{name:<40} {num_items:>6} {format_pct(overall):>10}")

    print(sep)
    print(header)
    print()


def print_detailed_table(
    all_items: Dict[str, List[Tuple[str, float, float]]],
    sort_by: str = "score",
    sort_desc: bool = False,
):
    """Print the detailed score comparison for each question."""
    # Find the union of all questions
    all_question_keys = set()
    for items in all_items.values():
        for k, _, _ in items:
            all_question_keys.add(k)
    all_question_keys = sorted(all_question_keys, key=lambda x: int(x.split("_")[1]))

    if not all_question_keys:
        return

    model_names = list(all_items.keys())

    # Build {question_key -> overall_score} for each model
    model_scores = {}
    for model_name, items in all_items.items():
        score_map = {}
        for k, raw, weight in items:
            score_map[k] = raw / weight if weight > 0 else 0.0
        model_scores[model_name] = score_map

    print()
    print("=" * (30 + 15 * len(model_names)))
    print("  Per-question score comparison")
    print("=" * (30 + 15 * len(model_names)))

    # Table header
    header = f"{'Question':<12}"
    for name in model_names:
        header += f" {name:>20}"
    print(header)
    print("-" * len(header))

    for qkey in all_question_keys:
        row = f"{qkey:<12}"
        for name in model_names:
            sc = model_scores.get(name, {}).get(qkey, None)
            if sc is not None:
                row += f" {format_pct(sc):>20}"
            else:
                row += f" {'N/A':>20}"
        print(row)

    print("-" * len(header))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="View RubricHub scoring results and display each model's score as a table"
    )
    parser.add_argument(
        "dir",
        nargs="?",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "output", "scores_ds"),
        help="Scoring results directory (default: output/scores_ds)",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Also show per-question score comparison",
    )
    parser.add_argument(
        "--sort-by",
        choices=["score", "name", "samples"],
        default="score",
        help="Sort field (default: score)",
    )
    parser.add_argument(
        "--sort-desc",
        action="store_true",
        help="Sort in descending order (default ascending)",
    )
    args = parser.parse_args()

    score_dir = args.dir
    if not os.path.isdir(score_dir):
        print(f"ERROR: directory does not exist: {score_dir}", file=sys.stderr)
        sys.exit(1)

    # Scan for scoring files
    score_files = sorted(
        [f for f in os.listdir(score_dir) if f.endswith("_scores.json")]
    )

    if not score_files:
        print(f"No *_scores.json files found in {score_dir}.")
        sys.exit(0)

    results = []
    all_items = OrderedDict()

    for sf in score_files:
        path = os.path.join(score_dir, sf)
        try:
            data_items, total_raw, total_weight, num_items, overall, model_name, per_item = load_score_file(
                path
            )
            results.append((model_name, num_items, total_raw, total_weight, overall))
            all_items[model_name] = per_item
        except Exception as e:
            print(f"  WARNING: failed to load {sf}: {e}", file=sys.stderr)

    # Print the summary table
    print_summary_table(results, sort_by=args.sort_by, sort_desc=args.sort_desc)

    # Print the detailed per-question comparison table
    if args.detail and len(all_items) > 1:
        print_detailed_table(all_items, sort_by=args.sort_by, sort_desc=args.sort_desc)


if __name__ == "__main__":
    main()
