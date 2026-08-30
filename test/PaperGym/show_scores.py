"""
Display the scoring result tables for each dataset (cs_new, physics, etc.) under the score folder.

Usage:
    python show_scores.py                      # display all datasets
    python show_scores.py --dataset cs_new      # display only the specified dataset
    python show_scores.py --sort-by mean         # sort by average score
    python show_scores.py --topk 5              # show only the top N models
"""

import argparse
import json
import os
import sys
import numpy as np

SCORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "score")


def collect_stats(score_dir: str) -> dict:
    """Scan all subdirectories (datasets) under score_dir, returning {dataset: [(model_name, stats_dict)]}."""
    results = {}

    if not os.path.isdir(score_dir):
        print(f"[Error] Score directory not found: {score_dir}", file=sys.stderr)
        sys.exit(1)

    for dataset in sorted(os.listdir(score_dir)):
        dataset_path = os.path.join(score_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue

        model_stats = []
        for fname in sorted(os.listdir(dataset_path)):
            if not fname.endswith(".json"):
                continue

            fpath = os.path.join(dataset_path, fname)
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[Warning] Failed to read {fpath}: {e}", file=sys.stderr)
                continue

            scores = []
            for key, value in data.items():
                s = value.get("score")
                if s is not None:
                    scores.append(float(s))

            if not scores:
                continue

            model_name = fname.replace("_scores.json", "")
            arr = np.array(scores)
            model_stats.append((
                model_name,
                {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "count": len(arr),
                },
            ))

        if model_stats:
            results[dataset] = model_stats

    return results


def print_table(results: dict, sort_by: str = None, topk: int = None, dataset_filter: str = None):
    """Print a formatted table."""
    for dataset, model_stats in results.items():
        if dataset_filter and dataset != dataset_filter:
            continue

        print(f"\n{'=' * 80}")
        print(f"  Dataset: {dataset}")
        print(f"{'=' * 80}")

        if not model_stats:
            print("  (no data)")
            continue

        # Sort
        if sort_by == "mean":
            model_stats.sort(key=lambda x: x[1]["mean"], reverse=True)
        elif sort_by == "std":
            model_stats.sort(key=lambda x: x[1]["std"], reverse=True)

        if topk and topk > 0:
            model_stats = model_stats[:topk]

        # Table header
        header = f"{'Model':<50} {'Samples':>6} {'Avg Score':>8}"
        sep = "-" * len(header)

        print(header)
        print(sep)

        for model_name, stats in model_stats:
            print(
                f"{model_name:<50} "
                f"{stats['count']:>6} "
                f"{stats['mean']:>8.4f}"
            )

    print()


def print_table_rich(results: dict, sort_by: str = None, topk: int = None, dataset_filter: str = None):
    """Print a colored table using the rich library (if rich is available)."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print_table(results, sort_by, topk, dataset_filter)
        return

    console = Console()

    for dataset, model_stats in results.items():
        if dataset_filter and dataset != dataset_filter:
            continue

        if not model_stats:
            continue

        if sort_by == "mean":
            model_stats.sort(key=lambda x: x[1]["mean"], reverse=True)
        elif sort_by == "std":
            model_stats.sort(key=lambda x: x[1]["std"], reverse=True)

        if topk and topk > 0:
            model_stats = model_stats[:topk]

        table = Table(title=f"Dataset: {dataset}", title_style="bold cyan")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Samples", justify="right")
        table.add_column("Avg Score", justify="right")

        for model_name, stats in model_stats:
            table.add_row(
                model_name,
                str(stats["count"]),
                f"{stats['mean']:.4f}",
            )

        console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Display the scoring results under the score directory as a table")
    parser.add_argument("--dataset", "-d", type=str, default=None,
                        help="Dataset name to show (e.g. cs_new, physics); show all if unset")
    parser.add_argument("--sort-by", "-s", type=str, choices=["mean"], default=None,
                        help="Field to sort by")
    parser.add_argument("--topk", "-k", type=int, default=None,
                        help="Show only the top N models")
    parser.add_argument("--no-rich", action="store_true",
                        help="Disable the rich colored table and use plain text output")
    args = parser.parse_args()

    results = collect_stats(SCORE_DIR)

    if not results:
        print(f"[Error] No scoring data found under the score directory", file=sys.stderr)
        sys.exit(1)

    if args.no_rich:
        print_table(results, sort_by=args.sort_by, topk=args.topk, dataset_filter=args.dataset)
    else:
        print_table_rich(results, sort_by=args.sort_by, topk=args.topk, dataset_filter=args.dataset)


if __name__ == "__main__":
    main()
