"""
Display the average scores of each model under the score folder, broken down by category (cs/econ/physics).

Difference from show_scores.py: show_scores.py only computes the average score of the whole dataset,
while this script assigns each sample to one of the cs / econ / physics categories based on the
data_source column of the parquet, computes each model's average score per category, and also
includes the overall average score for comparison.

Usage:
    python show_scores_by_category.py                          # process all folders
    python show_scores_by_category.py --folder solution        # process only solution
    python show_scores_by_category.py --folder experiment      # process only experiment
    python show_scores_by_category.py --sort-by physics        # sort by the physics average score
    python show_scores_by_category.py --topk 5                 # show only the top N models
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCORE_DIR = os.path.join(SCRIPT_DIR, "score")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CATEGORIES = ["cs", "econ", "physics"]


def load_category_map(folder: str) -> dict:
    """Read data/<folder>.parquet, returning the {idx: category} mapping."""
    parquet_path = os.path.join(DATA_DIR, f"{folder}.parquet")
    if not os.path.isfile(parquet_path):
        print(f"[Error] Dataset parquet not found: {parquet_path}", file=sys.stderr)
        return {}

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"[Error] Failed to read parquet {parquet_path}: {e}", file=sys.stderr)
        return {}

    if "data_source" not in df.columns:
        print(f"[Warning] {parquet_path} has no data_source column, cannot split by category", file=sys.stderr)
        return {}

    return {idx: str(src) for idx, src in enumerate(df["data_source"].tolist())}


def collect_stats_by_category(score_dir: str) -> dict:
    """Scan all subdirectories (datasets) under score_dir, returning
    {dataset: [(model_name, {cat: stats, "overall": stats})]}."""
    results = {}

    if not os.path.isdir(score_dir):
        print(f"[Error] Score directory not found: {score_dir}", file=sys.stderr)
        sys.exit(1)

    for dataset in sorted(os.listdir(score_dir)):
        dataset_path = os.path.join(score_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue

        cat_map = load_category_map(dataset)
        if not cat_map:
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

            # Collect scores by category
            cat_scores = {cat: [] for cat in CATEGORIES}
            for key, value in data.items():
                s = value.get("score") if isinstance(value, dict) else None
                if s is None:
                    continue
                # key is like "idx_5"; extract the row number and map it to a category
                try:
                    idx = int(key.split("_", 1)[1])
                except (IndexError, ValueError):
                    continue
                cat = cat_map.get(idx)
                if cat in cat_scores:
                    cat_scores[cat].append(float(s))

            # Skip models with no scores at all
            if not any(cat_scores.values()):
                continue

            model_name = fname.replace("_scores.json", "")
            stats = {}
            all_scores = []
            for cat in CATEGORIES:
                arr = np.array(cat_scores[cat])
                if arr.size > 0:
                    stats[cat] = {
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                        "count": int(arr.size),
                    }
                    all_scores.extend(arr.tolist())
                else:
                    stats[cat] = None

            arr_all = np.array(all_scores)
            stats["overall"] = {
                "mean": float(np.mean(arr_all)),
                "std": float(np.std(arr_all, ddof=1)) if arr_all.size > 1 else 0.0,
                "count": int(arr_all.size),
            }
            model_stats.append((model_name, stats))

        if model_stats:
            results[dataset] = model_stats

    return results


def print_table(results: dict, sort_by: str = None, topk: int = None, folder_filter: str = None):
    """Print a plain-text table."""
    for folder, model_stats in results.items():
        if folder_filter and folder != folder_filter:
            continue

        print(f"\n{'=' * 100}")
        print(f"  Dataset: {folder}")
        print(f"{'=' * 100}")

        if not model_stats:
            print("  (no data)")
            continue

        if sort_by and sort_by in CATEGORIES + ["overall"]:
            model_stats.sort(key=lambda x: x[1][sort_by]["mean"] if x[1][sort_by] else -1,
                             reverse=True)

        if topk and topk > 0:
            model_stats = model_stats[:topk]

        header = f"{'Model':<46}{'cs':>16}{'econ':>16}{'physics':>16}{'overall':>16}"
        sep = "-" * len(header)

        print(header)
        print(sep)

        for model_name, stats in model_stats:
            cells = []
            for cat in CATEGORIES + ["overall"]:
                st = stats[cat]
                if st:
                    cells.append(f"{st['mean']:>10.4f} ±{st['std']:<3.3f}")
                else:
                    cells.append(f"{'-':>14}")
            print(f"{model_name:<46}" + "".join(cells))

    print()


def print_table_rich(results: dict, sort_by: str = None, topk: int = None, folder_filter: str = None):
    """Print a colored table using the rich library (if rich is available)."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print_table(results, sort_by, topk, folder_filter)
        return

    console = Console()

    for folder, model_stats in results.items():
        if folder_filter and folder != folder_filter:
            continue

        if not model_stats:
            continue

        if sort_by and sort_by in CATEGORIES + ["overall"]:
            model_stats.sort(key=lambda x: x[1][sort_by]["mean"] if x[1][sort_by] else -1,
                             reverse=True)

        if topk and topk > 0:
            model_stats = model_stats[:topk]

        table = Table(title=f"Dataset: {folder} (average score by category)", title_style="bold cyan")
        table.add_column("Model", style="cyan", no_wrap=True)
        for cat in CATEGORIES + ["overall"]:
            table.add_column(cat, justify="right")

        for model_name, stats in model_stats:
            cells = [model_name]
            for cat in CATEGORIES + ["overall"]:
                st = stats[cat]
                if st:
                    cells.append(f"{st['mean']:.4f} ± {st['std']:.3f}")
                else:
                    cells.append("-")
            table.add_row(*cells)

        console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Show the average score of each model under the score directory, broken down by category (cs/econ/physics)"
    )
    parser.add_argument("--folder", "-f", type=str, default=None,
                        help="Dataset name to show (solution / experiment); show all if unset")
    parser.add_argument("--sort-by", "-s", type=str,
                        choices=["cs", "econ", "physics", "overall"], default=None,
                        help="Sort by the average score of a given category")
    parser.add_argument("--topk", "-k", type=int, default=None,
                        help="Show only the top N models")
    parser.add_argument("--no-rich", action="store_true",
                        help="Disable the rich colored table and use plain text output")
    args = parser.parse_args()

    results = collect_stats_by_category(SCORE_DIR)

    if not results:
        print(f"[Error] No scoring data found under the score directory", file=sys.stderr)
        sys.exit(1)

    if args.no_rich:
        print_table(results, sort_by=args.sort_by, topk=args.topk, folder_filter=args.folder)
    else:
        print_table_rich(results, sort_by=args.sort_by, topk=args.topk, folder_filter=args.folder)


if __name__ == "__main__":
    main()
