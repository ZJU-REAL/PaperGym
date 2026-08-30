#!/usr/bin/env python3
"""
Load scoring result JSON files, compute each model's average coverage, and output it as a table.

Usage:
  # Specify the scoring result file
  python show_results.py output/*_coverage_results.json

  # Specify a directory (automatically scans all *_coverage_results.json)
  python show_results.py --dir output

  # Specify output to a txt file
  python show_results.py --dir output --output results.txt

  # Show detailed coverage for each item
  python show_results.py --dir output --detail
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict

# ── Score conversion (kept consistent with compute_coverage_ds.py) ─────────────────────

TEXT_TO_SCORE = {
    'Not at all': 1,
    'not at all': 1,
    'Barely': 2,
    'barely': 2,
    'moderately': 3,
    'Moderately': 3,
    'Moderate': 3,
    'Mostly': 4,
    'mostly': 4,
    'Complete': 5,
    'Completely': 5,
    'completely': 5,
}


def normalize_5scale(x: int) -> float:
    return (x - 1) / 4


def compute_item_coverage(rubric_judges: List[dict]) -> float:
    """Compute the coverage (0~1) of a single item."""
    scores = [TEXT_TO_SCORE[j['score']] for j in rubric_judges]
    normalized = [normalize_5scale(s) for s in scores]
    return sum(normalized) / len(normalized)


# ── Model name inference ──────────────────────────────────────────────────────

def infer_model_name(filepath: str) -> str:
    """Infer a readable model name from the filename."""
    name = Path(filepath).stem
    # Strip the _coverage_results suffix
    name = re.sub(r'_coverage_results$', '', name)
    # Known mapping
    known = {
        'qwen3_1b': 'Qwen3-1B',
        'qwen3_1b_sft_200': 'Qwen3-1B-SFT-200',
        'qwen3_1b_opsd_200': 'Qwen3-1B-OPSD-200',
        'qwen3_1b_grpo_offline_100': 'Qwen3-1B-GRPO-Offline-100',
        'qwen3_1b_grpo_opsd_offline_28_100': 'Qwen3-1B-GRPO-OPSD-28-100',
        'qwen3_1b_grpo_opsd_offline_37_100': 'Qwen3-1B-GRPO-OPSD-37-100',
        'qwen3_1b_grpo_opsd_offline_46_100': 'Qwen3-1B-GRPO-OPSD-46-100',
        'qwen3_4b': 'Qwen3-4B',
        'qwen3_4b_opsd_200': 'Qwen3-4B-OPSD-200',
        'qwen3_4b_grpo_offline_200': 'Qwen3-4B-GRPO-Offline-200',
        'qwen3_4b_grpo_opsd_offline_150': 'Qwen3-4B-GRPO-OPSD-150',
        'qwen3_4b_grpo_opsd_offline_200': 'Qwen3-4B-GRPO-OPSD-200',
        'qwen3_8b': 'Qwen3-8B',
        'qwen3_8b_opsd_400': 'Qwen3-8B-OPSD-400',
        'qwen3_8b_grpo_opsd_offline_100': 'Qwen3-8B-GRPO-OPSD-100',
        'qwen3_32b': 'Qwen3-32B',
    }
    if name in known:
        return known[name]
    # Generic fallback: camel-case / upper-case conversion
    return name.replace('_', ' ').title()


# ── Loading and computing ────────────────────────────────────────────────────────

def load_result(path: str) -> Dict[str, list]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_file(filepath: str) -> dict:
    """Analyze a single scoring result file, returning statistics."""
    data = load_result(filepath)
    item_ids = list(data.keys())
    coverages = {}

    for item_id, rubric_judges in data.items():
        if not rubric_judges:  # empty list
            continue
        coverage = compute_item_coverage(rubric_judges)
        coverages[item_id] = coverage

    total_items = len(item_ids)
    valid_items = len(coverages)
    skipped = total_items - valid_items

    avg_coverage = sum(coverages.values()) / valid_items if valid_items > 0 else 0.0

    return {
        'filepath': filepath,
        'name': infer_model_name(filepath),
        'total_items': total_items,
        'valid_items': valid_items,
        'skipped': skipped,
        'avg_coverage': avg_coverage,
        'coverages': coverages,  # item_id -> float
    }


# ── Table output ──────────────────────────────────────────────────────────

def print_table(stats: List[dict], output_file=None, show_detail: bool = False):
    """Output the statistics as a table."""
    out = open(output_file, 'w', encoding='utf-8') if output_file else sys.stdout

    # ── Summary table ──
    header = f"{'Model':<35} {'Items':>6} {'Avg Coverage':>14}"
    sep = '-' * len(header)

    print(sep, file=out)
    print(header, file=out)
    print(sep, file=out)

    for s in stats:
        print(
            f"{s['name']:<35} {s['valid_items']:>6}  {s['avg_coverage']:>10.4f}",
            file=out,
        )

    print(sep, file=out)

    # Global average
    if stats:
        overall_avg = sum(s['avg_coverage'] * s['valid_items'] for s in stats) / sum(
            s['valid_items'] for s in stats
        )
        avg_items = sum(s['valid_items'] for s in stats)
        print(
            f"{'Average (weighted)':<35} {avg_items:>6}  {overall_avg:>10.4f}",
            file=out,
        )
        print(sep, file=out)

    # ── Detail information (coverage per item) ──
    if show_detail:
        print(file=out)
        for s in stats:
            print(f"── {s['name']} ──────────────────────────────", file=out)
            detail_header = f"{'Item ID':<35} {'Coverage':>10}"
            print(detail_header, file=out)
            print('-' * len(detail_header), file=out)
            for item_id in sorted(s['coverages'].keys()):
                print(
                    f"{item_id:<35} {s['coverages'][item_id]:>10.4f}",
                    file=out,
                )
            print(file=out)

    if output_file:
        out.close()


# ── Main entry ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Load scoring results and print a tabular coverage summary',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'files',
        nargs='*',
        default=None,
        help='Path to the scoring result JSON files (glob patterns supported)',
    )
    # output_mini
    parser.add_argument(
        '--dir',
        type=str,
        default=None,
        help='Scan all *_coverage_results.json files under the given directory',
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Write to a txt file (defaults to stdout if unset)',
    )
    parser.add_argument(
        '--detail',
        action='store_true',
        help='Show the detailed coverage of every item',
    )

    args = parser.parse_args()

    # Collect the file list
    filepaths: List[str] = []

    if args.dir:
        d = Path(args.dir)
        if not d.is_dir():
            print(f"Error: directory does not exist: {args.dir}", file=sys.stderr)
            sys.exit(1)
        filepaths.extend(sorted(str(p) for p in d.glob('*_coverage_results.json')))

    # Accept already-expanded file paths (shell wildcards already expanded externally)
    for pattern in args.files:
        p = Path(pattern).resolve()
        if p.is_file():
            filepaths.append(str(p))

    # Deduplicate while preserving order
    seen = set()
    filepaths = [f for f in filepaths if not (f in seen or seen.add(f))]

    if not filepaths:
        print("Error: no scoring result files found. Specify file paths or --dir.", file=sys.stderr)
        sys.exit(1)

    # Analyze all files
    stats = []
    for fp in filepaths:
        try:
            s = analyze_file(fp)
            stats.append(s)
        except Exception as e:
            print(f"Warning: failed to read {fp}: {e}", file=sys.stderr)

    if not stats:
        print("Error: no scoring result file could be read successfully.", file=sys.stderr)
        sys.exit(1)

    # Output the table
    print_table(stats, output_file=args.output, show_detail=args.detail)


if __name__ == '__main__':
    main()
