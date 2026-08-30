#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_three_results_with_expert.py — Three-model comparison result aggregation with expert review.

Difference from aggregate_three_results.py:
  - When voting in three folders results in NoMajority (one vote each for A/B/C), introduce a fourth expert (e.g. gemini)
    vote as the final result.
  - This way each dimension can determine a unique Winner, eliminating NoMajority cases.

Usage:
  python3 aggregate_three_results_with_expert.py results_three/kimi results_three/ds results_three/glm --expert results_three/gemini
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

DIMENSIONS = [
    "goal_alignment",
    "novel_insight",
    "scientific_soundness",
    "execution_quality",
    "expected_impact",
]

MODEL_FILES = [
    "ml_1b_responses.json",
    "ml_1b_rubrichub_responses.json",
    "ml_1b_cs_responses.json",
]


def short_name(name: str) -> str:
    return name.replace("_responses.json", "")


def load_dir(dirpath):
    """Load a result folder, return {pair_name: [entries]}."""
    pairs = {}
    for fp in sorted(glob.glob(os.path.join(dirpath, "*_comparison_results.json"))):
        name = os.path.basename(fp).replace("_comparison_results.json", "")
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                print(f"  [Warning] {fp} is not a list, skipped", file=sys.stderr)
                continue
            pairs[name] = data
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [Warning] Could not load {fp}: {e}", file=sys.stderr)
    return pairs


def majority_with_expert(votes, expert_vote):
    """Majority rules; if tied, refer to expert vote.

    votes: list of votes from three folders
    expert_vote: expert vote
    Returns: (winner, used_expert) — winner is None if undetermined
    """
    if not votes:
        return None, False

    counter = Counter(votes)
    best, cnt = counter.most_common(1)[0]

    # If an option exceeds half, return directly
    if cnt > len(votes) / 2:
        return best, False

    # Tie case (e.g. one vote each for A/B/C, or A/B), refer to expert vote
    if expert_vote in counter:
        # If expert vote is among tied options, choose expert vote
        return expert_vote, True
    else:
        # If expert vote is not among tied options (theoretically impossible), return None
        return None, False


def aggregate_three_pair_with_expert(pair_name, per_dir_entries, contributing, expert_dir, expert_entries):
    """Aggregate across folders for a single three-model comparison pair (with expert review), return statistics dict."""
    # Build index of folder -> (entry_key -> entry)
    index_maps = {}
    for d in contributing:
        m = {}
        for e in per_dir_entries[d]:
            key = e.get("idx_key") or str(e.get("index"))
            m[key] = e
        index_maps[d] = m

    # Expert index
    expert_map = {}
    for e in expert_entries:
        key = e.get("idx_key") or str(e.get("index"))
        expert_map[key] = e

    all_keys = set()
    for m in index_maps.values():
        all_keys |= set(m)

    # Only count entries present in all relevant folders
    counted = sorted(k for k in all_keys if all(k in m for m in index_maps.values()))
    skipped = len(all_keys) - len(counted)

    # Statistics container
    dim_wins = {d: {f: 0 for f in MODEL_FILES} for d in DIMENSIONS}
    dim_expert_used = {d: 0 for d in DIMENSIONS}  # count how many times the expert vote was used

    score_sums = {f: 0.0 for f in MODEL_FILES}
    score_counts = {f: 0 for f in MODEL_FILES}
    score_wins = {f: 0 for f in MODEL_FILES}

    for key in counted:
        entries = {d: index_maps[d][key] for d in contributing}
        expert_entry = expert_map.get(key)

        # ---- Dimensions: majority vote + expert review ----
        for dim in DIMENSIONS:
            votes = []
            for d in contributing:
                winner = entries[d]["parsed"].get(dim, "ParseError")
                if winner in MODEL_FILES:
                    votes.append(winner)

            # Expert vote
            expert_vote = None
            if expert_entry:
                expert_vote = expert_entry["parsed"].get(dim, "ParseError")
                if expert_vote not in MODEL_FILES:
                    expert_vote = None

            if votes:
                winner, used_expert = majority_with_expert(votes, expert_vote)
                if winner:
                    dim_wins[dim][winner] += 1
                    if used_expert:
                        dim_expert_used[dim] += 1
                # If winner is None, the expert vote also cannot resolve it; skip this entry

        # ---- Overall Score: average across folders, then compare highest ----
        avg_scores = {}
        for f in MODEL_FILES:
            vals = []
            for d in contributing:
                sc = entries[d]["parsed"].get("scores", {}).get(f)
                if sc is not None:
                    try:
                        vals.append(float(sc))
                    except (ValueError, TypeError):
                        pass
            if vals:
                avg_scores[f] = sum(vals) / len(vals)
                score_sums[f] += avg_scores[f]
                score_counts[f] += 1

        if len(avg_scores) == len(MODEL_FILES):
            best = max(avg_scores, key=avg_scores.get)
            score_wins[best] += 1

    return {
        "pair_name": pair_name,
        "contributing": contributing,
        "expert_dir": expert_dir,
        "counted": len(counted),
        "skipped": skipped,
        "dim_wins": dim_wins,
        "dim_expert_used": dim_expert_used,
        "score_sums": score_sums,
        "score_counts": score_counts,
        "score_wins": score_wins,
    }


def print_pair(stat):
    files = MODEL_FILES
    n = stat["counted"]

    print(f"\n{'='*72}")
    print(f"Comparison pair: {stat['pair_name']}")
    print(f"  Participating folders ({len(stat['contributing'])}): {', '.join(stat['contributing'])}")
    print(f"  Expert review: {stat['expert_dir']}")
    print(f"  Counted entries: {n}  (Skipped missing entries: {stat['skipped']})")
    print(f"{'='*72}")

    # Per-dimension accuracy
    print(f"\n  Best judgment per dimension (accuracy = times model judged best / total entries):")
    print(f"  {'Dimension':<20} " + " ".join([f"{short_name(f):<18}" for f in files]))
    print(f"  {'-' * (20 + 18 * len(files))}")
    for d in DIMENSIONS:
        row = f"  {d:<20}"
        for f in files:
            acc = stat["dim_wins"][d][f] / n * 100 if n > 0 else 0
            row += f" {acc:>5.1f}% ({stat['dim_wins'][d][f]:>3}/{n})"
        print(row, flush=True)
        if stat["dim_expert_used"][d] > 0:
            print(f"    [Expert review involved: {stat['dim_expert_used'][d]} times]", flush=True)

    # Overall Score statistics
    print(f"\n  Overall Score statistics (average across folders then compare):")
    print(f"  {'Model':<30} {'Avg Score':>8} {'Win Rate':>10} {'Valid Items':>8}")
    print(f"  {'-' * 70}")
    for f in files:
        avg = stat["score_sums"][f] / stat["score_counts"][f] if stat["score_counts"][f] > 0 else 0
        win_rate = stat["score_wins"][f] / n * 100 if n > 0 else 0
        print(f"  {short_name(f):<30} {avg:>8.2f} {win_rate:>9.1f}% {stat['score_counts'][f]:>8}", flush=True)

    # Overall accuracy
    print(f"\n  Overall accuracy (average of best judgment across all dimensions + total score win rate):")
    for f in files:
        total_wins = sum(stat["dim_wins"][d][f] for d in DIMENSIONS) + stat["score_wins"][f]
        total_possible = len(DIMENSIONS) * n + n
        overall_acc = total_wins / total_possible * 100 if total_possible > 0 else 0
        print(f"  {short_name(f):<30} {overall_acc:>6.2f}% ({total_wins}/{total_possible})", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate three-model comparison results with expert review (majority vote + expert review)"
    )
    parser.add_argument("result_dirs", nargs="+",
                        help="One or more result folders, e.g. results_three/kimi results_three/ds results_three/glm")
    parser.add_argument("--expert", type=str, required=True,
                        help="Expert review folder (e.g. results_three/gemini)")
    args = parser.parse_args()

    for d in args.result_dirs:
        if not os.path.isdir(d):
            print(f"Error: folder does not exist: {d}", file=sys.stderr)
            sys.exit(1)

    if not os.path.isdir(args.expert):
        print(f"Error: expert review folder does not exist: {args.expert}", file=sys.stderr)
        sys.exit(1)

    print(f"Aggregating three-model comparison results ({len(args.result_dirs)} folders + expert review)")
    print(f"  Participating folders: {', '.join(args.result_dirs)}")
    print(f"  Expert review: {args.expert}")
    print("Rules: an item is counted only if it exists in all relevant folders;")
    print("      dimension winner = majority vote; ties are resolved by expert review;")
    print("      overall score is compared after averaging across folders")

    # pair_name -> {dir: entries}
    all_pairs = {}
    for d in args.result_dirs:
        for name, entries in load_dir(d).items():
            all_pairs.setdefault(name, {})[d] = entries

    # Load the Expert review
    expert_pairs = load_dir(args.expert)

    if not all_pairs:
        print("No *_comparison_results.json files found", file=sys.stderr)
        sys.exit(1)

    pairs = []
    for name in sorted(all_pairs):
        per_dir = all_pairs[name]
        contributing = [d for d in args.result_dirs if d in per_dir]
        missing = [d for d in args.result_dirs if d not in per_dir]
        if missing:
            print(f"\n[Warning] Comparison pair '{name}' is missing folders: {', '.join(missing)}", file=sys.stderr)
        if len(contributing) < 2:
            print(f"[Skip] Comparison pair '{name}' appears in only {len(contributing)} folder(s), fewer than two, cannot take a majority vote",
                  file=sys.stderr)
            continue

        # Check whether the Expert review contains this comparison pair
        if name not in expert_pairs:
            print(f"[Skip] Comparison pair '{name}' does not exist in the expert review folder", file=sys.stderr)
            continue

        pairs.append(aggregate_three_pair_with_expert(
            name, per_dir, contributing, args.expert, expert_pairs[name]
        ))

    if not pairs:
        print("\nNo comparison pair satisfies the conditions.", file=sys.stderr)
        sys.exit(1)

    for st in pairs:
        print_pair(st)

    print(f"\n{'='*72}")
    print("All done!")


if __name__ == "__main__":
    main()
