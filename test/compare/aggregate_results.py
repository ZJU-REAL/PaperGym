#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_results.py — Aggregate comparison results across multiple result folders (majority vote + average score).

Unlike parse_results.py (which only looks at a single folder), this script treats
files with the same name *_comparison_results.json across multiple result folders as the same comparison pair, then:

  1. Win Rate: Uses "majority rules" — for the same entry, the conclusion given by the majority of folders
     is counted as valid (only counted if 2/3 or 3/3 agree, otherwise treated as NoMajority);
  2. Average score: Takes the mean Overall Score for the same entry across folders, then averages across all entries
     takes the average;
  3. Missing skip: entries must appear in all relevant folders to be counted — if folder A has it but
     B/C do not, the entry is skipped and not counted in any statistics.

Usage:
  python3 aggregate_results.py results_kimi_ml results_ds_ml results_qwen_ml

Notes:
  - The number of result folders is not limited to 3, can be any number;
  - When a comparison pair only appears in some folders, aggregation is done within the folders where it "appears",
    with a warning given; if fewer than 2 folders, the pair is skipped (cannot majority vote).
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

# Short names for dimensions (for compact display)
DIM_LABELS = {
    "goal_alignment": "GoalAlign",
    "novel_insight": "Novelty",
    "scientific_soundness": "Soundness",
    "execution_quality": "ExecQual",
    "expected_impact": "Impact",
}


def short_name(name: str) -> str:
    """Remove _responses.json suffix, show shorter model name."""
    return name.replace("_responses.json", "")


def load_dir(dirpath):
    """Load a result folder, return {pair_name: [entries]}.

    Files that cannot be parsed (empty / corrupted JSON / not a list) are skipped with a warning.
    """
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


def map_dim(entry, d):
    """Map the judgment on dimension d to the real source file.

    Returns: file_a / file_b / "Tie" / "ParseError"
    Handle random A/B swap via plan_a_actual_source / plan_b_actual_source.
    """
    parsed = entry.get("parsed") or {}
    j = parsed.get(d, "ParseError")
    if j in ("A", "B"):
        return entry["plan_a_actual_source"] if j == "A" else entry["plan_b_actual_source"]
    if j == "Tie":
        return "Tie"
    return "ParseError"


def map_scores(entry):
    """Returns {real source file: overall_score}, only includes parseable scores.

    When swapped=True, Plan A shows file_b content, scores need to be swapped.
    """
    parsed = entry.get("parsed") or {}
    sa = parsed.get("overall_score_a")
    sb = parsed.get("overall_score_b")

    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    sa, sb = _f(sa), _f(sb)
    swapped = entry.get("swapped", False)
    fa, fb = entry["file_a"], entry["file_b"]
    scores = {}
    if swapped:
        if sa is not None:
            scores[fb] = sa
        if sb is not None:
            scores[fa] = sb
    else:
        if sa is not None:
            scores[fa] = sa
        if sb is not None:
            scores[fb] = sb
    return scores


def majority(votes):
    """Majority rules: return conclusion with votes > len/2, otherwise "NoMajority"."""
    if not votes:
        return "NoMajority"
    best, cnt = Counter(votes).most_common(1)[0]
    return best if cnt > len(votes) / 2 else "NoMajority"


def win_rate(wins, losses, ties):
    denom = wins + losses + ties
    return (wins / denom * 100.0) if denom > 0 else 0.0


def aggregate_pair(pair_name, per_dir_entries, contributing):
    """Aggregate across folders for a single comparison pair, return statistics dict."""
    # file_a / file_b read from any entry in the pair (consistent across all folders)
    fa = fb = None
    for d in contributing:
        if per_dir_entries[d]:
            e0 = per_dir_entries[d][0]
            fa, fb = e0["file_a"], e0["file_b"]
            break

    # Build index of folder -> (entry_key -> entry)
    index_maps = {}
    for d in contributing:
        m = {}
        for e in per_dir_entries[d]:
            key = e.get("idx_key") or str(e.get("index"))
            m[key] = e
        index_maps[d] = m

    all_keys = set()
    for d in contributing:
        all_keys |= set(index_maps[d])
    # Only count entries present in all relevant folders (skip if A has but B/C does not)
    counted = sorted(k for k in all_keys if all(k in index_maps[d] for d in contributing))
    skipped = len(all_keys) - len(counted)

    # Statistics container
    dim_counts = {
        d: {fa: 0, fb: 0, "Tie": 0, "NoMajority": 0, "ParseError": 0}
        for d in DIMENSIONS
    }
    overall_counts = {fa: 0, fb: 0, "Tie": 0, "NoMajority": 0, "ParseError": 0}
    score_sum = {fa: 0.0, fb: 0.0}
    score_cnt = {fa: 0, fb: 0}

    for key in counted:
        entries = {d: index_maps[d][key] for d in contributing}

        # ---- Dimensions: majority vote ----
        for dim in DIMENSIONS:
            votes = []
            for d in contributing:
                o = map_dim(entries[d], dim)
                if o in ("Tie", fa, fb):
                    votes.append(o)
            if votes:
                dim_counts[dim][majority(votes)] += 1
            else:
                dim_counts[dim]["ParseError"] += 1  # all folders failed to parse

        # ---- Overall winner: judge by score in each folder → majority vote ----
        votes = []
        for d in contributing:
            sc = map_scores(entries[d])
            if fa in sc and fb in sc:
                if sc[fa] > sc[fb]:
                    votes.append(fa)
                elif sc[fb] > sc[fa]:
                    votes.append(fb)
                else:
                    votes.append("Tie")
        if votes:
            overall_counts[majority(votes)] += 1
        else:
            overall_counts["ParseError"] += 1

        # ---- Average score: same entry averaged across folders ----
        for model in (fa, fb):
            vals = []
            for d in contributing:
                sc = map_scores(entries[d])
                if model in sc:
                    vals.append(sc[model])
            if vals:
                score_sum[model] += sum(vals) / len(vals)
                score_cnt[model] += 1

    return {
        "pair_name": pair_name,
        "fa": fa,
        "fb": fb,
        "contributing": contributing,
        "counted": len(counted),
        "skipped": skipped,
        "dim_counts": dim_counts,
        "overall_counts": overall_counts,
        "score_sum": score_sum,
        "score_cnt": score_cnt,
    }


def _avg(stat, model):
    return stat["score_sum"][model] / stat["score_cnt"][model] if stat["score_cnt"][model] else 0.0


def print_pair(stat):
    fa, fb = stat["fa"], stat["fb"]
    sa, sb = short_name(fa), short_name(fb)
    print(f"\n{'='*72}")
    print(f"File pair: {stat['pair_name']}")
    print(f"  Contributing folders ({len(stat['contributing'])}): {', '.join(stat['contributing'])}")
    print(f"  Counted items: {stat['counted']}  (missing items skipped: {stat['skipped']})")
    print(f"  A = {fa}")
    print(f"  B = {fb}")
    print(f"{'='*72}")

    # Average scores
    print("\n  [Average score] (average each item across folders, then average over items)")
    print(f"    {sa:<30} {_avg(stat, fa):.2f}  ({stat['score_cnt'][fa]} items)")
    print(f"    {sb:<30} {_avg(stat, fb):.2f}  ({stat['score_cnt'][fb]} items)")

    # Overall Win Rate (majority vote)
    oc = stat["overall_counts"]
    wa = win_rate(oc[fa], oc[fb], oc["Tie"])
    wb = win_rate(oc[fb], oc[fa], oc["Tie"])
    print("\n  [Overall win rate] (majority vote)")
    print(f"    {sa:<30} {wa:6.1f}%")
    print(f"    {sb:<30} {wb:6.1f}%")
    print(f"    {'Tie':<30} {oc['Tie']} items | NoMajority: {oc['NoMajority']} items | ParseError: {oc['ParseError']} items")

    # Per-dimension Win Rate (majority vote)
    parts_a, parts_b, parts_tie = [], [], []
    for d in DIMENSIONS:
        c = stat["dim_counts"][d]
        parts_a.append(f"{DIM_LABELS[d]} {win_rate(c[fa], c[fb], c['Tie']):.0f}%")
        parts_b.append(f"{DIM_LABELS[d]} {win_rate(c[fb], c[fa], c['Tie']):.0f}%")
        parts_tie.append(f"{DIM_LABELS[d]} {c['Tie']}")
    print("\n  [Dimension win rate] (majority vote)")
    print(f"    {sa:<16} " + " | ".join(parts_a))
    print(f"    {sb:<16} " + " | ".join(parts_b))
    print(f"    {'Tie Count':<16} " + " | ".join(parts_tie))


def print_aggregate(pairs):
    """Aggregate across all comparison pairs: average score per model + overall Win Rate."""
    print(f"\n\n{'#'*72}")
    print("# Cross-pair summary (per model: average score + overall win rate)")
    print(f"{'#'*72}")

    agg = defaultdict(lambda: {"sum": 0.0, "cnt": 0, "wins": 0, "losses": 0, "ties": 0})
    for st in pairs:
        fa, fb = st["fa"], st["fb"]
        oc = st["overall_counts"]
        for model in (fa, fb):
            a = agg[model]
            a["sum"] += st["score_sum"][model]
            a["cnt"] += st["score_cnt"][model]
        a = agg[fa]
        a["wins"] += oc[fa]
        a["losses"] += oc[fb]
        a["ties"] += oc["Tie"]
        b = agg[fb]
        b["wins"] += oc[fb]
        b["losses"] += oc[fa]
        b["ties"] += oc["Tie"]

    rows = []
    for model, s in agg.items():
        if s["cnt"] == 0:
            continue
        avg = s["sum"] / s["cnt"]
        wr = win_rate(s["wins"], s["losses"], s["ties"])
        rows.append((short_name(model), avg, wr, s["cnt"]))

    if not rows:
        print("  No valid data")
        return

    rows.sort(key=lambda x: -x[1])
    colw = max(len(r[0]) for r in rows) + 2
    print(f"\n  {'Model':<{colw}} {'Avg Score':>8} {'Win Rate':>10} {'Items':>6}")
    print(f"  {'-' * (colw + 28)}")
    for name, avg, wr, cnt in rows:
        print(f"  {name:<{colw}} {avg:>8.2f} {wr:>9.1f}% {cnt:>6}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate comparison results across multiple result folders (majority vote + average score)"
    )
    parser.add_argument("result_dirs", nargs="+",
                        help="One or more result folders, e.g. results_kimi_ml results_ds_ml results_qwen_ml")
    args = parser.parse_args()

    for d in args.result_dirs:
        if not os.path.isdir(d):
            print(f"Error: folder does not exist: {d}", file=sys.stderr)
            sys.exit(1)

    print(f"Aggregating folders ({len(args.result_dirs)}): {', '.join(args.result_dirs)}")
    print("Rule: entries must exist in all relevant folders to be counted; Win Rate=majority vote; average score=mean across folders")

    # pair_name -> {dir: entries}
    all_pairs = {}
    for d in args.result_dirs:
        for name, entries in load_dir(d).items():
            all_pairs.setdefault(name, {})[d] = entries

    if not all_pairs:
        print("No *_comparison_results.json files found", file=sys.stderr)
        sys.exit(1)

    pairs = []
    for name in sorted(all_pairs):
        per_dir = all_pairs[name]
        contributing = [d for d in args.result_dirs if d in per_dir]
        missing = [d for d in args.result_dirs if d not in per_dir]
        if missing:
            print(f"\n[Warning] comparison pair '{name}' missing folders: {', '.join(missing)}", file=sys.stderr)
        if len(contributing) < 2:
            print(f"[Skip] comparison pair '{name}' appears in only {len(contributing)} folder(s), fewer than two, cannot take a majority vote",
                  file=sys.stderr)
            continue
        pairs.append(aggregate_pair(name, per_dir, contributing))

    if not pairs:
        print("\nNo comparison pair satisfies the majority-vote condition (at least 2 folders must contain the result).",
              file=sys.stderr)
        sys.exit(1)

    for st in pairs:
        print_pair(st)
    print_aggregate(pairs)


if __name__ == "__main__":
    main()
