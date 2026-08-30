#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_consistency.py — Comprehensive consistency analysis (intra-model + inter-model).

Input: answer/stablility/stability/<model>/round_{1..5}_scores.json
Output:
  1. Intra-model consistency: flip rate of a model's judgments across 5 rounds (rubric-level / item-level)
  2. Inter-model consistency:
       - Pairwise agreement / Cohen's Kappa
       - Agreement rate with the four-model consensus (majority vote) + lenient/strict direction bias
       - Conditional agreement rate (e.g., fraction where Y also judges 1 when X judges 1)
       - Per-rubric pass rate and cross-model disagreement
  All results are saved as JSON, and a readable report is printed.
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Use a relative path based on the script directory
DEFAULT_BASE_DIR = os.path.join(SCRIPT_DIR, "answer", "stablility", "stability")
DEFAULT_MODELS = ["qwen3_4b", "kimi_k2_6", "qwen3_vl_32b", "deepseek_v4_flash"]
NUM_RUBRICS = 10


# =============================================================================
# Loading
# =============================================================================
def load_rounds(base_dir: str, model: str) -> list:
    """Return the model's {idx_N: {score, individual_scores}} for all rounds."""
    mdir = os.path.join(base_dir, model)
    rounds = []
    r = 1
    while True:
        p = os.path.join(mdir, f"round_{r}_scores.json")
        if not os.path.isfile(p):
            break
        with open(p, encoding="utf-8") as fh:
            rounds.append(json.load(fh))
        r += 1
    return rounds


def per_rubric_per_round(rounds: list) -> dict:
    """{(idx, rid): [0/1 across rounds]}, only keeping rounds that have judgments."""
    out = defaultdict(list)
    for res in rounds:
        for key, v in res.items():
            try:
                idx = int(key.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            for ind in v.get("individual_scores", []):
                rid, sc = ind.get("rubric_id"), ind.get("score")
                if rid is not None and sc is not None:
                    out[(idx, rid)].append(int(sc))
    return dict(out)


def majority(seq: list) -> int:
    c = Counter(seq)
    return 1 if c[1] > c[0] else 0


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    p_obs = float(np.mean(a == b))
    p_exp = float(np.mean(a == 1)) * float(np.mean(b == 1)) + \
            float(np.mean(a == 0)) * float(np.mean(b == 0))
    denom = 1.0 - p_exp
    return float("nan") if denom <= 1e-12 else (p_obs - p_exp) / denom


# =============================================================================
# Intra-model consistency
# =============================================================================
def within_model_stats(rubric_series: dict) -> dict:
    """For a single model's (idx,rid) -> [round judgments], summarize internal flips."""
    n_pairs = len(rubric_series)
    flipped_pairs = 0
    flip_counts = Counter()          # distribution of flip counts
    stable_5 = 0                     # identical across all 5 rounds and all have values
    item_flips = defaultdict(int)    # idx -> number of flipped rubrics
    for (idx, rid), vals in rubric_series.items():
        if min(vals) == max(vals):
            if len(vals) >= 2:
                flip_counts[0] += 1
            if len(vals) >= 2 and len(set(vals)) == 1:
                stable_5 += 1
        else:
            flipped_pairs += 1
            item_flips[idx] += 1
            # flip count: switches between distinct values, approximated as occurrences - unique values
            flip_counts[len(set(vals)) - 1] += 1

    n_flip_events = sum(item_flips.values())
    return {
        "n_(idx,rubric)_pairs": n_pairs,
        "flip_rate": flipped_pairs / n_pairs if n_pairs else None,
        "n_items_with_any_flip": len(item_flips),
        "total_flip_events": n_flip_events,
        "mean_flip_per_item": n_flip_events / n_pairs * 10 if n_pairs else None,
        "flip_distribution": {str(k): v for k, v in sorted(flip_counts.items())},
    }


# =============================================================================
# Main function
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Comprehensive consistency analysis")
    ap.add_argument("--base-dir", "-d", default=DEFAULT_BASE_DIR)
    ap.add_argument("--models", "-m", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--out", "-o", default=None)
    args = ap.parse_args()

    models = args.models
    all_rubric = {}   # model -> {(idx,rid): [round judgments]}
    for m in models:
        rounds = load_rounds(args.base_dir, m)
        all_rubric[m] = per_rubric_per_round(rounds)
        print(f"[Load] {m}: {len(rounds)} rounds, {len(all_rubric[m])} (idx,rubric) judgments")

    report = {"models": models, "within_model": {}, "between_model": {}}

    # ---------------- Intra-model consistency ----------------
    print("\n" + "=" * 100)
    print("1. Intra-model consistency (whether the same model's judgments flip across 5 rounds)")
    print("=" * 100)
    print(f"{'Model':<18}{'Pairs':>8}{'FlipRate':>10}{'FlipItems':>10}{'TotalFlips':>12}")
    for m in models:
        s = within_model_stats(all_rubric[m])
        report["within_model"][m] = s
        print(f"{m:<18}{s['n_(idx,rubric)_pairs']:>8}"
              f"{s['flip_rate']:>10.4f}"
              f"{s['n_items_with_any_flip']:>10}"
              f"{s['total_flip_events']:>12}")

    # ---------------- Consensus ----------------
    # Four-model majority-vote consensus (for each (idx,rid), take the model's cross-round
    # majority vote, then vote across models)
    model_verdict = {}  # m -> {(idx,rid): 0/1}
    for m in models:
        model_verdict[m] = {k: majority(v) for k, v in all_rubric[m].items()}
    consensus = {}
    for key in set().union(*[set(v) for v in model_verdict.values()]):
        votes = [model_verdict[m][key] for m in models if key in model_verdict[m]]
        if len(votes) >= 2:   # at least two models have judgments
            consensus[key] = majority(votes)
    n_cons = len(consensus)
    print(f"\nConsensus coverage: {n_cons} (idx,rubric) judgments (>=2 models have judgments)")

    # ---------------- Inter-model consistency ----------------
    print("\n" + "=" * 100)
    print("2. Inter-model consistency")
    print("=" * 100)

    # 2a. Pairwise comparison
    print("\n[2a] Pairwise agreement / Cohen's Kappa")
    pairs = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            ma, mb = models[i], models[j]
            va = model_verdict[ma]
            vb = model_verdict[mb]
            common = [k for k in va if k in vb]
            a = np.array([va[k] for k in common])
            b = np.array([vb[k] for k in common])
            agree = float(np.mean(a == b))
            kappa = cohen_kappa(a, b)
            # Directionality: A=1&B=0 vs A=0&B=1
            n10 = int(np.sum((a == 1) & (b == 0)))
            n01 = int(np.sum((a == 0) & (b == 1)))
            pairs.append((ma, mb, len(common), agree, kappa, n10, n01))
            print(f"  {ma} vs {mb}: N={len(common)} agree={agree:.4f} kappa={kappa:.4f} "
                  f"(A1B0={n10}, A0B1={n01})")
    report["between_model"]["pairwise"] = [
        {"model_A": a, "model_B": b, "n": n, "agreement": ag, "kappa": ka,
         "A1_B0": x, "A0_B1": y}
        for a, b, n, ag, ka, x, y in pairs
    ]

    # 2b. Agreement with consensus + direction bias
    print("\n[2b] Each model vs four-model consensus (majority vote): agreement rate + lenient/strict bias")
    print(f"{'Model':<18}{'N':>7}{'Agree':>10}{'Kappa':>9}{'Bias(1vs0)':>16}{'Dir':>8}")
    consensus_bias = {}
    for m in models:
        vals = [(model_verdict[m].get(k, -1), consensus[k]) for k in consensus]
        vals = [(a, c) for a, c in vals if a != -1]
        a = np.array([x for x, _ in vals])
        c = np.array([y for _, y in vals])
        agree = float(np.mean(a == c))
        kappa = cohen_kappa(a, c)
        # Bias: model gives 1 and consensus gives 0 minus model gives 0 and consensus gives 1
        m1_c0 = int(np.sum((a == 1) & (c == 0)))
        m0_c1 = int(np.sum((a == 0) & (c == 1)))
        bias = m1_c0 - m0_c1
        direction = "lenient" if bias > 0 else ("strict" if bias < 0 else "balanced")
        consensus_bias[m] = {"n": int(len(a)), "agreement": agree, "kappa": kappa,
                             "over_approve(m1c0)": m1_c0, "under_approve(m0c1)": m0_c1,
                             "bias": bias, "direction": direction}
        print(f"{m:<18}{len(a):>7}{agree:>10.4f}{kappa:>9.4f}{bias:>16}{direction:>8}")
    report["between_model"]["vs_consensus"] = consensus_bias

    # 2c. Conditional agreement rate: fraction where Y also judges 1 when X judges 1
    print("\n[2c] Conditional agreement rate (p(Y=1|X=1) / p(Y=0|X=0))")
    header = "{:<16}".format("X\\Y") + "".join(f"{m:>16}" for m in models)
    print(header)
    cond = {}
    for x in models:
        row = [f"{x:<16}"]
        cond[x] = {}
        for y in models:
            va, vb = model_verdict[x], model_verdict[y]
            common = [k for k in va if k in vb and va[k] == 1]
            if common:
                p1 = float(np.mean([vb[k] for k in common]))
            else:
                p1 = float("nan")
            common0 = [k for k in va if k in vb and va[k] == 0]
            if common0:
                p0 = float(np.mean([vb[k] for k in common0]))
            else:
                p0 = float("nan")
            cond[x][y] = {"p(Y=1|X=1)": p1, "p(Y=0|X=0)": p0}
            row.append(f"{p1:.3f}/{p0:.3f}")
        print("".join(row))
    report["between_model"]["conditional"] = cond

    # 2d. Per-rubric pass rate and cross-model disagreement
    print("\n[2d] Per-rubric: each model's pass rate + cross-model agreement rate")
    print(f"{'Rubric':<8}" + "".join(f"{m:>10}" for m in models)
          + f"{'Agree':>12}")
    per_rubric = {}
    for rid in range(1, NUM_RUBRICS + 1):
        pass_rates = {}
        agree_list = []
        for m in models:
            vals = [v for (_, r), v in model_verdict[m].items() if r == rid]
            pass_rates[m] = float(np.mean(vals)) if vals else None
        # Average of pairwise agreement rates
        rates = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                ma, mb = models[i], models[j]
                va = np.array([v for (_, r), v in model_verdict[ma].items() if r == rid])
                vb = np.array([v for (_, r), v in model_verdict[mb].items() if r == rid])
                if len(va) == len(vb) and len(va):
                    rates.append(float(np.mean(va == vb)))
        per_rubric[str(rid)] = {"pass_rates": pass_rates,
                                "mean_pairwise_agreement": float(np.mean(rates)) if rates else None}
        print(f"  #{rid:<6}" + "".join(
            f"{pass_rates[m]:>10.3f}" for m in models)
              + f"{per_rubric[str(rid)]['mean_pairwise_agreement']:>12.4f}")
    report["between_model"]["per_rubric"] = per_rubric

    # ---------------- Saving ----------------
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()
