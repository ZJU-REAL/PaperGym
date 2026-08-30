#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_rubric_agreement.py — Compare scoring models pairwise on the judgment agreement
for each rubric.

For each model, take the cross-round majority vote of the 5 rounds as the model's final
binary judgment (0/1) for each (idx, rubric_id). Then compare two models rubric by rubric,
outputting:

  - Per-rubric judgment agreement rate
  - Cohen's Kappa (corrected for chance agreement)
  - Each model's pass rate per rubric (fraction judged as 1) — reveals strict/lenient differences
  - Confusion matrix (A=1&B=1 / A=1&B=0 / A=0&B=1 / A=0&B=0)

Usage:
    python analyze_rubric_agreement.py --models qwen3_4b qwen3_vl_32b
    python analyze_rubric_agreement.py --base-dir answer/stablility/stability
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Use a relative path based on the script directory
DEFAULT_BASE_DIR = os.path.join(SCRIPT_DIR, "answer", "stablility", "stability")
NUM_RUBRICS = 10


def load_model_verdicts(base_dir: str, model: str) -> dict:
    """Return {(idx, rubric_id): cross-round majority vote 0/1}."""
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
    if not rounds:
        return {}

    votes = defaultdict(list)  # (idx, rubric_id) -> [0/1 across rounds]
    for res in rounds:
        for key, v in res.items():
            try:
                idx = int(key.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            for ind in v.get("individual_scores", []):
                rid = ind.get("rubric_id")
                sc = ind.get("score")
                if rid is None or sc is None:
                    continue
                votes[(idx, rid)].append(int(sc))

    verdicts = {}
    for k, vals in votes.items():
        cnt = Counter(vals)
        verdicts[k] = 1 if cnt[1] > cnt[0] else 0  # majority vote (tie resolves to 0)
    return verdicts


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's Kappa for binary judgments."""
    n = len(a)
    if n == 0:
        return float("nan")
    p_obs = float(np.mean(a == b))
    p1a, p0a = float(np.mean(a == 1)), float(np.mean(a == 0))
    p1b, p0b = float(np.mean(b == 1)), float(np.mean(b == 0))
    p_exp = p1a * p1b + p0a * p0b
    denom = 1.0 - p_exp
    if denom <= 1e-12:
        return float("nan")
    return (p_obs - p_exp) / denom


def main() -> None:
    p = argparse.ArgumentParser(description="Pairwise comparison of scoring models' judgment agreement on each rubric")
    p.add_argument("--base-dir", "-d", default=DEFAULT_BASE_DIR,
                   help="Root directory containing each model's summary/round files")
    p.add_argument("--models", "-m", nargs="+", required=True,
                   help="Models to compare, e.g. qwen3_4b qwen3_vl_32b")
    p.add_argument("--out", "-o", default=None, help="Output JSON path")
    args = p.parse_args()

    if len(args.models) < 2:
        print("[ERROR] at least 2 models are required")
        return

    # Load all model judgments
    verdicts = {m: load_model_verdicts(args.base_dir, m) for m in args.models}
    for m, v in verdicts.items():
        print(f"[Load] {m}: {len(v)} (idx, rubric) judgments (cross-round majority vote)")

    # Common keys (items with judgments from both models)
    results = {}

    for i in range(len(args.models)):
        for j in range(i + 1, len(args.models)):
            ma, mb = args.models[i], args.models[j]
            va, vb = verdicts[ma], verdicts[mb]
            common = set(va) & set(vb)
            if not common:
                print(f"[WARNING] {ma} and {mb} have no common judgments, skipping")
                continue

            pair = {"n_common": len(common), "per_rubric": {}}
            all_a, all_b = [], []
            for rid in range(1, NUM_RUBRICS + 1):
                a = np.array([va[k] for k in common if k[1] == rid])
                b = np.array([vb[k] for k in common if k[1] == rid])
                if len(a) == 0:
                    continue
                agree = float(np.mean(a == b))
                kappa = cohen_kappa(a, b)
                conf = {
                    "11": int(np.sum((a == 1) & (b == 1))),
                    "10": int(np.sum((a == 1) & (b == 0))),
                    "01": int(np.sum((a == 0) & (b == 1))),
                    "00": int(np.sum((a == 0) & (b == 0))),
                }
                pair["per_rubric"][str(rid)] = {
                    "n": int(len(a)),
                    "agreement": agree,
                    "kappa": kappa,
                    "pass_rate_A": float(np.mean(a)),
                    "pass_rate_B": float(np.mean(b)),
                    "confusion": conf,
                }
                all_a.extend(a.tolist())
                all_b.extend(b.tolist())

            all_a = np.array(all_a)
            all_b = np.array(all_b)
            pair["overall_agreement"] = float(np.mean(all_a == all_b))
            pair["overall_kappa"] = cohen_kappa(all_a, all_b)
            results[f"{ma}__vs__{mb}"] = pair

            # ---- Printing ----
            print("\n" + "=" * 78)
            print(f"  {ma}  vs  {mb}   ({len(common)} common judgments)")
            print("=" * 78)
            print(f"  Overall agreement: {pair['overall_agreement']:.4f}  Overall Cohen's Kappa: {pair['overall_kappa']:.4f}")
            print(f"\n  {'Rubric':<8}{'N':>5}{'Agree':>10}{'Kappa':>9}{'PassA':>10}{'PassB':>10}  Confusion(11/10/01/00)")
            for rid, pr in pair["per_rubric"].items():
                conf = pr["confusion"]
                print(f"  #{rid:<7}{pr['n']:>5}{pr['agreement']:>10.4f}"
                      f"{pr['kappa'] if pr['kappa']==pr['kappa'] else float('nan'):>9.4f}"
                      f"{pr['pass_rate_A']:>10.3f}{pr['pass_rate_B']:>10.3f}"
                      f"  ({conf['11']}/{conf['10']}/{conf['01']}/{conf['00']})")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()
