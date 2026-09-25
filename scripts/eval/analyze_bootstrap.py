"""
Bootstrap 95% CI + effect size on the cross-source shortcut gap.

The valid PAIRED effect is on the SS-test (in-the-wild) set: SS-trained accuracy
minus Indian-trained accuracy over the SAME samples (the ~38 pp shortcut gap).
We bootstrap that difference (pooled across seeds + per seed) and report the
effect size (risk difference + McNemar odds ratio).

Consumes the per-sample correctness dumps written by eval_shortcut_test.py
(results["per_sample"]["ss_test"]). Pure CPU, no GPU, no re-eval.

    python scripts/eval/analyze_bootstrap.py --matrices outputs/shortcut_test_matrix_s*.json
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np


def _pool(matrices, which="ss_test"):
    ss, ind = [], []
    per_seed = []
    for m in matrices:
        r = json.load(open(m))
        ps = (r.get("per_sample") or {}).get(which)
        if not ps:
            continue
        s = np.asarray(ps["ss_trained_correct"], float)
        i = np.asarray(ps["in_trained_correct"], float)
        per_seed.append((Path(m).name, s, i))
        ss.append(s); ind.append(i)
    if not ss:
        return None, None, per_seed
    return np.concatenate(ss), np.concatenate(ind), per_seed


def _bootstrap(ss, ind, B=10000, seed=0):
    """Bootstrap CI of mean(ss) - mean(ind) over the same (paired) samples."""
    rng = np.random.default_rng(seed)
    n = len(ss)
    idx = rng.integers(0, n, size=(B, n))
    gaps = ss[idx].mean(1) - ind[idx].mean(1)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return float((ss.mean() - ind.mean())), float(lo), float(hi)


def _effect(ss, ind):
    """McNemar discordant odds ratio + risk difference on paired correctness."""
    b = int(((ss == 1) & (ind == 0)).sum())   # SS right, Indian wrong
    c = int(((ss == 0) & (ind == 1)).sum())   # SS wrong, Indian right
    odds = (b / c) if c else float("inf")
    return {"b_ss_right_in_wrong": b, "c_ss_wrong_in_right": c,
            "odds_ratio": odds, "risk_difference": float(ss.mean() - ind.mean())}


def _report(name, matrices, out):
    result = {"direction": name, "matrices": [Path(m).name for m in matrices]}
    for which, tag in (("ss_test", "collapse_gap_on_wild_test"),
                       ("in_test", "free_gap_on_studio_test")):
        ss, ind, per_seed = _pool(matrices, which)
        if ss is None:
            continue
        est, lo, hi = _bootstrap(ss, ind)
        eff = _effect(ss, ind)
        block = {"n_samples": int(len(ss)), "n_seeds": len(per_seed),
                 "gap_pp": round(est * 100, 3),
                 "ci95_pp": [round(lo * 100, 3), round(hi * 100, 3)],
                 "effect": eff,
                 "per_seed_gap_pp": [{"file": f, "gap_pp": round((s.mean() - i.mean()) * 100, 3)}
                                     for f, s, i in per_seed]}
        result[tag] = block
        print(f"\n[{tag}]  n={block['n_samples']} over {block['n_seeds']} seed(s)")
        print(f"  gap = {block['gap_pp']:.2f} pp   95% CI [{block['ci95_pp'][0]:.2f}, {block['ci95_pp'][1]:.2f}]")
        print(f"  odds ratio (discordant b/c) = {eff['odds_ratio']:.1f}  (b={eff['b_ss_right_in_wrong']}, c={eff['c_ss_wrong_in_right']})")
    json.dump(result, open(out, "w"), indent=2)
    print(f"  -> {out}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", nargs="+",
                    default=sorted(glob.glob("outputs/shortcut_test_matrix_s*.json")))
    ap.add_argument("--label", default="standard")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    mats = [m for m in args.matrices if "smoke" not in Path(m).name.lower()]
    if not mats:
        raise SystemExit("no matrices with per_sample dumps found")
    out = args.out or f"outputs/bootstrap_ci_{args.label}.json"
    _report(args.label, mats, out)


if __name__ == "__main__":
    main()
