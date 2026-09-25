"""
aggregate_shortcut_seeds.py — multi-seed aggregation of the cross-source 2x2.

Reads the per-seed shortcut matrices written by eval_shortcut_test.py and reports
mean +/- std (and min/max range) of each cell and the asymmetric shortcut tax
across seeds — the significance evidence for the headline finding.

    python scripts/eval/aggregate_shortcut_seeds.py --matrices \
        outputs/shortcut_test_matrix.json \
        outputs/shortcut_test_matrix_s1337.json \
        outputs/shortcut_test_matrix_s2024.json
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/shortcut_multiseed_aggregate.json")
    args = ap.parse_args()

    keys = ["ss_within", "ss_cross", "in_within", "in_cross", "ss_tax_pp", "in_tax_pp"]
    cells = {k: [] for k in keys}
    per_seed = []
    mcnemar_per_seed = []
    pooled_ss, pooled_in = [], []
    for path in args.matrices:
        r = json.load(open(path))
        a = r["analysis"]
        row = {"file": path,
               "ss_within": r["ss"]["ss"]["acc"], "ss_cross": r["ss"]["in"]["acc"],
               "in_within": r["in"]["in"]["acc"], "in_cross": r["in"]["ss"]["acc"],
               "ss_tax_pp": a["ss_shortcut_tax_pp"], "in_tax_pp": a["in_shortcut_tax_pp"]}
        per_seed.append(row)
        for k in keys:
            cells[k].append(row[k])
        mc = a.get("mcnemar_ss_test")
        if mc:
            mcnemar_per_seed.append({"file": path, "p_value": mc["p_value"],
                                     "chi2": mc["chi2"], "b": mc["b_A_right_B_wrong"],
                                     "c": mc["c_A_wrong_B_right"]})
        ps = (r.get("per_sample") or {}).get("ss_test")
        if ps:
            pooled_ss += ps["ss_trained_correct"]
            pooled_in += ps["in_trained_correct"]

    agg = {}
    for k in keys:
        v = np.asarray(cells[k], float)
        agg[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  "min": float(v.min()), "max": float(v.max()), "n": int(len(v)),
                  "values": v.tolist()}

    pooled_mcnemar = None
    if pooled_ss:
        a_ = np.asarray(pooled_ss, bool); b_ = np.asarray(pooled_in, bool)
        n01 = int((a_ & ~b_).sum()); n10 = int((~a_ & b_).sum()); n = n01 + n10
        chi2 = (abs(n01 - n10) - 1) ** 2 / n if n else 0.0
        try:
            from scipy.stats import chi2 as _c
            pv = float(_c.sf(chi2, 1))
        except Exception:
            import math
            pv = math.erfc(math.sqrt(chi2 / 2.0)) if chi2 else 1.0
        pooled_mcnemar = {"b_ss_right_in_wrong": n01, "c_ss_wrong_in_right": n10,
                          "chi2": float(chi2), "p_value": float(pv),
                          "n_discordant": n, "n_seeds_pooled": len(args.matrices)}

    out = {"n_seeds": len(args.matrices), "per_seed": per_seed, "aggregate": agg,
           "mcnemar_per_seed": mcnemar_per_seed, "mcnemar_pooled": pooled_mcnemar}
    json.dump(out, open(args.out, "w"), indent=2)

    print(f"\nMulti-seed cross-source aggregate (n={len(args.matrices)} seeds)")
    print("-" * 60)
    for k in ("ss_within", "ss_cross", "in_within", "in_cross"):
        s = agg[k]
        print(f"  {k:11s} {s['mean']*100:6.2f} +/- {s['std']*100:4.2f} %  "
              f"[{s['min']*100:.2f}, {s['max']*100:.2f}]")
    for k in ("ss_tax_pp", "in_tax_pp"):
        s = agg[k]
        print(f"  {k:11s} {s['mean']:6.2f} +/- {s['std']:4.2f} pp [{s['min']:.2f}, {s['max']:.2f}]")
    if pooled_mcnemar:
        print(f"  McNemar (pooled SS-test, SS- vs Indian-trained): "
              f"chi2={pooled_mcnemar['chi2']:.1f}  p={pooled_mcnemar['p_value']:.2e}  "
              f"(b={pooled_mcnemar['b_ss_right_in_wrong']}, c={pooled_mcnemar['c_ss_wrong_in_right']})")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
