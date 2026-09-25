#!/usr/bin/env python
"""Aggregate ARC-V Regime-A results across seeds into mean +/- std + a verdict.

Reads every outputs/arcv_regime_a_s<seed>.json produced by run_arcv.py (one per
seed) and reports, per method, the mean and standard deviation of the held-out
studio->wild accuracy and collapse. The headline is whether ARC-V beats ERM on the
held-out source by a margin that clears the seed spread.

    python scripts/eval/aggregate_arcv_seeds.py
"""
import os, sys, glob, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"


def mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, var ** 0.5


def main():
    files = sorted(f for f in glob.glob(str(OUT / "arcv_regime_a_s*.json"))
                   if "smoke" not in f)
    if not files:
        raise SystemExit("no per-seed result files (outputs/arcv_regime_a_s*.json). "
                         "Run: python pipelines/run_arcv.py --seed <seed>")
    seeds, per_method = [], {}   # method -> {"held": [...], "collapse": [...]}
    for f in files:
        d = json.load(open(f))
        seeds.append(d["seed"])
        for row in d["rows"]:
            method, sw, held, collapse = row[0], row[1], row[2], row[3]
            per_method.setdefault(method, {"held": [], "collapse": []})
            per_method[method]["held"].append(held)
            per_method[method]["collapse"].append(collapse)

    order = ["arcv", "fourier", "mixstyle", "erm"]
    methods = [m for m in order if m in per_method] + \
              [m for m in per_method if m not in order]

    lines = [f"# ARC-V Regime A -- aggregate over {len(seeds)} seeds {sorted(seeds)}\n",
             "Held-out = studio(Indian)-trained -> wild(SS)-test. Mean +/- std.\n",
             "| Method | Held-out acc (mean +/- std) | Collapse pp (mean +/- std) | seeds |",
             "|---|---|---|---|"]
    for m in methods:
        h_m, h_s = mean_std(per_method[m]["held"])
        c_m, c_s = mean_std(per_method[m]["collapse"])
        n = len(per_method[m]["held"])
        lines.append(f"| {m} | {h_m:.2f} +/- {h_s:.2f} | {c_m:.2f} +/- {c_s:.2f} | {n} |")
    text = "\n".join(lines)

    if "erm" in per_method and "arcv" in per_method:
        e_m, e_s = mean_std(per_method["erm"]["held"])
        a_m, a_s = mean_std(per_method["arcv"]["held"])
        gain = a_m - e_m
        pooled = (e_s ** 2 + a_s ** 2) ** 0.5
        verdict = "ARC-V beats ERM" if gain > 0 else "ARC-V does not beat ERM"
        note = ("margin exceeds the pooled seed spread" if gain > pooled
                else "margin is within the seed spread -- not yet conclusive")
        text += (f"\n\nVerdict: ERM {e_m:.2f} vs ARC-V {a_m:.2f} on the held-out source. "
                 f"{verdict} by {gain:+.2f} pp ({note}; pooled std {pooled:.2f}).")
        if len(seeds) < 3:
            text += f"\nNote: only {len(seeds)} seed(s); 3 are wanted for the paper."

    (OUT / "arcv_regime_a_aggregate.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
