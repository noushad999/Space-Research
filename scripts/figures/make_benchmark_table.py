"""
Assemble the cross-source benchmark: every backbone vs the studio->wild collapse.

Reads bench_<model>.json (from eval_backbone_shortcut.py) plus SpiceFusionNet
(ours, from shortcut_multiseed_aggregate.json) and GranuFormer, and writes a
comparison table (markdown + LaTeX) and a clean collapse-leaderboard figure. The
takeaway: the collapse is architecture-universal.
"""
import sys, os, glob, json
from pathlib import Path

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import numpy as np
from src import figstyle

ROOT = Path(_base)
OUT = ROOT / "outputs" / "benchmark_collapse"


def collect():
    rows = []  # (name, studio_within, studio_to_wild, tax, macroF1_cross)
    # ours (3-seed)
    agg = ROOT / "outputs" / "shortcut_multiseed_aggregate.json"
    if agg.exists():
        a = json.load(open(agg))["aggregate"]
        rows.append(("SpiceFusionNet (ours)", a["in_within"]["mean"], a["in_cross"]["mean"],
                     a["in_tax_pp"]["mean"], None))
    gf = ROOT / "outputs" / "shortcut_test_matrix_granuformer.json"
    if gf.exists():
        r = json.load(open(gf))
        rows.append(("GranuFormer", r["in"]["in"]["acc"], r["in"]["ss"]["acc"],
                     (r["in"]["in"]["acc"] - r["in"]["ss"]["acc"]) * 100, None))
    for f in sorted(glob.glob(str(ROOT / "outputs" / "bench_*.json"))):
        r = json.load(open(f))
        rows.append((r["model"], r["in"]["in"]["acc"], r["in"]["ss"]["acc"],
                     r["in_tax_pp"], r["in"]["ss"].get("macro_f1")))
    return rows


def main():
    rows = collect()
    if not rows:
        raise SystemExit("no benchmark results yet — run run_benchmark.py first")
    rows.sort(key=lambda r: r[3])  # by collapse tax ascending (least collapse first)

    md = ["# Cross-source benchmark (Indian/studio-trained -> SS/wild test)\n",
          "| Model | Studio within | Studio->Wild | Collapse (pp) | Macro-F1 (wild) |",
          "|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{lrrrr}", r"\hline",
           r"Model & Studio within & Studio$\rightarrow$Wild & Collapse (pp) & Macro-F1 \\", r"\hline"]
    for name, sw, stw, tax, f1 in rows:
        f1s = f"{f1*100:.1f}" if f1 is not None else "--"
        md.append(f"| {name} | {sw*100:.1f} | {stw*100:.1f} | {tax:.1f} | {f1s} |")
        tex.append(f"{name} & {sw*100:.1f} & {stw*100:.1f} & {tax:.1f} & {f1s} \\\\")
    tex += [r"\hline", r"\end{tabular}"]
    (ROOT / "outputs" / "benchmark_table.md").write_text("\n".join(md), encoding="utf-8")
    (ROOT / "outputs" / "benchmark_table.tex").write_text("\n".join(tex), encoding="utf-8")

    # ── clean collapse-leaderboard figure ───────────────────────────────────
    figstyle.apply()
    import matplotlib.pyplot as plt
    P = figstyle.PALETTE
    names = [r[0] for r in rows]
    within = np.array([r[1] for r in rows]) * 100
    cross = np.array([r[2] for r in rows]) * 100
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(names) + 1.5))
    ax.barh(y + 0.18, within, 0.34, color=P["studio"], label="studio test (within)")
    ax.barh(y - 0.18, cross, 0.34, color=P["cross_broken"], label="studio→wild (cross)")
    for yi, c in zip(y, cross):
        ax.text(c + 1, yi - 0.18, f"{c:.0f}", va="center", fontsize=8)
    ax.axvline(12.5, color=P["chance"], ls=":", lw=1, label="random (1/8)")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Top-1 accuracy (%)"); ax.set_xlim(0, 108)
    ax.set_title("Studio-trained models all collapse on wild data", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    figstyle.save(fig, str(OUT))
    print("\n".join(md))


if __name__ == "__main__":
    main()
