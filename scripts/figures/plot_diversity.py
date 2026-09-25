"""
Diversity fix — training on both sources closes the cross-source gap.

Single-source (studio-trained): studio-test ~100%, wild-test ~62% (big gap).
Both-source (diverse):          studio-test ~99%,  wild-test ~99%  (gap ~0).
The prescription: diversity, not architecture.

Reads diversity_aggregate.json + shortcut_multiseed_aggregate.json. Clean style.
"""
import sys, os, json
from pathlib import Path

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import numpy as np
from src import figstyle

ROOT = Path(_base)
OUT = ROOT / "outputs" / "diversity_fix"


def main():
    div_p = ROOT / "outputs" / "diversity_aggregate.json"
    sc_p = ROOT / "outputs" / "shortcut_multiseed_aggregate.json"
    if not div_p.exists():
        raise SystemExit("diversity_aggregate.json missing — run run_diversity.py first")
    div = json.load(open(div_p))
    both_studio, both_wild = div["acc_studio"]["mean"], div["acc_wild"]["mean"]
    if sc_p.exists():
        a = json.load(open(sc_p))["aggregate"]
        single_studio, single_wild = a["in_within"]["mean"], a["in_cross"]["mean"]
    else:
        single_studio, single_wild = 1.0, 0.6223

    figstyle.apply()
    import matplotlib.pyplot as plt
    P = figstyle.PALETTE

    groups = ["single-source\n(studio-trained)", "both-source\n(diverse)"]
    studio_test = [single_studio, both_studio]
    wild_test = [single_wild, both_wild]
    x = np.arange(2); w = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.bar(x - w / 2, studio_test, w, label="studio test", color=P["studio"])
    ax.bar(x + w / 2, wild_test, w, label="wild test", color=P["wild"])
    for xi, (s, wv) in enumerate(zip(studio_test, wild_test)):
        ax.text(xi - w / 2, s + 0.015, f"{s*100:.0f}", ha="center", fontsize=9)
        ax.text(xi + w / 2, wv + 0.015, f"{wv*100:.0f}", ha="center", fontsize=9)
        gap = abs(s - wv) * 100
        ax.text(xi, max(s, wv) + 0.07, f"gap {gap:.0f} pp", ha="center", fontsize=10,
                color=(P["cross_broken"] if gap > 5 else P["within"]), fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Top-1 accuracy"); ax.set_ylim(0, 1.15)
    ax.axhline(0.125, color=P["chance"], ls=":", lw=1, label="random (1/8)")
    ax.set_title("Training-source diversity closes the cross-source gap", fontsize=11)
    ax.legend(loc="lower center", ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    figstyle.save(fig, str(OUT))


if __name__ == "__main__":
    main()
