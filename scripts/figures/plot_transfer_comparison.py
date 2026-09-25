"""
Cross-source feature-transfer mechanism figure (publication-grade).

Reads the three source-probe JSONs and shows how class-discriminative features
TRANSFER across acquisition sources for each training regime. This is the
mechanistic evidence that the collapse lives in the FEATURES: a studio-trained
backbone's studio->wild transfer is low, while diverse/both-source training
transfers freely.

Fixes over the old figure:
  * internal jargon removed ("M1", "f_cnn", "fused", "Ind->SS") -> reader labels
  * chance line labeled; colorblind palette; vector PDF via figstyle
"""
import sys, os, json
from pathlib import Path

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import numpy as np
from src import figstyle

ROOT = Path(_base)
OUT = ROOT / "outputs" / "transfer_comparison"

REGIMES = [
    ("outputs/source_probe.json",        "Both-source\n(unified)"),
    ("outputs/source_probe_ss.json",     "Wild-trained\n(SS)"),
    ("outputs/source_probe_indian.json", "Studio-trained\n(Indian)"),
]
# (json-key, reader label, color)
SERIES = [
    ("ss_within",    "Wild within",   figstyle.PALETTE["within"]),
    ("indian_within","Studio within", figstyle.PALETTE["studio"]),
    ("ss->indian",   "Wild→Studio",   figstyle.PALETTE["cross_free"]),
    ("indian->ss",   "Studio→Wild",   figstyle.PALETTE["cross_broken"]),
]
STREAMS = [("f_cnn", "CNN features"), ("fused", "Fused decision features")]


def load():
    data = {}
    for path, label in REGIMES:
        p = ROOT / path
        if p.exists():
            data[label] = json.load(open(p)).get("transfer")
        else:
            print(f"  [missing] {path}")
    return data


def main():
    data = load()
    if not data:
        raise SystemExit("no source-probe JSONs found (run eval_source_probe.py first)")
    figstyle.apply()
    import matplotlib.pyplot as plt

    labels = [lab for _, lab in REGIMES if lab in data]
    x = np.arange(len(labels)); w = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
    for ax, (stream, stitle) in zip(axes, STREAMS):
        for i, (key, leg, col) in enumerate(SERIES):
            vals = []
            for lab in labels:
                s = (data[lab] or {}).get(stream, {})
                v = s.get(key) if s else None
                vals.append(v if v is not None else 0.0)
            ax.bar(x + (i - 1.5) * w, vals, w, label=leg, color=col)
        ax.axhline(0.125, ls=":", color=figstyle.PALETTE["chance"], lw=1,
                   label="random (1/8)")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title(stitle, fontweight="bold")
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Class balanced accuracy")
    # single shared legend outside, below
    handles, leglabels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leglabels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.06), framealpha=1.0)
    fig.suptitle("Cross-source feature transfer by training regime\n"
                 "Studio-only training makes class features source-specific (low Studio→Wild); "
                 "diversity restores transfer",
                 fontsize=12, fontweight="bold", y=1.04)
    figstyle.save(fig, str(OUT))


if __name__ == "__main__":
    main()
