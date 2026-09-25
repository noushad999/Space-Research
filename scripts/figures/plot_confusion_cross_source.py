"""
Cross-source confusion matrices — the 2x2 as four confusion heatmaps.

Kills the "accuracy is an artifact" objection: shows WHERE the studio-trained
model's predictions collapse in the wild (per-class error structure), not just a
scalar. Reads a shortcut_evidence_*.json (written by eval_confusion.py).
"""
import sys, os, glob, json
from pathlib import Path

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import numpy as np
from src import figstyle

ROOT = Path(_base)
OUT = ROOT / "outputs" / "confusion_cross_source"


def _pick():
    ev = [p for p in sorted(glob.glob(str(ROOT / "outputs" / "shortcut_evidence_s*.json")))
          if "smoke" not in Path(p).name.lower() and "dedup" not in Path(p).name.lower()]
    return ev[0] if ev else None


def main():
    src = _pick()
    if not src:
        raise SystemExit("no shortcut_evidence_s*.json found — run eval_confusion.py first")
    figstyle.apply()
    import matplotlib.pyplot as plt

    r = json.load(open(src))
    classes = r["classes"]
    cells = [("ss", "ss", "Wild-trained / wild-test (within)"),
             ("ss", "in", "Wild-trained / studio-test (free)"),
             ("in", "ss", "Studio-trained / wild-test (COLLAPSE)"),
             ("in", "in", "Studio-trained / studio-test (within)")]

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for ax, (mt, tt, title) in zip(axes.flat, cells):
        cm = np.array(r[mt][tt]["confusion"], float)
        cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
        im = ax.imshow(cmn, cmap=figstyle.MATRIX_CMAP, vmin=0, vmax=1, aspect="auto")
        acc = r[mt][tt]["acc"]; f1 = r[mt][tt]["macro_f1"]
        ax.set_title(f"{title}\nacc={acc*100:.1f}%  macro-F1={f1*100:.1f}%",
                     fontsize=10.5, fontweight="bold")
        ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_xlabel("predicted", fontsize=9); ax.set_ylabel("true", fontsize=9)
        for i in range(len(classes)):
            for j in range(len(classes)):
                v = cmn[i, j]
                if v > 0.02:
                    ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                            fontsize=6, color="white" if v < 0.6 else "black")
    fig.suptitle("Cross-source confusion structure (row-normalized)",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    figstyle.save(fig, str(OUT))


if __name__ == "__main__":
    main()
