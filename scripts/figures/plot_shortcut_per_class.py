"""
Per-class cross-source accuracy — the asymmetric shortcut signature.

Averages per-class accuracy across the per-seed matrices
(outputs/shortcut_test_matrix_s*.json) if present, else the single-seed file.

Fixes over the old figure:
  * legend moved OUTSIDE the axes -> coriander's ~-92 pp drop is finally visible
    (it used to be hidden behind a lower-left in-axes legend)
  * colorblind-safe Okabe-Ito colors, vector PDF via figstyle
"""
import sys, os, json, glob
from pathlib import Path

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import numpy as np
from src import figstyle

ROOT = Path(_base)
OUT = ROOT / "outputs" / "shortcut_per_class"


def _matrices():
    per_seed = sorted(glob.glob(str(ROOT / "outputs" / "shortcut_test_matrix_s*.json")))
    per_seed = [p for p in per_seed if "smoke" not in Path(p).name.lower()]
    if per_seed:
        return per_seed
    single = ROOT / "outputs" / "shortcut_test_matrix.json"
    return [str(single)] if single.exists() else []


def _mean_per_class(mats, train, test):
    """Average per-class acc across matrices for results[train][test]."""
    acc = {}
    for m in mats:
        pc = json.load(open(m))[train][test]["per_class"]
        for c, d in pc.items():
            acc.setdefault(c, []).append(d["acc"])
    return {c: float(np.mean(v)) for c, v in acc.items()}


def main():
    mats = _matrices()
    if not mats:
        raise SystemExit("no shortcut matrix JSON found — run eval first")
    figstyle.apply()
    import matplotlib.pyplot as plt

    ss_w = _mean_per_class(mats, "ss", "ss")   # SS-trained · SS-test (within)
    ss_c = _mean_per_class(mats, "ss", "in")   # SS-trained · Indian-test (cross, free)
    in_w = _mean_per_class(mats, "in", "in")   # Indian-trained · Indian-test (within)
    in_c = _mean_per_class(mats, "in", "ss")   # Indian-trained · SS-test (cross, broken)

    classes = sorted(in_w.keys())
    arr = lambda d: np.array([d.get(c, np.nan) for c in classes])
    sw, sc, iw, ic = arr(ss_w), arr(ss_c), arr(in_w), arr(in_c)

    drop = iw - ic                              # the broken-direction drop
    order = np.argsort(-drop)                    # biggest collapse first (coriander)
    classes = [classes[i] for i in order]
    sw, sc, iw, ic, drop = sw[order], sc[order], iw[order], ic[order], drop[order]

    x = np.arange(len(classes)); w = 0.2
    fig, ax = plt.subplots(figsize=(12, 5.6))
    P = figstyle.PALETTE
    ax.bar(x - 1.5*w, sw, w, label="SS-trained · SS-test (within)",        color=P["within"])
    ax.bar(x - 0.5*w, sc, w, label="SS-trained · Indian-test (free)",       color=P["cross_free"])
    ax.bar(x + 0.5*w, iw, w, label="Indian-trained · Indian-test (within)", color=P["studio"])
    ax.bar(x + 1.5*w, ic, w, label="Indian-trained · SS-test (collapse)",   color=P["cross_broken"])

    for i, d in enumerate(drop):
        if not np.isnan(d) and d > 0.02:
            ax.text(x[i] + 1.5*w, ic[i] + 0.015, f"−{d*100:.0f}pp",
                    ha="center", va="bottom", fontsize=8.5,
                    color=figstyle.OKABE_ITO["vermillion"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0.0, 1.08)
    ax.axhline(0.125, color=P["chance"], lw=1, ls=":", label="random (1/8)")
    n = len(mats)
    seed_note = f"mean over {n} seeds" if n > 1 else "single seed"
    ax.set_title("Per-class cross-source accuracy — the collapse is class-concentrated "
                 f"({seed_note})", fontweight="bold")
    # legend OUTSIDE the axes (right) — never occludes the coriander bar
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=1.0, borderaxespad=0)
    ax.grid(axis="y", alpha=0.25)

    figstyle.save(fig, str(OUT))
    print("\nPer-class collapse (Indian-trained: within - cross):")
    for c, d in zip(classes, drop):
        print(f"  {c:16s} {d*100:+.1f} pp")


if __name__ == "__main__":
    main()
