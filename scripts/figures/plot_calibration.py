"""
Calibration figure — the studio model is confidently WRONG in the wild.

Left : reliability diagram. Studio-trained within-source sits on the diagonal
       (well-calibrated); tested in the wild it falls far below (overconfident).
Right: confidence histogram of the studio model on the wild test — most of its
       WRONG predictions are still high-confidence.

Clean scientific style (white, minimal). Reads calibration_s*.json.
"""
import sys, os, glob, json
from pathlib import Path

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import numpy as np
from src import figstyle

ROOT = Path(_base)
OUT = ROOT / "outputs" / "calibration"


def _pool(cell):
    """Concatenate confidence/correct across per-seed calibration files for a cell."""
    files = [f for f in sorted(glob.glob(str(ROOT / "outputs" / "calibration_s*.json")))
             if "smoke" not in Path(f).name.lower()]
    conf, corr = [], []
    for f in files:
        d = json.load(open(f))[cell[0]][cell[1]]
        conf += d["confidence"]; corr += d["correct"]
    return np.asarray(conf), np.asarray(corr, float)


def _reliability(conf, correct, n_bins=12):
    bins = np.linspace(0, 1, n_bins + 1)
    cen, acc = [], []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            cen.append((bins[i] + bins[i + 1]) / 2); acc.append(correct[m].mean())
    return np.array(cen), np.array(acc)


def _ece(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0; N = len(conf)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            e += (m.sum() / N) * abs(correct[m].mean() - conf[m].mean())
    return e


def main():
    within_c, within_ok = _pool(("in", "in"))   # studio-trained, studio-test
    cross_c, cross_ok = _pool(("in", "ss"))      # studio-trained, wild-test
    if len(cross_c) == 0:
        raise SystemExit("no calibration_s*.json found — run eval_calibration.py first")

    figstyle.apply()
    import matplotlib.pyplot as plt
    P = figstyle.PALETTE

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))

    # ── reliability diagram ─────────────────────────────────────────────────
    axA.plot([0, 1], [0, 1], ls="--", color="#999", lw=1, label="perfect calibration")
    for (c, ok, color, lab) in [(within_c, within_ok, P["studio"], "studio-test (within)"),
                                (cross_c, cross_ok, P["cross_broken"], "wild-test (cross)")]:
        cen, acc = _reliability(c, ok)
        axA.plot(cen, acc, "-o", color=color, ms=5, lw=2,
                 label=f"{lab}  ECE={_ece(c, ok)*100:.0f}%")
    axA.set_xlabel("confidence"); axA.set_ylabel("accuracy")
    axA.set_xlim(0, 1.02); axA.set_ylim(0, 1.02)
    axA.set_title("Studio-trained model: reliability", fontsize=11)
    axA.legend(loc="upper left", fontsize=9)
    axA.grid(alpha=0.25)

    # ── confidence histogram on the wild test ───────────────────────────────
    bins = np.linspace(0, 1, 21)
    axB.hist(cross_c[cross_ok == 1], bins=bins, color=P["within"], alpha=0.75, label="correct")
    axB.hist(cross_c[cross_ok == 0], bins=bins, color=P["cross_broken"], alpha=0.75, label="wrong")
    axB.axvline(cross_c.mean(), color="#444", ls="--", lw=1,
                label=f"mean conf {cross_c.mean()*100:.0f}%")
    axB.set_xlabel("confidence"); axB.set_ylabel("samples")
    axB.set_title(f"Studio-trained on wild test  (acc {cross_ok.mean()*100:.0f}%)", fontsize=11)
    axB.legend(loc="upper left", fontsize=9)
    axB.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    figstyle.save(fig, str(OUT))
    print(f"cross ECE = {_ece(cross_c, cross_ok)*100:.1f}%  "
          f"(acc {cross_ok.mean()*100:.1f}%, mean-conf {cross_c.mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
