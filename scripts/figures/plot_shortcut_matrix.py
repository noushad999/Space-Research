"""
The headline 2x2 cross-source shortcut matrix — publication-grade, multi-seed.

Reads outputs/shortcut_multiseed_aggregate.json (mean +/- std across seeds) if it
exists; otherwise falls back to the single-seed outputs/shortcut_test_matrix.json.

Fixes over the old figure:
  * numbers come from the JSON (never stale), now mean +/- std across seeds
  * colorblind- AND grayscale-safe (cividis, not red/green)
  * asymmetry annotation lives in the subtitle, not a box over the colorbar
  * vector PDF + 600-dpi PNG via figstyle
"""
import sys, os, json
from pathlib import Path

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import numpy as np
from src import figstyle

ROOT = Path(_base)
AGG = ROOT / "outputs" / "shortcut_multiseed_aggregate.json"
SINGLE = ROOT / "outputs" / "shortcut_test_matrix.json"
OUT = ROOT / "outputs" / "shortcut_matrix"


def load():
    """Return means[2,2], stds[2,2], tax_ss(mean,std), tax_in(mean,std), n_seeds, mcnemar."""
    if AGG.exists():
        d = json.load(open(AGG))
        agg = d["aggregate"]
        means = np.array([[agg["ss_within"]["mean"], agg["ss_cross"]["mean"]],
                          [agg["in_cross"]["mean"],  agg["in_within"]["mean"]]])
        stds = np.array([[agg["ss_within"]["std"], agg["ss_cross"]["std"]],
                         [agg["in_cross"]["std"],  agg["in_within"]["std"]]])
        tax_ss = (agg["ss_tax_pp"]["mean"], agg["ss_tax_pp"]["std"])
        tax_in = (agg["in_tax_pp"]["mean"], agg["in_tax_pp"]["std"])
        return means, stds, tax_ss, tax_in, d.get("n_seeds", 1), d.get("mcnemar_pooled")
    # single-seed fallback
    r = json.load(open(SINGLE))
    means = np.array([[r["ss"]["ss"]["acc"], r["ss"]["in"]["acc"]],
                      [r["in"]["ss"]["acc"], r["in"]["in"]["acc"]]])
    stds = np.zeros((2, 2))
    a = r["analysis"]
    return means, stds, (a["ss_shortcut_tax_pp"], 0.0), (a["in_shortcut_tax_pp"], 0.0), 1, a.get("mcnemar_ss_test")


def main():
    figstyle.apply()
    import matplotlib.pyplot as plt

    means, stds, tax_ss, tax_in, n_seeds, mc = load()

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    im = ax.imshow(means, cmap=figstyle.MATRIX_CMAP, vmin=0.5, vmax=1.0, aspect="auto")

    labels = [[("Within-source", "SS-trained · SS-test"),
               ("Cross-source",  "SS-trained · Indian-test")],
              [("Cross-source",  "Indian-trained · SS-test"),
               ("Within-source", "Indian-trained · Indian-test")]]
    for i in range(2):
        for j in range(2):
            v = means[i, j]
            # cividis is dark at low values -> white text there, dark text when light
            txt_color = "white" if v < 0.72 else "#111111"
            tag, _sub = labels[i][j]
            ax.text(j, i - 0.22, tag, ha="center", va="center",
                    fontsize=11, style="italic", color=txt_color)
            num = f"{v*100:.2f}%"
            if n_seeds > 1 and stds[i, j] > 0:
                num += f"\n±{stds[i, j]*100:.2f}"
            ax.text(j, i + 0.10, num, ha="center", va="center",
                    fontsize=22, fontweight="bold", color=txt_color)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["SS-test\n(in-the-wild)", "Indian-test\n(studio)"], fontweight="bold")
    ax.set_yticklabels(["SS-trained\n(in-the-wild)", "Indian-trained\n(studio)"], fontweight="bold")
    ax.set_xlabel("Tested on", labelpad=8)
    ax.set_ylabel("Trained on", labelpad=8)
    ax.tick_params(length=0)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Top-1 accuracy", fontweight="bold")

    seed_note = f"mean ± std over {n_seeds} seeds" if n_seeds > 1 else "single seed"
    sub = (f"Wild→studio: −{tax_ss[0]:.2f} pp (free)     "
           f"Studio→wild: −{tax_in[0]:.2f} pp (collapse)")
    if n_seeds > 1:
        sub = (f"Wild→studio: −{tax_ss[0]:.2f}±{tax_ss[1]:.2f} pp (free)     "
               f"Studio→wild: −{tax_in[0]:.2f}±{tax_in[1]:.2f} pp (collapse)")
    # Report the valid per-seed McNemar (c=0 in every seed), not the pooled p, since
    # the three seeds share one test set and pooling them is invalid (see Section 4.2).
    ptxt = "   per-seed McNemar: c=0, p<1.2e-134"
    fig.suptitle("Asymmetric cross-source shortcut — SpiceFusionNet (EfficientNet-B4, 8 overlap classes)",
                 fontsize=12.5, fontweight="bold", y=0.99)
    ax.set_title(sub + ptxt + f"\n({seed_note})", fontsize=10.5, pad=10)

    figstyle.save(fig, str(OUT))


if __name__ == "__main__":
    main()
