"""
Teaser / graphical abstract — the one-glance asymmetric shortcut.

Left : the 2x2 cross-source matrix (mean across seeds if available).
Right: the same coriander model scoring ~100% on studio images and ~7-8% in the
       wild — the emotional core of the finding.

Elsevier graphical-abstract spec: >=1328x531 px, 300 dpi, TIFF/PDF, NO AI art.
This is built from matplotlib + real dataset images only. figstyle saves PDF,
600-dpi PNG, and (also_tiff) a 300-dpi TIFF.
"""
import sys, os, json, glob
from pathlib import Path

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import numpy as np
from PIL import Image
from src import figstyle

ROOT = Path(_base)
OUT = ROOT / "outputs" / "teaser"
SS_MANIFEST = ROOT / "outputs" / "manifest_overlap_ss.json"
IN_MANIFEST = ROOT / "outputs" / "manifest_overlap_indian.json"


def _resolve(p: str) -> str:
    """Resolve a manifest path onto the local data root."""
    from src.dataset import resolve_image_path
    return resolve_image_path(p)


def _img_for_class(manifest, class_name):
    m = json.load(open(manifest))
    classes = [c["name"] for c in sorted(m["classes"], key=lambda c: c["index"])]
    if class_name not in classes:
        return None
    idx = classes.index(class_name)
    for split in ("test", "val", "train"):
        for p, y in m["samples"][split]:
            if int(y) == idx:
                path = _resolve(p)
                if os.path.exists(path):
                    return path
    return None


def _matrix_and_coriander():
    agg = ROOT / "outputs" / "shortcut_multiseed_aggregate.json"
    mats = [p for p in sorted(glob.glob(str(ROOT / "outputs" / "shortcut_test_matrix_s*.json")))
            if "smoke" not in Path(p).name.lower()] \
        or [str(ROOT / "outputs" / "shortcut_test_matrix.json")]
    if agg.exists():
        a = json.load(open(agg))["aggregate"]
        M = np.array([[a["ss_within"]["mean"], a["ss_cross"]["mean"]],
                      [a["in_cross"]["mean"],  a["in_within"]["mean"]]])
    else:
        r = json.load(open(mats[0]))
        M = np.array([[r["ss"]["ss"]["acc"], r["ss"]["in"]["acc"]],
                      [r["in"]["ss"]["acc"], r["in"]["in"]["acc"]]])
    # coriander per-class, averaged across available matrices
    cs, cw = [], []
    for m in mats:
        r = json.load(open(m))
        pc_w = r["in"]["in"]["per_class"].get("coriander")   # studio within
        pc_c = r["in"]["ss"]["per_class"].get("coriander")   # wild cross
        if pc_w: cs.append(pc_w["acc"])
        if pc_c: cw.append(pc_c["acc"])
    cor_studio = float(np.mean(cs)) if cs else float("nan")
    cor_wild = float(np.mean(cw)) if cw else float("nan")
    return M, cor_studio, cor_wild


def main():
    figstyle.apply()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    M, cor_studio, cor_wild = _matrix_and_coriander()

    fig = plt.figure(figsize=(14, 5.6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.18)

    # ── Left: 2x2 matrix ────────────────────────────────────────────────
    axm = fig.add_subplot(gs[0, 0])
    im = axm.imshow(M, cmap=figstyle.MATRIX_CMAP, vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(2):
        for j in range(2):
            v = M[i, j]
            axm.text(j, i, f"{v*100:.1f}%", ha="center", va="center",
                     fontsize=17, fontweight="bold",
                     color="white" if v < 0.72 else "#111111")
    axm.set_xticks([0, 1]); axm.set_yticks([0, 1])
    axm.set_xticklabels(["wild\ntest", "studio\ntest"], fontsize=10)
    axm.set_yticklabels(["wild\ntrain", "studio\ntrain"], fontsize=10)
    axm.set_title("Cross-source accuracy", fontsize=11, fontweight="bold")
    axm.tick_params(length=0)

    # ── Middle + right: coriander studio vs wild ────────────────────────
    specs = [
        (gs[0, 1], IN_MANIFEST, "coriander", f"STUDIO: {cor_studio*100:.0f}%",
         figstyle.PALETTE["within"]),
        (gs[0, 2], SS_MANIFEST, "coriander", f"IN THE WILD: {cor_wild*100:.1f}%",
         figstyle.PALETTE["cross_broken"]),
    ]
    for spec, manifest, cls, caption, color in specs:
        ax = fig.add_subplot(spec)
        path = _img_for_class(manifest, cls)
        if path:
            ax.imshow(np.array(Image.open(path).convert("RGB")))
        else:
            ax.text(0.5, 0.5, "image\nunavailable", ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(caption, fontsize=13, fontweight="bold", color=color)
        for s in ax.spines.values():
            s.set_edgecolor(color); s.set_linewidth(3)

    fig.suptitle("Studio spice benchmarks lie: the cross-source shortcut is one-sided\n"
                 "A coriander model at ~100% on studio images collapses to single digits in the wild",
                 fontsize=13, fontweight="bold", y=1.02)
    figstyle.save(fig, str(OUT), also_tiff=True)


if __name__ == "__main__":
    main()
