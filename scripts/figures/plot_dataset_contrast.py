"""
Studio vs in-the-wild dataset contrast on the 8 overlap classes.

Two rows (studio = Indian_Spices, wild = Spice_Spectrum), one sample image per
class. Makes the paper's core visual argument visible: studio = uniform
background, wild = chaotic real-world scenes -> the domain gap you can SEE.
"""
import sys, os, json
from pathlib import Path

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import numpy as np
from PIL import Image
from src import figstyle

ROOT = Path(_base)
OUT = ROOT / "outputs" / "dataset_contrast"
SS_MANIFEST = ROOT / "outputs" / "manifest_overlap_ss.json"
IN_MANIFEST = ROOT / "outputs" / "manifest_overlap_indian.json"


def _resolve(p: str) -> str:
    """Resolve a manifest path onto the local data root."""
    from src.dataset import resolve_image_path
    return resolve_image_path(p)


def _classes(manifest):
    m = json.load(open(manifest))
    return [c["name"] for c in sorted(m["classes"], key=lambda c: c["index"])]


def _imgs_by_class(manifest):
    """One existing sample image path per class."""
    m = json.load(open(manifest))
    classes = _classes(manifest)
    out = {}
    for split in ("test", "val", "train"):
        for p, y in m["samples"][split]:
            c = classes[int(y)]
            if c not in out:
                path = _resolve(p)
                if os.path.exists(path):
                    out[c] = path
    return out


def _square(path, size=256):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return np.array(img.resize((size, size)))


def main():
    figstyle.apply()
    import matplotlib.pyplot as plt

    classes = _classes(SS_MANIFEST)
    wild = _imgs_by_class(SS_MANIFEST)
    studio = _imgs_by_class(IN_MANIFEST)

    n = len(classes)
    fig, axes = plt.subplots(2, n, figsize=(1.7 * n, 3.9))
    rows = [("STUDIO\n(Indian)", studio, figstyle.PALETTE["studio"]),
            ("IN THE WILD\n(SpiceSpectrum)", wild, figstyle.PALETTE["wild"])]

    for r, (rlabel, imgs, color) in enumerate(rows):
        for c, cls in enumerate(classes):
            ax = axes[r, c]
            path = imgs.get(cls)
            if path:
                ax.imshow(_square(path))
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cls.replace("_", "\n"), fontsize=9)
            if c == 0:
                ax.set_ylabel(rlabel, fontsize=10, fontweight="bold", color=color)
            for s in ax.spines.values():
                s.set_edgecolor(color); s.set_linewidth(2)

    fig.suptitle("Same 8 classes, two acquisition sources — the cross-source domain gap",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    figstyle.save(fig, str(OUT))


if __name__ == "__main__":
    main()
