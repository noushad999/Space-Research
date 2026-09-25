"""
figstyle.py — one shared, publication-grade figure design system.

Import this at the top of every plot script:

    import figstyle
    figstyle.apply()
    ...
    figstyle.save(fig, "outputs/shortcut_matrix")   # writes .pdf AND .png (600 dpi)

Design goals (Transactions/CEA-grade):
  * Colorblind-safe (Okabe-Ito categorical + cividis sequential — both survive
    grayscale printing, unlike the old red/green).
  * Embedded TrueType fonts in PDF (pdf.fonttype=42) so no Type-3 AQC warnings.
  * White background, uniform typography, vector PDF + 600-dpi PNG fallback.

Nothing here needs a GPU; it is pure matplotlib configuration.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Okabe-Ito colorblind-safe categorical palette ────────────────────────────
OKABE_ITO = {
    "black":        "#000000",
    "orange":       "#E69F00",
    "sky":          "#56B4E9",
    "green":        "#009E73",
    "yellow":       "#F0E442",
    "blue":         "#0072B2",
    "vermillion":   "#D55E00",
    "purple":       "#CC79A7",
    "grey":         "#8C8C8C",
}

# ── Semantic roles for THIS paper (kept consistent across every figure) ───────
# SS = Spice_Spectrum = in-the-wild (the robust/hero source)
# IN = Indian_Spices  = studio
PALETTE = {
    "wild":          OKABE_ITO["blue"],        # SS-trained / in-the-wild
    "studio":        OKABE_ITO["orange"],      # Indian-trained / studio
    "within":        OKABE_ITO["green"],       # within-source (good)
    "cross_free":    OKABE_ITO["sky"],         # the free cross direction (wild->studio)
    "cross_broken":  OKABE_ITO["vermillion"],  # the collapse (studio->wild)
    "chance":        OKABE_ITO["grey"],        # random baseline line
}

# Sequential map for the accuracy heatmap: colorblind- AND grayscale-safe.
MATRIX_CMAP = "cividis"


def apply():
    """Set global rcParams. Call once at the top of a plot script."""
    plt.rcParams.update({
        "pdf.fonttype":       42,     # embed TrueType (no Type-3)
        "ps.fonttype":        42,
        "svg.fonttype":       "none",
        "font.family":        "sans-serif",
        "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size":          11,
        "axes.titlesize":     12,
        "axes.labelsize":     11,
        "axes.labelweight":   "bold",
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    9,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "savefig.facecolor":  "white",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          False,
        "figure.dpi":         120,
    })


def save(fig, stem, dpi_png: int = 600, also_tiff: bool = False):
    """Save `fig` to `<stem>.pdf` (vector) and `<stem>.png` (600 dpi), white bg.

    `stem` may include a directory; parents are created. Returns the list of paths.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=dpi_png, bbox_inches="tight", facecolor="white")
    out += [pdf, png]
    if also_tiff:
        tif = stem.with_suffix(".tiff")
        fig.savefig(tif, dpi=300, bbox_inches="tight", facecolor="white",
                    pil_kwargs={"compression": "tiff_lzw"})
        out.append(tif)
    plt.close(fig)
    for p in out:
        print(f"  saved {p}")
    return out
