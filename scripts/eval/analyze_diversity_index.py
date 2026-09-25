"""
Quantified acquisition diversity per source (the measured "why").

The claim is that the studio source has a uniform, low-entropy background and the
wild source is varied. We measure it: per image we take the border region (a
background proxy) and its mean brightness/colour, and summarise the spread across
each source. The wild source should show far higher background variability and
colour entropy --- the diversity that makes wild-trained models generalize.

Pure CPU (PIL + numpy).
"""
import sys, os, json, glob
from pathlib import Path

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import numpy as np
from PIL import Image
from src import figstyle

ROOT = Path(_base)
OUT = ROOT / "outputs" / "diversity_index"
CAP = 700


def _resolve(p: str) -> str:
    """Resolve a manifest path onto the local data root."""
    from src.dataset import resolve_image_path
    return resolve_image_path(p)


def _paths(manifest):
    m = json.load(open(manifest))
    ps = []
    for split in m["samples"]:
        ps += [_resolve(p) for p, _ in m["samples"][split]]
    ps = [p for p in ps if os.path.exists(p)]
    if len(ps) > CAP:
        idx = np.random.RandomState(0).choice(len(ps), CAP, replace=False)
        ps = [ps[i] for i in idx]
    return ps


def _stats(path, n=128, f=16):
    im = np.asarray(Image.open(path).convert("RGB").resize((n, n)), float)
    mask = np.zeros((n, n), bool)
    mask[:f] = mask[-f:] = mask[:, :f] = mask[:, -f:] = True   # border frame
    border = im[mask]                                          # (P, 3)
    bmean = border.mean(0)                                     # mean border colour
    # colour entropy of the whole image (coarse 4x4x4 RGB histogram)
    q = (im // 64).astype(int).clip(0, 3)
    idx = q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]
    hist = np.bincount(idx.ravel(), minlength=64).astype(float)
    p = hist / hist.sum()
    ent = -(p[p > 0] * np.log2(p[p > 0])).sum()
    return bmean, ent


def _source(manifest):
    bmeans, ents = [], []
    for p in _paths(manifest):
        try:
            bm, e = _stats(p); bmeans.append(bm); ents.append(e)
        except Exception:
            pass
    bmeans = np.array(bmeans); ents = np.array(ents)
    bright = bmeans.mean(1)                          # border brightness per image
    return {"bg_brightness": bright, "entropy": ents,
            "bg_spread": float(bmeans.std(0).mean()),      # cross-image bg colour spread
            "entropy_mean": float(ents.mean()), "n": int(len(ents))}


def main():
    studio = _source(str(ROOT / "outputs/manifest_overlap_indian.json"))
    wild = _source(str(ROOT / "outputs/manifest_overlap_ss.json"))
    json.dump({"studio": {k: v for k, v in studio.items() if not isinstance(v, np.ndarray)},
               "wild": {k: v for k, v in wild.items() if not isinstance(v, np.ndarray)}},
              open(ROOT / "outputs" / "diversity_index.json", "w"), indent=2)

    figstyle.apply()
    import matplotlib.pyplot as plt
    P = figstyle.PALETTE
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))
    bins = np.linspace(0, 255, 40)
    axA.hist(studio["bg_brightness"], bins=bins, color=P["studio"], alpha=0.75,
             label=f"studio (spread {studio['bg_spread']:.1f})")
    axA.hist(wild["bg_brightness"], bins=bins, color=P["wild"], alpha=0.75,
             label=f"wild (spread {wild['bg_spread']:.1f})")
    axA.set_xlabel("background brightness (border region)"); axA.set_ylabel("images")
    axA.set_title("Background variability by source", fontsize=11)
    axA.legend(loc="upper left", fontsize=9); axA.grid(axis="y", alpha=0.25)

    axB.hist(studio["entropy"], bins=30, color=P["studio"], alpha=0.75,
             label=f"studio (mean {studio['entropy_mean']:.2f})")
    axB.hist(wild["entropy"], bins=30, color=P["wild"], alpha=0.75,
             label=f"wild (mean {wild['entropy_mean']:.2f})")
    axB.set_xlabel("image colour entropy (bits)"); axB.set_ylabel("images")
    axB.set_title("Colour entropy by source", fontsize=11)
    axB.legend(loc="upper left", fontsize=9); axB.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    figstyle.save(fig, str(OUT))
    print(f"studio: bg-spread {studio['bg_spread']:.1f}, entropy {studio['entropy_mean']:.2f}")
    print(f"wild:   bg-spread {wild['bg_spread']:.1f}, entropy {wild['entropy_mean']:.2f}")


if __name__ == "__main__":
    main()
