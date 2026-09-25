"""
t-SNE of backbone features colored by SOURCE — "features cluster by source".

Extracts EfficientNet-B4 backbone features on both sources' overlap test images
and projects to 2D. If the two sources form separate clusters, the backbone has
encoded acquisition source (the substrate of the shortcut). Direct visual analog
of the cross-dataset-bias result other agri-vision work reports.

Uses an overlap checkpoint produced by run_cross_source.py (backbone features are
class-count-independent, so the 8-class Phase-1 checkpoint is fine).
"""
import sys, os, glob, re
from pathlib import Path

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import figstyle
from src.model import SpiceFusionNet
from src.dataset import load_manifest_splits, SpiceDataset, get_val_transform

ROOT = Path(_base)
OUT = ROOT / "outputs" / "source_tsne"
SS_MANIFEST = str(ROOT / "outputs" / "manifest_overlap_ss.json")
IN_MANIFEST = str(ROOT / "outputs" / "manifest_overlap_indian.json")
CAP = 800   # max points per source for a readable/fast t-SNE


def _find_ckpt():
    ss = sorted(glob.glob(str(ROOT / "outputs/checkpoints/overlap_ss_s*/p1_best.pth")))
    if ss:
        return min(ss, key=lambda p: int(re.search(r"_s(\d+)", p).group(1)))
    return None


@torch.no_grad()
def _features(model, manifest, device):
    splits, _ = load_manifest_splits(manifest)
    paths, labels = splits["test"]
    paths = [p for p in paths if os.path.exists(p)]
    labels = labels[:len(paths)]
    if len(paths) > CAP:
        idx = np.random.RandomState(0).choice(len(paths), CAP, replace=False)
        paths = [paths[i] for i in idx]; labels = [labels[i] for i in idx]
    ds = SpiceDataset(paths, labels, get_val_transform(), multimodal=False)
    dl = DataLoader(ds, batch_size=64, num_workers=2, pin_memory=True)
    feats = []
    for imgs, tex, col, y in dl:
        feats.append(model.backbone(imgs.to(device)).cpu().numpy())
    return np.concatenate(feats)


def main():
    ckpt = _find_ckpt()
    if not ckpt:
        raise SystemExit("no overlap checkpoint found yet — run training first")
    from sklearn.manifold import TSNE
    figstyle.apply()
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpiceFusionNet(num_classes=8).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"]); model.eval()

    f_wild = _features(model, SS_MANIFEST, device)
    f_studio = _features(model, IN_MANIFEST, device)
    X = np.concatenate([f_wild, f_studio])
    src = np.array([0] * len(f_wild) + [1] * len(f_studio))

    proj = TSNE(n_components=2, perplexity=30, init="pca",
                learning_rate="auto", random_state=42).fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 7))
    for s, name, color in [(0, "in-the-wild (SS)", figstyle.PALETTE["wild"]),
                           (1, "studio (Indian)",  figstyle.PALETTE["studio"])]:
        m = src == s
        ax.scatter(proj[m, 0], proj[m, 1], s=10, c=color, label=name, alpha=0.6,
                   edgecolors="none")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title("Backbone features cluster by acquisition source\n"
                 "(overlap classes; if the two colors separate, the model encodes source)",
                 fontweight="bold")
    ax.legend(loc="best", framealpha=1.0, markerscale=1.8)
    ax.set_xticks([]); ax.set_yticks([])
    figstyle.save(fig, str(OUT))


if __name__ == "__main__":
    main()
