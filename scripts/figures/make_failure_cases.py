#!/usr/bin/env python
"""Qualitative failure cases: real in-the-wild images that the studio-trained
SpiceFusionNet misclassifies, captioned with the (wrong) predicted class. The
emotional core of a shortcut-learning result. Saves outputs/failure_cases.{png,pdf}.
"""
import sys, os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)
from src.model import SpiceFusionNet
from src.dataset import load_manifest_splits, SpiceDataset, get_val_transform

ROOT = Path(_base)
CKPT = ROOT / "outputs/checkpoints/overlap_indian_s42/p1_best.pth"   # studio-trained
SS_M = str(ROOT / "outputs/manifest_overlap_ss.json")                # wild test
PRIORITY = ["coriander", "black_pepper", "green_cardamom", "cumin", "cloves", "nutmeg"]


def nice(c): return c.replace("_", " ")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits, classes = load_manifest_splits(SS_M)
    paths, labels = splits["test"]
    labels = np.array(labels)
    model = SpiceFusionNet(num_classes=len(classes)).to(device)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"]); model.eval()

    ds = SpiceDataset(paths, labels.tolist(), get_val_transform(), multimodal=False)
    dl = DataLoader(ds, batch_size=64, num_workers=2)
    preds, conf = [], []
    with torch.no_grad():
        for imgs, tex, col, y in dl:
            p = torch.softmax(model.forward_image(imgs.to(device)), 1)
            c, k = p.max(1)
            preds.extend(k.cpu().tolist()); conf.extend(c.cpu().tolist())
    preds, conf = np.array(preds), np.array(conf)
    cls_idx = {c: i for i, c in enumerate(classes)}

    # one dramatic misclassification per priority class (highest-confidence wrong)
    picks = []
    for c in PRIORITY:
        if c not in cls_idx:
            continue
        ti = cls_idx[c]
        cand = np.where((labels == ti) & (preds != ti))[0]
        if len(cand):
            picks.append(cand[np.argmax(conf[cand])])
        if len(picks) == 6:
            break

    fig, axes = plt.subplots(2, 3, figsize=(9, 6.4))
    for ax, i in zip(axes.ravel(), picks):
        try:
            ax.imshow(mpimg.imread(paths[i]))
        except Exception:
            ax.set_facecolor("#eee")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"true: {nice(classes[labels[i]])}\npredicted: {nice(classes[preds[i]])}"
                     f"  ({conf[i]*100:.0f}%)", fontsize=9)
        for s in ax.spines.values():
            s.set_edgecolor("#c0504d"); s.set_linewidth(1.6)
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    fig.suptitle("In-the-wild images the studio-trained model gets wrong",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = ROOT / "outputs" / "failure_cases"
    fig.savefig(str(out) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    print("saved", out, "| picks:", [f"{nice(classes[labels[i]])}->{nice(classes[preds[i]])}" for i in picks])


if __name__ == "__main__":
    main()
