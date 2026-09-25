"""
Grad-CAM contrast: studio-trained vs wild-trained model on the SAME wild images.

Uses HiResCAM (element-wise grad x activation) + robust post-processing to avoid
the vanilla Grad-CAM corner/padding artifact that otherwise dominates the map:
  * HiResCAM instead of gradient-averaged Grad-CAM
  * percentile (2-98) normalization so one hot pixel can't wash out the map
  * Gaussian smoothing + gentle raised-cosine border suppression (kills padding
    hotspots at the image edges/corners)
  * CAM-proportional alpha so the background is not uniformly tinted

Uses the Phase-1 image-only overlap checkpoints from run_cross_source.py.
"""
import sys, os, glob, re
from pathlib import Path

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src import figstyle
import config
from src.model import SpiceFusionNet
from src.dataset import load_manifest_splits

ROOT = Path(_base)
OUT = ROOT / "outputs" / "gradcam_contrast"
SS_MANIFEST = str(ROOT / "outputs" / "manifest_overlap_ss.json")
_RESIZE = int(config.IMG_SIZE * 256 / 224)
_CROP = config.IMG_SIZE
FOCUS = ["coriander", "black_pepper", "green_cardamom", "cumin"]


class HiResCAM:
    """HiResCAM on a chosen conv layer: relu(sum_c(grad_c . act_c))."""
    def __init__(self, model, target_layer):
        self.model = model
        self.acts = self.grads = None
        self._f = target_layer.register_forward_hook(self._fwd)
        self._b = target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o): self.acts = o.detach()
    def _bwd(self, m, gi, go): self.grads = go[0].detach()
    def remove(self): self._f.remove(); self._b.remove()

    def __call__(self, x):
        self.model.zero_grad()
        logits = self.model.forward_image(x)
        idx = int(logits.argmax(1).item())
        logits[0, idx].backward()
        cam = F.relu((self.grads * self.acts).sum(1)).squeeze(0)   # (h, w)
        return cam.cpu().numpy(), idx


def _border_window(n, edge=0.12, hard=0.03):
    """Raised-cosine window with a hard-zero outer margin: kills padding/corner
    hotspots. 0 in the outer `hard` fraction, cosine ramp to 1 by `edge`."""
    r = np.ones(n)
    e = max(1, int(n * edge)); h = max(1, int(n * hard))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, e)))
    r[:e] = ramp; r[-e:] = ramp[::-1]
    r[:h] = 0.0; r[-h:] = 0.0
    return np.outer(r, r).astype(np.float32)


def _keep_main_blobs(cam, thresh=0.4, rel=0.22):
    """Keep attention blobs that are >= `rel` of the largest blob's area; drop
    small isolated specks (corner/padding artifacts), position-independent."""
    mask = (cam >= thresh).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return cam
    amax = stats[1:, cv2.CC_STAT_AREA].max()
    keep = np.zeros_like(cam, np.float32)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= rel * amax:
            keep[lbl == i] = 1.0
    keep = cv2.GaussianBlur(keep, (0, 0), sigmaX=cam.shape[0] * 0.025)  # soft edges
    return cam * np.clip(keep, 0, 1)


def _postprocess(cam, size):
    cam = cv2.resize(cam.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=size * 0.03)
    lo, hi = np.percentile(cam, 2), np.percentile(cam, 99)  # robust normalize FIRST
    cam = np.clip((cam - lo) / (hi - lo + 1e-8), 0, 1)
    cam *= _border_window(size)                            # THEN suppress border (not undone by norm)
    cam = _keep_main_blobs(cam)                            # drop isolated corner specks
    return cam


def _find_ckpts():
    ss = sorted(glob.glob(str(ROOT / "outputs/checkpoints/overlap_ss_s*/p1_best.pth")))
    ind = sorted(glob.glob(str(ROOT / "outputs/checkpoints/overlap_indian_s*/p1_best.pth")))
    if not ss or not ind:
        return None, None
    seed = lambda p: int(re.search(r"_s(\d+)", p).group(1))
    return min(ss, key=seed), min(ind, key=seed)


def _load(ckpt, device):
    m = SpiceFusionNet(num_classes=8).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m


def _view(path):
    img = np.array(Image.open(path).convert("RGB"))
    disp = A.Compose([A.Resize(_RESIZE, _RESIZE), A.CenterCrop(_CROP, _CROP)])(image=img)["image"]
    tensor = A.Compose([A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
                        ToTensorV2()])(image=disp)["image"].unsqueeze(0)
    return tensor, disp


def _samples(n_per=1):
    splits, classes = load_manifest_splits(SS_MANIFEST)
    paths, labels = splits["test"]
    picks = []
    for cls in FOCUS:
        if cls not in classes:
            continue
        idx = classes.index(cls); got = 0
        for p, y in zip(paths, labels):
            if y == idx and os.path.exists(p):
                picks.append((p, cls)); got += 1
                if got >= n_per:
                    break
    return picks, classes


def _overlay(disp, cam):
    """Classic Grad-CAM overlay: JET heatmap blended over the full image."""
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(disp, 0.55, heat, 0.45, 0)


def main():
    ss_ckpt, in_ckpt = _find_ckpts()
    if not ss_ckpt or not in_ckpt:
        raise SystemExit("overlap checkpoints not found yet — run training first")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "sans-serif",
                         "figure.facecolor": "white", "savefig.facecolor": "white"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wild_model, studio_model = _load(ss_ckpt, device), _load(in_ckpt, device)
    picks, classes = _samples()
    if not picks:
        raise SystemExit("no focus-class wild images found")

    def target(m):
        try:
            return m.backbone.blocks[-1]
        except Exception:
            return m.backbone.conv_head
    gc_w = HiResCAM(wild_model, target(wild_model))
    gc_s = HiResCAM(studio_model, target(studio_model))

    rows = len(picks)
    fig, axes = plt.subplots(rows, 3, figsize=(7.4, 2.6 * rows), squeeze=False)
    titles = ["input", "wild-trained", "studio-trained"]
    for r, (path, cls) in enumerate(picks):
        tensor, disp = _view(path); tensor = tensor.to(device)
        cam_w, pred_w = gc_w(tensor)
        cam_s, pred_s = gc_s(tensor)
        panels = [(disp, None),
                  (_overlay(disp, _postprocess(cam_w, disp.shape[0])), pred_w),
                  (_overlay(disp, _postprocess(cam_s, disp.shape[0])), pred_s)]
        for c, (im, pred) in enumerate(panels):
            ax = axes[r, c]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
            if pred is not None:
                name = classes[pred]
                ax.set_xlabel(f"Pred: {name}", fontsize=9,
                              color=("#2a7f3f" if name == cls else "#c0392b"))
        axes[r, 0].set_ylabel(cls, fontsize=9)
    gc_w.remove(); gc_s.remove()
    fig.tight_layout()
    figstyle.save(fig, str(OUT))


if __name__ == "__main__":
    main()
