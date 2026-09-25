"""
Evaluate a BOTH-source (diverse) overlap model on each source separately.

If training on both sources closes the shortcut, this model should score high on
BOTH the wild and the studio test sets (gap ~ 0) — unlike the single-source model
that collapses. GPU eval-only.

    python scripts/eval/eval_diversity.py --ckpt outputs/checkpoints/overlap_both_s42/p1_best.pth --out_suffix s42
"""
import sys, os, argparse, json
from pathlib import Path

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src.model import SpiceFusionNet
from src.dataset import SpiceDataset, get_val_transform, load_manifest_splits

ROOT = Path(_base)


def _loader(paths, labels):
    return DataLoader(SpiceDataset(paths, labels, get_val_transform(), multimodal=False),
                      batch_size=64, num_workers=2, pin_memory=True)


@torch.no_grad()
def _acc(model, loader, device, n):
    yt, yp = [], []
    for imgs, tex, col, labels in loader:
        yp.extend(model.forward_image(imgs.to(device)).argmax(1).cpu().tolist())
        yt.extend(labels.tolist())
    yt, yp = np.array(yt), np.array(yp)
    return float((yt == yp).mean()), float(f1_score(yt, yp, average="macro",
                                                    labels=list(range(n)), zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_suffix", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ss = load_manifest_splits(str(ROOT / "outputs/manifest_overlap_ss.json"))
    ins = load_manifest_splits(str(ROOT / "outputs/manifest_overlap_indian.json"))
    n = len(ss[1])
    model = SpiceFusionNet(num_classes=n).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"]); model.eval()

    acc_w, f1_w = _acc(model, _loader(*ss[0]["test"]), device, n)
    acc_s, f1_s = _acc(model, _loader(*ins[0]["test"]), device, n)
    out = {"acc_wild": acc_w, "acc_studio": acc_s, "gap_pp": round(abs(acc_s - acc_w) * 100, 2),
           "macro_f1_wild": f1_w, "macro_f1_studio": f1_s}
    json.dump(out, open(ROOT / "outputs" / f"diversity_{args.out_suffix}.json", "w"), indent=2)
    print(f"both-source: wild={acc_w*100:.1f}%  studio={acc_s*100:.1f}%  gap={out['gap_pp']} pp")


if __name__ == "__main__":
    main()
