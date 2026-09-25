"""
Calibration under cross-source shift — is the studio model confidently WRONG?

For each model x test-set, dumps per-sample (max-softmax confidence, correct)
and computes ECE. The headline: the studio-trained model, tested in the wild,
stays HIGH-confidence while its accuracy collapses — overconfidence under shift.
This turns the "100% accuracy is a red flag" concern into evidence.

    python scripts/eval/eval_calibration.py \
        --ss_ckpt outputs/checkpoints/overlap_ss_s42/p1_best.pth \
        --in_ckpt outputs/checkpoints/overlap_indian_s42/p1_best.pth --out_suffix s42

GPU eval-only. Run after run_cross_source.py.
"""
import sys, os, argparse, json
from pathlib import Path

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from src.model import SpiceFusionNet
from src.dataset import SpiceDataset, get_val_transform, load_manifest_splits

ROOT = Path(_base)
SS_MANIFEST = str(ROOT / "outputs" / "manifest_overlap_ss.json")
IN_MANIFEST = str(ROOT / "outputs" / "manifest_overlap_indian.json")


def _loader(paths, labels):
    ds = SpiceDataset(paths, labels, get_val_transform(), multimodal=False)
    return DataLoader(ds, batch_size=64, num_workers=2, pin_memory=True)


def _load(ckpt, n, device):
    m = SpiceFusionNet(num_classes=n).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m


@torch.no_grad()
def _eval(model, loader, device):
    conf, correct = [], []
    for imgs, tex, col, labels in loader:
        probs = F.softmax(model.forward_image(imgs.to(device)), dim=1)
        p, pred = probs.max(1)
        conf.extend(p.cpu().tolist())
        correct.extend((pred.cpu() == labels).int().tolist())
    return np.asarray(conf), np.asarray(correct)


def ece(conf, correct, n_bins=15):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    e, N = 0.0, len(conf)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            e += (m.sum() / N) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ss_ckpt", required=True)
    ap.add_argument("--in_ckpt", required=True)
    ap.add_argument("--ss_manifest", default=SS_MANIFEST)
    ap.add_argument("--in_manifest", default=IN_MANIFEST)
    ap.add_argument("--out_suffix", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ss_splits, classes = load_manifest_splits(args.ss_manifest)
    in_splits, _ = load_manifest_splits(args.in_manifest)
    n = len(classes)
    ss_loader, in_loader = _loader(*ss_splits["test"]), _loader(*in_splits["test"])
    models = {"ss": _load(args.ss_ckpt, n, device), "in": _load(args.in_ckpt, n, device)}
    loaders = {"ss": ss_loader, "in": in_loader}

    out = {}
    for mt in ("ss", "in"):
        out[mt] = {}
        for tt in ("ss", "in"):
            conf, correct = _eval(models[mt], loaders[tt], device)
            out[mt][tt] = {"confidence": conf.tolist(), "correct": correct.astype(int).tolist(),
                           "acc": float(correct.mean()), "mean_conf": float(conf.mean()),
                           "ece": ece(conf, correct)}
    out["classes"] = classes
    p = ROOT / "outputs" / f"calibration_{args.out_suffix}.json"
    json.dump(out, open(p, "w"))
    # headline: studio-trained (in) on wild-test (ss)
    h = out["in"]["ss"]
    direction = "over-confident" if h["mean_conf"] > h["acc"] else "under-confident"
    print(f"Studio-trained on WILD test: acc={h['acc']*100:.1f}%  mean-conf={h['mean_conf']*100:.1f}%  "
          f"ECE={h['ece']*100:.1f}%  ({direction})")
    print(f"saved -> {p}")


if __name__ == "__main__":
    main()
