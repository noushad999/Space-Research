#!/usr/bin/env python
"""Per-seed paired McNemar among the single-source methods on the wild held-out test.

Re-evaluates the studio(Indian)-trained checkpoints {erm, rsc, arcv, sd} on the wild
(SS) overlap test for seeds 42/1337/2024, then reports per-seed accuracy and the
discordant-pair counts for the comparisons that matter: each method vs ERM, and the
near-tie SD vs ARC-V. Inference only; force CPU with CUDA_VISIBLE_DEVICES="".
"""
import sys, os
from pathlib import Path
_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)
import numpy as np, torch, timm
from torch.utils.data import DataLoader
from scipy.stats import binomtest
from src.dataset import SpiceDataset, get_val_transform, load_manifest_splits

SS = str(Path(_base) / "outputs" / "manifest_overlap_ss.json")
CK = Path(_base) / "outputs" / "checkpoints"
SEEDS = [42, 1337, 2024]
METHODS = ["erm", "rsc", "arcv", "sd"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def correct_vec(ckpt):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    m = timm.create_model(ck["model_name"], pretrained=False,
                          num_classes=len(ck["classes"])).to(dev)
    m.load_state_dict(ck["model_state"]); m.eval()
    sp, _ = load_manifest_splits(SS); paths, labels = sp["test"]
    dl = DataLoader(SpiceDataset(paths, labels, get_val_transform(), multimodal=False),
                    batch_size=64, num_workers=2)
    yt, yp = [], []
    with torch.no_grad():
        for imgs, t, c, y in dl:
            yp.extend(m(imgs.to(dev)).argmax(1).cpu().tolist()); yt.extend(y.tolist())
    return np.array(yt) == np.array(yp)


def mcnemar(a, b):
    """discordant counts: bx = a right & b wrong, cx = b right & a wrong."""
    bx = int(np.sum(a & ~b)); cx = int(np.sum(b & ~a))
    p = binomtest(min(bx, cx), bx + cx, 0.5).pvalue if (bx + cx) else 1.0
    return bx, cx, p


def main():
    print(f"device={dev}")
    corr = {m: {} for m in METHODS}
    for s in SEEDS:
        for m in METHODS:
            corr[m][s] = correct_vec(CK / f"arcv_{m}_indian_s{s}" / "best.pth")
        accs = {m: 100 * corr[m][s].mean() for m in METHODS}
        print(f"seed {s}: " + "  ".join(f"{m} {accs[m]:.2f}" for m in METHODS))
    print("\nper-seed McNemar vs ERM (b=method fixes ERM error, c=method breaks):")
    for m in ["rsc", "arcv", "sd"]:
        for s in SEEDS:
            b, c, p = mcnemar(corr[m][s], corr["erm"][s])
            print(f"  {m} vs erm  seed {s}: b={b} c={c} p={p:.2e}")
    print("\nper-seed McNemar SD vs ARC-V (the near-tie):")
    for s in SEEDS:
        b, c, p = mcnemar(corr["sd"][s], corr["arcv"][s])
        print(f"  sd vs arcv seed {s}: b(sd fixes)={b} c(arcv fixes)={c} p={p:.2e}")


if __name__ == "__main__":
    main()
