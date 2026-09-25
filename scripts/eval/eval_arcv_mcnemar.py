#!/usr/bin/env python
"""Paired McNemar test: does ARC-V correct ERM's held-out errors more than the
reverse? Re-evaluates the on-disk studio(Indian)-trained checkpoints on the wild
(SS) held-out test, per seed, and pools the discordant pairs across seeds.

Inference only (no training). Force CPU with CUDA_VISIBLE_DEVICES="" to keep it cool.

    python scripts/eval/eval_arcv_mcnemar.py
"""
import sys, os
from pathlib import Path

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import chi2 as chi2dist
import timm

import config
from src.dataset import SpiceDataset, get_val_transform, load_manifest_splits

ROOT = Path(_base)
SS_M = str(ROOT / "outputs" / "manifest_overlap_ss.json")     # wild = held-out test
IN_M = str(ROOT / "outputs" / "manifest_overlap_indian.json")  # studio = train source
CKPT = ROOT / "outputs" / "checkpoints"
SEEDS = [42, 1337, 2024]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def correctness(ckpt):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m = timm.create_model(ck["model_name"], pretrained=False,
                          num_classes=len(ck["classes"])).to(device)
    m.load_state_dict(ck["model_state"]); m.eval()
    splits, _ = load_manifest_splits(SS_M)
    paths, labels = splits["test"]
    loader = DataLoader(SpiceDataset(paths, labels, get_val_transform(), multimodal=False),
                        batch_size=64, num_workers=2)
    yt, yp = [], []
    with torch.no_grad():
        for imgs, tex, col, y in loader:
            yp.extend(m(imgs.to(device)).argmax(1).cpu().tolist()); yt.extend(y.tolist())
    yt, yp = np.array(yt), np.array(yp)
    return yt == yp   # boolean per sample: correct?


def main():
    b_tot, c_tot = 0, 0   # b: ARC-V right & ERM wrong ; c: ERM right & ARC-V wrong
    print(f"device={device}")
    for s in SEEDS:
        erm = correctness(CKPT / f"arcv_erm_indian_s{s}" / "best.pth")
        arc = correctness(CKPT / f"arcv_arcv_indian_s{s}" / "best.pth")
        b = int(np.sum(arc & ~erm)); c = int(np.sum(erm & ~arc))
        b_tot += b; c_tot += c
        print(f"  seed {s:>4}: ERM acc {erm.mean()*100:.2f}  ARC-V acc {arc.mean()*100:.2f}  "
              f"| b(arcv fixes)={b}  c(arcv breaks)={c}")

    n = b_tot + c_tot
    chi2 = (abs(b_tot - c_tot) - 1) ** 2 / n if n else 0.0   # continuity-corrected
    p = float(chi2dist.sf(chi2, df=1))
    print("\n" + "=" * 60)
    print(f"Pooled over {len(SEEDS)} seeds:  b={b_tot} (ARC-V right, ERM wrong)  "
          f"c={c_tot} (ERM right, ARC-V wrong)")
    print(f"McNemar chi2={chi2:.1f}, p={p:.3e}")
    better = "ARC-V" if b_tot > c_tot else "ERM"
    print(f"Discordant pairs favor {better}. "
          f"{'Significant' if p < 0.05 else 'Not significant'} at 0.05.")
    (ROOT / "outputs" / "arcv_mcnemar.txt").write_text(
        f"pooled 3 seeds: b={b_tot} c={c_tot} chi2={chi2:.1f} p={p:.3e} favors {better}\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
