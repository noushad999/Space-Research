"""
Cross-source evaluation harness for SpiceNet-Bench.

Splits the unified benchmark test set into source-specific subsets and reports
per-source accuracy. This is the core evaluation that exposes whether a model
has memorized source-specific shortcuts (uniform white background of Mendeley
vs. "in-the-wild" SpiceSpectrum).

For a given trained model checkpoint, produces:
  * Full unified test accuracy
  * SpiceSpectrum-only test accuracy
  * Mendeley-only test accuracy
  * Per-class breakdown
  * Drop from unified -> single-source-only (the "shortcut tax")

Usage:
  python scripts/eval/eval_cross_source.py --ckpt outputs/checkpoints/unified/best.pth
  python scripts/eval/eval_cross_source.py --ckpt outputs/checkpoints/granuformer/best.pth --arch granuformer
"""
import sys, os
_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config
from src.dataset import (
    SpiceDataset, get_val_transform, load_manifest_splits, _translate_path,
)
from torch.utils.data import DataLoader


MANIFEST = Path(_base) / "outputs" / "unified_benchmark.json"


def _source_of(path: str) -> str:
    p = path.replace("\\", "/").lower()
    if "/spice_spectrum/" in p:
        return "spice_spectrum"
    if "/indian_spices/" in p:
        return "indian"
    return "unknown"


def _build_filtered_loader(paths, labels, multimodal: bool, batch_size: int = 64):
    ds = SpiceDataset(paths, labels, get_val_transform(), multimodal=multimodal)
    return DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)


def _load_model(ckpt_path: str, arch: str, num_classes: int, device):
    if arch == "spicefusion":
        from src.model import SpiceFusionNet
        model = SpiceFusionNet(num_classes=num_classes)
        ck = torch.load(ckpt_path, map_location=device)
        # The trainer.save_checkpoint format keys are "model_state"
        state = ck.get("model_state", ck)
        model.load_state_dict(state)
        return model.to(device), True   # multimodal
    elif arch == "granuformer":
        raise NotImplementedError("GranuFormer is not part of this release; its numbers in the paper are reported from saved results (see results/).")
    else:
        raise ValueError(f"Unknown arch: {arch}")


@torch.no_grad()
def _predict(model, loader, device, multimodal: bool):
    model.eval()
    preds, gts = [], []
    for imgs, tex, col, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if multimodal:
            tex, col = tex.to(device), col.to(device)
            logits, _ = model.forward_fusion(imgs, tex, col)
        else:
            logits = model(imgs)
        preds.extend(logits.argmax(1).cpu().tolist())
        gts.extend(labels.cpu().tolist())
    return np.asarray(gts), np.asarray(preds)


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--arch", default="spicefusion",
                        choices=["spicefusion", "granuformer"])
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--out_suffix", default=None,
                        help="Suffix for output JSON filename")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits, classes = load_manifest_splits(MANIFEST)
    NUM_CLASSES = len(classes)
    print(f"Classes: {NUM_CLASSES}")

    # Patch config so SpiceFusionNet sees correct class count
    config.NUM_CLASSES = NUM_CLASSES
    config.CLASSES = classes

    model, multimodal = _load_model(args.ckpt, args.arch, NUM_CLASSES, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {args.arch} ({n_params:,} params) from {args.ckpt}")
    print(f"Mode: {'multimodal' if multimodal else 'image-only'}")

    # ── Build three filtered evaluation sets ─────────────────────────────
    paths, labels = splits["test"]
    src_paths = defaultdict(list)
    src_labels = defaultdict(list)
    for p, y in zip(paths, labels):
        s = _source_of(p)
        src_paths[s].append(p)
        src_labels[s].append(y)
        src_paths["all"].append(p)
        src_labels["all"].append(y)

    print(f"\nTest split by source:")
    for s in ("all", "spice_spectrum", "indian"):
        print(f"  {s:18s} n={len(src_paths[s])}")

    # ── Evaluate each ────────────────────────────────────────────────────
    results = {"checkpoint": args.ckpt, "arch": args.arch,
               "n_params": n_params, "by_source": {}}

    for src in ("all", "spice_spectrum", "indian"):
        if not src_paths[src]:
            continue
        loader = _build_filtered_loader(src_paths[src], src_labels[src],
                                        multimodal=multimodal, batch_size=args.batch)
        y_true, y_pred = _predict(model, loader, device, multimodal)
        acc = _accuracy(y_true, y_pred)
        # Per-class
        per_class = {}
        for c_idx, c_name in enumerate(classes):
            mask = (y_true == c_idx)
            if mask.sum() > 0:
                per_class[c_name] = {
                    "n": int(mask.sum()),
                    "acc": float((y_pred[mask] == c_idx).mean()),
                }
        results["by_source"][src] = {
            "n_total": len(y_true),
            "accuracy": acc,
            "per_class": per_class,
        }
        print(f"\n{src}:  acc = {acc:.4f}  (n={len(y_true)})")

    # ── Compute the "shortcut tax" ────────────────────────────────────────
    if all(s in results["by_source"] for s in ("all", "spice_spectrum", "indian")):
        ss_acc = results["by_source"]["spice_spectrum"]["accuracy"]
        in_acc = results["by_source"]["indian"]["accuracy"]
        all_acc = results["by_source"]["all"]["accuracy"]
        results["analysis"] = {
            "overall_test_acc": all_acc,
            "spice_spectrum_test_acc": ss_acc,
            "indian_test_acc": in_acc,
            "source_gap_pp":   round(abs(ss_acc - in_acc) * 100, 2),
            "weaker_source": "spice_spectrum" if ss_acc < in_acc else "indian",
        }
        print("\n──── Cross-source summary ────")
        print(f"  Overall      : {all_acc:.4f}")
        print(f"  SpiceSpectrum: {ss_acc:.4f}")
        print(f"  Indian       : {in_acc:.4f}")
        print(f"  Source gap   : {results['analysis']['source_gap_pp']:.2f} pp "
              f"(weaker: {results['analysis']['weaker_source']})")

    # ── Save ─────────────────────────────────────────────────────────────
    suffix = args.out_suffix or args.arch
    out = config.OUTPUT_DIR / f"cross_source_{suffix}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
