"""
The shortcut-learning test: train on one source, evaluate on the other.

Given two checkpoints trained on the overlap-only manifests of each source,
this script computes the FULL 2×2 cross-source accuracy matrix.

Predicted finding (the central claim of the paper):
    diag(within-source) >> off-diag(cross-source)

If diag - off_diag > ~5 pp, the model is exploiting source-specific shortcuts.
If gap is small, the unified training actually generalizes.

Usage:
  python scripts/eval/eval_shortcut_test.py \\
    --ss_ckpt outputs/checkpoints/overlap_ss/p1_best.pth \\
    --in_ckpt outputs/checkpoints/overlap_indian/p1_best.pth
  python scripts/eval/eval_shortcut_test.py --arch granuformer \\
    --ss_ckpt outputs/checkpoints/granuformer_overlap_ss/best.pth \\
    --in_ckpt outputs/checkpoints/granuformer_overlap_indian/best.pth \\
    --out_suffix granuformer
"""
import sys, os
_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from src.dataset import SpiceDataset, get_val_transform, load_manifest_splits, _translate_path


SS_MANIFEST = Path(_base) / "outputs" / "manifest_overlap_ss.json"
IN_MANIFEST = Path(_base) / "outputs" / "manifest_overlap_indian.json"


def _load_ckpt(path: str, arch: str, num_classes: int, device):
    if arch == "spicefusion":
        from src.model import SpiceFusionNet
        model = SpiceFusionNet(num_classes=num_classes).to(device)
    elif arch == "granuformer":
        raise NotImplementedError("GranuFormer is not part of this release; its numbers in the paper are reported from saved results (see results/).")
    else:
        raise ValueError(f"Unknown arch: {arch}")
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck.get("epoch", 0), ck.get("best_val_acc", 0.0)


def _build_loader(paths, labels, batch_size=64):
    ds = SpiceDataset(paths, labels, get_val_transform(), multimodal=False)
    return DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)


@torch.no_grad()
def _eval(model, loader, device, arch: str):
    preds, gts = [], []
    for imgs, tex, col, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if arch == "spicefusion":
            # Phase-1 image-only forward path
            logits = model.forward_image(imgs)
        else:
            logits = model(imgs)
        preds.extend(logits.argmax(1).cpu().tolist())
        gts.extend(labels.cpu().tolist())
    return np.asarray(gts), np.asarray(preds)


def _mcnemar(correct_a, correct_b):
    """Paired McNemar test on two boolean correctness vectors over the SAME
    samples (here: SS-trained vs Indian-trained, both evaluated on SS-test).

      b = A right & B wrong,  c = A wrong & B right
      chi2 = (|b-c| - 1)^2 / (b+c)   [continuity-corrected], df=1
    """
    a = np.asarray(correct_a, bool)
    b = np.asarray(correct_b, bool)
    n01 = int((a & ~b).sum())   # A right, B wrong
    n10 = int((~a & b).sum())   # A wrong, B right
    n = n01 + n10
    if n == 0:
        return {"b_A_right_B_wrong": n01, "c_A_wrong_B_right": n10,
                "chi2": 0.0, "p_value": 1.0, "n_discordant": 0}
    chi2 = (abs(n01 - n10) - 1) ** 2 / n
    try:
        from scipy.stats import chi2 as _chi2dist
        p = float(_chi2dist.sf(chi2, 1))
    except Exception:
        import math
        p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"b_A_right_B_wrong": n01, "c_A_wrong_B_right": n10,
            "chi2": float(chi2), "p_value": float(p), "n_discordant": n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ss_ckpt", required=True, help="Checkpoint trained on SS-only overlap")
    parser.add_argument("--in_ckpt", required=True, help="Checkpoint trained on Indian-only overlap")
    parser.add_argument("--arch", default="spicefusion",
                        choices=["spicefusion", "granuformer"])
    parser.add_argument("--out_suffix", default=None,
                        help="Suffix for output JSON filename (default arch-based)")
    parser.add_argument("--ss_manifest", default=str(SS_MANIFEST),
                        help="SS overlap manifest (use *_dedup.json for leakage-corrected eval)")
    parser.add_argument("--in_manifest", default=str(IN_MANIFEST),
                        help="Indian overlap manifest (use *_dedup.json for leakage-corrected eval)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Sanity-check that both manifests have identical class lists ──────
    ss_splits, ss_classes = load_manifest_splits(args.ss_manifest)
    in_splits, in_classes = load_manifest_splits(args.in_manifest)
    assert ss_classes == in_classes, "Manifests must share class set"
    NUM = len(ss_classes)
    print(f"Overlap classes ({NUM}): {ss_classes}")

    # Build test loaders for each source
    ss_loader = _build_loader(*ss_splits["test"])
    in_loader = _build_loader(*in_splits["test"])

    # ── Load both models ─────────────────────────────────────────────────
    ss_model, _, ss_bva = _load_ckpt(args.ss_ckpt, args.arch, NUM, device)
    in_model, _, in_bva = _load_ckpt(args.in_ckpt, args.arch, NUM, device)
    print(f"\narch={args.arch}")
    print(f"SS-trained model:     best_val_acc={ss_bva:.4f} | {args.ss_ckpt}")
    print(f"Indian-trained model: best_val_acc={in_bva:.4f} | {args.in_ckpt}")

    # ── 2×2 cross-source matrix ──────────────────────────────────────────
    results = {}
    preds_store = {}   # (train_tag, test_tag) -> (y_true, y_pred) for McNemar
    print("\n┌─────────────────┬───────────────┬───────────────┐")
    print("│ Train \\ Test    │ SS-test       │ Indian-test   │")
    print("├─────────────────┼───────────────┼───────────────┤")

    for model_name, model, tag in (("SS-trained", ss_model, "ss"), ("Indian-trained", in_model, "in")):
        row = {}
        for test_name, loader, ttag in (("SS-test", ss_loader, "ss"), ("Indian-test", in_loader, "in")):
            y_true, y_pred = _eval(model, loader, device, args.arch)
            acc = float((y_true == y_pred).mean())
            per_class = {}
            for c_idx, c_name in enumerate(ss_classes):
                mask = (y_true == c_idx)
                if mask.sum() > 0:
                    per_class[c_name] = {
                        "n": int(mask.sum()),
                        "acc": float((y_pred[mask] == c_idx).mean()),
                    }
            row[ttag] = {"n": len(y_true), "acc": acc, "per_class": per_class}
            preds_store[(tag, ttag)] = (y_true, y_pred)
        results[tag] = row
        print(f"│ {model_name:15s} │ {row['ss']['acc']:.4f} ({row['ss']['n']:4d}) │ "
              f"{row['in']['acc']:.4f} ({row['in']['n']:4d}) │")

    print("└─────────────────┴───────────────┴───────────────┘")

    # ── Shortcut tax ─────────────────────────────────────────────────────
    ss_within = results["ss"]["ss"]["acc"]
    ss_cross  = results["ss"]["in"]["acc"]
    in_within = results["in"]["in"]["acc"]
    in_cross  = results["in"]["ss"]["acc"]

    analysis = {
        "ss_trained_within_source":  ss_within,
        "ss_trained_cross_source":   ss_cross,
        "ss_shortcut_tax_pp":        round((ss_within - ss_cross) * 100, 2),
        "in_trained_within_source":  in_within,
        "in_trained_cross_source":   in_cross,
        "in_shortcut_tax_pp":        round((in_within - in_cross) * 100, 2),
        "avg_shortcut_tax_pp":       round(((ss_within - ss_cross) + (in_within - in_cross)) * 50, 2),
    }
    results["analysis"] = analysis

    # ── McNemar: on the SAME SS-test samples, does the TRAINING SOURCE change
    #    the error pattern? (SS-trained vs Indian-trained — paired, valid.) ────
    yt_ss = preds_store[("ss", "ss")][0]                    # SS-test ground truth
    ss_correct = (preds_store[("ss", "ss")][1] == yt_ss)   # SS-trained on SS-test
    in_correct = (preds_store[("in", "ss")][1] == yt_ss)   # Indian-trained on SS-test
    mc = _mcnemar(ss_correct, in_correct)
    analysis["mcnemar_ss_test"] = mc
    results["per_sample"] = {"ss_test": {
        "ss_trained_correct": ss_correct.astype(int).tolist(),
        "in_trained_correct": in_correct.astype(int).tolist(),
    }}
    print(f"\nMcNemar (SS-test · SS-trained vs Indian-trained): "
          f"b={mc['b_A_right_B_wrong']} c={mc['c_A_wrong_B_right']} "
          f"chi2={mc['chi2']:.1f} p={mc['p_value']:.2e}")

    print(f"\nShortcut tax:")
    print(f"  SS-trained:     within={ss_within:.4f}  cross={ss_cross:.4f}  "
          f"tax={analysis['ss_shortcut_tax_pp']:+.2f} pp")
    print(f"  Indian-trained: within={in_within:.4f}  cross={in_cross:.4f}  "
          f"tax={analysis['in_shortcut_tax_pp']:+.2f} pp")
    print(f"  Average shortcut tax: {analysis['avg_shortcut_tax_pp']:+.2f} pp")

    suffix = args.out_suffix or (args.arch if args.arch != "spicefusion" else None)
    fname = f"shortcut_test_matrix_{suffix}.json" if suffix else "shortcut_test_matrix.json"
    out = config.OUTPUT_DIR / fname
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
