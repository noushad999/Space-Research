"""
Rich cross-source eval — paper-grade metrics beyond top-1.

Same 2x2 protocol as eval_shortcut_test.py but for each of the 4 cells also
computes macro-F1 and the full confusion matrix, and dumps per-sample
correctness for BOTH test sets (so bootstrap covers both directions). Reads
the overlap checkpoints; runs on standard OR dedup manifests.

    python scripts/eval/eval_confusion.py --ss_ckpt .../overlap_ss_s42/p1_best.pth \
        --in_ckpt .../overlap_indian_s42/p1_best.pth --out_suffix s42

GPU eval-only (no training). Run after run_cross_source.py.
"""
import sys, os, argparse, json
from pathlib import Path

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix

import config
from src.model import SpiceFusionNet
from src.dataset import SpiceDataset, get_val_transform, load_manifest_splits

ROOT = Path(_base)
SS_MANIFEST = str(ROOT / "outputs" / "manifest_overlap_ss.json")
IN_MANIFEST = str(ROOT / "outputs" / "manifest_overlap_indian.json")


def _loader(paths, labels):
    ds = SpiceDataset(paths, labels, get_val_transform(), multimodal=False)
    return DataLoader(ds, batch_size=64, num_workers=2, pin_memory=True)


@torch.no_grad()
def _eval(model, loader, device):
    preds, gts = [], []
    for imgs, tex, col, labels in loader:
        logits = model.forward_image(imgs.to(device))
        preds.extend(logits.argmax(1).cpu().tolist()); gts.extend(labels.tolist())
    return np.asarray(gts), np.asarray(preds)


def _load(ckpt, n, device):
    m = SpiceFusionNet(num_classes=n).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m


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
    in_splits, in_classes = load_manifest_splits(args.in_manifest)
    assert classes == in_classes, "manifests must share classes"
    n = len(classes)
    labels_range = list(range(n))

    ss_loader = _loader(*ss_splits["test"])
    in_loader = _loader(*in_splits["test"])
    ss_model = _load(args.ss_ckpt, n, device)
    in_model = _load(args.in_ckpt, n, device)

    results, preds_store = {}, {}
    for mtag, model in (("ss", ss_model), ("in", in_model)):
        row = {}
        for ttag, loader in (("ss", ss_loader), ("in", in_loader)):
            yt, yp = _eval(model, loader, device)
            preds_store[(mtag, ttag)] = (yt, yp)
            row[ttag] = {
                "n": int(len(yt)), "acc": float((yt == yp).mean()),
                "macro_f1": float(f1_score(yt, yp, average="macro", labels=labels_range, zero_division=0)),
                "confusion": confusion_matrix(yt, yp, labels=labels_range).tolist(),
                "per_class": {classes[c]: {"n": int((yt == c).sum()),
                                           "acc": float((yp[yt == c] == c).mean()) if (yt == c).any() else 0.0}
                              for c in labels_range},
            }
        results[mtag] = row

    sw, sc = results["ss"]["ss"]["acc"], results["ss"]["in"]["acc"]
    iw, ic = results["in"]["in"]["acc"], results["in"]["ss"]["acc"]
    results["analysis"] = {"ss_shortcut_tax_pp": round((sw - sc) * 100, 2),
                           "in_shortcut_tax_pp": round((iw - ic) * 100, 2),
                           "avg_shortcut_tax_pp": round(((sw - sc) + (iw - ic)) * 50, 2)}
    results["classes"] = classes
    # per-sample correctness for BOTH directions (bootstrap both ways)
    yt_ss = preds_store[("ss", "ss")][0]; yt_in = preds_store[("ss", "in")][0]
    results["per_sample"] = {
        "ss_test": {"ss_trained_correct": (preds_store[("ss", "ss")][1] == yt_ss).astype(int).tolist(),
                    "in_trained_correct": (preds_store[("in", "ss")][1] == yt_ss).astype(int).tolist()},
        "in_test": {"ss_trained_correct": (preds_store[("ss", "in")][1] == yt_in).astype(int).tolist(),
                    "in_trained_correct": (preds_store[("in", "in")][1] == yt_in).astype(int).tolist()},
    }
    out = ROOT / "outputs" / f"shortcut_evidence_{args.out_suffix}.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"acc: SS/SS={sw:.4f} SS/IN={sc:.4f} IN/SS={ic:.4f} IN/IN={iw:.4f} | "
          f"macroF1 IN/SS={results['in']['ss']['macro_f1']:.4f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
