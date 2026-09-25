"""
Cross-source 2x2 matrix for a timm backbone (both source checkpoints).

    python scripts/eval/eval_backbone_shortcut.py --model resnet50 \
        --ss_ckpt outputs/checkpoints/bench_resnet50_ss/best.pth \
        --in_ckpt outputs/checkpoints/bench_resnet50_indian/best.pth
"""
import sys, os, argparse, json
from pathlib import Path

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import timm

import config
from src.dataset import SpiceDataset, get_val_transform, load_manifest_splits

ROOT = Path(_base)
SS_M = str(ROOT / "outputs" / "manifest_overlap_ss.json")
IN_M = str(ROOT / "outputs" / "manifest_overlap_indian.json")


def _loader(paths, labels):
    return DataLoader(SpiceDataset(paths, labels, get_val_transform(), multimodal=False),
                      batch_size=64, num_workers=2, pin_memory=True)


def _load(model_name, ckpt, n, device):
    m = timm.create_model(model_name, pretrained=False, num_classes=n).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m


@torch.no_grad()
def _eval(model, loader, device, n):
    yt, yp = [], []
    for imgs, tex, col, labels in loader:
        yp.extend(model(imgs.to(device)).argmax(1).cpu().tolist()); yt.extend(labels.tolist())
    yt, yp = np.array(yt), np.array(yp)
    return float((yt == yp).mean()), float(f1_score(yt, yp, average="macro",
                                                    labels=list(range(n)), zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ss_ckpt", required=True)
    ap.add_argument("--in_ckpt", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ss, classes = load_manifest_splits(SS_M)
    ins, _ = load_manifest_splits(IN_M)
    n = len(classes)
    ss_loader, in_loader = _loader(*ss["test"]), _loader(*ins["test"])
    ss_model = _load(args.model, args.ss_ckpt, n, device)
    in_model = _load(args.model, args.in_ckpt, n, device)

    res = {"model": args.model}
    for tag, model in (("ss", ss_model), ("in", in_model)):
        row = {}
        for ttag, loader in (("ss", ss_loader), ("in", in_loader)):
            acc, f1 = _eval(model, loader, device, n)
            row[ttag] = {"acc": acc, "macro_f1": f1}
        res[tag] = row
    res["ss_tax_pp"] = round((res["ss"]["ss"]["acc"] - res["ss"]["in"]["acc"]) * 100, 2)
    res["in_tax_pp"] = round((res["in"]["in"]["acc"] - res["in"]["ss"]["acc"]) * 100, 2)
    out = ROOT / "outputs" / f"bench_{args.model}.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"{args.model}: within SS {res['ss']['ss']['acc']*100:.1f} / IN {res['in']['in']['acc']*100:.1f} | "
          f"collapse IN->SS tax {res['in_tax_pp']} pp  ->  {out}")


if __name__ == "__main__":
    main()
