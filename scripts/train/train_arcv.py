"""Train a backbone on ONE source's 8-class overlap with a chosen DG method, for
the ARC-V Regime-A comparison (train single-source -> test the *held-out* source).

Methods (all single-source, the P1 regime -- see ARCV_METHOD_DESIGN.md §5.1 A):
  erm       plain cross-entropy (the bar Gulrajani says kills most DG methods)
  mixstyle  M1 only  -- MixStyle feature-statistic randomization
  fourier   M2 only  -- Fourier amplitude-mix + phase-consistency (FACT)
  arcv      M1 + M2  -- the full ARC-V method
  rsc       RSC self-challenging (Huang 2020), a single-source DG baseline
  sd        Spectral Decoupling (Pezeshki 2021), a shortcut-mitigation regularizer

    python scripts/train/train_arcv.py --method arcv --manifest outputs/manifest_overlap_indian.json \
        --suffix arcv_arcv_indian --seed 42 [--smoke]

ERM is the ARCVObjective with the mechanisms disabled, so every method shares one
train path -- an honest apples-to-apples comparison.
"""
import sys, os, argparse, time
from pathlib import Path

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm

import config
from src.dataset import get_dataloaders, load_manifest_splits
from src.utils import set_seed
from src.dg.mixstyle import register_mixstyle_hooks
from src.dg.arcv import ARCVObjective
from src.dg.rsc import rsc_step
from src.dg.sd import spectral_decoupling_loss

METHODS = ("erm", "mixstyle", "fourier", "arcv", "rsc", "sd")


def early_stage_modules(model, n_stages):
    """The first `n_stages` conv stages of a timm backbone (where channel stats
    encode acquisition style), for MixStyle hooks."""
    if hasattr(model, "blocks"):                       # efficientnet, convnext, ...
        blocks = model.blocks
        return [blocks[i] for i in range(min(n_stages, len(blocks)))]
    if hasattr(model, "layer1"):                       # resnet family
        return [getattr(model, f"layer{i}") for i in range(1, min(n_stages, 4) + 1)]
    return []


def build_objective(method):
    if method in ("fourier", "arcv"):
        return ARCVObjective(p_f=config.ARCV_FOURIER_P, eta=config.ARCV_FOURIER_ETA,
                             eta_weak=config.ARCV_FOURIER_ETA_WEAK,
                             gamma_pc=config.ARCV_GAMMA_PC)
    return ARCVObjective(p_f=0.0, gamma_pc=0.0)          # erm / mixstyle -> plain CE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS, default="arcv")
    ap.add_argument("--model", default="efficientnet_b4", help="timm backbone")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--smoke", action="store_true", help="1 epoch, 2 batches -- pipeline check")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, classes = load_manifest_splits(args.manifest)
    n = len(classes)
    tr, val, *_ = get_dataloaders(multimodal=False, manifest_path=args.manifest, strong_aug=True)

    model = timm.create_model(args.model, pretrained=True, num_classes=n).to(device)
    if args.method in ("mixstyle", "arcv"):
        hooked = early_stage_modules(model, config.ARCV_MIXSTYLE_STAGES)
        register_mixstyle_hooks(hooked, p=config.ARCV_MIXSTYLE_P, alpha=config.ARCV_MIXSTYLE_ALPHA)
        print(f"  MixStyle hooks on {len(hooked)} early stages", flush=True)

    objective = build_objective(args.method)

    rsc_cfg = None
    if args.method == "rsc":
        feature_fn = lambda x: model.forward_head(model.forward_features(x), pre_logits=True)
        rsc_cfg = (feature_fn, model.get_classifier(),
                   getattr(config, "ARCV_RSC_DROP", 1.0 / 3),
                   getattr(config, "ARCV_RSC_APPLY", 0.5))
        print(f"  RSC self-challenging: drop={rsc_cfg[2]:.3f} apply={rsc_cfg[3]:.2f}", flush=True)
    sd_lambda = getattr(config, "ARCV_SD_LAMBDA", 0.1) if args.method == "sd" else 0.0
    if args.method == "sd":
        print(f"  Spectral Decoupling regularizer: lambda={sd_lambda}", flush=True)

    print(f"ARC-V[{args.method}] {args.model} | classes={n} | "
          f"train={len(tr.dataset)} val={len(val.dataset)} | device={device}", flush=True)

    epochs = 1 if args.smoke else args.epochs
    max_batches = 2 if args.smoke else None
    opt = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    ckdir = config.CHECKPOINT_DIR / args.suffix
    ckdir.mkdir(parents=True, exist_ok=True)
    best, patience = 0.0, 0
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time()
        for i, (imgs, tex, col, labels) in enumerate(tr):
            if max_batches is not None and i >= max_batches:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            if rsc_cfg is not None:
                feature_fn, head, drop, appl = rsc_cfg
                loss, _ = rsc_step(feature_fn, head, imgs, labels, drop_frac=drop, apply_frac=appl)
            elif args.method == "sd":
                loss = spectral_decoupling_loss(model(imgs), labels, sd_lambda)
            else:
                loss, _ = objective(model, imgs, labels, training=True)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        model.eval(); c = t = 0
        with torch.no_grad():
            for i, (imgs, tex, col, labels) in enumerate(val):
                if max_batches is not None and i >= max_batches:
                    break
                pred = model(imgs.to(device)).argmax(1).cpu()
                c += (pred == labels).sum().item(); t += labels.size(0)
        acc = c / max(t, 1); sched.step()
        print(f"  ep {ep:02d}/{epochs} | val {acc:.4f} | {time.time()-t0:.0f}s", flush=True)

        if acc >= best:
            best, patience = acc, 0
            torch.save({"model_state": model.state_dict(), "model_name": args.model,
                        "method": args.method, "best_val_acc": best, "classes": classes},
                       ckdir / "best.pth")
        else:
            patience += 1
            if patience >= 8 and not args.smoke:
                print("  early stop", flush=True); break
    print(f"done arcv[{args.method}] {args.suffix} | best val {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
