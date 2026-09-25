"""
3-Phase Trainer for SpiceFusionNet.
  Phase 1 — Backbone pre-training    (CE loss, image-only)
  Phase 2 — Contrastive fine-tuning  (SupCon loss, backbone only)
  Phase 3 — Full fusion end-to-end   (CE + SupCon, all branches)
"""
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

import config
from src.model import SpiceFusionNet, save_checkpoint
from src.losses import SupConLoss, CombinedLoss

try:
    import wandb as _wandb
    _WANDB = True
except ImportError:
    _WANDB = False


def _log(d: dict):
    if config.USE_WANDB and _WANDB:
        _wandb.log(d)


def _make_scheduler(optimizer, warmup: int, total: int, min_lr: float):
    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup),
            CosineAnnealingLR(optimizer, T_max=total - warmup, eta_min=min_lr),
        ],
        milestones=[warmup],
    )


class PhaseTrainer:
    def __init__(self, model: SpiceFusionNet, device: torch.device, ckpt_dir: Path):
        self.model    = model
        self.device   = device
        self.ckpt_dir = ckpt_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1 ──────────────────────────────────────────────────────

    def phase1(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        print("\n" + "="*60)
        print("  PHASE 1 — Backbone Pre-training")
        print("="*60)

        model = self.model.to(self.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=config.P1_LABEL_SMOOTH)

        optimizer = AdamW(model.parameters(), lr=config.P1_LR, weight_decay=config.P1_WEIGHT_DECAY)
        scheduler = _make_scheduler(optimizer, config.P1_WARMUP, config.P1_EPOCHS, config.P1_MIN_LR)

        best_acc, patience, history = 0.0, 0, {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

        for epoch in range(1, config.P1_EPOCHS + 1):
            t0 = time.time()

            # Train
            model.train()
            tl, tc, tt = 0.0, 0, 0
            for imgs, tex, col, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits = model.forward_image(imgs)
                loss   = criterion(logits, labels)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()
                tl += loss.item() * imgs.size(0)
                tc += (logits.argmax(1) == labels).sum().item()
                tt += imgs.size(0)

            # Val
            val_loss, val_acc = self._eval_image(model, val_loader, criterion)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            for k, v in zip(["train_loss","train_acc","val_loss","val_acc","lr"],
                             [tl/tt, tc/tt, val_loss, val_acc, lr]):
                history[k].append(v)

            print(f"P1 Ep {epoch:03d}/{config.P1_EPOCHS} | "
                  f"tr_loss {tl/tt:.4f} tr_acc {tc/tt:.4f} | "
                  f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | "
                  f"lr {lr:.2e} | {time.time()-t0:.1f}s")
            _log({"p1/train_loss": tl/tt, "p1/val_acc": val_acc, "p1/lr": lr})

            if val_acc > best_acc:
                best_acc, patience = val_acc, 0
                save_checkpoint(self.ckpt_dir/"p1_best.pth", model, optimizer, epoch, best_acc, history)
                print(f"  --> P1 best: {best_acc:.4f}")
            else:
                patience += 1
                if patience >= config.PATIENCE:
                    print(f"  Early stop at epoch {epoch}")
                    break

        save_checkpoint(self.ckpt_dir/"p1_last.pth", model, optimizer, epoch, best_acc, history)
        print(f"\nPhase 1 complete. Best val acc: {best_acc:.4f}")
        return history

    # ── Phase 2 ──────────────────────────────────────────────────────

    def phase2(self, train_loader: DataLoader) -> None:
        print("\n" + "="*60)
        print("  PHASE 2 — Contrastive Fine-tuning (SupCon)")
        print("="*60)

        # Load best P1 weights
        p1_ckpt = self.ckpt_dir / "p1_best.pth"
        if p1_ckpt.exists():
            ckpt = torch.load(p1_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            print(f"  Loaded P1 best checkpoint (val acc: {ckpt.get('best_val_acc',0):.4f})")

        model = self.model.to(self.device)
        supcon = SupConLoss()

        # Only train backbone + proj_head; freeze branches and heads
        for name, p in model.named_parameters():
            p.requires_grad = any(k in name for k in ("backbone", "proj_head"))

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable:,}")

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.P2_LR,
        )

        for epoch in range(1, config.P2_EPOCHS + 1):
            t0 = time.time()
            model.train()
            total_loss, n = 0.0, 0

            for imgs, tex, col, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                proj = model.forward_contrastive(imgs)
                loss = supcon(proj, labels)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()
                total_loss += loss.item() * imgs.size(0)
                n += imgs.size(0)

            avg = total_loss / n
            print(f"P2 Ep {epoch:02d}/{config.P2_EPOCHS} | SupCon loss {avg:.4f} | {time.time()-t0:.1f}s")
            _log({"p2/supcon_loss": avg})

        # Re-enable all params for Phase 3
        for p in model.parameters():
            p.requires_grad = True

        save_checkpoint(self.ckpt_dir/"p2_last.pth", model, optimizer, epoch, 0.0, {})
        print("Phase 2 complete.")

    # ── Phase 3 ──────────────────────────────────────────────────────

    def phase3(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        print("\n" + "="*60)
        print("  PHASE 3 — Full Fusion End-to-end Training")
        print("="*60)

        p2_ckpt = self.ckpt_dir / "p2_last.pth"
        if p2_ckpt.exists():
            ckpt = torch.load(p2_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            print("  Loaded P2 checkpoint.")

        model   = self.model.to(self.device)
        loss_fn = CombinedLoss()

        optimizer = AdamW(model.parameters(), lr=config.P3_LR, weight_decay=config.P3_WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.P3_EPOCHS, eta_min=1e-7)

        best_acc, patience, history = 0.0, 0, {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

        for epoch in range(1, config.P3_EPOCHS + 1):
            t0 = time.time()
            model.train()
            tl, tc, tt = 0.0, 0, 0

            for imgs, tex, col, labels in train_loader:
                imgs, tex, col, labels = (
                    imgs.to(self.device), tex.to(self.device),
                    col.to(self.device), labels.to(self.device),
                )
                logits, proj = model.forward_fusion(imgs, tex, col)
                loss = loss_fn(logits, proj, labels)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()
                tl += loss.item() * imgs.size(0)
                tc += (logits.argmax(1) == labels).sum().item()
                tt += imgs.size(0)

            val_loss, val_acc = self._eval_fusion(model, val_loader)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            for k, v in zip(["train_loss","train_acc","val_loss","val_acc","lr"],
                             [tl/tt, tc/tt, val_loss, val_acc, lr]):
                history[k].append(v)

            print(f"P3 Ep {epoch:02d}/{config.P3_EPOCHS} | "
                  f"tr_loss {tl/tt:.4f} tr_acc {tc/tt:.4f} | "
                  f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | "
                  f"lr {lr:.2e} | {time.time()-t0:.1f}s")
            _log({"p3/val_acc": val_acc})

            if val_acc > best_acc:
                best_acc, patience = val_acc, 0
                save_checkpoint(self.ckpt_dir/"best.pth", model, optimizer, epoch, best_acc, history)
                print(f"  --> Best: {best_acc:.4f}")
            else:
                patience += 1
                if patience >= config.PATIENCE:
                    print(f"  Early stop at epoch {epoch}")
                    break

        save_checkpoint(self.ckpt_dir/"last.pth", model, optimizer, epoch, best_acc, history)
        print(f"\nPhase 3 complete. Best val acc: {best_acc:.4f}")
        return history

    # ── Helpers ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _eval_image(self, model, loader, criterion):
        model.eval()
        tl, tc, tt = 0.0, 0, 0
        for imgs, tex, col, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = model.forward_image(imgs)
            tl += criterion(logits, labels).item() * imgs.size(0)
            tc += (logits.argmax(1) == labels).sum().item()
            tt += imgs.size(0)
        return tl / tt, tc / tt

    @torch.no_grad()
    def _eval_fusion(self, model, loader):
        model.eval()
        criterion = nn.CrossEntropyLoss()
        tl, tc, tt = 0.0, 0, 0
        for imgs, tex, col, labels in loader:
            imgs, tex, col, labels = (
                imgs.to(self.device), tex.to(self.device),
                col.to(self.device), labels.to(self.device),
            )
            logits, _ = model.forward_fusion(imgs, tex, col)
            tl += criterion(logits, labels).item() * imgs.size(0)
            tc += (logits.argmax(1) == labels).sum().item()
            tt += imgs.size(0)
        return tl / tt, tc / tt
