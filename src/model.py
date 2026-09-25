"""
SpiceFusionNet — Multi-Modal Fusion CNN
  • CNN Branch   : EfficientNet-B4 (ImageNet pretrained)
  • Texture Branch: LBP + GLCM features → MLP (→ 256-d)
  • Color Branch  : HSV histogram       → MLP (→ 128-d)
  • Fusion        : learned attention weights across three branches
  • Head          : FC → BN → ReLU → Dropout(0.4) → Softmax
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import config


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AttentionFusion(nn.Module):
    """Gate each branch with learned scalar weights (sum-to-1 via softmax)."""
    def __init__(self, cnn_dim: int, tex_dim: int, col_dim: int):
        super().__init__()
        self.total = cnn_dim + tex_dim + col_dim
        self.gate = nn.Sequential(
            nn.Linear(self.total, 3),
            nn.Softmax(dim=1),
        )

    def forward(self, f_cnn, f_tex, f_col):
        cat = torch.cat([f_cnn, f_tex, f_col], dim=1)   # (B, total)
        g   = self.gate(cat)                              # (B, 3)
        out = torch.cat([
            g[:, 0:1] * f_cnn,
            g[:, 1:2] * f_tex,
            g[:, 2:3] * f_col,
        ], dim=1)
        return out   # (B, total)


class SpiceFusionNet(nn.Module):
    def __init__(
        self,
        num_classes: int  = config.NUM_CLASSES,
        pretrained: bool  = config.PRETRAINED,
        drop_rate: float  = config.DROP_RATE,
        cnn_dim: int      = config.CNN_DIM,
        tex_dim: int      = config.TEX_DIM,
        col_dim: int      = config.COL_DIM,
        proj_dim: int     = config.PROJ_DIM,
        tex_in: int       = config.TEX_INPUT_DIM,
        col_in: int       = config.COL_INPUT_DIM,
    ):
        super().__init__()
        # ── CNN Branch ───────────────────────────────────────────────
        self.backbone = timm.create_model(
            config.BACKBONE,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_rate=drop_rate,
        )

        # ── Texture Branch ───────────────────────────────────────────
        self.tex_branch = _MLP(tex_in, 128, tex_dim, drop=0.3)

        # ── Color Branch ─────────────────────────────────────────────
        self.col_branch = _MLP(col_in, 64, col_dim, drop=0.3)

        # ── Attention Fusion ─────────────────────────────────────────
        self.fusion = AttentionFusion(cnn_dim, tex_dim, col_dim)
        total_dim = cnn_dim + tex_dim + col_dim

        # ── Projection Head (Phase 2 — SupCon) ───────────────────────
        self.proj_head = nn.Sequential(
            nn.Linear(cnn_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

        # ── Image-only Classifier (Phase 1) ──────────────────────────
        self.img_head = nn.Sequential(
            nn.Linear(cnn_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(512, num_classes),
        )

        # ── Fusion Classifier (Phase 3) ───────────────────────────────
        self.fusion_head = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(512, num_classes),
        )

    # ── Forward modes ────────────────────────────────────────────────

    def forward_image(self, x: torch.Tensor) -> torch.Tensor:
        """Phase 1: image → logits."""
        return self.img_head(self.backbone(x))

    def forward_contrastive(self, x: torch.Tensor) -> torch.Tensor:
        """Phase 2: image → L2-normalized projection features (for SupCon)."""
        feats = self.backbone(x)
        proj  = self.proj_head(feats)
        return F.normalize(proj, dim=1)

    def forward_fusion(
        self,
        x: torch.Tensor,
        tex: torch.Tensor,
        col: torch.Tensor,
    ):
        """Phase 3: image + texture + color → (logits, proj_feats)."""
        f_cnn  = self.backbone(x)
        f_tex  = self.tex_branch(tex)
        f_col  = self.col_branch(col)
        fused  = self.fusion(f_cnn, f_tex, f_col)
        logits = self.fusion_head(fused)
        proj   = F.normalize(self.proj_head(f_cnn), dim=1)
        return logits, proj

    def forward(self, x, tex=None, col=None):
        """Default forward — uses fusion if tex/col provided, else image-only."""
        if tex is not None and col is not None:
            logits, _ = self.forward_fusion(x, tex, col)
            return logits
        return self.forward_image(x)


def save_checkpoint(path, model, optimizer, epoch, best_val_acc, history):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "history": history,
    }, path)


def load_checkpoint(path: str, device: torch.device):
    ckpt  = torch.load(path, map_location=device)
    model = SpiceFusionNet()
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    return model, ckpt.get("epoch", 0), ckpt.get("best_val_acc", 0.0), ckpt.get("history", {})
