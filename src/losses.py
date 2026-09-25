"""
Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
Used in Phase 2 for hard-negative spice pair fine-tuning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = config.P2_TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) — L2-normalized projection features
            labels:   (B,)  — class indices
        """
        device = features.device
        B = features.shape[0]

        # Cosine similarity matrix scaled by temperature
        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # Numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Positive mask: same label, excluding self
        labels_col = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels_col, labels_col.T).float().to(device)
        eye = torch.eye(B, device=device)
        pos_mask = pos_mask - eye  # remove self-pairs

        # Denominator: exp(sim) over all pairs except self
        exp_sim = torch.exp(sim) * (1.0 - eye)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean log-likelihood over positives
        n_pos = pos_mask.sum(dim=1)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-8)
        loss = -mean_log_prob_pos

        # Exclude anchors with no positive in batch
        valid = n_pos > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return loss[valid].mean()


class CombinedLoss(nn.Module):
    """alpha * CE  +  (1-alpha) * SupCon  — used in Phase 3."""
    def __init__(self, alpha: float = config.P3_ALPHA, label_smoothing: float = config.P3_LABEL_SMOOTH):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.supcon = SupConLoss()

    def forward(self, logits, proj_feats, labels):
        return self.alpha * self.ce(logits, labels) + (1 - self.alpha) * self.supcon(proj_feats, labels)
