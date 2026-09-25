"""Spectral Decoupling (Pezeshki et al., NeurIPS 2021) -- a single-source,
shortcut-mitigation regularizer used here as a DG baseline for ARC-V.

Gradient starvation lets a network lock onto one dominant, label-correlated feature
(here the acquisition background) and starve the rest. Spectral Decoupling adds an
L2 penalty on the logits, which decouples the learning dynamics of the features and
lets the weaker, class-intrinsic cues keep growing. It is a one-line addition to the
loss, so it slots into the same single-source training path as ERM, RSC, and ARC-V.
With ``lam=0`` it is exactly cross-entropy.
"""
import torch
import torch.nn.functional as F


def spectral_decoupling_loss(logits: torch.Tensor, labels: torch.Tensor,
                             lam: float = 0.1) -> torch.Tensor:
    """Cross-entropy plus the Spectral Decoupling penalty ``(lam/2) * ||logits||^2``,
    averaged over the batch. ``lam=0`` reduces to plain cross-entropy."""
    ce = F.cross_entropy(logits, labels)
    penalty = 0.5 * lam * logits.pow(2).sum(dim=1).mean()
    return ce + penalty
