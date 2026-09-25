"""IRMv1 penalty (Arjovsky et al., 2019) -- DG baseline.

Penalizes the sensitivity of the per-environment risk to a dummy scalar classifier
of value 1: ``(d/dw CE(w * logits, y))^2`` at ``w = 1``. When the representation is
truly invariant the optimal classifier is simultaneously optimal on every
environment, so this gradient vanishes. Rarely beats tuned ERM (Gulrajani &
Lopez-Paz); included as a required baseline (ARCV_METHOD_DESIGN.md §3, row 8).
"""
import torch
import torch.nn.functional as F


def irm_penalty(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    scale = torch.ones(1, device=logits.device, requires_grad=True)
    loss = F.cross_entropy(logits * scale, labels)
    grad, = torch.autograd.grad(loss, scale, create_graph=True)
    return (grad ** 2).sum()
