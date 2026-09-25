"""Deep CORAL (Sun & Saenko, ECCV-W 2016) -- DG baseline / marginal-alignment control.

Aligns second-order feature statistics: the squared Frobenius distance between the
feature covariance matrices of two sources, normalized by 4 d^2. In ARC-V this is the
*marginal* alignment control that the class-conditional M3 (cross-source SupCon) is
meant to beat (ARCV_METHOD_DESIGN.md §2d, §4.2).
"""
import torch


def _covariance(x: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    xc = x - x.mean(dim=0, keepdim=True)
    return (xc.t() @ xc) / max(n - 1, 1)


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared Frobenius distance between the two feature covariances / (4 d^2)."""
    d = source.size(1)
    diff = _covariance(source) - _covariance(target)
    return (diff * diff).sum() / (4.0 * d * d)
