"""Fourier amplitude-mix with phase preservation (FACT; Xu et al., CVPR 2021).

Acquisition shift (white background, warm cast, JPEG, gamma) concentrates in the
*amplitude* spectrum; granule shape/micro-texture lives in *phase*. During
training, with probability ``p``, mix each image's amplitude spectrum with a
shuffled peer's while KEEPING its own phase -- perturbing acquisition style
without touching semantics. Identity at eval. ARC-V's M2 mechanism; the
phase-stable output is also what the (previously dormant) frequency branch
consumes.

Pair with :func:`phase_consistency_loss` -- a FACT-style co-teaching term that
keeps the prediction on a strongly amplitude-mixed image consistent with a
weakly-mixed copy (they share phase, hence semantics).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierAmplitudeMix(nn.Module):
    def __init__(self, p: float = 0.5, eta: float = 1.0):
        super().__init__()
        self.p = float(p)
        self.eta = float(eta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        if torch.rand(1).item() > self.p:
            return x
        b = x.size(0)
        f = torch.fft.rfft2(x)
        amp, phase = f.abs(), f.angle()
        perm = torch.randperm(b, device=x.device)
        lam = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.0, self.eta)
        amp_mix = lam * amp + (1.0 - lam) * amp[perm]
        out = torch.fft.irfft2(torch.polar(amp_mix, phase), s=x.shape[-2:])
        return out.to(dtype=x.dtype)

    def extra_repr(self) -> str:
        return f"p={self.p}, eta={self.eta}"


def phase_consistency_loss(logits_strong: torch.Tensor,
                           logits_weak: torch.Tensor) -> torch.Tensor:
    """Symmetric Jensen-Shannon divergence between the class posteriors of a
    strongly amplitude-mixed image and a weakly-mixed copy. Both share this image's
    phase (semantics), so their predictions should agree (FACT co-teaching)."""
    p = F.softmax(logits_strong, dim=1)
    q = F.softmax(logits_weak, dim=1)
    m = (0.5 * (p + q)).clamp_min(1e-8)
    kl_pm = (p * (p.clamp_min(1e-8).log() - m.log())).sum(dim=1)
    kl_qm = (q * (q.clamp_min(1e-8).log() - m.log())).sum(dim=1)
    return (0.5 * (kl_pm + kl_qm)).mean()
