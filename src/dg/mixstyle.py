"""MixStyle (Zhou et al., ICLR 2021) -- feature-statistic randomization.

During training, with probability ``p``, replace each sample's per-channel
statistics ``(mu, sigma)`` with a random convex combination of its own and a
shuffled peer's, then re-apply -- synthesizing pseudo-acquisition-domains the
single training source lacks. Identity at eval. This is ARC-V's M1 mechanism and
also a standalone DG baseline; insert as a hook after early backbone stages.

The operation rewrites only per-channel affine statistics, so the
instance-normalized *content* is preserved by construction.
"""
import torch
import torch.nn as nn


class MixStyle(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self._beta = torch.distributions.Beta(self.alpha, self.alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        if torch.rand(1).item() > self.p:
            return x
        b = x.size(0)
        mu = x.mean(dim=(2, 3), keepdim=True)
        sig = (x.var(dim=(2, 3), keepdim=True, unbiased=False) + self.eps).sqrt()
        x_norm = (x - mu) / sig
        lam = self._beta.sample((b, 1, 1, 1)).to(device=x.device, dtype=x.dtype)
        perm = torch.randperm(b, device=x.device)
        mu_mix = lam * mu + (1.0 - lam) * mu[perm]
        sig_mix = lam * sig + (1.0 - lam) * sig[perm]
        return x_norm * sig_mix + mu_mix

    def extra_repr(self) -> str:
        return f"p={self.p}, alpha={self.alpha}"


def register_mixstyle_hooks(modules, p: float = 0.5, alpha: float = 0.1):
    """Attach a MixStyle forward-hook to each module in ``modules`` (e.g. the first
    few backbone stages) so their outputs are style-randomized during training and
    passed through unchanged at eval. The hook syncs the MixStyle to the hooked
    module's train/eval mode and replaces its output. Returns the hook handles."""
    handles = []
    for m in modules:
        ms = MixStyle(p=p, alpha=alpha)

        def _hook(module, inputs, output, _ms=ms):
            _ms.train(module.training)
            return _ms(output)

        handles.append(m.register_forward_hook(_hook))
    return handles
