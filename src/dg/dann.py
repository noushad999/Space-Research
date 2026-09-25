"""DANN gradient reversal (Ganin & Lempitsky, ICML 2015) -- DG baseline.

Retired as the ARC-V *contribution* -- a marginal source-adversary optimizes source
decodability, which this project showed is uncorrelated with robustness, and it
plateaus once the discriminator is fooled (ARCV_METHOD_DESIGN.md §2). Kept as a
required baseline. The gradient-reversal layer is the identity on the forward pass
and negates (scaled by ``lambda_``) the gradient on the backward pass, so a source
discriminator stacked on top pushes the backbone toward source-confusion.
"""
import torch


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Identity forward; gradient is negated and scaled by ``lambda_`` backward."""
    return _GradReverse.apply(x, lambda_)
