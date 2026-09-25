"""RSC -- Representation Self-Challenging (Huang et al., ECCV 2020) -- DG ablation.

Each step, mute the features with the largest loss-gradient -- the dominant,
most-relied-upon directions, i.e. exactly the studio shortcut -- and force the
prediction from the remainder, so the model cannot lean on a single acquisition
cue. This module is the core masking primitive; the trainer computes the feature
gradient, builds the mask on a random subset of the batch, and re-forwards through
the head. Identity at eval (the trainer skips it).
"""
import torch
import torch.nn.functional as F


def rsc_feature_mask(grad: torch.Tensor, drop_frac: float = 1.0 / 3.0) -> torch.Tensor:
    """Return a 0/1 mask (shape of ``grad``) that zeros, per sample, the
    ``drop_frac`` fraction of features with the largest absolute gradient."""
    _, c = grad.shape
    k = int(round(drop_frac * c))
    mask = torch.ones_like(grad)
    if k <= 0:
        return mask
    _, idx = grad.abs().topk(k, dim=1)
    return mask.scatter(1, idx, 0.0)


def rsc_step(feature_fn, head, imgs, labels,
             drop_frac: float = 1.0 / 3.0, apply_frac: float = 0.5):
    """One Representation Self-Challenging step (Huang et al., ECCV 2020).

    ``feature_fn(imgs)`` gives pooled features ``[B, C]`` and ``head(features)`` the
    logits ``[B, K]``. For a random ``apply_frac`` of the batch we mute, per sample,
    the ``drop_frac`` fraction of channels carrying the largest loss-gradient, the
    most-relied-upon directions, then predict from the remainder so the network
    cannot lean on a single dominant (acquisition) cue. Returns
    ``(loss, logits_challenged)``. With ``drop_frac <= 0`` or ``apply_frac <= 0`` it
    is plain cross-entropy, so the method reduces exactly to ERM. Use only in
    training; at eval the caller does an ordinary forward.
    """
    features = feature_fn(imgs)
    if drop_frac <= 0 or apply_frac <= 0:
        logits = head(features)
        return F.cross_entropy(logits, labels), logits
    loss_pre = F.cross_entropy(head(features), labels)
    grad = torch.autograd.grad(loss_pre, features, retain_graph=True)[0]
    mask = rsc_feature_mask(grad, drop_frac)               # zeros top-k channels per row
    if apply_frac < 1.0:                                   # challenge a random subset only
        b = features.size(0)
        keep = torch.rand(b, device=features.device) >= apply_frac
        mask = torch.where(keep.view(b, 1), torch.ones_like(mask), mask)
    logits = head(features * mask)
    return F.cross_entropy(logits, labels), logits
