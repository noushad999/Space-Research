"""Tests for src/dg/ domain-generalization modules (ARC-V, Paper 2).

Each module is a well-defined operation with checkable invariants; we test the
invariants, not an implementation. Run: python -m pytest test_dg.py -q
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import torch
import torch.nn as nn


def _styled_feats(b=4, c=8, h=6, w=6, seed=0):
    """Feature map where each sample has a distinct per-channel style (mu,sigma),
    so a style-mixing op is detectable and content-preservation is meaningful."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, c, h, w, generator=g)
    scale = torch.arange(1, b + 1).float().view(b, 1, 1, 1)   # per-sample sigma
    shift = torch.arange(0, b).float().view(b, 1, 1, 1) * 3.0  # per-sample mu
    return x * scale + shift


def _inorm(t, eps=1e-5):
    mu = t.mean(dim=(2, 3), keepdim=True)
    sd = t.std(dim=(2, 3), keepdim=True, unbiased=False) + eps
    return (t - mu) / sd


# ---------------------------------------------------------------- MixStyle (M1)
def test_mixstyle_is_identity_in_eval():
    from src.dg.mixstyle import MixStyle
    m = MixStyle(p=1.0, alpha=0.1).eval()
    x = _styled_feats()
    assert torch.equal(m(x), x), "MixStyle must be a no-op in eval mode"


def test_mixstyle_p0_never_mixes_even_in_train():
    from src.dg.mixstyle import MixStyle
    m = MixStyle(p=0.0, alpha=0.1).train()
    x = _styled_feats()
    assert torch.equal(m(x), x), "p=0 must never mix"


def test_mixstyle_preserves_shape():
    from src.dg.mixstyle import MixStyle
    m = MixStyle(p=1.0, alpha=0.1).train()
    x = _styled_feats()
    assert m(x).shape == x.shape


def test_mixstyle_preserves_normalized_content():
    """It rewrites only per-channel (mu,sigma); the instance-normalized content
    must be unchanged regardless of the random mixing."""
    torch.manual_seed(0)
    from src.dg.mixstyle import MixStyle
    m = MixStyle(p=1.0, alpha=0.1).train()
    x = _styled_feats()
    out = m(x)
    assert torch.allclose(_inorm(out), _inorm(x), atol=1e-3), \
        "MixStyle must preserve instance-normalized content"


def test_mixstyle_changes_style_when_applied():
    """With p=1 on a batch of distinct styles, per-sample channel means move."""
    torch.manual_seed(0)
    from src.dg.mixstyle import MixStyle
    m = MixStyle(p=1.0, alpha=0.1).train()
    x = _styled_feats()
    out = m(x)
    assert not torch.allclose(out.mean(dim=(2, 3)), x.mean(dim=(2, 3)), atol=1e-2), \
        "with p=1 the per-sample styles should change"


# ---------------------------------------------------- Fourier amplitude-mix (M2)
def test_fourier_preserves_phase():
    """Amplitude-mix keeps THIS image's phase spectrum (carries semantics)."""
    torch.manual_seed(0)
    from src.dg.fourier import FourierAmplitudeMix
    aug = FourierAmplitudeMix(p=1.0, eta=1.0).train()
    x = _styled_feats(c=3)
    out = aug(x)
    fx, fo = torch.fft.rfft2(x), torch.fft.rfft2(out)
    mask = (fx.abs() > 1e-2) & (fo.abs() > 1e-2)   # phase measurable in both
    px = (fx / (fx.abs() + 1e-8))[mask]
    po = (fo / (fo.abs() + 1e-8))[mask]
    assert torch.allclose(px.real, po.real, atol=1e-3), "phase (real) must be kept"
    assert torch.allclose(px.imag, po.imag, atol=1e-3), "phase (imag) must be kept"


def test_fourier_output_is_real_with_same_shape():
    from src.dg.fourier import FourierAmplitudeMix
    aug = FourierAmplitudeMix(p=1.0, eta=1.0).train()
    x = _styled_feats(c=3)
    out = aug(x)
    assert out.shape == x.shape and out.dtype == x.dtype and not out.is_complex()


def test_fourier_is_identity_in_eval():
    from src.dg.fourier import FourierAmplitudeMix
    aug = FourierAmplitudeMix(p=1.0, eta=1.0).eval()
    x = _styled_feats(c=3)
    assert torch.equal(aug(x), x)


def test_fourier_p0_never_mixes():
    from src.dg.fourier import FourierAmplitudeMix
    aug = FourierAmplitudeMix(p=0.0, eta=1.0).train()
    x = _styled_feats(c=3)
    assert torch.equal(aug(x), x)


def test_fourier_changes_amplitude_when_applied():
    torch.manual_seed(0)
    from src.dg.fourier import FourierAmplitudeMix
    aug = FourierAmplitudeMix(p=1.0, eta=1.0).train()
    x = _styled_feats(c=3)
    assert not torch.allclose(aug(x), x, atol=1e-2), "amplitude should be mixed"


# ------------------------------------------------ DANN gradient reversal (baseline)
def test_grl_forward_is_identity():
    from src.dg.dann import grad_reverse
    x = torch.randn(4, 8, requires_grad=True)
    assert torch.equal(grad_reverse(x, 1.0), x)


def test_grl_reverses_and_scales_gradient():
    from src.dg.dann import grad_reverse
    x = torch.randn(4, 8, requires_grad=True)
    grad_reverse(x, 2.0).sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -2.0)), \
        "GRL must negate and scale the gradient by lambda"


# ------------------------------------------------------------ Deep CORAL (baseline)
def test_coral_is_zero_for_identical_features():
    from src.dg.coral import coral_loss
    torch.manual_seed(0)
    x = torch.randn(64, 16)
    assert coral_loss(x, x).item() < 1e-6, "same covariance -> zero CORAL"


def test_coral_is_positive_for_different_covariance():
    from src.dg.coral import coral_loss
    torch.manual_seed(0)
    x = torch.randn(64, 16)
    y = torch.randn(64, 16) * 5.0 + 3.0
    assert coral_loss(x, y).item() > 0.1


# --------------------------------------------------------------- IRM penalty (baseline)
def test_irm_penalty_is_zero_for_scale_invariant_logits():
    from src.dg.irm import irm_penalty
    labels = torch.tensor([0, 1, 0, 1])
    logits = torch.zeros(4, 2)   # w*0 == 0 for any w -> risk constant in scale -> 0
    assert irm_penalty(logits, labels).item() < 1e-8


def test_irm_penalty_is_positive_when_scale_matters():
    from src.dg.irm import irm_penalty
    labels = torch.tensor([0, 1, 0, 1])
    logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0], [1.5, 0.0], [0.0, 1.5]])
    assert irm_penalty(logits, labels).item() > 1e-6


# ------------------------------------------------ phase-consistency loss (M2 term)
def test_phase_consistency_is_zero_for_identical_predictions():
    from src.dg.fourier import phase_consistency_loss
    torch.manual_seed(0)
    z = torch.randn(4, 6)
    assert phase_consistency_loss(z, z).item() < 1e-7, "JS(p,p) must be 0"


def test_phase_consistency_is_positive_and_symmetric():
    from src.dg.fourier import phase_consistency_loss
    torch.manual_seed(0)
    a, b = torch.randn(4, 6), torch.randn(4, 6)
    lab, lba = phase_consistency_loss(a, b), phase_consistency_loss(b, a)
    assert lab.item() > 1e-4, "different predictions -> positive"
    assert abs(lab.item() - lba.item()) < 1e-6, "JS is symmetric"


# --------------------------------------------------- RSC self-challenging (ablation)
def test_rsc_drops_the_highest_gradient_features():
    from src.dg.rsc import rsc_feature_mask
    grad = torch.tensor([[0.1, 0.9, 0.5, 0.2]])   # drop 25% -> top-1 is index 1
    mask = rsc_feature_mask(grad, drop_frac=0.25)
    assert mask[0, 1].item() == 0.0, "the dominant (highest-|grad|) feature is masked"
    assert mask.sum().item() == 3.0, "exactly 1 of 4 dropped"


def test_rsc_keeps_everything_when_drop_frac_zero():
    from src.dg.rsc import rsc_feature_mask
    grad = torch.randn(2, 8)
    mask = rsc_feature_mask(grad, drop_frac=0.0)
    assert mask.sum().item() == 16.0, "drop_frac=0 masks nothing"


def test_rsc_step_reduces_to_ce_when_drop_frac_zero():
    """drop_frac=0 challenges nothing, so one RSC step is plain cross-entropy (ERM)."""
    import torch.nn.functional as F
    from src.dg.rsc import rsc_step
    torch.manual_seed(0)
    imgs = torch.randn(4, 16)
    labels = torch.tensor([0, 1, 2, 3])
    feat = nn.Linear(16, 16); head = nn.Linear(16, 4)
    loss, logits = rsc_step(feat, head, imgs, labels, drop_frac=0.0)
    assert torch.allclose(loss, F.cross_entropy(head(feat(imgs)), labels), atol=1e-6), \
        "with nothing dropped, RSC must equal plain CE"


def test_rsc_step_challenges_and_stays_differentiable():
    """With a positive drop, the step must run the masked re-forward, stay finite,
    and pass gradient back into both the head and the feature extractor."""
    from src.dg.rsc import rsc_step
    torch.manual_seed(0)
    imgs = torch.randn(4, 16)
    labels = torch.tensor([0, 1, 2, 3])
    feat = nn.Linear(16, 16); head = nn.Linear(16, 4)
    loss, logits = rsc_step(feat, head, imgs, labels, drop_frac=1.0 / 3, apply_frac=1.0)
    assert torch.isfinite(loss) and loss.requires_grad and logits.shape == (4, 4)
    loss.backward()
    assert head.weight.grad is not None and feat.weight.grad is not None, \
        "gradient must reach the head and the backbone through the masked features"


# --------------------------------------- Spectral Decoupling (shortcut regularizer)
def test_sd_reduces_to_ce_when_lambda_zero():
    """lambda=0 removes the logit penalty, so SD is plain cross-entropy (ERM)."""
    import torch.nn.functional as F
    from src.dg.sd import spectral_decoupling_loss
    torch.manual_seed(0)
    logits = torch.randn(4, 5)
    labels = torch.tensor([0, 1, 2, 3])
    loss = spectral_decoupling_loss(logits, labels, lam=0.0)
    assert torch.allclose(loss, F.cross_entropy(logits, labels), atol=1e-6)


def test_sd_adds_positive_logit_penalty():
    """With lambda>0 the loss exceeds plain CE by the L2 logit penalty, and the
    penalty grows with logit magnitude."""
    import torch.nn.functional as F
    from src.dg.sd import spectral_decoupling_loss
    torch.manual_seed(0)
    logits = torch.randn(4, 5)
    labels = torch.tensor([0, 1, 2, 3])
    ce = F.cross_entropy(logits, labels)
    loss = spectral_decoupling_loss(logits, labels, lam=0.5)
    assert loss.item() > ce.item(), "SD must add a positive penalty over CE"
    bigger = spectral_decoupling_loss(logits * 2, labels, lam=0.5)
    smaller = spectral_decoupling_loss(logits * 0.5, labels, lam=0.5)
    assert (bigger - F.cross_entropy(logits * 2, labels)).item() > \
           (smaller - F.cross_entropy(logits * 0.5, labels)).item(), \
        "the penalty must grow with logit magnitude"


# --------------------------------------------- MixStyle backbone hook (integration)
def test_mixstyle_hook_is_identity_in_eval_and_perturbs_in_train():
    from src.dg.mixstyle import register_mixstyle_hooks
    torch.manual_seed(0)
    layer = nn.Conv2d(3, 6, 3, padding=1)
    x = _styled_feats(b=4, c=3, h=8, w=8, seed=0)   # distinct per-sample styles
    layer.eval()
    ref = layer(x).clone()                       # raw output, before hooking
    register_mixstyle_hooks([layer], p=1.0, alpha=0.1)
    layer.eval()
    assert torch.equal(layer(x), ref), "hooked MixStyle must be identity at eval"
    layer.train()
    out = layer(x)
    assert out.shape == ref.shape
    assert not torch.allclose(out, ref, atol=1e-3), "train -> MixStyle perturbs the output"


# ----------------------------------------------------- ARC-V objective (training core)
def test_arcv_objective_reduces_to_ce_when_disabled():
    """p_f=0 (no aug) + gamma_pc=0 (no consistency) == plain ERM cross-entropy."""
    from src.dg.arcv import ARCVObjective
    import torch.nn.functional as F
    torch.manual_seed(0)
    imgs = torch.randn(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 3])
    head = nn.Linear(3 * 8 * 8, 4)
    fwd = lambda z: head(z.flatten(1))
    loss, _ = ARCVObjective(p_f=0.0, gamma_pc=0.0)(fwd, imgs, labels, training=True)
    assert torch.allclose(loss, F.cross_entropy(fwd(imgs), labels), atol=1e-6)


def test_arcv_objective_is_finite_and_differentiable_with_consistency():
    from src.dg.arcv import ARCVObjective
    torch.manual_seed(0)
    imgs = torch.randn(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 3])
    head = nn.Linear(3 * 8 * 8, 4)
    fwd = lambda z: head(z.flatten(1))
    loss, logits = ARCVObjective(p_f=1.0, eta=1.0, gamma_pc=1.0)(fwd, imgs, labels, training=True)
    assert torch.isfinite(loss) and loss.requires_grad and logits.shape == (4, 4)
    loss.backward()
    assert head.weight.grad is not None
