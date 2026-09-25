"""ARC-V training objective (ARCV_METHOD_DESIGN.md §4.3).

Combines the loss-side ARC-V mechanisms on top of any model that maps images to
logits: the Fourier amplitude-mix (M2) augmentation plus a FACT-style
phase-consistency co-teaching term. MixStyle (M1) lives inside the backbone via
:func:`src.dg.mixstyle.register_mixstyle_hooks` (architectural, no loss term); the
cross-source SupCon (M3) is a LOSO-only add-on for Regime B.

With ``p_f=0`` and ``gamma_pc=0`` this reduces *exactly* to ERM cross-entropy, so a
single code path covers both the ERM baseline and the ARC-V method (toggled by
config) -- the honest apples-to-apples comparison the design demands.

``forward`` is a callable mapping an image batch to class logits, e.g.
``model.forward_image``.
"""
import torch.nn.functional as F

from src.dg.fourier import FourierAmplitudeMix, phase_consistency_loss


class ARCVObjective:
    def __init__(self, p_f: float = 0.5, eta: float = 1.0,
                 eta_weak: float = 0.3, gamma_pc: float = 1.0):
        self.strong = FourierAmplitudeMix(p=p_f, eta=eta)
        self.weak = FourierAmplitudeMix(p=p_f, eta=eta_weak)
        self.gamma_pc = float(gamma_pc)

    def __call__(self, forward, imgs, labels, training: bool = True):
        self.strong.train(training)
        self.weak.train(training)
        logits = forward(self.strong(imgs))
        loss = F.cross_entropy(logits, labels)
        if training and self.gamma_pc > 0.0:
            logits_weak = forward(self.weak(imgs))
            loss = loss + self.gamma_pc * phase_consistency_loss(logits, logits_weak)
        return loss, logits
