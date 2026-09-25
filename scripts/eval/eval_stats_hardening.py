#!/usr/bin/env python
"""Two statistics the reviewer panel asked for, computed from stored artifacts.

1. An image-level cluster bootstrap for the studio-to-wild collapse. The three seeds
   share one 1218-image wild test set, so resampling all 3x1218 rows independently
   (as the previous pooled bootstrap did) triple-counts each image and shrinks the
   interval by about sqrt(3). Here we resample the 1218 image indices once per
   bootstrap draw and average the per-seed collapse over that shared resample.

2. Seed-level paired t-tests for the single-source table. With three seeds the paired
   difference per seed is the right unit; this is both valid and, for the ARC-V and SD
   gains over ERM, stronger than the per-seed McNemar we already report.

Inputs are the per-sample correctness arrays in outputs/shortcut_evidence_s*.json and
the per-seed accuracies. No model is run.
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parents[2]
SEEDS = [42, 1337, 2024]
RNG = np.random.default_rng(20260721)
B = 10000

# Per-seed held-out wild accuracy, single-source (studio-trained), all four methods.
# From eval_baseline_mcnemar.py (CPU); reproduces the paper's Table means to 0.03.
ACC = {
    "erm":  [61.99, 59.85, 63.38],
    "rsc":  [65.35, 61.00, 64.70],
    "arcv": [67.73, 65.44, 68.80],
    "sd":   [68.31, 66.67, 67.82],
}


def cluster_bootstrap_collapse():
    # correct[seed] = studio-trained correctness on the 1218 wild-test images.
    # ss_test is the wild test set; in_trained_correct is the studio-trained model on it.
    correct = []
    for s in SEEDS:
        j = json.loads((BASE / "outputs" / f"shortcut_evidence_s{s}.json").read_text())
        correct.append(np.asarray(j["per_sample"]["ss_test"]["in_trained_correct"], float))
    correct = np.vstack(correct)            # (3, 1218); studio within-source is 100%
    n = correct.shape[1]
    point = float(np.mean([100 * (1 - c.mean()) for c in correct]))
    draws = np.empty(B)
    for b in range(B):
        idx = RNG.integers(0, n, n)         # one shared resample across seeds
        draws[b] = np.mean([100 * (1 - c[idx].mean()) for c in correct])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    # naive pooled bootstrap for contrast (resample 3*n rows independently)
    flat = correct.reshape(-1)
    m = flat.size
    pooled = np.empty(B)
    for b in range(B):
        pooled[b] = 100 * (1 - flat[RNG.integers(0, m, m)].mean())
    plo, phi = np.percentile(pooled, [2.5, 97.5])
    print("=== studio->wild collapse, bootstrap ===")
    print(f"point estimate           : {point:.2f} pp")
    print(f"cluster (image-level) 95%: [{lo:.2f}, {hi:.2f}]  width {hi-lo:.2f}")
    print(f"pooled (rows) 95%        : [{plo:.2f}, {phi:.2f}]  width {phi-plo:.2f}  (invalid, shown for contrast)")


def paired_t(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    t, p = stats.ttest_rel(a, b)
    return d.mean(), d.std(ddof=1), float(t), float(p)


def seed_level_tests():
    print("\n=== seed-level paired t-tests (single-source held-out wild) ===")
    for m in ("rsc", "arcv", "sd"):
        md, sd_, t, p = paired_t(ACC[m], ACC["erm"])
        print(f"{m:>4} - erm : {md:+.2f} +- {sd_:.2f}  t(2)={t:.1f}  p={p:.4f}")
    md, sd_, t, p = paired_t(ACC["sd"], ACC["arcv"])
    print(f" sd - arcv : {md:+.2f} +- {sd_:.2f}  t(2)={t:.2f}  p={p:.4f}  (the near-tie)")


if __name__ == "__main__":
    cluster_bootstrap_collapse()
    seed_level_tests()
