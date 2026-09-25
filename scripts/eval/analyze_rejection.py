"""Summarize selective prediction from saved cross-source predictions.

This script reads existing calibration JSON files. It does not run a model. The
three seeds share one test set so their rows are analyzed separately and are not
pooled as independent observations.
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import numpy as np

from src import figstyle

ROOT = Path(_base)
OUT = ROOT / "outputs" / "rejection"
TARGETS = (0.90, 0.95, 0.99)


def selective_curve(confidence, correct):
    confidence = np.asarray(confidence)
    correct = np.asarray(correct, dtype=float)
    accepted = correct[np.argsort(-confidence)]
    k = np.arange(1, len(accepted) + 1)
    coverage = k / len(accepted)
    accuracy = np.cumsum(accepted) / k
    return coverage, accuracy, float(correct.mean())


def main():
    files = [
        f for f in sorted(glob.glob(str(ROOT / "outputs" / "calibration_s*.json")))
        if "smoke" not in Path(f).name.lower()
    ]
    if not files:
        raise SystemExit("no calibration_s*.json - run eval_calibration.py first")

    per_seed = []
    curves = []
    coverage = None
    for filename in files:
        data = json.load(open(filename, encoding="utf-8"))["in"]["ss"]
        coverage, accuracy, base_accuracy = selective_curve(
            data["confidence"], data["correct"]
        )
        marks = {}
        for target in TARGETS:
            valid = np.where(accuracy >= target)[0]
            marks[str(target)] = float(coverage[valid[-1]]) if len(valid) else 0.0
        seed_match = re.search(r"_s(\d+)", Path(filename).stem)
        per_seed.append({
            "seed": int(seed_match.group(1)) if seed_match else Path(filename).stem,
            "n": int(len(accuracy)),
            "base_accuracy": base_accuracy,
            "coverage_at_accuracy": marks,
        })
        curves.append(accuracy)

    base = np.asarray([row["base_accuracy"] for row in per_seed])
    aggregate_marks = {}
    for target in TARGETS:
        values = np.asarray([
            row["coverage_at_accuracy"][str(target)] for row in per_seed
        ])
        aggregate_marks[str(target)] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
        }
    aggregate = {
        "base_accuracy_mean": float(base.mean()),
        "base_accuracy_sd": float(base.std(ddof=1)),
        "coverage_at_accuracy": aggregate_marks,
    }
    output = {
        "unit": "seed",
        "note": "The same test images are used in each seed. Seed rows are not pooled.",
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    with open(ROOT / "outputs" / "rejection.json", "w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)

    figstyle.apply()
    import matplotlib.pyplot as plt

    palette = figstyle.PALETTE
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for row, accuracy in zip(per_seed, curves):
        ax.plot(
            coverage * 100,
            accuracy * 100,
            color=palette["wild"],
            lw=1,
            alpha=0.28,
            label=f"seed {row['seed']}",
        )
    mean_curve = np.mean(np.vstack(curves), axis=0)
    ax.plot(
        coverage * 100,
        mean_curve * 100,
        color=palette["wild"],
        lw=2.4,
        label="mean across seeds",
    )
    ax.axhline(
        base.mean() * 100,
        color=palette["chance"],
        ls=":",
        lw=1,
        label=f"full-coverage mean {base.mean()*100:.0f}%",
    )
    for target in TARGETS:
        mean_coverage = aggregate_marks[str(target)]["mean"]
        if mean_coverage > 0:
            ax.plot(
                mean_coverage * 100,
                target * 100,
                "o",
                color=palette["cross_broken"],
                ms=6,
            )
    ax.set_xlabel("coverage (% of wild predictions kept)")
    ax.set_ylabel("accuracy on accepted (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(50, 101)
    ax.set_title("Selective prediction on the wild test (studio-trained model)", fontsize=11)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    figstyle.save(fig, str(OUT))

    summary = []
    for target in TARGETS:
        item = aggregate_marks[str(target)]
        summary.append(
            f">= {int(target*100)}% acc @ "
            f"{item['mean']*100:.1f}+/-{item['sd']*100:.1f}% coverage"
        )
    print(
        f"base acc {base.mean()*100:.1f}+/-{base.std(ddof=1)*100:.1f}%  |  "
        + "  ".join(summary)
    )


if __name__ == "__main__":
    main()
