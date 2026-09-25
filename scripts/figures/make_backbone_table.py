"""
Assemble the within-source architecture + augmentation tables.

Kills two objections at once:
  * "you cherry-picked a weak backbone"  -> within-source is >=99% across ResNet-50,
    EfficientNet-B4, ViT and our fusion; the classical SVM baseline (~32%) shows the
    task is non-trivial.
  * "the numbers are just accuracy"      -> macro-F1 reported alongside top-1.

Reads existing outputs/baseline_comparison.json + outputs/ablation_results.json.
Pure assembly (no GPU, no re-eval). Writes markdown + LaTeX.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_MD = ROOT / "outputs" / "backbone_baseline_table.md"
OUT_TEX = ROOT / "outputs" / "backbone_baseline_table.tex"


def _load(name):
    p = ROOT / "outputs" / name
    return json.load(open(p)) if p.exists() else None


def main():
    base = _load("baseline_comparison.json")
    abl = _load("ablation_results.json")

    md = ["# Within-source robustness tables (assembled)\n",
          "## Architecture comparison (within-source, full Spice_Spectrum test)\n",
          "| Model | Top-1 | Top-5 | Macro-F1 |", "|---|---|---|---|"]
    tex = [r"\begin{tabular}{lccc}", r"\hline",
           r"Model & Top-1 & Top-5 & Macro-F1 \\", r"\hline"]
    if base:
        for b in base["baselines"]:
            t1 = f"{b['top1']*100:.2f}"
            t5 = f"{b['top5']*100:.2f}" if b.get("top5") is not None else "--"
            fm = f"{b['f1_macro']*100:.2f}" if b.get("f1_macro") is not None else "--"
            md.append(f"| {b['model']} | {t1} | {t5} | {fm} |")
            tex.append(f"{b['model']} & {t1} & {t5} & {fm} \\\\")
    tex += [r"\hline", r"\end{tabular}"]

    if abl and "A3" in abl:
        md += ["\n## Augmentation ablation (within-source)\n",
               "| Augmentation | Top-1 | Macro-F1 |", "|---|---|---|"]
        for a in abl["A3"]:
            md.append(f"| {a['name']} | {a['top1_accuracy']*100:.2f} | {a['f1_macro']*100:.2f} |")

    md.append("\n> Note: these are WITHIN-source metrics (they justify the model choice "
              "and show the task is non-trivial). The cross-source collapse is reported "
              "separately in the shortcut matrix — augmentation does not close it.")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    OUT_TEX.write_text("\n".join(tex), encoding="utf-8")
    print(f"saved -> {OUT_MD}\nsaved -> {OUT_TEX}")
    print("\n".join(md[:12]))


if __name__ == "__main__":
    main()
