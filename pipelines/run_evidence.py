#!/usr/bin/env python
"""
run_evidence.py — the reviewer-proof evidence pass (run AFTER run_cross_source.py).

Reuses the anchor's overlap checkpoints. NO training. Adds the cheap, high-value
rigor that preempts the standard objections:

  1. rich eval   — macro-F1 + full confusion per 2x2 cell, both directions
  2. dedup 2x2   — same eval on perceptual-hash dedup test manifests
                   (proves the collapse survives leakage removal)
  3. bootstrap   — 95% CI + effect size on the shortcut gap (standard + dedup)
  4. confusion   — 4-cell confusion figure
  5. table       — within-source architecture + augmentation tables

    python pipelines/run_evidence.py                 # after the anchor finishes

Refuses to start while run_cross_source.py is still running (GPU contention).
"""
import argparse, atexit, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "outputs"
CKPT = OUT / "checkpoints"
LOGS = ROOT / "logs"
ANCHOR_LOCK = OUT / ".cross_source.lock"
EV_LOCK = OUT / ".evidence.lock"
SS_M = str(OUT / "manifest_overlap_ss.json")
IN_M = str(OUT / "manifest_overlap_indian.json")
SS_D = str(OUT / "manifest_overlap_ss_dedup.json")
IN_D = str(OUT / "manifest_overlap_indian_dedup.json")


def _alive(lock):
    if not lock.exists():
        return False
    try:
        os.kill(int(lock.read_text().strip()), 0); return True
    except Exception:
        return False


def acquire_lock():
    if _alive(ANCHOR_LOCK):
        sys.exit(f"[LOCK] run_cross_source.py is still running (PID {ANCHOR_LOCK.read_text().strip()}). "
                 "Wait for it to finish, then run the evidence pass.")
    if _alive(EV_LOCK):
        sys.exit(f"[LOCK] run_evidence.py already running (PID {EV_LOCK.read_text().strip()}).")
    OUT.mkdir(exist_ok=True)
    EV_LOCK.write_text(str(os.getpid()))
    atexit.register(lambda: EV_LOCK.exists() and EV_LOCK.unlink())


def banner(m): print("\n" + "=" * 72 + f"\n  {m}\n" + "=" * 72, flush=True)


def run(cmd, log=None, guard=False):
    print(">> " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=guard, text=guard)
    if guard and log:
        LOGS.mkdir(exist_ok=True)
        (LOGS / log).write_text((r.stdout or "") + "\n" + (r.stderr or ""), encoding="utf-8")
    return r.returncode


def rich_eval(seeds, skip):
    banner("STAGE 1+2 — rich eval (standard + dedup)")
    for seed in seeds:
        ss = CKPT / f"overlap_ss_s{seed}" / "p1_best.pth"
        ind = CKPT / f"overlap_indian_s{seed}" / "p1_best.pth"
        if not (ss.exists() and ind.exists()):
            print(f"  [skip seed {seed}] missing overlap checkpoints"); continue
        for suffix, sm, im in [(f"s{seed}", SS_M, IN_M), (f"dedup_s{seed}", SS_D, IN_D)]:
            outp = OUT / f"shortcut_evidence_{suffix}.json"
            if outp.exists() and skip:
                print(f"  [skip] {outp.name} exists"); continue
            if "dedup" in suffix and not (Path(sm).exists() and Path(im).exists()):
                print(f"  [skip dedup] dedup manifests missing"); continue
            run([PY, "scripts/eval/eval_confusion.py", "--ss_ckpt", str(ss), "--in_ckpt", str(ind),
                 "--ss_manifest", sm, "--in_manifest", im, "--out_suffix", suffix])


def analyze():
    banner("STAGE 3 — bootstrap CI + effect size")
    import glob
    for label, pat in [("standard", "shortcut_evidence_s*.json"),
                       ("dedup", "shortcut_evidence_dedup_s*.json")]:
        mats = sorted(glob.glob(str(OUT / pat)))
        if label == "standard":
            mats = [m for m in mats if "dedup" not in Path(m).name]
        if not mats:
            print(f"  [skip {label}] no evidence files"); continue
        run([PY, "scripts/eval/analyze_bootstrap.py", "--matrices", *mats, "--label", label,
             "--out", str(OUT / f"bootstrap_ci_{label}.json")])


def figures_tables():
    banner("STAGE 4+5 — confusion figure + backbone/aug tables (guarded)")
    for script, log in [("scripts/figures/plot_confusion_cross_source.py", "ev_confusion.log"),
                        ("scripts/figures/make_backbone_table.py", "ev_table.log")]:
        rc = run([PY, script], log=log, guard=True)
        print(f"  [{'ok' if rc == 0 else 'FAILED'}] {script}"
              + ("" if rc == 0 else f" (see logs/{log})"))


def summary(seeds):
    banner("EVIDENCE SUMMARY")
    import json
    lines = ["# Reviewer-proof evidence summary", ""]
    for label in ("standard", "dedup"):
        p = OUT / f"bootstrap_ci_{label}.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        blk = d.get("collapse_gap_on_wild_test")
        if blk:
            e = blk["effect"]
            s = (f"[{label}] shortcut gap on wild test = {blk['gap_pp']:.2f} pp, "
                 f"95% CI [{blk['ci95_pp'][0]:.2f}, {blk['ci95_pp'][1]:.2f}] "
                 f"(n={blk['n_samples']}, odds ratio {e['odds_ratio']:.0f})")
            print("  " + s); lines.append(s)
    for extra in ["", "Artifacts: shortcut_evidence_*.json, bootstrap_ci_*.json, "
                  "confusion_cross_source.pdf, backbone_baseline_table.{md,tex}"]:
        lines.append(extra)
    (OUT / "EVIDENCE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  written -> {OUT/'EVIDENCE_SUMMARY.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    ap.add_argument("--stage", default="all",
                    choices=["all", "eval", "bootstrap", "figures", "summary"])
    ap.add_argument("--no-skip", action="store_true")
    args = ap.parse_args()
    acquire_lock()
    t0 = time.time()
    if args.stage in ("all", "eval"):
        rich_eval(args.seeds, not args.no_skip)
    if args.stage in ("all", "bootstrap"):
        analyze()
    if args.stage in ("all", "figures"):
        figures_tables()
    if args.stage in ("all", "summary"):
        summary(args.seeds)
    print(f"\n  total wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
