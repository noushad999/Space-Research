#!/usr/bin/env python
"""
run_diversity.py — the diversity-fix experiment (the prescription).

Trains the 8-class overlap on BOTH sources (diverse), 3 seeds, then evaluates
each model on the wild and studio test sets separately. If diversity closes the
shortcut, the cross-source gap collapses from ~38 pp (single-source) to ~0.

    python pipelines/run_diversity.py --smoke            # 1-epoch, 1-seed pipeline check
    nohup python pipelines/run_diversity.py > div.log 2>&1 &   # the real run (bounded GPU)

Faithful setup: Phase-1 image-only + strong-aug (same as the anchor). Resumable,
thermal cooldown, lock-guarded (refuses while the anchor is still running).
"""
import argparse, atexit, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "outputs"
CKPT = OUT / "checkpoints"
MANIFEST = str(OUT / "manifest_overlap_only.json")
ANCHOR_LOCK = OUT / ".cross_source.lock"
DIV_LOCK = OUT / ".diversity.lock"


def _alive(lock):
    if not lock.exists():
        return False
    try:
        os.kill(int(lock.read_text().strip()), 0); return True
    except Exception:
        return False


def acquire_lock():
    if _alive(ANCHOR_LOCK):
        sys.exit(f"[LOCK] run_cross_source.py still running (PID {ANCHOR_LOCK.read_text().strip()}). Wait.")
    if _alive(DIV_LOCK):
        sys.exit(f"[LOCK] run_diversity.py already running (PID {DIV_LOCK.read_text().strip()}).")
    OUT.mkdir(exist_ok=True)
    DIV_LOCK.write_text(str(os.getpid()))
    atexit.register(lambda: DIV_LOCK.exists() and DIV_LOCK.unlink())


def banner(m): print("\n" + "=" * 72 + f"\n  {m}\n" + "=" * 72, flush=True)


def run(cmd):
    print(">> " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    ap.add_argument("--cooldown", type=int, default=120)
    ap.add_argument("--no-skip", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    acquire_lock()
    seeds = [args.seeds[0]] if args.smoke else args.seeds
    sfx = (lambda s: f"overlap_both_s{s}" + ("_smoke" if args.smoke else ""))
    msfx = (lambda s: f"s{s}_smoke" if args.smoke else f"s{s}")
    t0 = time.time()

    banner(f"TRAIN — both-source overlap, {len(seeds)} seeds")
    for i, s in enumerate(seeds):
        ck = CKPT / sfx(s) / "p1_best.pth"
        if ck.exists() and not args.no_skip and not args.smoke:
            print(f"  [skip] {sfx(s)} exists"); continue
        cmd = [PY, "scripts/train/train_unified.py", "--phase", "1", "--strong-aug",
               "--manifest", MANIFEST, "--suffix", sfx(s), "--seed", str(s)]
        if args.smoke:
            cmd.append("--smoke")
        if run(cmd) != 0:
            sys.exit(f"training failed ({sfx(s)})")
        if args.cooldown and i < len(seeds) - 1 and not args.smoke:
            print(f"  cooling {args.cooldown}s ...", flush=True); time.sleep(args.cooldown)

    banner("EVAL — both-source model on each source")
    for s in seeds:
        ck = CKPT / sfx(s) / "p1_best.pth"
        if ck.exists():
            run([PY, "scripts/eval/eval_diversity.py", "--ckpt", str(ck), "--out_suffix", msfx(s)])

    banner("AGGREGATE + figure")
    import json, glob
    import numpy as np
    files = [f for f in sorted(glob.glob(str(OUT / "diversity_s*.json")))
             if ("smoke" in f) == args.smoke]
    if files:
        w = np.array([json.load(open(f))["acc_wild"] for f in files])
        st = np.array([json.load(open(f))["acc_studio"] for f in files])
        gap = np.array([json.load(open(f))["gap_pp"] for f in files])
        agg = {"n_seeds": len(files),
               "acc_wild": {"mean": float(w.mean()), "std": float(w.std(ddof=1) if len(w) > 1 else 0)},
               "acc_studio": {"mean": float(st.mean()), "std": float(st.std(ddof=1) if len(st) > 1 else 0)},
               "gap_pp": {"mean": float(gap.mean()), "std": float(gap.std(ddof=1) if len(gap) > 1 else 0)}}
        out = OUT / ("diversity_aggregate_smoke.json" if args.smoke else "diversity_aggregate.json")
        json.dump(agg, open(out, "w"), indent=2)
        print(f"\n  both-source: wild {agg['acc_wild']['mean']*100:.2f}%  "
              f"studio {agg['acc_studio']['mean']*100:.2f}%  "
              f"gap {agg['gap_pp']['mean']:.2f} +/- {agg['gap_pp']['std']:.2f} pp  (n={len(files)})")
        if not args.smoke:
            rc = subprocess.run([PY, "scripts/figures/plot_diversity.py"], cwd=str(ROOT)).returncode
            print(f"  figure: {'ok' if rc == 0 else 'FAILED'}")
    print(f"\n  total wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
