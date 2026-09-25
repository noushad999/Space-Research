#!/usr/bin/env python
"""
run_benchmark.py — cross-source benchmark of many published backbones.

Trains each timm backbone on each source's 8-class overlap split, evaluates the
2x2 cross-source matrix, and assembles the comparison table + figure. The point:
EVERY architecture collapses studio->wild; the shortcut is not model-specific.

    python pipelines/run_benchmark.py --smoke                    # 1 model, 1 epoch, pipeline check
    nohup python pipelines/run_benchmark.py --cooldown 180 > bench.log 2>&1 &   # real (sessioned)

Resumable (skips finished checkpoints), thermal cooldown between runs,
lock-guarded (refuses while the anchor/diversity runs are active).
"""
import argparse, atexit, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "outputs"
CKPT = OUT / "checkpoints"
SS_M = str(OUT / "manifest_overlap_ss.json")
IN_M = str(OUT / "manifest_overlap_indian.json")
LOCKS = [OUT / ".cross_source.lock", OUT / ".diversity.lock"]
MY_LOCK = OUT / ".benchmark.lock"

# representative published architectures (timm names): CNNs + transformers
MODELS = ["resnet50", "efficientnet_b0", "efficientnet_b4", "mobilenetv3_large_100",
          "convnext_tiny", "densenet121", "vit_base_patch16_224",
          "swin_tiny_patch4_window7_224", "deit_small_patch16_224"]
SOURCES = [("ss", SS_M), ("indian", IN_M)]


def _alive(lock):
    try:
        return lock.exists() and (os.kill(int(lock.read_text().strip()), 0) or True)
    except Exception:
        return False


def acquire_lock():
    for L in LOCKS:
        if _alive(L):
            sys.exit(f"[LOCK] {L.name} is active (PID {L.read_text().strip()}). Wait for it to finish.")
    if _alive(MY_LOCK):
        sys.exit(f"[LOCK] run_benchmark.py already running (PID {MY_LOCK.read_text().strip()}).")
    OUT.mkdir(exist_ok=True); MY_LOCK.write_text(str(os.getpid()))
    atexit.register(lambda: MY_LOCK.exists() and MY_LOCK.unlink())


def banner(m): print("\n" + "=" * 72 + f"\n  {m}\n" + "=" * 72, flush=True)


def run(cmd):
    print(">> " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cooldown", type=int, default=180)
    ap.add_argument("--no-skip", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    acquire_lock()
    models = ["resnet50"] if args.smoke else args.models
    # Seed 42 keeps its original checkpoint names (already trained); other seeds are
    # scoped with _s{seed} so multi-seeding a model does not collide with seed 42.
    def sfx(m, s):
        base = f"bench_{m}_{s}" + (f"_s{args.seed}" if args.seed != 42 else "")
        return base + ("_smoke" if args.smoke else "")
    t0 = time.time()

    banner(f"TRAIN — {len(models)} models x 2 sources")
    runs = [(m, st, mf) for m in models for st, mf in SOURCES]
    for i, (m, st, mf) in enumerate(runs):
        ck = CKPT / sfx(m, st) / "best.pth"
        if ck.exists() and not args.no_skip and not args.smoke:
            print(f"  [skip] {sfx(m, st)}"); continue
        cmd = [PY, "scripts/train/train_backbone_overlap.py", "--model", m, "--manifest", mf,
               "--suffix", sfx(m, st), "--seed", str(args.seed), "--strong-aug"]
        if args.smoke:
            cmd.append("--smoke")
        if run(cmd) != 0:
            print(f"  [warn] training failed for {m} {st} — skipping this model")
            continue
        if args.cooldown and i < len(runs) - 1 and not args.smoke:
            print(f"  cooling {args.cooldown}s ...", flush=True); time.sleep(args.cooldown)

    # The per-model eval JSONs and the assembled table are seed-42 artifacts; extra
    # seeds only add checkpoints (aggregated separately) and must not overwrite them.
    if args.seed == 42:
        banner("EVAL — 2x2 per model")
        for m in models:
            ss_ck = CKPT / sfx(m, "ss") / "best.pth"
            in_ck = CKPT / sfx(m, "indian") / "best.pth"
            if ss_ck.exists() and in_ck.exists():
                run([PY, "scripts/eval/eval_backbone_shortcut.py", "--model", m,
                     "--ss_ckpt", str(ss_ck), "--in_ckpt", str(in_ck)])
        banner("TABLE + figure")
        if not args.smoke:
            run([PY, "scripts/figures/make_benchmark_table.py"])
    else:
        banner(f"seed {args.seed}: checkpoints only (eval/table are seed-42 artifacts)")
    print(f"\n  total wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
