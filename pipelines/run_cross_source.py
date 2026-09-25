#!/usr/bin/env python
"""
run_cross_source.py — one command for the headline cross-source experiment (3 seeds).

Run it once on a GPU and leave it. It does everything end-to-end and
prints the answer at the end (also written to outputs/FINAL_ANSWER.md):

  0. preflight   — verify src/, manifests, CUDA
  1. train       — SpiceFusionNet Phase-1 (image-only) + strong-aug, on each
                   source's 8-class overlap manifest, for 3 seeds  (6 runs)
  2. eval        — full 2x2 cross-source matrix per seed + McNemar significance
  3. aggregate   — mean +/- std across seeds + pooled McNemar (THE answer)
  4. figures     — 7 publication-grade figures (guarded; never abort the run)

Typical use (recommended — smoke first, then the real run in the background):

    python pipelines/run_cross_source.py --smoke                 # ~minutes, validates pipeline
    nohup python pipelines/run_cross_source.py > cross_source.log 2>&1 &   # the real multi-hour run

Faithful to the published setup: Phase-1 image-only + strong augmentation is the
exact configuration behind the -38 pp headline (eval uses forward_image on
p1_best.pth), and is ~3x cooler than full 3-phase training.

Flags:
  --stage {all,preflight,train,eval,aggregate,figures}   (default all)
  --seeds 42 1337 2024
  --cooldown 90            seconds to idle between training runs (thermal safety)
  --no-skip                retrain even if a checkpoint already exists
  --allow-cpu              proceed without CUDA (WARNING: extremely slow)
  --smoke                  1-epoch, 1-seed end-to-end validation (skips figures)
"""
import argparse
import atexit
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "outputs"
CKPT = OUT / "checkpoints"
LOGS = ROOT / "logs"
LOCK = OUT / ".cross_source.lock"
SS_MANIFEST = OUT / "manifest_overlap_ss.json"
IN_MANIFEST = OUT / "manifest_overlap_indian.json"

SOURCES = [("ss", SS_MANIFEST), ("indian", IN_MANIFEST)]
FIGSCRIPTS = [
    f"scripts/figures/{name}.py" for name in (
        "plot_shortcut_matrix", "plot_shortcut_per_class", "plot_transfer_comparison",
        "plot_teaser", "plot_dataset_contrast", "plot_gradcam_contrast", "plot_source_tsne",
    )
]


def acquire_lock():
    """Refuse to start if another run_cross_source.py is already active (prevents the
    double-launch collision: two runs writing the same checkpoints + double GPU heat)."""
    if LOCK.exists():
        try:
            old = int(LOCK.read_text().strip())
            os.kill(old, 0)   # raises if that PID is not alive
            sys.exit(f"[LOCK] another run_cross_source.py is already running (PID {old}).\n"
                     f"       Do NOT start a second one. If this is stale, remove {LOCK}.")
        except (ValueError, ProcessLookupError):
            pass              # stale/garbage lock -> take over
        except PermissionError:
            sys.exit(f"[LOCK] a live process holds {LOCK}. Remove it only if you are sure it is stale.")
    OUT.mkdir(exist_ok=True)
    LOCK.write_text(str(os.getpid()))
    atexit.register(lambda: LOCK.exists() and LOCK.unlink())


def banner(msg):
    print("\n" + "=" * 72 + f"\n  {msg}\n" + "=" * 72, flush=True)


def run(cmd, capture=False, log=None):
    print(">> " + " ".join(str(c) for c in cmd), flush=True)
    if capture:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if log:
            LOGS.mkdir(exist_ok=True)
            (LOGS / log).write_text((r.stdout or "") + "\n" + (r.stderr or ""), encoding="utf-8")
        return r.returncode, r.stdout or "", r.stderr or ""
    return subprocess.run(cmd, cwd=str(ROOT)).returncode, "", ""


# ── stages ───────────────────────────────────────────────────────────────────

def preflight(args):
    banner("STAGE 0 — preflight")
    ok = True
    ds = ROOT / "src" / "dataset.py"
    if not ds.exists() or "load_manifest_splits" not in ds.read_text(encoding="utf-8", errors="ignore"):
        print("  [FATAL] src/dataset.py missing the manifest-aware pipeline "
              "(load_manifest_splits). Update src/dataset.py.")
        ok = False
    else:
        print("  [ok] src/dataset.py has the manifest pipeline")
    for m in (SS_MANIFEST, IN_MANIFEST):
        if m.exists():
            print(f"  [ok] {m.name}")
        else:
            print(f"  [FATAL] missing manifest {m}"); ok = False
    try:
        import torch
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "CPU"
        print(f"  [{'ok' if cuda else 'WARN'}] torch {torch.__version__} | device: {dev}")
        if not cuda and not (args.allow_cpu or args.smoke):
            print("  [FATAL] no CUDA. Real training on CPU is impractical. "
                  "Re-run with --allow-cpu to override, or fix the GPU env.")
            ok = False
    except Exception as e:
        print(f"  [FATAL] cannot import torch: {e}"); ok = False
    if not ok:
        sys.exit("preflight failed — see messages above")
    print("  preflight OK")


def _suffix(base, seed, smoke):
    return f"overlap_{base}_s{seed}" + ("_smoke" if smoke else "")


def _matrix_suffix(seed, smoke):
    return f"s{seed}_smoke" if smoke else f"s{seed}"


def train(args, seeds):
    banner(f"STAGE 1 — train  ({len(seeds)} seeds x 2 sources = {len(seeds)*2} runs)")
    runs = [(seed, base, man) for seed in seeds for base, man in SOURCES]
    for i, (seed, base, man) in enumerate(runs):
        suffix = _suffix(base, seed, args.smoke)
        ckpt = CKPT / suffix / "p1_best.pth"
        if ckpt.exists() and not args.no_skip and not args.smoke:
            print(f"  [skip] {suffix} — checkpoint exists ({ckpt})")
            continue
        cmd = [PY, "scripts/train/train_unified.py", "--phase", "1", "--strong-aug",
               "--manifest", str(man), "--suffix", suffix, "--seed", str(seed)]
        if args.smoke:
            cmd.append("--smoke")
        rc, _, _ = run(cmd)
        if rc != 0:
            sys.exit(f"training run failed ({suffix}, rc={rc}) — stopping")
        if args.cooldown and i < len(runs) - 1 and not args.smoke:
            print(f"  cooling down {args.cooldown}s (thermal) ...", flush=True)
            time.sleep(args.cooldown)


def evaluate(args, seeds):
    banner("STAGE 2 — eval (2x2 matrix + McNemar per seed)")
    done = []
    for seed in seeds:
        ss_ckpt = CKPT / _suffix("ss", seed, args.smoke) / "p1_best.pth"
        in_ckpt = CKPT / _suffix("indian", seed, args.smoke) / "p1_best.pth"
        if not (ss_ckpt.exists() and in_ckpt.exists()):
            print(f"  [skip seed {seed}] missing checkpoints")
            continue
        suf = _matrix_suffix(seed, args.smoke)
        rc, _, _ = run([PY, "scripts/eval/eval_shortcut_test.py", "--ss_ckpt", str(ss_ckpt),
                        "--in_ckpt", str(in_ckpt), "--out_suffix", suf])
        if rc == 0:
            done.append(OUT / f"shortcut_test_matrix_{suf}.json")
        else:
            print(f"  [warn] eval failed for seed {seed} (rc={rc})")
    return done


def aggregate(args, seeds):
    banner("STAGE 3 — aggregate (mean +/- std + pooled McNemar)")
    mats = [OUT / f"shortcut_test_matrix_{_matrix_suffix(s, args.smoke)}.json" for s in seeds]
    mats = [m for m in mats if m.exists()]
    if not mats:
        print("  [warn] no per-seed matrices found; skipping aggregate")
        return None
    out = OUT / ("shortcut_multiseed_aggregate_smoke.json" if args.smoke
                 else "shortcut_multiseed_aggregate.json")
    rc, _, _ = run([PY, "scripts/eval/aggregate_shortcut_seeds.py", "--matrices", *map(str, mats),
                    "--out", str(out)])
    return out if rc == 0 else None


def figures(args):
    banner("STAGE 4 — figures (guarded; a failure never aborts the run)")
    status = {}
    for script in FIGSCRIPTS:
        rc, so, se = run([PY, script], capture=True, log=f"fig_{script}.log")
        status[script] = (rc == 0)
        if rc == 0:
            print(f"  [ok] {script}")
        else:
            tail = (se or so).strip().splitlines()[-1:] or [""]
            print(f"  [FAILED] {script} (rc={rc}) — {tail[0]}  (see logs/fig_{script}.log)")
    return status


def final_answer(agg_path, fig_status):
    banner("RESULT")
    import json
    lines = ["# Cross-source experiment: result", ""]
    if agg_path and Path(agg_path).exists():
        d = json.load(open(agg_path))
        a = d["aggregate"]
        n = d.get("n_seeds", "?")
        pm = d.get("mcnemar_pooled") or {}

        def cell(k): return f"{a[k]['mean']*100:.2f} +/- {a[k]['std']*100:.2f}%"

        headline = (f"Studio->wild collapse: {-a['in_tax_pp']['mean']:+.2f} +/- "
                    f"{a['in_tax_pp']['std']:.2f} pp   (over {n} seeds)")
        free = (f"Wild->studio: {-a['ss_tax_pp']['mean']:+.2f} +/- "
                f"{a['ss_tax_pp']['std']:.2f} pp (free)")
        pv = f"pooled McNemar p = {pm.get('p_value'):.2e}" if pm.get("p_value") is not None else "McNemar n/a"
        for s in [headline, free, pv, "",
                  f"  SS-trained   / SS-test     (within):  {cell('ss_within')}",
                  f"  SS-trained   / Indian-test (free)  :  {cell('ss_cross')}",
                  f"  Indian-train / SS-test     (collapse): {cell('in_cross')}",
                  f"  Indian-train / Indian-test (within):  {cell('in_within')}"]:
            print("  " + s)
            lines.append(s)
    else:
        print("  [warn] no aggregate produced — check the eval/train stages")
        lines.append("No aggregate produced — check eval/train stages.")

    if fig_status is not None:
        lines += ["", "## Figures"]
        for k, ok in fig_status.items():
            lines.append(f"- {'[ok]' if ok else '[FAILED]'} outputs/{Path(k).stem.replace('plot_', '')}")
    (OUT / "FINAL_ANSWER.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  written -> {OUT/'FINAL_ANSWER.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["all", "preflight", "train", "eval", "aggregate", "figures"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    ap.add_argument("--cooldown", type=int, default=90)
    ap.add_argument("--no-skip", action="store_true", help="retrain even if checkpoint exists")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1-epoch, 1-seed pipeline validation")
    args = ap.parse_args()
    acquire_lock()

    seeds = [args.seeds[0]] if args.smoke else args.seeds
    if args.smoke:
        print("** SMOKE MODE ** — 1 epoch, 1 seed, figures skipped, cooldown off")

    t0 = time.time()
    agg_path, fig_status = None, None
    if args.stage in ("all", "preflight"):
        preflight(args)
    if args.stage in ("all", "train"):
        train(args, seeds)
    if args.stage in ("all", "eval"):
        evaluate(args, seeds)
    if args.stage in ("all", "aggregate"):
        agg_path = aggregate(args, seeds)
    else:
        agg_path = OUT / ("shortcut_multiseed_aggregate_smoke.json" if args.smoke
                          else "shortcut_multiseed_aggregate.json")
    if args.stage in ("all", "figures") and not args.smoke:
        fig_status = figures(args)
    if args.stage in ("all", "aggregate", "figures"):
        final_answer(agg_path, fig_status)

    print(f"\n  total wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
