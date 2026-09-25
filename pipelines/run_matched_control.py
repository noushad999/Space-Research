"""Size-matched control for the cross-source asymmetry, end to end.

The published matrix trains the in-the-wild arm on 5,551 images and the studio
arm on 2,688, so the direction of the collapse is confounded with the amount of
training data. This runs the control: subsample the in-the-wild training split
to the studio count, retrain that arm over three seeds, re-evaluate the 2x2
matrix against the existing studio checkpoints, and report the matched collapse
next to the published one.

Only the in-the-wild arm is retrained. The studio checkpoints already exist and
are unchanged, and val/test splits are untouched, so the numbers stay directly
comparable with the published matrix.

    python pipelines/run_matched_control.py                 # full run, 3 seeds
    python pipelines/run_matched_control.py --smoke         # 1 epoch, 1 seed, ~2 min
    python pipelines/run_matched_control.py --cooldown 300  # longer thermal pause

Stages can be skipped once done, e.g. --skip-train to only re-evaluate.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "outputs"
CKPT = OUT / "checkpoints"

SEEDS = [42, 1337, 2024]

# Published three-seed figures this control is testing (Section 5 of the paper).
PUB_COLLAPSE, PUB_COLLAPSE_SD = 37.77, 0.89
PUB_FREE, PUB_FREE_SD = 0.48, 0.09


def banner(msg):
    print("\n" + "=" * 72 + f"\n  {msg}\n" + "=" * 72, flush=True)


def run(cmd):
    print(">> " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def die(msg):
    sys.exit(f"\n!! {msg}\n   stopping; nothing further was run.")


def stage_manifests(seeds):
    banner(f"STAGE 1 - build size-matched manifests ({len(seeds)} seeds)")
    for seed in seeds:
        if run([PY, "scripts/data/make_matched_manifest.py", "--seed", str(seed)]) != 0:
            die(f"manifest build failed (seed {seed})")


def stage_train(seeds, smoke, cooldown):
    banner(f"STAGE 2 - train matched in-the-wild arm ({len(seeds)} runs)")
    for i, seed in enumerate(seeds):
        suffix = f"overlap_ssmatched_s{seed}" + ("_smoke" if smoke else "")
        ckpt = CKPT / suffix / "p1_best.pth"
        if ckpt.exists() and not smoke:
            print(f"  [skip] {suffix} - checkpoint exists")
            continue
        cmd = [PY, "scripts/train/train_unified.py", "--phase", "1", "--strong-aug",
               "--manifest", str(OUT / f"manifest_overlap_ss_matched_s{seed}.json"),
               "--suffix", suffix, "--seed", str(seed)]
        if smoke:
            cmd.append("--smoke")
        if run(cmd) != 0:
            die(f"training failed (seed {seed})")
        if cooldown and i < len(seeds) - 1 and not smoke:
            print(f"  cooling down {cooldown}s (thermal) ...", flush=True)
            time.sleep(cooldown)


def stage_eval(seeds, smoke):
    banner("STAGE 3 - evaluate 2x2 matrix against existing studio checkpoints")
    mats = []
    for seed in seeds:
        tag = "_smoke" if smoke else ""
        ss = CKPT / f"overlap_ssmatched_s{seed}{tag}" / "p1_best.pth"
        indian = CKPT / f"overlap_indian_s{seed}{tag}" / "p1_best.pth"
        if not ss.exists():
            print(f"  [skip seed {seed}] missing matched checkpoint")
            continue
        if not indian.exists():
            print(f"  [skip seed {seed}] missing studio checkpoint {indian}")
            continue
        suf = f"matched_s{seed}{tag}"
        if run([PY, "scripts/eval/eval_shortcut_test.py", "--ss_ckpt", str(ss),
                "--in_ckpt", str(indian), "--out_suffix", suf]) != 0:
            print(f"  [warn] eval failed for seed {seed}")
            continue
        mats.append(OUT / f"shortcut_test_matrix_{suf}.json")
    return mats


def stage_aggregate(mats, smoke):
    banner("STAGE 4 - aggregate and compare against the published matrix")
    if not mats:
        die("no matrices produced; nothing to aggregate")
    out = OUT / ("shortcut_matched_aggregate_smoke.json" if smoke
                 else "shortcut_matched_aggregate.json")
    if run([PY, "scripts/eval/aggregate_shortcut_seeds.py", "--matrices", *map(str, mats),
            "--out", str(out)]) != 0:
        die("aggregation failed")
    report(out, smoke)
    return out


def _find(obj, *names):
    """Pull a (mean, std) pair for the first key matching one of `names`.

    aggregate_shortcut_seeds.py nests each quantity as
    {"aggregate": {"in_tax_pp": {"mean": .., "std": ..}}}, so match on the
    parent key and read its mean. A bare numeric value is accepted too.
    """
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if any(n in k.lower() for n in names):
                    if isinstance(v, dict) and isinstance(v.get("mean"), (int, float)):
                        return v["mean"], v.get("std")
                    if isinstance(v, (int, float)):
                        return v, None
                stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return None, None


def report(path, smoke):
    try:
        agg = json.loads(Path(path).read_text())
    except Exception as exc:
        print(f"  [warn] could not read {path}: {exc}")
        return

    print(f"\n  aggregate written to {path}")
    collapse, collapse_sd = _find(agg, "in_tax", "collapse", "studio_to_wild")
    free, free_sd = _find(agg, "ss_tax", "wild_to_studio")

    def fmt(v, sd):
        if v is None:
            return "n/a"
        return f"{v:.2f}" + (f" +/-{sd:.2f}" if isinstance(sd, (int, float)) else "")

    print("\n  " + "-" * 66)
    print(f"  {'quantity':<26}{'published':>18}{'size-matched':>20}")
    print("  " + "-" * 66)
    print(f"  {'studio -> wild collapse':<26}"
          f"{PUB_COLLAPSE:>11.2f} +/-{PUB_COLLAPSE_SD:<4.2f}{fmt(collapse, collapse_sd):>20}")
    print(f"  {'wild -> studio (free)':<26}"
          f"{PUB_FREE:>11.2f} +/-{PUB_FREE_SD:<4.2f}{fmt(free, free_sd):>20}")
    print("  " + "-" * 66)

    if smoke:
        print("\n  SMOKE RUN - numbers are meaningless, this only proves the pipeline runs.")
        return
    if collapse is None:
        print("\n  Could not locate the collapse field automatically.")
        print("  Open the aggregate JSON and compare by hand.")
        return

    print("\n  NOTE: only the in-the-wild arm was retrained. The studio arm and the")
    print("  test splits are unchanged, so the collapse row CANNOT move and proves")
    print("  nothing here. The row that tests the confound is the free direction.")

    if free is None or collapse is None:
        print("  Could not locate both directions; compare the aggregate JSON by hand.")
        return

    print(f"\n  Free direction at matched size: {free:.2f} pp "
          f"(published {PUB_FREE:.2f} pp at 5,551 images).")
    if free > 0:
        print(f"  Asymmetry at matched training size: {collapse / free:.0f} to 1.")
    if free < 5:
        print("  READ: the diverse arm still transfers freely on the studio image")
        print("        count, so training-set SIZE was not driving the direction.")
        print("        Acquisition variety was. The confound is answered.")
    elif free < 15:
        print("  READ: the free direction degraded materially once size was matched,")
        print("        so part of the published asymmetry was sample count.")
    else:
        print("  READ: the free direction collapses once size is matched, so the")
        print("        published asymmetry was largely a data-quantity effect.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 epoch, 1 seed, ~2 min")
    ap.add_argument("--cooldown", type=int, default=180,
                    help="seconds to idle between training runs (default 180)")
    ap.add_argument("--skip-manifests", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    seeds = SEEDS[:1] if args.smoke else SEEDS
    t0 = time.time()

    banner("SIZE-MATCHED CONTROL" + ("  [SMOKE]" if args.smoke else ""))
    print(f"  seeds     : {seeds}")
    print(f"  cooldown  : {args.cooldown}s between training runs")
    print(f"  retraining: in-the-wild arm only; studio checkpoints reused")

    if not args.skip_manifests:
        stage_manifests(seeds)
    if not args.skip_train:
        stage_train(seeds, args.smoke, args.cooldown)
    mats = stage_eval(seeds, args.smoke)
    stage_aggregate(mats, args.smoke)

    mins = (time.time() - t0) / 60
    banner(f"DONE in {mins:.1f} min")


if __name__ == "__main__":
    main()
