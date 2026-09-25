#!/usr/bin/env python
"""run_arcv.py -- the ONE command for the ARC-V Regime-A experiment.

Trains {ERM, MixStyle, Fourier, ARC-V} on each source's 8-class overlap, evaluates
every checkpoint cross-source, and writes the comparison table. The falsifiable
headline (ARCV_METHOD_DESIGN.md §5): does ARC-V -- and its ablations -- beat ERM on
the HELD-OUT source (studio->wild)?

    python pipelines/run_arcv.py                 # real run (uses GPU if present); sessioned + resumable
    python pipelines/run_arcv.py --smoke         # CPU pipeline check (ERM + ARC-V, 2 batches each)

Resumable (skips finished checkpoints), thermal cooldown between runs, lock-guarded
against the anchor / diversity / benchmark runs.
"""
import argparse, atexit, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "outputs"
CKPT = OUT / "checkpoints"
SS_M = str(OUT / "manifest_overlap_ss.json")
IN_M = str(OUT / "manifest_overlap_indian.json")
LOCKS = [OUT / ".cross_source.lock", OUT / ".diversity.lock", OUT / ".benchmark.lock"]
MY_LOCK = OUT / ".arcv.lock"

METHODS = ["erm", "mixstyle", "fourier", "arcv", "rsc", "sd"]
SOURCES = [("ss", SS_M), ("indian", IN_M)]


def _alive(lock):
    try:
        return lock.exists() and (os.kill(int(lock.read_text().strip()), 0) or True)
    except Exception:
        return False


def acquire_lock():
    for L in LOCKS:
        if _alive(L):
            sys.exit(f"[LOCK] {L.name} is active (PID {L.read_text().strip()}). Wait for it.")
    if _alive(MY_LOCK):
        sys.exit(f"[LOCK] run_arcv.py already running (PID {MY_LOCK.read_text().strip()}).")
    OUT.mkdir(exist_ok=True); MY_LOCK.write_text(str(os.getpid()))
    atexit.register(lambda: MY_LOCK.exists() and MY_LOCK.unlink())


def banner(m): print("\n" + "=" * 72 + f"\n  {m}\n" + "=" * 72, flush=True)


def run(cmd):
    print(">> " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def sfx(method, source, smoke, seed):
    return f"arcv_{method}_{source}_s{seed}" + ("_smoke" if smoke else "")


def evaluate(methods, smoke, seed):
    """Cross-source eval: each method's per-source checkpoint on both test sets."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(ROOT))
    from src.dataset import SpiceDataset, get_val_transform, load_manifest_splits
    import timm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def acc_on(ckpt, manifest):
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        m = timm.create_model(ck["model_name"], pretrained=False,
                              num_classes=len(ck["classes"])).to(device)
        m.load_state_dict(ck["model_state"]); m.eval()
        splits, _ = load_manifest_splits(manifest)
        paths, labels = splits["test"]
        loader = DataLoader(SpiceDataset(paths, labels, get_val_transform(), multimodal=False),
                            batch_size=64, num_workers=2)
        yt, yp = [], []
        with torch.no_grad():
            for imgs, tex, col, y in loader:
                yp.extend(m(imgs.to(device)).argmax(1).cpu().tolist()); yt.extend(y.tolist())
        yt, yp = np.array(yt), np.array(yp)
        return round(100 * float((yt == yp).mean()), 2)

    rows = []
    for method in methods:
        ss_ck = CKPT / sfx(method, "ss", smoke, seed) / "best.pth"
        in_ck = CKPT / sfx(method, "indian", smoke, seed) / "best.pth"
        if not (ss_ck.exists() and in_ck.exists()):
            continue
        studio_within = acc_on(in_ck, IN_M)      # Indian-trained on Indian-test
        studio_to_wild = acc_on(in_ck, SS_M)     # Indian-trained on SS-test (held-out)
        wild_within = acc_on(ss_ck, SS_M)
        wild_to_studio = acc_on(ss_ck, IN_M)
        rows.append([method, studio_within, studio_to_wild,
                     round(studio_within - studio_to_wild, 2), wild_within, wild_to_studio])
        print(f"  {method:9s} | studio->wild {studio_to_wild:.1f} "
              f"(collapse {studio_within - studio_to_wild:.1f} pp)", flush=True)
    return rows


def write_table(rows, seed, smoke=False):
    import json
    rows.sort(key=lambda r: r[3])   # by studio->wild collapse ascending (most robust first)
    tag = f"s{seed}" + ("_smoke" if smoke else "")   # smoke never pollutes real per-seed files
    md = [f"# ARC-V Regime A (seed {seed}{' SMOKE' if smoke else ''}) -- single-source -> held-out source\n",
          "Headline metric: studio(Indian)-trained -> wild(SS)-test. Smaller collapse = more robust.\n",
          "| Method | Studio within | Studio->Wild | Collapse (pp) | Wild within | Wild->Studio |",
          "|---|---|---|---|---|---|"]
    for name, sw, stw, tax, ww, wts in rows:
        md.append(f"| {name} | {sw} | {stw} | {tax:.2f} | {ww} | {wts} |")
    text = "\n".join(md)
    (OUT / f"arcv_regime_a_{tag}.md").write_text(text, encoding="utf-8")
    json.dump({"seed": seed, "rows": rows}, open(OUT / f"arcv_regime_a_{tag}.json", "w"), indent=2)
    print("\n" + text)
    d = {r[0]: r for r in rows}
    if "erm" in d and "arcv" in d:
        erm_hw, arcv_hw = d["erm"][2], d["arcv"][2]     # studio->wild held-out acc
        verdict = "ARC-V BEATS ERM" if arcv_hw > erm_hw else "ARC-V does NOT beat ERM -> honest null"
        print(f"\n  VERDICT (studio->wild held-out): ERM {erm_hw}  vs  ARC-V {arcv_hw}  ->  {verdict}")
        print("  (multi-seed + paired test still needed before this is paper-final.)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cooldown", type=int, default=180)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    acquire_lock()
    # Smoke exercises the new single-source baselines (RSC, SD) so the pipeline is
    # verified end to end before any GPU run.
    methods = ["erm", "rsc", "sd"] if args.smoke else args.methods
    t0 = time.time()

    banner(f"TRAIN -- {len(methods)} methods x 2 sources" + (" [SMOKE]" if args.smoke else ""))
    runs = [(m, st, mf) for m in methods for st, mf in SOURCES]
    for i, (method, st, mf) in enumerate(runs):
        ck = CKPT / sfx(method, st, args.smoke, args.seed) / "best.pth"
        if ck.exists() and not args.smoke:
            print(f"  [skip] {sfx(method, st, args.smoke, args.seed)}"); continue
        cmd = [PY, "scripts/train/train_arcv.py", "--method", method, "--manifest", mf,
               "--suffix", sfx(method, st, args.smoke, args.seed), "--seed", str(args.seed),
               "--epochs", str(args.epochs)]
        if args.smoke:
            cmd.append("--smoke")
        if run(cmd) != 0:
            print(f"  [warn] training failed for {method} {st}"); continue
        if args.cooldown and i < len(runs) - 1 and not args.smoke:
            print(f"  cooling {args.cooldown}s ..."); time.sleep(args.cooldown)

    banner("EVAL -- cross-source per method")
    rows = evaluate(methods, args.smoke, args.seed)
    if rows:
        write_table(rows, args.seed, args.smoke)
    else:
        print("  no complete method checkpoints yet.")
    print(f"\n  total wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
