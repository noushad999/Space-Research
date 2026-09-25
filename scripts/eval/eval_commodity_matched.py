#!/usr/bin/env python
"""Recompute the cross-source matrix on the commodity-matched subset.

The overlap set pairs each class across the two sources by name, but one pairing
is not a commodity pairing: the wild source's `coriander` is fresh coriander leaf
(cilantro) while the studio source's is dried coriander seed. Those are different
plant organs, so a model trained on one and tested on the other is not being tested
on an acquisition shift. This script re-derives the headline numbers with that class
excluded, from the per-class counts already stored in the evidence artifacts, and
reports both figures side by side. Arithmetic only, no model is run.

Ginger is deliberately KEPT: fresh and dried ginger are the same organ (rhizome) in
two preparations, and the transfer accuracy (95 percent) confirms the pairing holds.
"""
import json
from pathlib import Path
from statistics import mean, stdev

BASE = Path(__file__).resolve().parents[2]
SEEDS = [42, 1337, 2024]
EXCLUDE = {"coriander"}


def subset_acc(per_class, drop):
    """Weighted accuracy over the classes we keep."""
    n = sum(c["n"] for k, c in per_class.items() if k not in drop)
    hit = sum(c["n"] * c["acc"] for k, c in per_class.items() if k not in drop)
    return 100.0 * hit / n, n


def msd(v):
    return mean(v), (stdev(v) if len(v) > 1 else 0.0)


def main():
    rows = {k: {"all": [], "matched": []} for k in ("in_ss", "in_in", "ss_in", "ss_ss")}
    per_seed = []
    for s in SEEDS:
        j = json.loads((BASE / "outputs" / f"shortcut_evidence_s{s}.json").read_text())
        rec = {"seed": s}
        for tag, block in (("in", j["in"]), ("ss", j["ss"])):
            for split in ("ss", "in"):
                pc = block[split]["per_class"]
                a_all, n_all = subset_acc(pc, set())
                a_mat, n_mat = subset_acc(pc, EXCLUDE)
                rows[f"{tag}_{split}"]["all"].append(a_all)
                rows[f"{tag}_{split}"]["matched"].append(a_mat)
                rec[f"{tag}->{split}"] = (a_all, a_mat, n_all, n_mat)
        per_seed.append(rec)

    def line(key):
        a = msd(rows[key]["all"])
        m = msd(rows[key]["matched"])
        return a, m

    (in_ss_a, in_ss_m) = line("in_ss")
    (in_in_a, in_in_m) = line("in_in")
    (ss_in_a, ss_in_m) = line("ss_in")
    (ss_ss_a, ss_ss_m) = line("ss_ss")

    coll_a = in_in_a[0] - in_ss_a[0]
    coll_m = in_in_m[0] - in_ss_m[0]
    free_a = ss_ss_a[0] - ss_in_a[0]
    free_m = ss_ss_m[0] - ss_in_m[0]

    print(f"excluded classes: {sorted(EXCLUDE)}")
    print(f"test images per seed: all={per_seed[0]['in->ss'][2]}  matched={per_seed[0]['in->ss'][3]}")
    print()
    print("STUDIO-TRAINED (in)")
    print(f"  within studio : all {in_in_a[0]:.2f}+-{in_in_a[1]:.2f}   matched {in_in_m[0]:.2f}+-{in_in_m[1]:.2f}")
    print(f"  -> wild       : all {in_ss_a[0]:.2f}+-{in_ss_a[1]:.2f}   matched {in_ss_m[0]:.2f}+-{in_ss_m[1]:.2f}")
    print(f"  COLLAPSE      : all {coll_a:.2f}                matched {coll_m:.2f}")
    print()
    print("WILD-TRAINED (ss)")
    print(f"  within wild   : all {ss_ss_a[0]:.2f}+-{ss_ss_a[1]:.2f}   matched {ss_ss_m[0]:.2f}+-{ss_ss_m[1]:.2f}")
    print(f"  -> studio     : all {ss_in_a[0]:.2f}+-{ss_in_a[1]:.2f}   matched {ss_in_m[0]:.2f}+-{ss_in_m[1]:.2f}")
    print(f"  COST          : all {free_a:.2f}                matched {free_m:.2f}")
    print()
    print("per seed, studio->wild:")
    for r in per_seed:
        a, m, _, _ = r["in->ss"]
        print(f"  seed {r['seed']:>4}: all {a:.2f}   matched {m:.2f}")
    print()
    print(f"asymmetry ratio: all {coll_a/max(free_a,1e-9):.1f}x   matched {coll_m/max(free_m,1e-9):.1f}x")


if __name__ == "__main__":
    main()
