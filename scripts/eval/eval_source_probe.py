"""
eval_source_probe.py — Mechanistic source analysis (M1).

Three complementary probes on a trained SpiceFusionNet checkpoint, over the
overlap-class subset of the test split (per-class where relevant, so the
class->source confound is controlled; binary-source chance = 0.50):

  (1) DECODABILITY  — can a linear probe predict SOURCE from each feature stream?
                      (is source information present?)

  (2) RELIANCE (INLP) — remove linearly source-predictive directions, then measure
                      CLASS-accuracy retention. (does a both-source probe need
                      source directions?) NOTE: M1 found this does NOT separate
                      robust from collapsing models — it measures geometry, not
                      the deployed head's rule. Kept for completeness.

  (3) TRANSFER      — train a class probe on ONE source's features, test on the
                      OTHER. Denies the probe target-source labels, exposing
                      whether class-discriminative directions are SHARED across
                      sources or source-specific. This is the decisive probe for
                      localizing the cross-source collapse to features vs head.

Feature extraction is cached under outputs/_probe_cache/ (keyed by ckpt+manifest+
split+max_per_class) so repeated analyses don't re-run the GPU.

No training. Usage:
    python scripts/eval/eval_source_probe.py
    python scripts/eval/eval_source_probe.py --ckpt outputs/checkpoints/overlap_indian/p1_best.pth
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
import argparse
import json
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score

import config
from src.model import SpiceFusionNet
from src.features import extract_all
from src.dataset import load_manifest_splits, get_val_transform

STREAMS = ["f_cnn", "f_tex", "f_col", "fused"]


# ── source recovery ──────────────────────────────────────────────────────────

def source_of(path: str) -> str:
    p = path.lower().replace("\\", "/")
    if "indian_spices" in p:
        return "indian"            # studio (white background)
    if "spice_spectrum" in p:
        return "spice_spectrum"    # in-the-wild
    return "unknown"


# ── checkpoint loading ───────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device, arch: str = "spicefusion"):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt.get("model", ckpt))
    n_cls = state["fusion_head.4.weight"].shape[0] if "fusion_head.4.weight" in state else (
        state.get("img_head.4.weight", torch.empty(config.NUM_CLASSES, 1)).shape[0])
    if arch == "aifnet":
        raise NotImplementedError("AIFNet is not part of this release; its numbers in the paper are reported from saved results (see results/).")
    else:
        model = SpiceFusionNet(num_classes=n_cls, pretrained=False)
    missing, _ = model.load_state_dict(state, strict=False)
    feat_keys = [k for k in missing if k.startswith(("backbone", "tex_branch", "col_branch", "fusion", "freq_branch"))]
    if feat_keys:
        print(f"  [warn] {len(feat_keys)} feature-module weights missing from "
              f"{Path(ckpt_path).name}: e.g. {feat_keys[:2]}")
    model.to(device).eval()
    return model


# ── feature extraction (+ cache) ─────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, samples, idx2name, device, batch_size, transform, arch="spicefusion"):
    streams = defaultdict(list)
    sources, class_names = [], []
    img_buf, tex_buf, col_buf, meta_buf = [], [], [], []

    def flush():
        if not img_buf:
            return
        x = torch.stack(img_buf).to(device)
        tex = torch.from_numpy(np.stack(tex_buf)).float().to(device)
        col = torch.from_numpy(np.stack(col_buf)).float().to(device)
        f_cnn = model.backbone(x)
        f_tex = model.tex_branch(tex)
        f_col = model.col_branch(col)
        if arch == "aifnet":
            sd = OrderedDict(f_cnn=f_cnn, f_tex=f_tex, f_col=f_col)
            if getattr(model, "use_freq", False):
                sd["f_freq"] = model.freq_branch(x)
            fused = model.fusion(sd)[0]
        else:
            fused = model.fusion(f_cnn, f_tex, f_col)
        streams["f_cnn"].append(f_cnn.cpu().numpy())
        streams["f_tex"].append(f_tex.cpu().numpy())
        streams["f_col"].append(f_col.cpu().numpy())
        streams["fused"].append(fused.cpu().numpy())
        for src, cname in meta_buf:
            sources.append(src); class_names.append(cname)
        img_buf.clear(); tex_buf.clear(); col_buf.clear(); meta_buf.clear()

    for path, label in samples:
        try:
            img_np = np.array(Image.open(path).convert("RGB"))
        except Exception as e:
            print(f"  [skip] {path}: {e}")
            continue
        tex_np, col_np = extract_all(img_np)
        img_buf.append(transform(image=img_np)["image"])
        tex_buf.append(tex_np); col_buf.append(col_np)
        meta_buf.append((source_of(path), idx2name[label]))
        if len(img_buf) >= batch_size:
            flush()
    flush()
    feats = {k: np.concatenate(v, axis=0) for k, v in streams.items()}
    return feats, np.array(sources), np.array(class_names)


def cache_path(args):
    if args.no_cache:
        return None
    p = Path(args.ckpt)
    key = f"{p.parent.name}_{p.stem}__{Path(args.manifest).stem}__{args.split}__mpc{args.max_per_class}"
    return Path("outputs/_probe_cache") / f"{key}.npz"


def save_cache(path, feats, sources, class_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, sources=sources, class_names=class_names,
             **{f"feat_{k}": v for k, v in feats.items()})


def load_cache(path):
    z = np.load(path, allow_pickle=False)
    feats = {k[5:]: z[k] for k in z.files if k.startswith("feat_")}
    return feats, z["sources"], z["class_names"]


# ── (1) decodability ─────────────────────────────────────────────────────────

def cv_balanced_acc(X, y, folds, seed=42):
    counts = np.bincount(y)
    k = min(folds, int(counts.min()))
    if k < 2 or len(np.unique(y)) < 2:
        return None, k
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return cross_val_score(clf, X, y, cv=skf, scoring="balanced_accuracy"), k


def per_class_source_probe(feats, sources, class_names, folds, min_per_source):
    classes = sorted(set(class_names))
    overlap = []
    for c in classes:
        m = class_names == c
        n_ind = int((sources[m] == "indian").sum())
        n_ss = int((sources[m] == "spice_spectrum").sum())
        if n_ind >= min_per_source and n_ss >= min_per_source:
            overlap.append((c, n_ind, n_ss))
    results = {stream: {} for stream in feats}
    for stream, X_all in feats.items():
        for c, n_ind, n_ss in overlap:
            m = class_names == c
            scores, k = cv_balanced_acc(X_all[m], (sources[m] == "indian").astype(int), folds)
            if scores is not None:
                results[stream][c] = {"balanced_acc_mean": float(scores.mean()),
                                      "balanced_acc_std": float(scores.std()),
                                      "n_indian": n_ind, "n_ss": n_ss, "folds": k}
    return results, [c for c, _, _ in overlap]


def class_probe(feats, class_names, folds):
    out = {}
    le = {c: i for i, c in enumerate(sorted(set(class_names)))}
    y = np.array([le[c] for c in class_names])
    for stream, X in feats.items():
        scores, _ = cv_balanced_acc(X, y, folds)
        out[stream] = float(scores.mean()) if scores is not None else None
    return out


# ── (2) reliance (INLP) ──────────────────────────────────────────────────────

def _within_class_center(X, y_class):
    Xc = X.copy()
    for c in np.unique(y_class):
        m = y_class == c
        Xc[m] = Xc[m] - Xc[m].mean(axis=0, keepdims=True)
    return Xc


def _nullspace_proj(w):
    w = np.atleast_2d(w)
    _, s, vt = np.linalg.svd(w, full_matrices=False)
    basis = vt[s > 1e-10]
    return np.eye(w.shape[1]) - basis.T @ basis


def _src_decodability(X, y_class, y_src, folds=3, seed=42):
    s, _ = cv_balanced_acc(_within_class_center(X, y_class), y_src, folds, seed)
    return float(s.mean()) if s is not None else 0.5


def inlp_reliance(feats, sources, class_names, max_iter=15, src_floor=0.55):
    le = {c: i for i, c in enumerate(sorted(set(class_names)))}
    y_class = np.array([le[c] for c in class_names])
    y_src = (sources == "indian").astype(int)
    out = {}
    for stream, X0 in feats.items():
        X = StandardScaler().fit_transform(X0)
        base_cls, _ = cv_balanced_acc(X, y_class, 5)
        base_src = _src_decodability(X, y_class, y_src)
        Xp, removed = X.copy(), 0
        for _ in range(max_iter):
            if _src_decodability(Xp, y_class, y_src) <= src_floor:
                break
            w = LogisticRegression(max_iter=2000).fit(_within_class_center(Xp, y_class), y_src).coef_
            Xp = Xp @ _nullspace_proj(w)
            removed += 1
        after_cls, _ = cv_balanced_acc(Xp, y_class, 5)
        bc, ac = float(base_cls.mean()), float(after_cls.mean())
        out[stream] = {"class_acc_before": bc, "class_acc_after_source_removal": ac,
                       "class_acc_retention": (ac / bc) if bc > 0 else float("nan"),
                       "source_acc_before": base_src,
                       "source_acc_after": _src_decodability(Xp, y_class, y_src),
                       "directions_removed": removed}
    return out


# ── (3) cross-source transfer ────────────────────────────────────────────────

def cross_source_transfer(feats, sources, class_names, folds=5):
    """Train a class probe on one source, test on the other. Localizes the
    cross-source collapse to features (low transfer) vs head (high transfer)."""
    le = {c: i for i, c in enumerate(sorted(set(class_names)))}
    y = np.array([le[c] for c in class_names])
    ind = sources == "indian"
    ss = sources == "spice_spectrum"
    out = {}
    for stream, X in feats.items():
        res = {}
        for name, tr, te in [("indian->ss", ind, ss), ("ss->indian", ss, ind)]:
            common = sorted(set(y[tr]) & set(y[te]))
            mtr = np.isin(y[tr], common); mte = np.isin(y[te], common)
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(X[tr][mtr], y[tr][mtr])
            res[name] = float(balanced_accuracy_score(y[te][mte], clf.predict(X[te][mte])))
        s_ind, _ = cv_balanced_acc(X[ind], y[ind], folds)
        s_ss, _ = cv_balanced_acc(X[ss], y[ss], folds)
        res["indian_within"] = float(s_ind.mean()) if s_ind is not None else None
        res["ss_within"] = float(s_ss.mean()) if s_ss is not None else None
        res["transfer_gap_ind_to_ss"] = (res["ss_within"] - res["indian->ss"]
                                         if res["ss_within"] is not None else None)
        res["transfer_gap_ss_to_ind"] = (res["indian_within"] - res["ss->indian"]
                                         if res["indian_within"] is not None else None)
        out[stream] = res
    return out


# ── plotting ─────────────────────────────────────────────────────────────────

LABELS = {"f_cnn": "CNN\n(contextual)", "f_tex": "Texture", "f_col": "Color",
          "fused": "Fused\n(decision)"}


def plot_decodability(agg, class_acc, out_png):
    src = [agg[s]["mean"] for s in STREAMS]; err = [agg[s]["std"] for s in STREAMS]
    cls = [class_acc[s] for s in STREAMS]
    x = np.arange(len(STREAMS)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, src, w, yerr=err, capsize=4, label="Source-probe", color="#d1495b")
    b2 = ax.bar(x + w/2, cls, w, label="Class-probe", color="#3b7a8c")
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[s] for s in STREAMS])
    ax.set_ylabel("Balanced accuracy (5-fold CV)"); ax.set_ylim(0, 1.08)
    ax.set_title("(1) Source DECODABILITY by stream"); ax.legend(loc="lower center", fontsize=9)
    for b in list(b1) + list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{b.get_height():.2f}",
                ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); print(f"  figure -> {out_png}")


def plot_transfer(transfer, out_png):
    keys = [("indian_within", "Indian within", "#3b7a8c"),
            ("ss_within", "SS within", "#5b8c5a"),
            ("indian->ss", "Indian->SS", "#edae49"),
            ("ss->indian", "SS->Indian", "#d1495b")]
    x = np.arange(len(STREAMS)); w = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (k, lab, col) in enumerate(keys):
        vals = [transfer[s][k] if transfer[s][k] is not None else 0 for s in STREAMS]
        ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[s] for s in STREAMS])
    ax.set_ylabel("Class balanced accuracy"); ax.set_ylim(0, 1.08)
    ax.set_title("(3) Cross-source TRANSFER (within vs train-A/test-B)")
    ax.legend(loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); print(f"  figure -> {out_png}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/unified/best.pth")
    ap.add_argument("--manifest", default="outputs/unified_benchmark.json")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--max-per-class", type=int, default=250)
    ap.add_argument("--min-per-source", type=int, default=15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="outputs/source_probe.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-reliance", action="store_true")
    ap.add_argument("--no-transfer", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--arch", default="spicefusion", choices=["spicefusion", "aifnet"])
    args = ap.parse_args()

    if not Path(args.ckpt).exists():
        d = Path(args.ckpt).parent
        avail = sorted(p.name for p in d.glob("*.pth")) if d.exists() else []
        raise SystemExit(f"checkpoint not found: {args.ckpt}\n  available in {d}: {avail}")

    device = torch.device(args.device)
    print(f"device={device}  ckpt={args.ckpt}")
    splits, classes = load_manifest_splits(args.manifest)
    idx2name = {i: name for i, name in enumerate(classes)}
    paths, labels = splits[args.split]

    by_class = defaultdict(list)
    for p, y in zip(paths, labels):
        by_class[y].append((p, y))
    rng = np.random.default_rng(42)
    samples = []
    for y, items in by_class.items():
        if not ({"indian", "spice_spectrum"} <= {source_of(p) for p, _ in items}):
            continue
        if len(items) > args.max_per_class:
            items = [items[i] for i in rng.choice(len(items), args.max_per_class, replace=False)]
        samples.extend(items)
    print(f"overlap-class samples: {len(samples)} across {len({y for _, y in samples})} classes")

    cp = cache_path(args)
    if cp is not None and cp.exists():
        print(f"loading cached features <- {cp}")
        feats, sources, class_names = load_cache(cp)
    else:
        model = load_model(args.ckpt, device, args.arch)
        print("extracting features ...")
        feats, sources, class_names = extract_features(
            model, samples, idx2name, device, args.batch, get_val_transform(), args.arch)
        if cp is not None:
            save_cache(cp, feats, sources, class_names)
            print(f"cached features -> {cp}")
    print("  " + ", ".join(f"{k}{v.shape}" for k, v in feats.items()))
    print(f"  source counts: indian={int((sources=='indian').sum())} "
          f"spice_spectrum={int((sources=='spice_spectrum').sum())}")

    print("(1) decodability ...")
    results, overlap_classes = per_class_source_probe(
        feats, sources, class_names, args.folds, args.min_per_source)
    class_acc = class_probe(feats, class_names, args.folds)
    agg = {}
    for stream in feats:
        vals = [d["balanced_acc_mean"] for d in results[stream].values()]
        agg[stream] = {"mean": float(np.mean(vals)) if vals else float("nan"),
                       "std": float(np.std(vals)) if vals else float("nan"),
                       "n_classes": len(vals)}
    reliance = None if args.no_reliance else (print("(2) reliance (INLP) ...") or
                                              inlp_reliance(feats, sources, class_names))
    transfer = None if args.no_transfer else (print("(3) cross-source transfer ...") or
                                              cross_source_transfer(feats, sources, class_names, args.folds))

    name = Path(args.ckpt).name
    print("\n" + "=" * 70)
    print(f"(1) SOURCE DECODABILITY (chance 0.50) -- {name} [{Path(args.ckpt).parent.name}]")
    print("-" * 70)
    print(f"{'stream':<8} {'source-probe':>16} {'class-probe':>14}")
    for s in STREAMS:
        cp_s = f"{class_acc[s]:.3f}" if class_acc[s] is not None else "n/a"
        print(f"{s:<8} {agg[s]['mean']:.3f} +/- {agg[s]['std']:.3f}  {cp_s:>14}")
    if transfer is not None:
        print("\n(3) CROSS-SOURCE TRANSFER -- class probe trained on A, tested on B")
        print("-" * 70)
        print(f"{'stream':<8} {'ind_within':>10} {'ss_within':>10} {'ind->ss':>9} {'ss->ind':>9}")
        def _f(v):
            return f"{v:.3f}" if v is not None else "n/a"
        for s in STREAMS:
            t = transfer[s]
            print(f"{s:<8} {_f(t['indian_within']):>10} {_f(t['ss_within']):>10} "
                  f"{_f(t['indian->ss']):>9} {_f(t['ss->indian']):>9}")
    print("=" * 70)

    out = {"checkpoint": args.ckpt, "manifest": args.manifest, "split": args.split,
           "overlap_classes": overlap_classes, "n_samples": len(class_names),
           "aggregate_source_probe": agg, "class_probe": class_acc,
           "reliance_inlp": reliance, "transfer": transfer, "per_class": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  results -> {args.out}")
    plot_decodability(agg, class_acc, args.out.replace(".json", ".png"))
    if transfer is not None:
        plot_transfer(transfer, args.out.replace(".json", "_transfer.png"))


if __name__ == "__main__":
    main()
