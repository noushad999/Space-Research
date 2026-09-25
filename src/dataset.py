import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

import config

try:
    from src.features import extract_all as _extract_all
    _FEATURES_OK = True
except Exception:
    _FEATURES_OK = False


def source_of(path: str) -> str:
    """Recover the acquisition source (domain) from a sample path."""
    p = str(path).lower().replace("\\", "/")
    if "indian_spices" in p:
        return "indian"            # studio (white background)
    if "spice_spectrum" in p:
        return "spice_spectrum"    # in-the-wild
    return "unknown"


# ── Transforms ───────────────────────────────────────────────────────────────

def get_train_transform(img_size: int = config.IMG_SIZE) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=config.AUG_SCALE, ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=config.AUG_ROTATION, p=0.6),
        A.ColorJitter(
            brightness=config.AUG_BRIGHTNESS,
            contrast=config.AUG_CONTRAST,
            saturation=config.AUG_SATURATION,
            hue=config.AUG_HUE,
            p=0.7,
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
        ], p=0.3),
        A.GaussNoise(p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=0,
            p=0.3,
        ),
        A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ToTensorV2(),
    ])


def get_train_transform_strong(img_size: int = config.IMG_SIZE) -> A.Compose:
    """Anti-shortcut augmentation pipeline.

    Built from peer-reviewed building blocks:
      * Random Erasing with random-uniform & inpaint fills (Zhong et al. 2017,
        AAAI 2020) — denies the model fixed background context.
      * Heavier ColorJitter + Hue shift — denies stable color shortcuts.
      * RandomResizedCrop + Affine — denies stable position shortcuts.
      * Stronger blur/noise mixture — denies texture shortcuts.

    Returns a Compose that should be drop-in compatible with the default
    train transform (same input dtype, same output tensor shape).
    """
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Affine(rotate=(-config.AUG_ROTATION, config.AUG_ROTATION),
                 translate_percent=(0.0, 0.08), scale=(0.9, 1.1), p=0.5),
        A.ColorJitter(
            brightness=0.35, contrast=0.35,
            saturation=0.35, hue=0.12, p=0.75,
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
        ], p=0.30),
        A.GaussNoise(std_range=(0.03, 0.10), p=0.20),
        # Anti-shortcut erasure: 1-3 medium holes per image with varied fills.
        # Sized so total occluded area stays <~20% of the image, leaving the
        # spice visible while breaking any fixed-background dependency.
        A.OneOf([
            A.CoarseDropout(num_holes_range=(1, 3),
                            hole_height_range=(20, 48),
                            hole_width_range=(20, 48),
                            fill="random_uniform", p=1.0),
            A.CoarseDropout(num_holes_range=(1, 3),
                            hole_height_range=(20, 48),
                            hole_width_range=(20, 48),
                            fill="random", p=1.0),
            A.CoarseDropout(num_holes_range=(1, 2),
                            hole_height_range=(30, 60),
                            hole_width_range=(30, 60),
                            fill=0, p=1.0),
        ], p=0.55),
        A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ToTensorV2(),
    ])


def get_val_transform(img_size: int = config.IMG_SIZE) -> A.Compose:
    resize = int(img_size * 256 / 224)
    return A.Compose([
        A.Resize(resize, resize),
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ToTensorV2(),
    ])


# ── Splits ────────────────────────────────────────────────────────────────────

def _collect(data_dir: Path, classes: List[str]):
    paths, labels = [], []
    for idx, cls in enumerate(classes):
        for fp in sorted((data_dir / cls).iterdir()):
            if fp.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(str(fp))
                labels.append(idx)
    return paths, labels


def build_splits(data_dir: Path = config.DATA_DIR):
    paths, labels = _collect(data_dir, config.CLASSES)
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(
        paths, labels, test_size=1 - config.TRAIN_RATIO,
        stratify=labels, random_state=config.RANDOM_SEED,
    )
    val_frac = config.VAL_RATIO / (1 - config.TRAIN_RATIO)
    x_val, x_te, y_val, y_te = train_test_split(
        x_tmp, y_tmp, test_size=1 - val_frac,
        stratify=y_tmp, random_state=config.RANDOM_SEED,
    )
    return x_tr, y_tr, x_val, y_val, x_te, y_te


_SOURCE_DIRS = ("Spice_Spectrum", "Indian_Spices")


def _translate_path(p: str) -> str:
    """Resolve a manifest sample path to a file under config.DATA_ROOT.

    Released manifests store paths relative to the data root
    ("Spice_Spectrum/cumin/x.jpg"). Older manifests hold absolute paths from the
    machine that built them; those are re-rooted onto DATA_ROOT the same way.
    """
    q = str(p).replace("\\", "/")
    for d in _SOURCE_DIRS:
        i = q.find(d + "/")
        if i >= 0:
            return str(config.DATA_ROOT / q[i:])
    return str(p)


resolve_image_path = _translate_path


def load_manifest_splits(manifest_path):
    """Load (paths, labels) per split from a unified-benchmark manifest JSON.

    Sample paths are resolved onto config.DATA_ROOT (see _translate_path).
    """
    import json
    with open(manifest_path) as f:
        m = json.load(f)
    out = {}
    for split in ("train", "val", "test"):
        paths = [_translate_path(p) for p, _ in m["samples"][split]]
        labels = [int(y) for _, y in m["samples"][split]]
        out[split] = (paths, labels)
    classes = [c["name"] for c in sorted(m["classes"], key=lambda c: c["index"])]
    return out, classes


# ── Dataset ───────────────────────────────────────────────────────────────────

class SpiceDataset(Dataset):
    """Unified dataset — returns (image, texture, color, label).
    texture/color are zero tensors if multimodal=False or features unavailable.
    """
    def __init__(
        self,
        paths: List[str],
        labels: List[int],
        transform: Optional[A.Compose] = None,
        multimodal: bool = False,
        return_source: bool = False,
        source_map: Optional[dict] = None,
    ):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.multimodal = multimodal and _FEATURES_OK
        self.return_source = return_source
        self.source_map = source_map or {}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_np = np.array(Image.open(self.paths[idx]).convert("RGB"))

        # Extract hand-crafted features BEFORE augmentation (on original image)
        if self.multimodal:
            tex_np, col_np = _extract_all(img_np)
            tex = torch.from_numpy(tex_np)
            col = torch.from_numpy(col_np)
        else:
            tex = torch.zeros(config.TEX_INPUT_DIM)
            col = torch.zeros(config.COL_INPUT_DIM)

        if self.transform:
            img_np = self.transform(image=img_np)["image"]

        if self.return_source:
            src = self.source_map.get(source_of(self.paths[idx]), -1)
            return img_np, tex, col, self.labels[idx], src
        return img_np, tex, col, self.labels[idx]


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_dataloaders(
    multimodal: bool = False,
    data_dir: Path = config.DATA_DIR,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    strong_aug: bool = False,
    manifest_path: Optional[str] = None,
):
    """Build train/val/test dataloaders.

    If `manifest_path` is given, splits come from the unified-benchmark manifest;
    otherwise the legacy `build_splits` on `data_dir` is used.
    """
    if manifest_path is not None:
        splits, _ = load_manifest_splits(manifest_path)
        x_tr, y_tr = splits["train"]
        x_val, y_val = splits["val"]
        x_te,  y_te  = splits["test"]
    else:
        x_tr, y_tr, x_val, y_val, x_te, y_te = build_splits(data_dir)

    train_tf = get_train_transform_strong() if strong_aug else get_train_transform()
    tr_ds  = SpiceDataset(x_tr,  y_tr,  train_tf, multimodal)
    val_ds = SpiceDataset(x_val, y_val, get_val_transform(), multimodal)
    te_ds  = SpiceDataset(x_te,  y_te,  get_val_transform(), multimodal)

    mk = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(tr_ds,  shuffle=True,  drop_last=True, **mk),
        DataLoader(val_ds, shuffle=False, **mk),
        DataLoader(te_ds,  shuffle=False, **mk),
        x_te, y_te,
    )


def get_aifnet_dataloaders(
    manifest_path: str,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    strong_aug: bool = True,
):
    """Source-aware multimodal loaders for AIFNet.

    Each batch is (image, texture, color, label, source_id). The source_map maps
    acquisition-source name -> integer id (built deterministically from all paths
    in the manifest, so train/val/test share the same id space).
    """
    splits, classes = load_manifest_splits(manifest_path)
    all_paths = splits["train"][0] + splits["val"][0] + splits["test"][0]
    source_map = {s: i for i, s in enumerate(sorted({source_of(p) for p in all_paths}))}

    train_tf = get_train_transform_strong() if strong_aug else get_train_transform()

    def _ds(split, tf):
        p, y = splits[split]
        return SpiceDataset(p, y, tf, multimodal=True, return_source=True, source_map=source_map)

    mk = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(_ds("train", train_tf),          shuffle=True,  drop_last=True, **mk),
        DataLoader(_ds("val",   get_val_transform()), shuffle=False, **mk),
        DataLoader(_ds("test",  get_val_transform()), shuffle=False, **mk),
        classes, source_map,
    )
