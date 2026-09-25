"""
Baseline models for comparison:
  B1 — ResNet-50     (fine-tuned, image-only)
  B2 — EfficientNet-B4 (fine-tuned, image-only)
  B3 — ViT-Base/16   (fine-tuned, image-only)
  B4 — SVM on HOG + Color Histogram (traditional ML)
"""
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import timm
import cv2
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
import joblib

import config


# ── Generic fine-tuning loop ─────────────────────────────────────────────────

def finetune(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-4,
    name: str = "baseline",
    ckpt_dir: Path = config.CHECKPOINT_DIR,
) -> float:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, tex, col, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs), labels)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_acc = evaluate_nn(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / f"{name}_best.pth")

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  [{name}] Ep {epoch:02d}/{epochs} | val acc {val_acc:.4f}")

    return best_acc


@torch.no_grad()
def evaluate_nn(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for imgs, tex, col, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


@torch.no_grad()
def predict_nn(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, tex, col, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_labels, all_preds


# ── Model factories ───────────────────────────────────────────────────────────

def make_resnet50(num_classes: int = config.NUM_CLASSES) -> nn.Module:
    m = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)
    return m


def make_efficientnet_b4(num_classes: int = config.NUM_CLASSES) -> nn.Module:
    m = timm.create_model("efficientnet_b4", pretrained=True, num_classes=num_classes)
    return m


def make_vit_base(num_classes: int = config.NUM_CLASSES) -> nn.Module:
    m = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    return m


# ── SVM baseline ─────────────────────────────────────────────────────────────

def _hog_features(img_bgr: np.ndarray) -> np.ndarray:
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray   = cv2.resize(gray, (224, 224))
    win_size = (224, 224)
    hog = cv2.HOGDescriptor(win_size, (16,16), (8,8), (8,8), 9)
    return hog.compute(gray).ravel()


def _color_hist(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [36], [0, 180]).ravel()
    s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).ravel()
    v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).ravel()
    feat = np.concatenate([h, s, v])
    return feat / (feat.sum() + 1e-8)


def build_svm_features(paths: list) -> np.ndarray:
    feats, expected_dim = [], None
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            if expected_dim is not None:
                feats.append(np.zeros(expected_dim, dtype=np.float32))
            else:
                feats.append(None)  # patch later
            continue
        img = cv2.resize(img, (224, 224))
        f = np.concatenate([_hog_features(img), _color_hist(img)]).astype(np.float32)
        if expected_dim is None:
            expected_dim = f.shape[0]
        feats.append(f)
    # Patch any None entries (failed image loads) with zeros of correct shape
    feats = [np.zeros(expected_dim, dtype=np.float32) if f is None else f for f in feats]
    return np.stack(feats)


def train_svm(
    x_train, y_train, x_val, y_val,
    ckpt_dir: Path = config.CHECKPOINT_DIR,
) -> float:
    print("  [SVM] Extracting HOG + Color features for train set...")
    t0 = time.time()
    X_tr = build_svm_features(x_train)
    X_val = build_svm_features(x_val)
    print(f"  Feature extraction: {time.time()-t0:.1f}s | shape {X_tr.shape}")

    # LinearSVC + PCA(256) — orders of magnitude faster than RBF-SVM on large feature sets
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=256, random_state=42)),
        ("svm",    LinearSVC(C=1.0, max_iter=2000)),
    ])
    print("  [SVM] Training (LinearSVC + PCA-256)...")
    pipe.fit(X_tr, y_train)

    val_acc = accuracy_score(y_val, pipe.predict(X_val))
    print(f"  [SVM] Val acc: {val_acc:.4f}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, ckpt_dir / "svm_best.pkl")
    return val_acc, pipe
