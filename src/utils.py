import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)

import config


def set_seed(seed: int = config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, k: int = 5) -> float:
    with torch.no_grad():
        batch = target.size(0)
        _, pred = output.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        return correct.any(dim=1).float().sum().item() / batch


@torch.no_grad()
def measure_inference_time(model, loader, device, n_batches: int = 20) -> float:
    """Returns mean inference time in ms per image. Works with both SpiceFusionNet
    (uses forward_image) and plain timm models (uses __call__)."""
    model.eval()
    forward_fn = getattr(model, "forward_image", model)
    times = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        imgs = batch[0].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = forward_fn(imgs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        times.append(elapsed / imgs.size(0))
    return float(np.mean(times))


def plot_training_curves(history: dict, output_dir: Path, prefix: str = ""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"],   label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"],   label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    axes[2].plot(epochs, history["lr"])
    axes[2].set_title("Learning Rate"); axes[2].set_xlabel("Epoch"); axes[2].set_yscale("log")

    plt.tight_layout()
    name = f"{prefix}_training_curves.png" if prefix else "training_curves.png"
    plt.savefig(output_dir / name, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, output_dir: Path, prefix: str = ""):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    for ax, data, title, fmt in [
        (axes[0], cm,      "Counts",     "d"),
        (axes[1], cm_norm, "Normalized", ".2f"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax, linewidths=0.5)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    name = f"{prefix}_confusion_matrix.png" if prefix else "confusion_matrix.png"
    plt.savefig(output_dir / name, dpi=150)
    plt.close()


def compute_and_print_metrics(y_true, y_pred, classes, top5_acc=None, infer_ms=None) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted")
    f1_m = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)

    print(f"\n{'='*50}")
    print(f"  Top-1 Accuracy   : {acc:.4f}")
    if top5_acc is not None:
        print(f"  Top-5 Accuracy   : {top5_acc:.4f}")
    print(f"  Weighted F1      : {f1_w:.4f}")
    print(f"  Macro F1         : {f1_m:.4f}")
    if infer_ms is not None:
        print(f"  Inference time   : {infer_ms:.2f} ms/image")
    print(f"{'='*50}")
    print(f"\n{report}")

    metrics = {"top1_accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_m}
    if top5_acc  is not None: metrics["top5_accuracy"]    = top5_acc
    if infer_ms  is not None: metrics["inference_ms"]     = infer_ms
    return metrics


def save_metrics(metrics: dict, path: Path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> {path}")
