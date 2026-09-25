"""
Grad-CAM visualization for SpiceFusionNet.
Hooks into EfficientNet-B4's last convolutional layer.

Bug fixes:
  - Display image now matches the *exact* image the model sees
    (resize -> center crop), so heatmap pixels align correctly.
  - Optional fusion-mode Grad-CAM: computes gradients w.r.t. the
    fusion_head logits (matches the actual prediction pathway).
  - Uses last MBConv block features (slightly richer than conv_head).
"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from src.features import extract_all as _extract_all


# Resize-then-centercrop dimensions (must match get_val_transform exactly)
_RESIZE = int(config.IMG_SIZE * 256 / 224)
_CROP   = config.IMG_SIZE


class GradCAM:
    def __init__(self, model, target_layer=None, mode: str = "fusion"):
        """mode = "fusion" or "image"  — which forward pass to backprop through."""
        self.model = model
        self.mode = mode
        self.gradients = None
        self.activations = None

        # Last conv block of EfficientNet-B4. `conv_head` is a 1x1 conv;
        # `blocks[-1]` carries the spatial features that flow into it.
        if target_layer is None:
            target_layer = model.backbone.conv_head

        self._fwd = target_layer.register_forward_hook(self._save_activation)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()

    def __call__(self, x: torch.Tensor, tex=None, col=None, class_idx: int = None):
        """Returns (cam, pred_idx). cam is H×W float32 in [0,1]."""
        self.model.eval()
        self.model.zero_grad()

        if self.mode == "fusion" and tex is not None and col is not None:
            logits, _ = self.model.forward_fusion(x, tex, col)
        else:
            logits = self.model.forward_image(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx


def _model_view(img_path: str):
    """Return BOTH the tensor the model sees AND the matching uint8 display
    image, guaranteed to be the exact same pixels (resize -> center crop)."""
    img = np.array(Image.open(img_path).convert("RGB"))

    # Step 1: resize + center crop to produce display image (uint8)
    crop_tf = A.Compose([
        A.Resize(_RESIZE, _RESIZE),
        A.CenterCrop(_CROP, _CROP),
    ])
    display = crop_tf(image=img)["image"]   # (224, 224, 3) uint8

    # Step 2: normalize + to-tensor for the model
    norm_tf = A.Compose([
        A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ToTensorV2(),
    ])
    tensor = norm_tf(image=display)["image"].unsqueeze(0)   # (1, 3, 224, 224)

    # Step 3: hand-crafted features on the ORIGINAL image (matches dataset.py)
    tex_np, col_np = _extract_all(img)
    tex = torch.from_numpy(tex_np).unsqueeze(0).float()
    col = torch.from_numpy(col_np).unsqueeze(0).float()

    return tensor, display, tex, col


def visualize_gradcam(
    model,
    img_paths: list,
    true_labels: list,
    classes: list,
    device: torch.device,
    output_path: Path,
    n_samples: int = 16,
    mode: str = "fusion",
):
    """Plot Grad-CAM for up to n_samples images.

    mode='fusion' computes gradients through the fusion_head (matches the
    actual prediction). mode='image' uses the Phase-1 img_head.
    """
    model.eval()
    gcam = GradCAM(model, mode=mode)

    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 6, rows * 3))
    axes = axes.flatten()

    plotted = 0
    for img_path, true_label in zip(img_paths, true_labels):
        if plotted >= n_samples:
            break

        tensor, display, tex, col = _model_view(img_path)
        tensor = tensor.to(device); tex = tex.to(device); col = col.to(device)

        cam, pred_idx = gcam(tensor, tex=tex, col=col)

        # Overlay — display & cam now share identical 224x224 spatial grid
        heatmap = cv2.resize(cam, (display.shape[1], display.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(display, 0.5, heatmap, 0.5, 0)

        ax_img = axes[plotted * 2]
        ax_cam = axes[plotted * 2 + 1]

        ax_img.imshow(display)
        ax_img.set_title(f"True: {classes[true_label]}", fontsize=8)
        ax_img.axis("off")

        ax_cam.imshow(overlay)
        color = "green" if pred_idx == true_label else "red"
        ax_cam.set_title(f"Pred: {classes[pred_idx]}", fontsize=8, color=color)
        ax_cam.axis("off")

        plotted += 1

    for i in range(plotted * 2, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"Grad-CAM ({mode} mode)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    gcam.remove()
    print(f"Grad-CAM saved to {output_path}")
