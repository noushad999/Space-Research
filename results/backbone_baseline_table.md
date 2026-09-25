# Within-source robustness tables (assembled)

## Architecture comparison (within-source, full Spice_Spectrum test)

| Model | Top-1 | Top-5 | Macro-F1 |
|---|---|---|---|
| SVM (HOG + Color) | 32.49 | -- | 30.69 |
| ResNet-50 (fine-tuned) | 99.36 | 99.95 | 99.36 |
| EfficientNet-B4 (image-only) | 99.59 | 100.00 | 99.59 |
| ViT-Base/16 (fine-tuned) | 99.73 | 99.95 | 99.73 |
| SpiceFusionNet (ours) | 99.68 | 100.00 | 99.68 |

## Augmentation ablation (within-source)

| Augmentation | Top-1 | Macro-F1 |
|---|---|---|
| a3_no_aug | 98.55 | 98.54 |
| a3_std_aug | 99.05 | 99.05 |
| a3_spice_aug | 98.91 | 98.91 |

> Note: these are WITHIN-source metrics (they justify the model choice and show the task is non-trivial). The cross-source collapse is reported separately in the shortcut matrix — augmentation does not close it.