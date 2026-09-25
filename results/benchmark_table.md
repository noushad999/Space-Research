# Cross-source benchmark (Indian/studio-trained -> SS/wild test)

| Model | Studio within | Studio->Wild | Collapse (pp) | Macro-F1 (wild) |
|---|---|---|---|---|
| convnext_tiny | 100.0 | 67.7 | 32.4 | 64.7 |
| SpiceFusionNet (ours) | 100.0 | 62.2 | 37.8 | -- |
| densenet121 | 100.0 | 60.4 | 39.6 | 56.4 |
| resnet50 | 99.8 | 57.5 | 42.4 | 55.0 |
| efficientnet_b4 | 100.0 | 46.1 | 53.9 | 43.8 |
| efficientnet_b0 | 100.0 | 45.9 | 54.1 | 41.6 |
| mobilenetv3_large_100 | 100.0 | 37.9 | 62.1 | 36.1 |
| GranuFormer | 100.0 | 14.1 | 85.9 | -- |