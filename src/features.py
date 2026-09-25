"""
Hand-crafted feature extraction: LBP (texture) + GLCM (texture) + HSV histogram (color).
Used by the texture branch and color branch of SpiceFusionNet.
"""
import numpy as np
import cv2

try:
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False

import config


def _require_skimage():
    if not _SKIMAGE:
        raise ImportError(
            "scikit-image is required for texture features.\n"
            "Install: pip install scikit-image"
        )


def extract_lbp(img_rgb: np.ndarray) -> np.ndarray:
    """Local Binary Pattern histogram — 10-d vector."""
    _require_skimage()
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, config.LBP_P, config.LBP_R, method="uniform")
    n_bins = config.LBP_P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def extract_glcm(img_rgb: np.ndarray) -> np.ndarray:
    """GLCM texture features — 48-d vector (6 props × 2 dist × 4 angles)."""
    _require_skimage()
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = (gray // (256 // config.GLCM_LEVELS)).astype(np.uint8)
    gray = np.clip(gray, 0, config.GLCM_LEVELS - 1)

    angles_rad = [np.deg2rad(a) for a in config.GLCM_ANGLES_DEG]
    glcm = graycomatrix(
        gray,
        distances=config.GLCM_DISTANCES,
        angles=angles_rad,
        levels=config.GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    feats = []
    for p in props:
        feats.extend(graycoprops(glcm, p).ravel())
    return np.array(feats, dtype=np.float32)


def extract_texture(img_rgb: np.ndarray) -> np.ndarray:
    """Concatenate LBP + GLCM → 58-d texture vector."""
    return np.concatenate([extract_lbp(img_rgb), extract_glcm(img_rgb)])


def extract_hsv(img_rgb: np.ndarray) -> np.ndarray:
    """HSV histogram — 100-d color vector (H=36 + S=32 + V=32)."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h = np.histogram(hsv[:, :, 0], bins=config.HSV_H_BINS, range=(0, 180),  density=True)[0]
    s = np.histogram(hsv[:, :, 1], bins=config.HSV_S_BINS, range=(0, 256),  density=True)[0]
    v = np.histogram(hsv[:, :, 2], bins=config.HSV_V_BINS, range=(0, 256),  density=True)[0]
    return np.concatenate([h, s, v]).astype(np.float32)


def extract_all(img_rgb: np.ndarray):
    """Returns (texture: 58-d, color: 100-d)."""
    return extract_texture(img_rgb), extract_hsv(img_rgb)
