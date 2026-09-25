"""Sanity tests: feature extraction + model forward shapes."""
import os, sys, traceback
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import config
from src import features, model as M
import torch

PASS, FAIL = "[PASS]", "[FAIL]"
results = []

def _run(name, fn):
    try:
        fn()
        results.append((PASS, name, ""))
        print(f"{PASS} {name}")
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        results.append((FAIL, name, msg))
        print(f"{FAIL} {name} -> {msg}")
        traceback.print_exc()

# --- Feature extraction ---
rng = np.random.default_rng(0)
img = rng.integers(0, 256, size=(config.IMG_FULL, config.IMG_FULL, 3), dtype=np.uint8)

def test_lbp():
    v = features.extract_lbp(img)
    assert v.shape == (config.LBP_P + 2,), v.shape
    assert v.dtype == np.float32

def test_glcm():
    v = features.extract_glcm(img)
    expected = 6 * len(config.GLCM_DISTANCES) * len(config.GLCM_ANGLES_DEG)
    assert v.shape == (expected,), v.shape

def test_texture():
    v = features.extract_texture(img)
    assert v.shape == (config.TEX_INPUT_DIM,), v.shape

def test_hsv():
    v = features.extract_hsv(img)
    assert v.shape == (config.COL_INPUT_DIM,), v.shape

def test_all():
    tex, col = features.extract_all(img)
    assert tex.shape == (config.TEX_INPUT_DIM,)
    assert col.shape == (config.COL_INPUT_DIM,)

# --- Model forward ---
def test_model_image_forward():
    net = M.SpiceFusionNet(pretrained=False).eval()
    x = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE)
    with torch.no_grad():
        logits = net.forward_image(x)
    assert logits.shape == (2, config.NUM_CLASSES), logits.shape

def test_model_contrastive_forward():
    net = M.SpiceFusionNet(pretrained=False).eval()
    x = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE)
    with torch.no_grad():
        proj = net.forward_contrastive(x)
    assert proj.shape == (2, config.PROJ_DIM), proj.shape
    norms = proj.norm(dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5), norms

def test_model_fusion_forward():
    net = M.SpiceFusionNet(pretrained=False).eval()
    x = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE)
    tex = torch.randn(2, config.TEX_INPUT_DIM)
    col = torch.randn(2, config.COL_INPUT_DIM)
    with torch.no_grad():
        logits, proj = net.forward_fusion(x, tex, col)
    assert logits.shape == (2, config.NUM_CLASSES), logits.shape
    assert proj.shape == (2, config.PROJ_DIM), proj.shape

def test_attention_sum_to_one():
    fusion = M.AttentionFusion(config.CNN_DIM, config.TEX_DIM, config.COL_DIM).eval()
    f_cnn = torch.randn(4, config.CNN_DIM)
    f_tex = torch.randn(4, config.TEX_DIM)
    f_col = torch.randn(4, config.COL_DIM)
    with torch.no_grad():
        cat = torch.cat([f_cnn, f_tex, f_col], dim=1)
        gates = fusion.gate(cat)
    assert gates.shape == (4, 3)
    assert torch.allclose(gates.sum(dim=1), torch.ones(4), atol=1e-5)

def test_config_dims_consistent():
    assert config.TEX_INPUT_DIM == 58, config.TEX_INPUT_DIM
    assert config.COL_INPUT_DIM == 100, config.COL_INPUT_DIM
    assert config.NUM_CLASSES == len(config.CLASSES) == 11

# Run standalone: python tests/test_sanity.py (pytest collects the test_* functions)
if __name__ == "__main__":
    for name, fn in [
        ("config dims consistent", test_config_dims_consistent),
        ("extract_lbp shape",      test_lbp),
        ("extract_glcm shape",     test_glcm),
        ("extract_texture shape",  test_texture),
        ("extract_hsv shape",      test_hsv),
        ("extract_all shapes",     test_all),
        ("AttentionFusion gates sum-to-1", test_attention_sum_to_one),
        ("model.forward_image shape",     test_model_image_forward),
        ("model.forward_contrastive L2-norm", test_model_contrastive_forward),
        ("model.forward_fusion shape",    test_model_fusion_forward),
    ]:
        _run(name, fn)

    n_pass = sum(1 for r in results if r[0] == PASS)
    n_fail = sum(1 for r in results if r[0] == FAIL)
    print(f"\n=== {n_pass} passed, {n_fail} failed ===")
    sys.exit(0 if n_fail == 0 else 1)
