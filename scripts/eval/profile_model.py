"""Compute SpiceFusionNet model profile: params, FLOPs, latency."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import json
import time
import torch
import numpy as np

import config
from src.model import SpiceFusionNet, load_checkpoint


def count_params(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def measure_latency(model, device, n_warmup=20, n_runs=100, batch_size=1, mode="fusion"):
    model.eval()
    x   = torch.randn(batch_size, 3, config.IMG_SIZE, config.IMG_SIZE, device=device)
    tex = torch.randn(batch_size, config.TEX_INPUT_DIM, device=device)
    col = torch.randn(batch_size, config.COL_INPUT_DIM, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            if mode == "fusion":
                model.forward_fusion(x, tex, col)
            else:
                model.forward_image(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            if mode == "fusion":
                model.forward_fusion(x, tex, col)
            else:
                model.forward_image(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / n_runs

    return elapsed * 1000.0  # ms per inference


def count_flops(model, mode="fusion"):
    """Use thop to count FLOPs for a single inference."""
    from thop import profile
    model.eval()
    x   = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
    tex = torch.randn(1, config.TEX_INPUT_DIM)
    col = torch.randn(1, config.COL_INPUT_DIM)

    # Wrap to handle multi-input
    class Wrapper(torch.nn.Module):
        def __init__(self, m, mode):
            super().__init__(); self.m = m; self.mode = mode
        def forward(self, x, tex=None, col=None):
            if self.mode == "fusion":
                logits, _ = self.m.forward_fusion(x, tex, col)
                return logits
            return self.m.forward_image(x)

    wrapped = Wrapper(model, mode)
    if mode == "fusion":
        macs, params = profile(wrapped, inputs=(x, tex, col), verbose=False)
    else:
        macs, params = profile(wrapped, inputs=(x,), verbose=False)
    return {"flops_g": (2 * macs) / 1e9, "params_m": params / 1e6}


def main():
    print("SpiceFusionNet — Model Profile\n" + "=" * 50)

    # Load best.pth on CPU first for param counting
    cpu = torch.device("cpu")
    ckpt = torch.load(config.CHECKPOINT_DIR / "best.pth", map_location=cpu)
    model = SpiceFusionNet()
    model.load_state_dict(ckpt["model_state"])

    # Param counts
    p = count_params(model)
    print(f"Parameters (total)     : {p['total']:,}  ({p['total']/1e6:.2f}M)")
    print(f"Parameters (trainable) : {p['trainable']:,}")

    # FLOPs
    print("\nFLOPs (single inference):")
    fl_fusion = count_flops(model, mode="fusion")
    fl_image  = count_flops(model, mode="image")
    print(f"  fusion mode : {fl_fusion['flops_g']:.2f} GFLOPs  ({fl_fusion['params_m']:.2f}M counted params)")
    print(f"  image-only  : {fl_image['flops_g']:.2f} GFLOPs   ({fl_image['params_m']:.2f}M counted params)")

    # Latency on GPU
    print("\nLatency (ms / image, batch=1):")
    if torch.cuda.is_available():
        gpu = torch.device("cuda")
        model_gpu = model.to(gpu)
        for mode in ["fusion", "image"]:
            t = measure_latency(model_gpu, gpu, mode=mode)
            print(f"  GPU {mode:7s} : {t:.2f} ms")

    # Latency on CPU
    model_cpu = model.cpu()
    for mode in ["fusion", "image"]:
        t = measure_latency(model_cpu, cpu, n_runs=20, mode=mode)
        print(f"  CPU {mode:7s} : {t:.2f} ms")

    # Latency on GPU at batch 32
    if torch.cuda.is_available():
        model_gpu = model.to(gpu)
        t = measure_latency(model_gpu, gpu, batch_size=32, mode="fusion")
        print(f"  GPU batch=32  : {t:.2f} ms ({t/32:.2f} ms/img amortized)")

    # Save profile
    out = {
        "params_total":     p["total"],
        "params_trainable": p["trainable"],
        "flops_g_fusion":   fl_fusion["flops_g"],
        "flops_g_image":    fl_image["flops_g"],
    }
    devices = {"cpu": torch.device("cpu")}
    if torch.cuda.is_available():
        devices["gpu"] = torch.device("cuda")

    for dname, dev in devices.items():
        m = model.to(dev)
        n_runs = 20 if dname == "cpu" else 100
        for mode in ["fusion", "image"]:
            ms = measure_latency(m, dev, n_runs=n_runs, mode=mode)
            out[f"latency_ms_{dname}_{mode}"] = ms

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.OUTPUT_DIR / "compute_profile.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {config.OUTPUT_DIR / 'compute_profile.json'}")


if __name__ == "__main__":
    main()
