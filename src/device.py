"""
device.py — OTTER device selection utility

Single source of truth for hardware backend selection across the pipeline.
Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU.
"""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """
    Return the best available PyTorch device.

    Priority
    --------
    1. MPS  — Apple Silicon GPU (M1/M2/M3/M4/M5)
    2. CUDA — NVIDIA GPU
    3. CPU  — universal fallback

    Returns
    -------
    torch.device
    """
    if torch.backends.mps.is_available():
        print("Device: Apple MPS (Metal Performance Shaders)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Device: CUDA GPU")
        return torch.device("cuda")
    else:
        print("Device: CPU")
        return torch.device("cpu")


def log_device_info() -> torch.device:
    """
    Print detailed hardware info and return the selected device.

    Returns
    -------
    torch.device
    """
    device = get_device()
    if device.type == "mps":
        print(f"MPS available : {torch.backends.mps.is_available()}")
        print(f"MPS built     : {torch.backends.mps.is_built()}")
    elif device.type == "cuda":
        print(f"GPU  : {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    return device


# ---------------------------------------------------------------------------
# Quick check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log_device_info()
