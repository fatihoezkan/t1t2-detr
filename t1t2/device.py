"""Device selection: CUDA, then Apple MPS, then CPU."""
from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the torch device. An explicit name ("cuda", "cpu", "mps") wins over auto-detection."""
    # an explicit choice wins
    if prefer:
        return torch.device(prefer)
    # otherwise CUDA, then Apple MPS, then CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info() -> str:
    """One-line device description for the run log."""
    # name of the accelerator for the log line
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps (Apple GPU)"
    return "cpu"
