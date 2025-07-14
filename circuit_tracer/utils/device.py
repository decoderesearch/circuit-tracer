# circuit_tracer/utils/device.py
import torch

def get_default_device() -> torch.device:
    """Smart device detection - CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
