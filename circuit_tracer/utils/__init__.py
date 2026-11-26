import torch
from circuit_tracer.utils.create_graph_files import create_graph_files as create_graph_files


def get_default_device() -> torch.device:
    """Get the default device, preferring CUDA, then MPS if available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


__all__ = ["create_graph_files", "get_default_device"]
