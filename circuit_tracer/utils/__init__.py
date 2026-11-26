import torch
from circuit_tracer.utils.create_graph_files import create_graph_files as create_graph_files
from circuit_tracer.utils.memory_estimation import estimate_graph_memory as estimate_graph_memory


def get_default_device() -> torch.device:
    """Get the default device, preferring CUDA if available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


__all__ = ["create_graph_files", "estimate_graph_memory", "get_default_device"]
