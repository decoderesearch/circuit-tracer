import torch
from circuit_tracer.utils.create_graph_files import create_graph_files as create_graph_files


def get_default_device() -> torch.device:
    """Get the default device, preferring CUDA if available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def as_token_sequence(prompt: torch.Tensor) -> torch.Tensor:
    """Return `prompt` as a 1-D sequence of token ids.

    A `(1, n_pos)` batch of one loses its batch dimension. Anything else that is not already
    1-D is rejected rather than reshaped, since `squeeze()` with no dimension also drops a
    sequence axis of length one: it turns a single-token prompt into a 0-D tensor, and a
    `(batch, 1)` stack of separate sequences into one sequence of `batch` tokens.

    Args:
        prompt: a 1-D tensor of token ids, or a 2-D tensor whose batch size is 1

    Returns:
        torch.Tensor: the 1-D token sequence

    Raises:
        ValueError: if `prompt` is neither 1-D nor 2-D with a batch size of 1
    """
    if prompt.ndim == 2 and prompt.shape[0] == 1:
        prompt = prompt.squeeze(0)
    if prompt.ndim != 1:
        raise ValueError(
            f"Prompt must be 1-D or 2-D with batch size 1, got shape {tuple(prompt.shape)}"
        )
    return prompt


__all__ = ["as_token_sequence", "create_graph_files", "get_default_device"]
