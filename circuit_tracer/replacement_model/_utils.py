import torch


def zero_special_positions(
    activations: torch.Tensor,
    zero_positions: int | slice,
) -> torch.Tensor:
    """Zero special-token positions on the position axis for single or batched activations."""

    if activations.ndim < 2:
        raise ValueError(
            f"Expected activations with at least 2 dimensions, got shape {tuple(activations.shape)}"
        )

    if activations.ndim == 2:
        activations[zero_positions, :] = 0
    else:
        activations[:, zero_positions, :] = 0
    return activations
