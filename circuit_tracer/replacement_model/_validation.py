from collections.abc import Sequence

import torch


def validate_single_sequence_inputs(
    inputs: str | torch.Tensor | list[int],
    method_name: str,
) -> None:
    """Raise a clear error when a single-sequence path receives batched inputs."""

    if isinstance(inputs, torch.Tensor):
        if inputs.ndim > 2 or (inputs.ndim == 2 and inputs.shape[0] != 1):
            raise ValueError(
                f"{method_name} only supports a single sequence, got tensor input with shape "
                f"{tuple(inputs.shape)}. Loop over the batch instead."
            )
        return

    if isinstance(inputs, Sequence) and not isinstance(inputs, str):
        if inputs and not all(isinstance(token_id, int) for token_id in inputs):
            raise ValueError(
                f"{method_name} only supports a single sequence. Expected a single string, "
                "a 1D token tensor, or a single list of token ids. Batched list inputs are "
                "not supported."
            )
