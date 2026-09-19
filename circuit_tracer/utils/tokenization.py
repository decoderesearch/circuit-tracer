"""Prompt -> token ids, shared by every backend.

Attribution compares graphs across backends, so the backends have to agree on the token
sequence before anything else can be compared. This lived as a byte-identical copy in each
``ReplacementModel`` implementation; it is one function here so a third backend does not become
a third copy that can drift.
"""

import warnings

import torch


def ensure_tokenized(
    prompt: str | torch.Tensor | list[int],
    tokenizer,
    device: torch.device | str | None,
    model_name: str,
) -> torch.Tensor:
    """Convert prompt to 1-D tensor of token ids with proper special token handling.

    This function ensures that a special token (BOS/PAD) is prepended to the input sequence.
    The first token position in transformer models typically exhibits unusually high norm
    and an excessive number of active features due to how models process the beginning of
    sequences. By prepending a special token, we ensure that actual content tokens have
    more consistent and interpretable feature activations, avoiding the artifacts present
    at position 0. This prepended token is later ignored during attribution analysis.

    Args:
        prompt: String, tensor, or list of token ids representing a single sequence
        tokenizer: The model's tokenizer
        device: Device to place the returned token ids on. ``None`` leaves them where they are,
            which is what ``Tensor.to`` already does and what both existing backends pass when
            their config carries no device.
        model_name: Model name, used to detect the Gemma-3-it chat prefix special case

    Returns:
        1-D tensor of token ids with BOS/PAD token at the beginning

    Raises:
        TypeError: If prompt is not str, tensor, or list
        ValueError: If tensor has wrong shape (must be 1-D or 2-D with batch size 1)
    """

    if isinstance(prompt, str):
        tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(
            0
        )
    elif isinstance(prompt, torch.Tensor):
        tokens = prompt.squeeze()
    elif isinstance(prompt, list):
        tokens = torch.tensor(prompt, dtype=torch.long).squeeze()
    else:
        raise TypeError(f"Unsupported prompt type: {type(prompt)}")

    if tokens.ndim > 1:
        raise ValueError(f"Tensor must be 1-D, got shape {tokens.shape}")

    tokens = tokens.to(device)

    gemma_3_it = "gemma-3" in model_name and model_name.endswith("-it")
    if gemma_3_it:
        ignore_prefix = torch.tensor([2, 105, 2364, 107], dtype=tokens.dtype, device=tokens.device)
        tokenization_error = (
            "Input tokens should start with <bos><start_of_turn>user\n, but got {tokens}"
        )
        assert tokens.size(0) >= 4 and torch.all(tokens[:4] == ignore_prefix), (
            tokenization_error.format(tokens=tokenizer.decode(tokens.cpu().tolist()))
        )
        return tokens

    # Check if a special token is already present at the beginning
    if tokens[0] in tokenizer.all_special_ids:
        return tokens

    # Prepend a special token to avoid artifacts at position 0
    candidate_bos_token_ids = [
        tokenizer.bos_token_id,
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
    ]
    candidate_bos_token_ids += tokenizer.all_special_ids

    dummy_bos_token_id = next(filter(None, candidate_bos_token_ids))
    if dummy_bos_token_id is None:
        warnings.warn(
            "No suitable special token found for BOS token replacement. "
            "The first token will be ignored."
        )
    else:
        tokens = torch.cat([torch.tensor([dummy_bos_token_id], device=tokens.device), tokens])

    return tokens.to(device)


def zero_positions(model_name: str) -> slice:
    """Positions whose features are zeroed out before attribution.

    Position 0 always goes, for the high-norm reason in :func:`ensure_tokenized`. The Gemma
    Scope 2 ``-it`` transcoders additionally need the whole ``<bos><start_of_turn>user\\n``
    chat prefix dropped, which is four tokens.
    """
    gemma_3_it = "gemma-3" in model_name and model_name.endswith("-it")
    return slice(0, 4) if gemma_3_it else slice(0, 1)
