"""Tests for the prompt shape normalisation shared by both ReplacementModel backends."""

import pytest
import torch

from circuit_tracer.utils import as_token_sequence


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (torch.tensor([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5]),
        (torch.tensor([[1, 2, 3, 4, 5]]), [1, 2, 3, 4, 5]),
        # a single-token prompt used to collapse to a 0-D tensor, which then raised an
        # IndexError further down ensure_tokenized rather than being tokenised
        (torch.tensor([7]), [7]),
        (torch.tensor([[7]]), [7]),
    ],
)
def test_returns_a_one_dimensional_sequence(prompt: torch.Tensor, expected: list[int]):
    """A 1-D prompt passes through and a batch of one loses only its batch dimension."""
    tokens = as_token_sequence(prompt)
    assert tokens.ndim == 1
    assert tokens.tolist() == expected


@pytest.mark.parametrize(
    "prompt",
    [
        # three separate one-token sequences used to come back as one three-token sequence
        torch.tensor([[1], [2], [3]]),
        torch.tensor([[1, 2], [3, 4], [5, 6]]),
        torch.tensor(7),
        torch.zeros(1, 2, 3, dtype=torch.long),
    ],
)
def test_rejects_anything_that_is_not_a_single_sequence(prompt: torch.Tensor):
    """Only a 1-D prompt or a 2-D batch of one names a single sequence."""
    with pytest.raises(ValueError, match="1-D or 2-D with batch size 1"):
        as_token_sequence(prompt)


def test_does_not_copy_a_prompt_that_is_already_one_dimensional():
    """The common path should hand back the same storage, not a copy."""
    prompt = torch.tensor([1, 2, 3])
    assert as_token_sequence(prompt) is prompt
