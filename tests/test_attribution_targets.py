"""Unit tests for AttributionTargets class."""

import torch
import pytest

from circuit_tracer.attribution.targets import AttributionTargets


class MockTokenizer:
    """Mock tokenizer for testing."""

    vocab_size = 100  # Define vocab size for testing

    def encode(self, text, add_special_tokens=False):
        # Simple mock: return token indices within valid range (0-99)
        if not text:
            return []
        # Use hash to generate consistent indices within range
        return [hash(text) % 100]

    def decode(self, token_id):
        """Decode a single token ID to a string."""
        # Simple mock: return string representation prefixed with "tok_"
        if isinstance(token_id, int):
            return f"tok_{token_id}"
        return str(token_id)


@pytest.fixture
def mock_data():
    """Create mock logits and unembedding projection."""
    vocab_size = 100
    d_model = 64

    # Create reproducible random data
    torch.manual_seed(42)
    logits = torch.randn(vocab_size)
    unembed_proj = torch.randn(d_model, vocab_size)
    tokenizer = MockTokenizer()

    return logits, unembed_proj, tokenizer


@pytest.mark.parametrize(
    "targets_list,expected_len,expected_key_types,expected_keys,test_id",
    [
        (
            [("arbitrary_token", 0.5, torch.randn(64)), 5, ("another", 0.3, torch.randn(64))],
            3,
            # LogitTarget instances have both str and int, but check token_str type
            ["str", "int", "str"],
            ["arbitrary_token", None, "another"],  # None for dynamic int keys
            "mixed",
        ),
        (
            [
                ("token1", 0.4, torch.randn(64)),
                ("token2", 0.3, torch.randn(64)),
                ("token3", 0.3, torch.randn(64)),
            ],
            3,
            ["str", "str", "str"],
            ["token1", "token2", "token3"],
            "all_tuples",
        ),
        (
            ["hello", "world", "test"],
            3,
            ["int", "int", "int"],  # Strings get tokenized to ints
            [None, None, None],  # Dynamic keys
            "all_strings",
        ),
    ],
    ids=["mixed", "all_tuples", "all_strings"],
)
def test_attribution_targets_list_mode(
    mock_data, targets_list, expected_len, expected_key_types, expected_keys, test_id
):
    """Test AttributionTargets with list input (most flexible mode)."""
    logits, unembed_proj, tokenizer = mock_data

    targets = AttributionTargets(
        attribution_targets=targets_list,
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
    )

    # Verify basic structure
    from circuit_tracer.attribution.targets import LogitTarget

    assert isinstance(targets.logit_targets, list)
    assert len(targets) == expected_len
    assert all(isinstance(t, LogitTarget) for t in targets.logit_targets)
    assert targets.logit_probabilities.shape == (expected_len,)
    assert targets.logit_vectors.shape == (expected_len, 64)

    # Verify token_str and vocab_idx based on expected types
    for i, expected_type in enumerate(expected_key_types):
        target = targets.logit_targets[i]
        assert isinstance(target.token_str, str), f"Target {i} token_str should be str"
        assert isinstance(target.vocab_idx, int), f"Target {i} vocab_idx should be int"

        # Check token_str matches expected_keys when provided
        expected_key = expected_keys[i]
        if expected_key is not None:
            assert target.token_str == expected_key, f"Target {i} token_str mismatch"

        # Check vocab_idx type based on whether this was an arbitrary string/
        # function thereof (tuple)
        if expected_type == "str":  # Was a tuple with arbitrary string
            # Should have virtual index >= vocab_size
            assert target.vocab_idx >= tokenizer.vocab_size, f"Target {i} should have virtual index"
        else:  # Was int or tokenized string
            # Should have real vocab index < vocab_size
            assert target.vocab_idx < tokenizer.vocab_size, (
                f"Target {i} should have real vocab index"
            )

    # Test-specific assertions
    if test_id == "mixed":
        # First and third elements from tuples should have provided probs
        assert abs(targets.logit_probabilities[0].item() - 0.5) < 1e-6
        assert abs(targets.logit_probabilities[2].item() - 0.3) < 1e-6
    elif test_id == "all_tuples":
        assert torch.allclose(targets.logit_probabilities, torch.tensor([0.4, 0.3, 0.3]))
    elif test_id == "all_strings":
        # All should be tokenized - check via tokens property
        assert all(len(t) > 0 for t in targets.tokens)


@pytest.mark.parametrize(
    "attribution_targets,max_n_logits,desired_prob,test_id",
    [
        (None, 5, 0.8, "salient"),
        (torch.tensor([5, 10, 15]), None, None, "specific_indices"),
    ],
    ids=["salient", "specific_indices"],
)
def test_attribution_targets_auto_modes(
    mock_data, attribution_targets, max_n_logits, desired_prob, test_id
):
    """Test AttributionTargets with automatic modes (None and Tensor)."""
    logits, unembed_proj, tokenizer = mock_data

    kwargs = {}
    if max_n_logits is not None:
        kwargs["max_n_logits"] = max_n_logits
    if desired_prob is not None:
        kwargs["desired_logit_prob"] = desired_prob

    targets = AttributionTargets(
        attribution_targets=attribution_targets,
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
        **kwargs,
    )

    # Verify basic structure - all targets should be LogitTarget instances
    from circuit_tracer.attribution.targets import LogitTarget

    assert isinstance(targets.logit_targets, list)
    assert all(isinstance(t, LogitTarget) for t in targets.logit_targets)
    # All should have real vocab indices (< vocab_size)
    assert all(t.vocab_idx < tokenizer.vocab_size for t in targets.logit_targets)

    if test_id == "salient":
        assert len(targets) <= max_n_logits
        assert len(targets) >= 1
        # Probabilities should sum to at least desired_prob (or hit max_n_logits)
        prob_sum = targets.logit_probabilities.sum().item()
        assert prob_sum >= desired_prob or len(targets) == max_n_logits
    elif test_id == "specific_indices":
        # Check vocab_idx matches expected
        assert [t.vocab_idx for t in targets.logit_targets] == [5, 10, 15]
        assert targets.logit_probabilities.shape == (3,)
        assert targets.logit_vectors.shape == (3, 64)


@pytest.mark.parametrize(
    "targets_list,error_match",
    [
        (
            [("token", 0.5)],  # Only 2 elements, should be 3
            "exactly 3 elements",
        ),
        (
            [(5, 0.5, torch.randn(64))],  # int instead of str
            "str as first element",
        ),
        (
            [],  # Empty list
            "cannot be empty",
        ),
    ],
    ids=["invalid_tuple_length", "invalid_tuple_token_type", "empty_list"],
)
def test_attribution_targets_errors(mock_data, targets_list, error_match):
    """Test AttributionTargets error handling."""
    logits, unembed_proj, tokenizer = mock_data

    with pytest.raises(ValueError, match=error_match):
        AttributionTargets(
            attribution_targets=targets_list,  # type: ignore
            logits=logits,
            unembed_proj=unembed_proj,
            tokenizer=tokenizer,
        )


def test_attribution_targets_consistency(mock_data):
    """Test that the same inputs produce consistent results."""
    logits, unembed_proj, tokenizer = mock_data

    targets_list = [5, "hello", ("custom", 0.5, torch.randn(64))]

    targets1 = AttributionTargets(
        attribution_targets=targets_list,
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
    )
    targets2 = AttributionTargets(
        attribution_targets=targets_list,
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
    )

    assert targets1.logit_targets == targets2.logit_targets


def test_attribution_targets_tokens_property(mock_data):
    """Test tokens property decodes ints and preserves strings."""
    logits, unembed_proj, tokenizer = mock_data

    targets_list = [
        5,
        ("arbitrary", 0.5, torch.randn(64)),
        10,
    ]

    targets = AttributionTargets(
        attribution_targets=targets_list,
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
    )

    tokens = targets.tokens

    assert isinstance(tokens, list)
    assert len(tokens) == 3
    assert tokens[0] == "tok_5"  # int decoded with tokenizer
    assert tokens[1] == "arbitrary"  # str kept as-is
    assert tokens[2] == "tok_10"  # int decoded with tokenizer


@pytest.mark.parametrize(
    "test_method,expected_value",
    [
        ("to_device", "cpu"),
        ("repr", "AttributionTargets(n=5, keys=[1, 2, 3]...)"),
        ("len", 5),
    ],
    ids=["to_device", "repr", "len"],
)
def test_attribution_targets_utility_methods(mock_data, test_method, expected_value):
    """Test utility methods: to(), __repr__(), and __len__()."""
    logits, unembed_proj, tokenizer = mock_data

    # Use same targets for all tests
    targets_list = [1, 2, 3, 4, 5]

    targets = AttributionTargets(
        attribution_targets=targets_list,
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
    )

    if test_method == "to_device":
        # Test device transfer
        targets_cpu = targets.to("cpu")
        assert isinstance(targets_cpu, AttributionTargets)
        assert targets_cpu.logit_targets == targets.logit_targets
        assert targets_cpu.logit_probabilities.device.type == expected_value
        assert targets_cpu.logit_vectors.device.type == expected_value
        assert targets_cpu.tokenizer is tokenizer  # Verify tokenizer preserved
    elif test_method == "repr":
        # Test string representation
        repr_str = repr(targets)
        assert "AttributionTargets" in repr_str
        assert "n=5" in repr_str
        # Check for "targets=" since keys are now LogitTarget instances
        assert "targets=" in repr_str
    elif test_method == "len":
        # Test __len__
        assert len(targets) == expected_value


@pytest.mark.parametrize(
    "targets_list,expected_indices,test_id",
    [
        # All real vocab tokens
        ([5, 10, 15], [5, 10, 15], "all_real"),
        # Mixed real and virtual (arbitrary strings)
        ([5, ("arb", 0.5, torch.randn(64)), 10], lambda vs: [5, vs + 1, 10], "mixed"),
        # All virtual (arbitrary strings)
        (
            [
                ("t1", 0.3, torch.randn(64)),
                ("t2", 0.4, torch.randn(64)),
                ("t3", 0.3, torch.randn(64)),
            ],
            lambda vs: [vs + 0, vs + 1, vs + 2],
            "all_virtual",
        ),
    ],
    ids=["all_real", "mixed", "all_virtual"],
)
def test_attribution_targets_vocab_indices(mock_data, targets_list, expected_indices, test_id):
    """Test vocab_indices property with various combinations of real and virtual tokens."""
    logits, unembed_proj, tokenizer = mock_data
    vocab_size = tokenizer.vocab_size  # 100

    targets = AttributionTargets(
        attribution_targets=targets_list,
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
    )

    # Compute expected indices (may depend on vocab_size for virtual indices)
    if callable(expected_indices):
        expected = expected_indices(vocab_size)
    else:
        expected = expected_indices

    vocab_indices = targets.vocab_indices
    assert vocab_indices == expected
    assert all(isinstance(idx, int) for idx in vocab_indices)

    # Verify virtual index detection
    if test_id == "all_real":
        assert not targets.has_virtual_indices
        # Should be able to get token_ids
        token_ids = targets.token_ids
        assert torch.equal(token_ids, torch.tensor(expected, dtype=torch.long))
    else:
        assert targets.has_virtual_indices
        # Should raise when trying to get token_ids
        with pytest.raises(ValueError, match="virtual indices"):
            _ = targets.token_ids


@pytest.mark.parametrize(
    "targets_list,error_match",
    [
        # Out of range token ID
        ([110], "out of vocabulary range.*100"),
        # Negative token ID
        ([-5], "out of vocabulary range"),
        # Tensor with out of range
        (torch.tensor([5, 105, 10]), "Token indices must be in range"),
    ],
    ids=["token_id_out_of_range", "token_id_negative", "tensor_out_of_range"],
)
def test_attribution_targets_validation_errors(mock_data, targets_list, error_match):
    """Test validation catches various invalid token ID errors."""
    logits, unembed_proj, tokenizer = mock_data

    with pytest.raises(ValueError, match=error_match):
        AttributionTargets(
            attribution_targets=targets_list,  # type: ignore
            logits=logits,
            unembed_proj=unembed_proj,
            tokenizer=tokenizer,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
