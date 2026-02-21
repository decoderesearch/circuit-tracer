"""Tests for AttributionTargets class."""

import gc
from collections.abc import Sequence
from typing import cast

import torch
import pytest

from circuit_tracer import Graph, ReplacementModel
from circuit_tracer.attribution.attribute import attribute
from circuit_tracer.attribution.targets import AttributionTargets, CustomTarget, LogitTarget


class MockTokenizer:
    """Mock tokenizer for testing.

    This tokenizer supports bijective encode/decode for strings of the form
    ``"tok_<id>"`` so that roundtrip consistency tests work correctly.
    """

    vocab_size = 100  # Define vocab size for testing

    def encode(self, text, add_special_tokens=False):
        # Simple mock: return token indices within valid range (0-99)
        if not text:
            return []
        # Support roundtrip: if text is "tok_<N>", decode back to N
        if text.startswith("tok_"):
            try:
                idx = int(text[4:])
                if 0 <= idx < self.vocab_size:
                    return [idx]
            except ValueError:
                pass
        # Fallback: use hash to generate consistent indices within range
        return [hash(text) % self.vocab_size]

    def decode(self, token_id):
        """Decode a single token ID to a string."""
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


# === Sequence[str] mode tests ===


def test_attribution_targets_str_list(mock_data):
    """Test AttributionTargets with Sequence[str] input (list)."""
    logits, unembed_proj, tokenizer = mock_data
    targets = AttributionTargets(
        attribution_targets=["hello", "world", "test"],
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
    )

    assert len(targets) == 3
    assert all(isinstance(t, LogitTarget) for t in targets.logit_targets)
    assert targets.logit_probabilities.shape == (3,)
    assert targets.logit_vectors.shape == (3, 64)
    # All should have real vocab indices
    assert all(t.vocab_idx < tokenizer.vocab_size for t in targets.logit_targets)
    # token_ids should work (all real indices)
    token_ids = targets.token_ids
    assert token_ids.shape == (3,)
    # tokens property should return decoded strings
    assert all(len(t) > 0 for t in targets.tokens)


# === Sequence[TargetSpec] mode tests ===


@pytest.mark.parametrize(
    "target_tuples,expected_keys",
    [
        (
            [
                ("token1", 0.4, torch.randn(64)),
                ("token2", 0.3, torch.randn(64)),
                ("token3", 0.3, torch.randn(64)),
            ],
            ["token1", "token2", "token3"],
        ),
    ],
    ids=["all_tuples"],
)
def test_attribution_targets_tuple_list(mock_data, target_tuples, expected_keys):
    """Test AttributionTargets with Sequence[tuple[str, float, Tensor]] input."""
    logits, unembed_proj, tokenizer = mock_data
    targets = AttributionTargets(
        attribution_targets=target_tuples,
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
    )

    assert len(targets) == len(expected_keys)
    assert all(isinstance(t, LogitTarget) for t in targets.logit_targets)
    # Tuple targets get virtual indices
    assert all(t.vocab_idx >= tokenizer.vocab_size for t in targets.logit_targets)
    # Check token_str matches expected keys
    for i, expected_key in enumerate(expected_keys):
        assert targets.logit_targets[i].token_str == expected_key
    assert torch.allclose(targets.logit_probabilities, torch.tensor([0.4, 0.3, 0.3]))


def test_attribution_targets_custom_target_namedtuple(mock_data):
    """Test AttributionTargets with Sequence[CustomTarget] input."""
    logits, unembed_proj, tokenizer = mock_data

    custom_targets = [
        CustomTarget(token_str="target_a", prob=0.6, vec=torch.randn(64)),
        CustomTarget(token_str="target_b", prob=0.4, vec=torch.randn(64)),
    ]
    targets = AttributionTargets(
        attribution_targets=custom_targets,
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
    )

    assert len(targets) == 2
    assert all(isinstance(t, LogitTarget) for t in targets.logit_targets)
    # CustomTarget targets get virtual indices
    assert all(t.vocab_idx >= tokenizer.vocab_size for t in targets.logit_targets)
    assert targets.logit_targets[0].token_str == "target_a"
    assert targets.logit_targets[1].token_str == "target_b"
    assert torch.allclose(targets.logit_probabilities, torch.tensor([0.6, 0.4]))


# === Auto modes (None and Tensor) ===


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

    assert isinstance(targets.logit_targets, list)
    assert all(isinstance(t, LogitTarget) for t in targets.logit_targets)
    assert all(t.vocab_idx < tokenizer.vocab_size for t in targets.logit_targets)

    if test_id == "salient":
        assert len(targets) <= max_n_logits
        assert len(targets) >= 1
        prob_sum = targets.logit_probabilities.sum().item()
        assert prob_sum >= desired_prob or len(targets) == max_n_logits
    elif test_id == "specific_indices":
        assert [t.vocab_idx for t in targets.logit_targets] == [5, 10, 15]
        assert targets.logit_probabilities.shape == (3,)
        assert targets.logit_vectors.shape == (3, 64)


# === Error handling ===


@pytest.mark.parametrize(
    "targets_input,error_type,error_match",
    [
        (
            [("token", 0.5)],  # Only 2 elements, should be 3
            ValueError,
            "exactly 3 elements",
        ),
        (
            [(5, 0.5, torch.randn(64))],  # int instead of str
            TypeError,
            "Custom target token_str must be str",
        ),
        (
            [],  # Empty list
            ValueError,
            "cannot be empty",
        ),
        (
            [42],  # int in list (no longer supported)
            TypeError,
            "Sequence elements must be str or TargetSpec",
        ),
        (
            torch.tensor([5, 105, 10]),  # Tensor with out of range
            ValueError,
            "Token indices must be in range",
        ),
    ],
    ids=[
        "invalid_tuple_length",
        "invalid_tuple_token_type",
        "empty_list",
        "int_in_list_rejected",
        "tensor_out_of_range",
    ],
)
def test_attribution_targets_errors(mock_data, targets_input, error_type, error_match):
    """Test AttributionTargets error handling."""
    logits, unembed_proj, tokenizer = mock_data

    with pytest.raises(error_type, match=error_match):
        AttributionTargets(
            attribution_targets=targets_input,  # type: ignore
            logits=logits,
            unembed_proj=unembed_proj,
            tokenizer=tokenizer,
        )


# === Consistency tests ===


def test_attribution_targets_str_list_consistency(mock_data):
    """Test that the same string list inputs produce consistent results."""
    logits, unembed_proj, tokenizer = mock_data

    targets1 = AttributionTargets(
        attribution_targets=["hello", "world"],
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
    )
    targets2 = AttributionTargets(
        attribution_targets=["hello", "world"],
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
    )
    assert targets1.logit_targets == targets2.logit_targets
    assert torch.equal(targets1.logit_probabilities, targets2.logit_probabilities)
    assert torch.equal(targets1.logit_vectors, targets2.logit_vectors)


def test_attribution_targets_none_vs_str_list_consistency(mock_data):
    """Test that None (auto-select) and equivalent Sequence[str] produce same results.

    Runs with None to auto-select salient logits, then constructs equivalent
    Sequence[str] from the auto-selected token strings and verifies consistency.
    """
    logits, unembed_proj, tokenizer = mock_data

    # Auto-select
    targets_auto = AttributionTargets(
        attribution_targets=None,
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
        max_n_logits=5,
        desired_logit_prob=0.8,
    )

    # Reconstruct using the auto-selected token strings
    auto_token_strs = targets_auto.tokens
    targets_explicit = AttributionTargets(
        attribution_targets=auto_token_strs,
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
    )

    # Same logit targets
    assert targets_auto.logit_targets == targets_explicit.logit_targets
    # Same probabilities
    assert torch.allclose(targets_auto.logit_probabilities, targets_explicit.logit_probabilities)
    # Same vectors
    assert torch.allclose(targets_auto.logit_vectors, targets_explicit.logit_vectors)


def test_attribution_targets_none_vs_tuple_list_consistency(mock_data):
    """Test that None and equivalent Sequence[TargetSpec] produce same results.

    Auto-selects, then constructs equivalent Sequence[TargetSpec] with the same
    probabilities and vectors, and verifies consistency.
    """
    logits, unembed_proj, tokenizer = mock_data

    # Auto-select
    targets_auto = AttributionTargets(
        attribution_targets=None,
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
        max_n_logits=3,
        desired_logit_prob=0.5,
    )

    # Reconstruct as tuple list with same probs and vecs
    tuple_targets = [
        (tok, prob.item(), vec)
        for tok, prob, vec in zip(
            targets_auto.tokens,
            targets_auto.logit_probabilities,
            targets_auto.logit_vectors,
        )
    ]
    targets_tuple = AttributionTargets(
        attribution_targets=tuple_targets,
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
    )

    # Same probabilities
    assert torch.allclose(targets_auto.logit_probabilities, targets_tuple.logit_probabilities)
    # Same vectors
    assert torch.allclose(targets_auto.logit_vectors, targets_tuple.logit_vectors)
    # Same token strings
    assert targets_auto.tokens == targets_tuple.tokens


# === Tuple (non-list Sequence) input tests ===


def test_attribution_targets_tuple_of_strs(mock_data):
    """Test AttributionTargets accepts tuple[str, ...] as Sequence[str] input."""
    logits, unembed_proj, tokenizer = mock_data
    targets = AttributionTargets(
        attribution_targets=("hello", "world", "test"),
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
    )

    assert len(targets) == 3
    assert all(isinstance(t, LogitTarget) for t in targets.logit_targets)
    assert targets.logit_probabilities.shape == (3,)
    assert targets.logit_vectors.shape == (3, 64)
    assert all(t.vocab_idx < tokenizer.vocab_size for t in targets.logit_targets)


def test_attribution_targets_tuple_of_target_specs(mock_data):
    """Test AttributionTargets accepts tuple[TargetSpec, ...] as Sequence[TargetSpec] input."""
    logits, unembed_proj, tokenizer = mock_data
    ct1 = CustomTarget(token_str="alpha", prob=0.6, vec=torch.randn(64))
    ct2 = CustomTarget(token_str="beta", prob=0.4, vec=torch.randn(64))
    targets = AttributionTargets(
        attribution_targets=(ct1, ct2),
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
    )

    assert len(targets) == 2
    assert all(isinstance(t, LogitTarget) for t in targets.logit_targets)
    assert targets.logit_targets[0].token_str == "alpha"
    assert targets.logit_targets[1].token_str == "beta"
    assert torch.allclose(targets.logit_probabilities, torch.tensor([0.6, 0.4]))


# === Property and utility tests ===


def test_attribution_targets_tokens_property(mock_data):
    """Test tokens property returns correct strings for tuple targets."""
    logits, unembed_proj, tokenizer = mock_data

    targets = AttributionTargets(
        attribution_targets=[
            ("arbitrary", 0.5, torch.randn(64)),
            ("custom_func", 0.3, torch.randn(64)),
        ],
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
    )

    tokens = targets.tokens
    assert tokens == ["arbitrary", "custom_func"]


def test_attribution_targets_virtual_token_ids(mock_data):
    """Test token_ids property for tuple targets (virtual indices)."""
    logits, unembed_proj, tokenizer = mock_data
    vocab_size = tokenizer.vocab_size

    targets = AttributionTargets(
        attribution_targets=[
            ("t1", 0.3, torch.randn(64)),
            ("t2", 0.4, torch.randn(64)),
            ("t3", 0.3, torch.randn(64)),
        ],
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
    )

    expected = [vocab_size + 0, vocab_size + 1, vocab_size + 2]
    assert targets.token_ids.tolist() == expected


def test_attribution_targets_token_ids_real(mock_data):
    """Test token_ids property for real vocab indices (str list and tensor)."""
    logits, unembed_proj, tokenizer = mock_data

    # Tensor input
    targets = AttributionTargets(
        attribution_targets=torch.tensor([5, 10, 15]),
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
    )
    token_ids = targets.token_ids
    assert torch.equal(token_ids, torch.tensor([5, 10, 15], dtype=torch.long))


@pytest.mark.parametrize(
    "test_method,expected_value",
    [
        ("to_device", "cpu"),
        ("repr", "AttributionTargets"),
        ("len", 3),
    ],
    ids=["to_device", "repr", "len"],
)
def test_attribution_targets_utility_methods(mock_data, test_method, expected_value):
    """Test utility methods: to(), __repr__(), and __len__()."""
    logits, unembed_proj, tokenizer = mock_data

    targets = AttributionTargets(
        attribution_targets=["a", "b", "c"],
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
    )

    if test_method == "to_device":
        targets_cpu = targets.to("cpu")
        assert isinstance(targets_cpu, AttributionTargets)
        assert targets_cpu.logit_probabilities.device.type == expected_value
        assert targets_cpu.logit_vectors.device.type == expected_value
        assert targets_cpu.tokenizer is tokenizer
    elif test_method == "repr":
        repr_str = repr(targets)
        assert "AttributionTargets" in repr_str
        assert "n=3" in repr_str
    elif test_method == "len":
        assert len(targets) == expected_value


# === Multi-token encoding tests ===


def test_attribution_targets_multi_token_error(mock_data):
    """Test that multi-token strings raise a ValueError."""
    logits, unembed_proj, tokenizer = mock_data

    # Mock tokenizer to return multi-token encoding for a specific string
    original_encode = tokenizer.encode

    def multi_token_encode(text, add_special_tokens=False):
        if text == "multi_token_string":
            return [10, 20, 30]  # Three tokens
        return original_encode(text, add_special_tokens)

    tokenizer.encode = multi_token_encode

    with pytest.raises(ValueError, match="encoded to 3 tokens"):
        AttributionTargets(
            attribution_targets=["multi_token_string"],
            logits=logits,
            unembed_proj=unembed_proj,
            tokenizer=tokenizer,
        )

    # Restore original encode
    tokenizer.encode = original_encode


# === Type validation ===


def test_attribution_targets_tuple_invalid_prob_type(mock_data):
    """Test that invalid prob type raises TypeError."""
    logits, unembed_proj, tokenizer = mock_data

    with pytest.raises(TypeError, match="Custom target prob must be int or float"):
        from circuit_tracer.attribution.targets import TargetSpec

        invalid_targets = cast(
            Sequence[TargetSpec],
            [
                (
                    "token1",
                    "0.5",
                    torch.randn(64),
                ),  # String instead of float - intentionally invalid
            ],
        )
        AttributionTargets(
            attribution_targets=invalid_targets,
            logits=logits,
            unembed_proj=unembed_proj,
            tokenizer=tokenizer,
        )


def test_attribution_targets_tuple_invalid_vec_type(mock_data):
    """Test that invalid vec type raises TypeError."""
    logits, unembed_proj, tokenizer = mock_data

    with pytest.raises(TypeError, match="Custom target vec must be torch.Tensor"):
        from circuit_tracer.attribution.targets import TargetSpec

        invalid_targets = cast(
            Sequence[TargetSpec],
            [
                ("token1", 0.5, [1.0, 2.0, 3.0]),  # List instead of Tensor - intentionally invalid
            ],
        )
        AttributionTargets(
            attribution_targets=invalid_targets,
            logits=logits,
            unembed_proj=unembed_proj,
            tokenizer=tokenizer,
        )


def test_attribution_targets_tuple_valid_int_prob(mock_data):
    """Test that int probability is accepted (not just float)."""
    logits, unembed_proj, tokenizer = mock_data

    targets = AttributionTargets(
        attribution_targets=[
            ("token1", 1, torch.randn(64)),  # Int probability
        ],
        logits=logits,
        unembed_proj=unembed_proj,
        tokenizer=tokenizer,
    )

    assert len(targets) == 1
    assert targets.logit_probabilities[0].item() == 1.0


# =============================================================================
# Integration tests: custom target correctness & format consistency
# =============================================================================

# === Shared helpers for integration tests ===


def _get_top_features(graph: Graph, n: int = 10) -> list[tuple[int, int, int]]:
    """Extract the top-N feature nodes from the graph based on attribution scores.

    Returns list of (layer, pos, feature_idx) tuples.
    """
    error_node_offset = graph.active_features.shape[0]
    _, first_order_indices = torch.topk(graph.adjacency_matrix[-1, :error_node_offset], n)
    top_features = [tuple(x) for x in graph.active_features[first_order_indices].tolist()]
    return top_features


def _get_unembed_weights(model, backend: str):
    """Helper to get unembedding weights in a backend-agnostic way."""
    if backend == "transformerlens":
        return model.unembed.W_U  # (d_model, d_vocab)
    else:
        return model.unembed_weight  # (d_vocab, d_model) for NNSight


def _build_custom_diff_target(
    model, prompt: str, token_x: str, token_y: str, backend: str
) -> tuple[CustomTarget, int, int]:
    """Build a CustomTarget representing logit(x) - logit(y) from the model's unembed matrix.

    Returns:
        Tuple of (custom_target, idx_x, idx_y) where idx_x and idx_y are
        the token indices for x and y respectively.
    """
    tokenizer = model.tokenizer
    idx_x = tokenizer.encode(token_x, add_special_tokens=False)[-1]
    idx_y = tokenizer.encode(token_y, add_special_tokens=False)[-1]

    input_ids = model.ensure_tokenized(prompt)
    with torch.no_grad():
        logits, _ = model.get_activations(input_ids)
    last_logits = logits.squeeze(0)[-1]  # (d_vocab,)

    # Auto-detect matrix orientation by matching against vocabulary size
    d_vocab = tokenizer.vocab_size
    unembed = _get_unembed_weights(model, backend)
    if unembed.shape[0] == d_vocab:
        vec_x = unembed[idx_x]  # (d_model,)
        vec_y = unembed[idx_y]  # (d_model,)
    else:
        # Shape is (d_model, d_vocab) – second axis is vocabulary (e.g., TransformerLens)
        vec_x = unembed[:, idx_x]  # (d_model,)
        vec_y = unembed[:, idx_y]  # (d_model,)

    diff_vec = vec_x - vec_y
    # Use the absolute difference in softmax probabilities as weight
    probs = torch.softmax(last_logits, dim=-1)
    diff_prob = (probs[idx_x] - probs[idx_y]).abs().item()
    if diff_prob < 1e-6:
        diff_prob = 0.5  # fallback weight if probs are nearly equal

    custom_target = CustomTarget(
        token_str=f"logit({token_x})-logit({token_y})",
        prob=diff_prob,
        vec=diff_vec,
    )
    return custom_target, idx_x, idx_y


def _cfg_backend(backend: str):
    """Return (model, n_layers_range, unembed_proj) for the given backend."""
    if backend == "transformerlens":
        model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma")
        n_layers_range = range(model.cfg.n_layers)  # type: ignore
        unembed_proj = model.unembed.W_U
    else:
        model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", backend="nnsight")
        n_layers_range = range(model.config.num_hidden_layers)  # type: ignore
        unembed_proj = model.unembed_weight
    return model, n_layers_range, unembed_proj


def _run_attribution_format_consistency(backend: str):
    """Backend-agnostic logic for attribution target format consistency test.

    Runs attribution with None (auto-select), then constructs equivalent Sequence[str]
    and Sequence[CustomTarget] from the auto-selected targets and verifies consistency.
    """
    prompt = "Entropy spares no entity"

    model, _, unembed_proj = _cfg_backend(backend)

    # Run with None (auto-select salient logits)
    graph_none = attribute(prompt, model, attribution_targets=None, max_n_logits=5, batch_size=256)

    # Extract the auto-selected token strings and their internal data
    auto_token_strs = [t.token_str for t in graph_none.logit_targets]

    # Run with Sequence[str] using the same token strings
    graph_str = attribute(prompt, model, attribution_targets=auto_token_strs, batch_size=256)

    # Run with Sequence[CustomTarget] using the same tokens, probs, and vectors
    # Reconstruct the unembed vectors for each auto-selected token
    input_ids = model.ensure_tokenized(prompt)
    with torch.no_grad():
        logits, _ = model.get_activations(input_ids)
    last_logits = logits.squeeze(0)[-1]

    # Build the same AttributionTargets that _from_salient would produce to extract the exact vectors
    assert isinstance(unembed_proj, torch.Tensor)
    auto_targets_obj = AttributionTargets(
        attribution_targets=None,
        logits=last_logits,
        unembed_proj=unembed_proj,
        tokenizer=model.tokenizer,
        max_n_logits=5,
        desired_logit_prob=0.8,
    )

    custom_targets = [
        CustomTarget(token_str=tok, prob=prob.item(), vec=vec)
        for tok, prob, vec in zip(
            auto_targets_obj.tokens,
            auto_targets_obj.logit_probabilities,
            auto_targets_obj.logit_vectors,
        )
    ]

    graph_tuple = attribute(prompt, model, attribution_targets=custom_targets, batch_size=256)

    # Verify consistency between None and Sequence[str]
    # Same number of targets
    assert len(graph_none.logit_targets) == len(graph_str.logit_targets), (
        f"None ({len(graph_none.logit_targets)}) vs str ({len(graph_str.logit_targets)}) "
        f"target count mismatch"
    )

    # Same token strings
    none_tokens = [t.token_str for t in graph_none.logit_targets]
    str_tokens = [t.token_str for t in graph_str.logit_targets]
    assert none_tokens == str_tokens, f"Token strings differ: {none_tokens} vs {str_tokens}"

    # Same probabilities (within tolerance)
    assert torch.allclose(
        graph_none.logit_probabilities,
        graph_str.logit_probabilities,
        atol=1e-6,
    ), "Probabilities differ between None and Sequence[str] modes"

    # Same adjacency matrix (within tolerance)
    assert torch.allclose(
        graph_none.adjacency_matrix,
        graph_str.adjacency_matrix,
        atol=1e-5,
        rtol=1e-4,
    ), "Adjacency matrices differ between None and Sequence[str] modes"

    # Verify consistency between None and Sequence[CustomTarget]
    assert len(graph_none.logit_targets) == len(graph_tuple.logit_targets), (
        f"None ({len(graph_none.logit_targets)}) vs tuple ({len(graph_tuple.logit_targets)}) "
        f"target count mismatch"
    )

    # Token strings should match
    tuple_tokens = [t.token_str for t in graph_tuple.logit_targets]
    assert none_tokens == tuple_tokens, f"Token strings differ: {none_tokens} vs {tuple_tokens}"

    # Probabilities should match
    assert torch.allclose(
        graph_none.logit_probabilities,
        graph_tuple.logit_probabilities.to(graph_none.logit_probabilities.device),
        atol=1e-6,
    ), "Probabilities differ between None and Sequence[CustomTarget] modes"

    # Adjacency matrices should match (tuple targets use the same unembed vecs)
    assert torch.allclose(
        graph_none.adjacency_matrix,
        graph_tuple.adjacency_matrix.to(graph_none.adjacency_matrix.device),
        atol=1e-5,
        rtol=1e-4,
    ), "Adjacency matrices differ between None and Sequence[CustomTarget] modes"


def _run_custom_target_correctness(
    backend: str,
    n_samples: int = 20,
    act_atol: float = 5e-4,
    act_rtol: float = 1e-5,
    logit_atol: float = 1e-4,
    logit_rtol: float = 1e-3,
):
    """Verify custom target direction feature attribution driven interventions produce expected activation/logit changes

    For a ``logit(x) − logit(y)`` custom direction, randomly samples features, doubles each feature's pre-activation
    value (under constrained/frozen-layer conditions), and checks that both the activation changes and the custom
    logit-difference change match the adjacency matrix predictions within acceptable tolerances.

    * **Activation changes** match ``adjacency_matrix[:n_features, node]`` within act_atol/act_rtol.
    * **Custom logit-difference change** matches the adjacency logit-node prediction within logit_atol/logit_rtol.

    We use the same linear-regime conditions as our other attribution validation tests, e.g. ``verify_feature_edges``:

    * ``constrained_layers=range(n_layers)`` — freezes all layer norms, MLPs, and attention, preventing non-linear
      propagation.
    * ``apply_activation_function=False`` — operates on pre-activation values.
    * ``model.zero_softcap()`` — removes the final logit softcap.
    * Intervention = doubling the pre-activation (delta = old_activation). Because the adjacency column already encodes
      the full effect of the feature at its current activation level, doubling adds exactly one copy of the predicted
      effect.
    """
    prompt = "The capital of the state containing Dallas is"
    token_x, token_y = "▁Austin", "▁Dallas"

    model, n_layers_range, _ = _cfg_backend(backend)
    custom_target, idx_x, idx_y = _build_custom_diff_target(
        model, prompt, token_x, token_y, backend
    )

    graph = attribute(prompt, model, attribution_targets=[custom_target], batch_size=256)

    device = next(model.parameters()).device
    adjacency_matrix = graph.adjacency_matrix.to(device)
    active_features = graph.active_features.to(device)
    n_features = active_features.size(0)
    n_logits = len(graph.logit_targets)

    # --- baseline (pre-activation, unsoftcapped) ---
    with model.zero_softcap():
        logits, activation_cache = model.get_activations(
            graph.input_tokens, apply_activation_function=False
        )
    logits = logits.squeeze(0)

    relevant_activations = activation_cache[
        active_features[:, 0], active_features[:, 1], active_features[:, 2]
    ]
    baseline_logit_diff = logits[-1, idx_x] - logits[-1, idx_y]

    # --- per-feature exact checks ---
    random_order = torch.randperm(n_features)
    chosen_nodes = random_order[: min(n_samples, n_features)]

    for chosen_node in chosen_nodes:
        layer, pos, feat_idx = active_features[chosen_node].tolist()
        old_activation = activation_cache[layer, pos, feat_idx]
        new_activation = old_activation * 2

        expected_effects = adjacency_matrix[:, chosen_node]
        expected_act_diff = expected_effects[:n_features]
        expected_logit_diff = expected_effects[-n_logits:]  # (1,) for single target

        with model.zero_softcap():
            new_logits, new_act_cache = model.feature_intervention(
                graph.input_tokens,
                [(layer, pos, feat_idx, new_activation)],
                constrained_layers=n_layers_range,
                apply_activation_function=False,
            )
        new_logits = new_logits.squeeze(0)

        # -- activation check --
        assert new_act_cache is not None
        new_relevant_activations = new_act_cache[
            active_features[:, 0], active_features[:, 1], active_features[:, 2]
        ]
        assert torch.allclose(
            new_relevant_activations,
            relevant_activations + expected_act_diff,
            atol=act_atol,
            rtol=act_rtol,
        ), (
            f"Activation mismatch for feature ({layer}, {pos}, {feat_idx}): "
            f"max diff = {(new_relevant_activations - relevant_activations - expected_act_diff).abs().max():.6e}"
        )

        # -- logit-difference check --
        new_logit_diff = new_logits[-1, idx_x] - new_logits[-1, idx_y]
        actual_logit_change = (new_logit_diff - baseline_logit_diff).unsqueeze(0)
        assert torch.allclose(
            actual_logit_change,
            expected_logit_diff,
            atol=logit_atol,
            rtol=logit_rtol,
        ), (
            f"Logit-diff mismatch for feature ({layer}, {pos}, {feat_idx}): "
            f"predicted={expected_logit_diff.item():.6e}, "
            f"actual={actual_logit_change.item():.6e}"
        )


@pytest.fixture(autouse=False)
def cleanup_cuda():
    yield
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("backend", ["transformerlens", "nnsight"])
def test_custom_target_correctness(cleanup_cuda, backend):
    """Verify custom target direction feature attribution driven interventions produce expected activation/logit changes

    For a ``logit(x) − logit(y)`` custom direction, randomly samples features, doubles each feature's pre-activation
    value (under constrained/frozen-layer conditions), and checks that both the activation changes and the custom
    logit-difference change match the adjacency matrix predictions within acceptable tolerances.

    Args:
        cleanup_cuda: Fixture for CUDA cleanup after test
        backend: Model backend to test ("transformerlens" or "nnsight")
    """
    _run_custom_target_correctness(backend)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("backend", ["transformerlens", "nnsight"])
def test_attribution_format_consistency(cleanup_cuda, backend):
    """Verify None, Sequence[str], and Sequence[CustomTarget] produce consistent results.

    Runs attribution with None (auto-select), then with equivalent Sequence[str] and
    Sequence[CustomTarget] targets, and verifies the graphs are consistent.

    Args:
        cleanup_cuda: Fixture for CUDA cleanup after test
        backend: Model backend to test ("transformerlens" or "nnsight")
    """
    _run_attribution_format_consistency(backend)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
