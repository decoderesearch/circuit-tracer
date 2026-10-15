import pytest

from circuit_tracer.utils.memory_estimation import estimate_graph_memory, format_bytes


def test_estimate_graph_memory_node_counts_and_dense_bytes():
    estimate = estimate_graph_memory(
        n_tokens=100,
        n_layers=4,
        max_feature_nodes=50,
        n_logits=5,
        dtype="float32",
    )

    assert estimate.n_error_nodes == 400
    assert estimate.n_token_nodes == 100
    assert estimate.n_logit_nodes == 5
    assert estimate.n_total_nodes == 555
    assert estimate.dense_adjacency_bytes == 555 * 555 * 4
    assert estimate.dense_bool_mask_bytes == 555 * 555
    assert estimate.pruning_peak_bytes == 4 * estimate.dense_adjacency_bytes + 555 * 555


def test_estimate_graph_memory_dtype_alias_halves_adjacency():
    fp32 = estimate_graph_memory(n_tokens=128, n_layers=8, dtype="fp32")
    bf16 = estimate_graph_memory(n_tokens=128, n_layers=8, dtype="bf16")

    assert fp32.dtype == "float32"
    assert bf16.dtype == "bfloat16"
    assert fp32.bytes_per_value == 4
    assert bf16.bytes_per_value == 2
    assert bf16.dense_adjacency_bytes == fp32.dense_adjacency_bytes // 2


def test_estimate_graph_memory_flags_h100_sized_long_context_risk():
    estimate = estimate_graph_memory(
        n_tokens=6000,
        n_layers=26,
        max_feature_nodes=7500,
        n_logits=10,
        dtype="float16",
        available_memory_gib=80,
    )

    assert estimate.fits_usable_memory is False
    assert estimate.n_error_nodes == 156_000
    assert estimate.estimated_peak_bytes > estimate.usable_memory_bytes
    assert any("exceeds the usable memory budget" in rec for rec in estimate.recommendations)
    assert any("sparse or blockwise graph backend" in rec for rec in estimate.recommendations)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"n_tokens": 0, "n_layers": 2}, "n_tokens must be positive"),
        ({"n_tokens": 5, "n_layers": 0}, "n_layers must be positive"),
        ({"n_tokens": 5, "n_layers": 2, "dtype": "int8"}, "Unsupported dtype"),
        ({"n_tokens": 5, "n_layers": 2, "safety_fraction": 0}, "safety_fraction"),
    ],
)
def test_estimate_graph_memory_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        estimate_graph_memory(**kwargs)


def test_estimate_graph_memory_serialization_helpers():
    estimate = estimate_graph_memory(
        n_tokens=16,
        n_layers=2,
        max_feature_nodes=8,
        n_logits=2,
        dtype="float16",
        available_memory_gib=1,
    )

    as_dict = estimate.to_dict()
    assert as_dict["nodes"]["total_nodes"] == 58
    assert "estimated_peak" in as_dict["memory"]
    assert "Recommendations" in estimate.to_markdown()
    assert '"fits_usable_memory"' in estimate.to_json()
    assert format_bytes(1024**3) == "1.00 GiB"
