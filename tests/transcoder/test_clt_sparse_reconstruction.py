"""Sparse CLT reconstruction must retain tokens with no active features."""

import pytest
import torch
from safetensors.torch import save_file

from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder


def make_clt(tmp_path, lazy_encoder=False, lazy_decoder=False, skip=False):
    clt = CrossLayerTranscoder(
        2,
        2,
        2,
        skip_connection=skip,
        lazy_decoder=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert clt.W_dec is not None
    with torch.no_grad():
        clt.W_enc.copy_(torch.eye(2).expand(2, 2, 2))
        clt.b_dec.copy_(torch.tensor([[0.5, -0.25], [1.0, 0.75]]))
        for layer, dec in enumerate(clt.W_dec):
            dec.copy_(torch.arange(dec.numel()).reshape(dec.shape) / 4 + layer + 1)
        if skip:
            assert clt.W_skip is not None
            clt.W_skip.copy_(torch.tensor([[[2.0, 1.0], [0.0, 3.0]], [[1.0, 0.0], [2.0, 1.0]]]))
    # Save the weights needed for lazy reads; keep affine state in memory.
    for layer in range(2):
        save_file(
            {f"W_enc_{layer}": clt.W_enc[layer].contiguous()},
            str(tmp_path / f"W_enc_{layer}.safetensors"),
        )
        save_file(
            {f"W_dec_{layer}": clt.W_dec[layer].contiguous()},
            str(tmp_path / f"W_dec_{layer}.safetensors"),
        )
    weights = [dec.detach().clone() for dec in clt.W_dec]
    if lazy_encoder:
        del clt.W_enc
        clt.lazy_encoder = True
    if lazy_decoder:
        clt.W_dec = None
        clt.lazy_decoder = True
    clt.clt_path = str(tmp_path)
    return clt, weights


def dense_oracle(clt, weights, inputs, zero_positions):
    # Identity encoders/zero encoder biases give an independent dense oracle.
    features = inputs.relu()
    features[:, zero_positions] = 0
    expected = clt.b_dec[:, None].expand_as(inputs).clone()
    for source in range(2):
        for target in range(source, 2):
            expected[target] += features[source] @ weights[source][:, target - source]
    if clt.W_skip is not None:
        for layer in range(2):
            expected[layer] += inputs[layer] @ clt.W_skip[layer]
    return features, expected


@pytest.mark.parametrize("lazy_encoder", [False, True])
@pytest.mark.parametrize("lazy_decoder", [False, True])
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize(
    "values,zero_positions",
    [
        ([1.0, 2.0, -1.0], slice(0, 1)),
        ([-1.0, -2.0, -3.0], slice(0, 1)),
        ([2.0], slice(0, 1)),
        ([2.0, -1.0, -1.0], slice(0, 0)),
        ([1.0, 2.0, 3.0], slice(0, 1)),
    ],
    ids=[
        "trailing-inactive",
        "all-inactive",
        "one-token-bos",
        "first-active-only",
        "active-control",
    ],
)
def test_sparse_token_extent(tmp_path, lazy_encoder, lazy_decoder, skip, values, zero_positions):
    clt, weights = make_clt(tmp_path, lazy_encoder, lazy_decoder, skip)
    inputs = torch.tensor(values)[None, :, None].repeat(2, 1, 2)
    features, expected = dense_oracle(clt, weights, inputs, zero_positions)
    kwargs = {} if zero_positions == slice(0, 1) else {"zero_positions": zero_positions}
    components = clt.compute_attribution_components(inputs, **kwargs)
    actual = components["reconstruction"]
    assert actual.shape == inputs.shape
    torch.testing.assert_close(actual, expected)
    # The replacement-model subtraction previously broadcast length-one outputs.
    torch.testing.assert_close(torch.zeros_like(inputs) - actual, -expected)
    for supplied_features in (features, features.to_sparse()):
        decoded = clt.decode(supplied_features, inputs if skip else None)
        assert decoded.shape == inputs.shape
        torch.testing.assert_close(decoded, expected)
    if not features.any():
        assert components["encoder_vecs"].shape == (0, 2)
        assert components["decoder_vecs"].shape == (0, 2)
        assert components["decoder_vecs"].dtype == inputs.dtype
        assert components["decoder_vecs"].device == inputs.device
        assert components["decoder_locations"].shape == (2, 0)
        assert components["decoder_locations"].dtype == torch.long
        assert components["encoder_to_decoder_map"].shape == (0,)
        assert components["encoder_to_decoder_map"].dtype == torch.long


def test_compute_reconstruction_legacy_positional_call(tmp_path):
    clt, _ = make_clt(tmp_path)
    pos = torch.tensor([0, 1])
    layers = torch.tensor([0, 1])
    vectors = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    expected = clt.b_dec[:, None].expand(2, 2, 2).clone()
    expected[0, 0] += vectors[0]
    expected[1, 1] += vectors[1]
    torch.testing.assert_close(clt.compute_reconstruction(pos, layers, vectors), expected)
    torch.testing.assert_close(clt.compute_reconstruction(pos, layers, vectors, None), expected)


@pytest.mark.parametrize("with_inputs", [False, True])
def test_empty_reconstruction_extent(tmp_path, with_inputs):
    clt, _ = make_clt(tmp_path, skip=with_inputs)
    empty = torch.empty(0, dtype=torch.long)
    vectors = torch.empty(0, 2)
    inputs = torch.ones(2, 3, 2) if with_inputs else None
    kwargs = {} if with_inputs else {"n_pos": 3}
    actual = clt.compute_reconstruction(empty, empty, vectors, inputs, **kwargs)
    expected = clt.b_dec[:, None].expand(2, 3, 2).clone()
    if with_inputs:
        assert inputs is not None and clt.W_skip is not None
        expected += inputs @ clt.W_skip
    torch.testing.assert_close(actual, expected)


def test_empty_reconstruction_requires_extent(tmp_path):
    clt, _ = make_clt(tmp_path)
    empty = torch.empty(0, dtype=torch.long)
    with pytest.raises(ValueError, match="n_pos.*input_acts"):
        clt.compute_reconstruction(empty, empty, torch.empty(0, 2))


def test_empty_decoder_vectors_preserve_promoted_dtype(tmp_path):
    clt, _ = make_clt(tmp_path)
    positions, layers, features, vectors, mapping = clt.select_decoder_vectors(
        torch.zeros(2, 3, 2, dtype=torch.float64)
    )
    assert vectors.shape == (0, 2)
    assert vectors.dtype == torch.float64
    for indices in (positions, layers, features, mapping):
        assert indices.shape == (0,)
        assert indices.dtype == torch.long
        assert indices.device == vectors.device
