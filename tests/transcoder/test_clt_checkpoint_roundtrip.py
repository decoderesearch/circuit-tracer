"""Checkpoint parity for affine and non-affine cross-layer transcoders."""

import pytest
import torch
from safetensors.torch import load_file, save_file

from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder, load_clt


@pytest.fixture
def make_clt():
    def create(skip_connection):
        clt = CrossLayerTranscoder(
            n_layers=2,
            d_transcoder=1,
            d_model=2,
            skip_connection=skip_connection,
            lazy_decoder=False,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        with torch.no_grad():
            clt.W_enc.fill_(1)
            assert clt.W_dec is not None
            for weights in clt.W_dec:
                weights.fill_(1)
            clt.b_dec.fill_(0.5)
            if skip_connection:
                assert clt.W_skip is not None
                clt.W_skip.copy_(torch.tensor([[[1, 2], [3, 4]], [[2, 0], [1, 3]]]))
        return clt

    return create


@pytest.mark.parametrize("skip_connection", [False, True])
@pytest.mark.parametrize("lazy_encoder", [False, True])
@pytest.mark.parametrize("lazy_decoder", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@torch.no_grad()
def test_checkpoint_preserves_reconstruction(
    tmp_path, make_clt, skip_connection, lazy_encoder, lazy_decoder, dtype
):
    original = make_clt(skip_connection)
    inputs = torch.tensor([[[1, 2], [2, 3], [3, 4]]] * 2, dtype=torch.float64)
    expected = original.compute_attribution_components(inputs)["reconstruction"].to(dtype)
    original.to_safetensors(str(tmp_path / "original"))

    loaded = load_clt(
        str(tmp_path / "original"),
        lazy_encoder=lazy_encoder,
        lazy_decoder=lazy_decoder,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    assert loaded.skip_connection == skip_connection
    assert loaded.dtype == dtype
    assert loaded.device == torch.device("cpu")
    if skip_connection:
        assert loaded.W_skip is not None
        assert loaded.W_skip.dtype == dtype
        assert loaded.W_skip.device == torch.device("cpu")
        torch.testing.assert_close(loaded.W_skip, original.W_skip.to(dtype))
        for layer in range(original.n_layers):
            torch.testing.assert_close(
                loaded.compute_skip(layer, inputs[layer].to(dtype)),
                original.compute_skip(layer, inputs[layer]).to(dtype),
            )
    else:
        assert loaded.W_skip is None
    torch.testing.assert_close(
        loaded.compute_attribution_components(inputs.to(dtype))["reconstruction"], expected
    )

    # Re-saving a lazily loaded checkpoint must preserve the same learned function.
    loaded.to_safetensors(str(tmp_path / "resaved"))
    resaved = load_clt(
        str(tmp_path / "resaved"),
        lazy_encoder=False,
        lazy_decoder=False,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    assert resaved.skip_connection == skip_connection
    torch.testing.assert_close(
        resaved.compute_attribution_components(inputs.to(dtype))["reconstruction"], expected
    )


@pytest.mark.parametrize("missing_layer", [0, 1])
def test_checkpoint_rejects_inconsistent_skip_weights(tmp_path, make_clt, missing_layer):
    with torch.no_grad():
        make_clt(True).to_safetensors(str(tmp_path))
    # Build a partially affine checkpoint independently of the serializer.
    for layer in range(2):
        path = tmp_path / f"W_enc_{layer}.safetensors"
        tensors = load_file(str(path))
        if layer == missing_layer:
            tensors.pop(f"W_skip_{layer}", None)
        else:
            tensors[f"W_skip_{layer}"] = torch.eye(2)
        save_file(tensors, str(path))

    with pytest.raises(ValueError, match="Inconsistent skip weights"):
        load_clt(str(tmp_path), device=torch.device("cpu"))
