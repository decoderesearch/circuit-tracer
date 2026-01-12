import gc
import os
import tempfile

import pytest
import torch
from safetensors.torch import save_file

from circuit_tracer.transcoder.single_layer_transcoder import (
    load_relu_transcoder,
    load_transcoder_set,
)


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


@pytest.fixture
def create_test_transcoder_file():
    """Create a temporary transcoder safetensors file for testing."""

    def _create_file(d_model=128, d_sae=512, skip_connection=False):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            W_enc = torch.randn(d_sae, d_model)
            W_dec = torch.randn(d_sae, d_model)
            b_enc = torch.randn(d_sae)
            b_dec = torch.randn(d_model)

            state_dict = {
                "W_enc": W_enc,
                "W_dec": W_dec,
                "b_enc": b_enc,
                "b_dec": b_dec,
            }

            if skip_connection:
                state_dict["W_skip"] = torch.randn(d_model, d_model)

            save_file(state_dict, f.name)
            return f.name, state_dict

    files_to_cleanup = []

    def _create_and_track(*args, **kwargs):
        path, state_dict = _create_file(*args, **kwargs)
        files_to_cleanup.append(path)
        return path, state_dict

    yield _create_and_track

    # Cleanup
    for path in files_to_cleanup:
        if os.path.exists(path):
            os.unlink(path)


# === Attribution Tests ===


@pytest.mark.parametrize("skip_connection", [False, True])
def test_transcoder_set_attribution_components(create_test_transcoder_file, skip_connection):
    """Test compute_attribution_components functionality."""
    # Create test files for multiple layers
    n_layers = 3
    paths = {}
    for layer in range(n_layers):
        path, _ = create_test_transcoder_file(
            d_model=128, d_sae=512, skip_connection=skip_connection
        )
        paths[layer] = path

    transcoder_set = load_transcoder_set(
        transcoder_paths=paths,
        scan="test_scan",
        feature_input_hook="hook_resid_mid",
        feature_output_hook="hook_mlp_out",
        device=torch.device("cpu"),
        lazy_encoder=False,
        lazy_decoder=True,
    )

    # Create test MLP inputs
    n_pos = 10
    d_model = 128
    mlp_inputs = torch.randn(n_layers, n_pos, d_model)

    # Compute attribution components
    components = transcoder_set.compute_attribution_components(
        mlp_inputs, zero_positions=slice(0, 1)
    )

    # Verify all required components are present
    assert "activation_matrix" in components
    assert "reconstruction" in components
    assert "encoder_vecs" in components
    assert "decoder_vecs" in components
    assert "encoder_to_decoder_map" in components
    assert "decoder_locations" in components

    # Check activation matrix
    act_matrix = components["activation_matrix"]
    assert act_matrix.is_sparse
    assert act_matrix.shape == (n_layers, n_pos, 512)

    # Check reconstruction (only positions 1 and beyond)
    reconstruction = components["reconstruction"]
    assert reconstruction.shape == (n_layers, n_pos, d_model)
    for layer, transcoder in enumerate(transcoder_set.transcoders):
        assert torch.allclose(
            reconstruction[layer, 1:], transcoder(mlp_inputs[layer])[1:], rtol=1e-4, atol=1e-4
        )

    # Check encoder/decoder vectors have matching counts
    n_active = act_matrix._nnz()
    assert components["encoder_vecs"].shape[0] == n_active
    assert components["decoder_vecs"].shape[0] == n_active
    assert components["encoder_to_decoder_map"].shape[0] == n_active

    # Check decoder locations
    decoder_locs = components["decoder_locations"]
    assert decoder_locs.shape == (2, n_active)


def test_sparse_encode_decode(create_test_transcoder_file):
    """Test sparse encoding and decoding functionality."""
    path, _ = create_test_transcoder_file(d_model=128, d_sae=512)

    transcoder = load_relu_transcoder(
        path, layer=0, device=torch.device("cpu"), lazy_encoder=False, lazy_decoder=False
    )

    # Create test input
    n_pos = 10
    d_model = 128
    test_input = torch.randn(n_pos, d_model)

    # Test sparse encoding
    sparse_acts, active_encoders = transcoder.encode_sparse(test_input)

    # Check that first position is zeroed
    assert sparse_acts[0].sum() == 0

    # Check sparse format
    assert sparse_acts.is_sparse
    assert sparse_acts.shape == (n_pos, 512)

    # Test sparse decoding
    reconstruction, scaled_decoders = transcoder.decode_sparse(sparse_acts)

    assert reconstruction.shape == (n_pos, d_model)
    assert len(scaled_decoders) == sparse_acts._nnz()


def test_decoder_slice_access(create_test_transcoder_file):
    """Test _get_decoder_vectors with both lazy and eager decoder."""
    path, expected_state = create_test_transcoder_file(d_model=128, d_sae=512)

    # Test eager decoder
    eager_transcoder = load_relu_transcoder(
        path, layer=0, lazy_encoder=True, lazy_decoder=False, device=torch.device("cpu")
    )

    # Test lazy decoder
    lazy_transcoder = load_relu_transcoder(
        path, layer=0, lazy_encoder=True, lazy_decoder=True, device=torch.device("cpu")
    )

    # Test full access
    eager_full = eager_transcoder._get_decoder_vectors()
    lazy_full = lazy_transcoder._get_decoder_vectors()
    assert torch.allclose(eager_full, lazy_full)
    assert torch.allclose(eager_full, expected_state["W_dec"])

    # Test slice access
    indices = torch.tensor([10, 50, 100, 200])
    eager_slice = eager_transcoder._get_decoder_vectors(indices)
    lazy_slice = lazy_transcoder._get_decoder_vectors(indices)
    assert torch.allclose(eager_slice, lazy_slice)
    assert torch.allclose(eager_slice, expected_state["W_dec"][indices])


def test_encode_decode_operations(create_test_transcoder_file):
    """Test that encode/decode operations work with lazy loading."""
    path, expected_state = create_test_transcoder_file(d_model=128, d_sae=512)

    # Create test input
    batch_size = 2
    seq_len = 10
    d_model = 128
    test_input = torch.randn(batch_size * seq_len, d_model)

    # Test all combinations
    configs = [
        (False, False),  # Both eager
        (True, False),  # Lazy encoder
        (False, True),  # Lazy decoder
        (True, True),  # Both lazy
    ]

    outputs = []
    for lazy_enc, lazy_dec in configs:
        transcoder = load_relu_transcoder(
            path, layer=0, lazy_encoder=lazy_enc, lazy_decoder=lazy_dec, device=torch.device("cpu")
        )

        # Test encode
        encoded = transcoder.encode(test_input)
        assert encoded.shape == (batch_size * seq_len, 512)

        # Test decode
        decoded = transcoder.decode(encoded)
        assert decoded.shape == (batch_size * seq_len, d_model)

        outputs.append((encoded, decoded))

    # All outputs should be identical regardless of lazy loading
    for i in range(1, len(outputs)):
        assert torch.allclose(outputs[0][0], outputs[i][0])  # encoded
        assert torch.allclose(outputs[0][1], outputs[i][1])  # decoded


# === Lazy Loading Tests ===


def test_lazy_encoder_only(create_test_transcoder_file):
    """Test lazy loading of encoder while decoder is eager."""
    path, expected_state = create_test_transcoder_file()

    transcoder = load_relu_transcoder(
        path, layer=0, lazy_encoder=True, lazy_decoder=False, device=torch.device("cpu")
    )

    # W_dec should be a parameter, W_enc should not exist as attribute
    assert hasattr(transcoder, "W_dec")
    assert isinstance(transcoder.W_dec, torch.nn.Parameter)
    assert "W_enc" not in transcoder._parameters

    # Accessing W_enc should trigger lazy loading
    W_enc = transcoder.W_enc
    assert W_enc.shape == expected_state["W_enc"].shape
    assert torch.allclose(W_enc, expected_state["W_enc"])

    # W_enc should NOT become a parameter after access
    assert "W_enc" not in transcoder._parameters


def test_lazy_decoder_only(create_test_transcoder_file):
    """Test lazy loading of decoder while encoder is eager."""
    path, expected_state = create_test_transcoder_file()

    transcoder = load_relu_transcoder(
        path, layer=0, lazy_encoder=False, lazy_decoder=True, device=torch.device("cpu")
    )

    # W_enc should be a parameter, W_dec should not exist as attribute
    assert hasattr(transcoder, "W_enc")
    assert isinstance(transcoder.W_enc, torch.nn.Parameter)
    assert "W_dec" not in transcoder._parameters

    # Accessing W_dec should trigger lazy loading
    W_dec = transcoder.W_dec
    assert W_dec.shape == expected_state["W_dec"].shape
    assert torch.allclose(W_dec, expected_state["W_dec"])

    # W_dec should NOT become a parameter after access
    assert "W_dec" not in transcoder._parameters


def test_lazy_loading_both(create_test_transcoder_file):
    """Test lazy loading of both encoder and decoder."""
    path, expected_state = create_test_transcoder_file()

    transcoder = load_relu_transcoder(
        path, layer=0, lazy_encoder=True, lazy_decoder=True, device=torch.device("cpu")
    )

    # Neither weight should exist as parameters
    assert "W_enc" not in transcoder._parameters
    assert "W_dec" not in transcoder._parameters

    # Accessing weights should trigger lazy loading
    W_enc = transcoder.W_enc
    W_dec = transcoder.W_dec

    assert W_enc.shape == expected_state["W_enc"].shape
    assert W_dec.shape == expected_state["W_dec"].shape
    assert torch.allclose(W_enc, expected_state["W_enc"])
    assert torch.allclose(W_dec, expected_state["W_dec"])
