import gc
import os
import tempfile

import pytest
import torch
from safetensors.torch import save_file

from circuit_tracer.transcoder.cross_layer_transcoder import (
    load_clt,
)


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


@pytest.fixture
def create_test_clt_files():
    """Create temporary CLT safetensors files for testing."""

    def _create_files(n_layers=4, d_model=128, d_transcoder=512, skip_connection=False):
        tmpdir = tempfile.mkdtemp()

        # Create encoder and decoder files for each layer
        for i in range(n_layers):
            # Encoder weights and biases
            enc_dict = {
                f"W_enc_{i}": torch.randn(d_transcoder, d_model),
                f"b_enc_{i}": torch.randn(d_transcoder),
                f"b_dec_{i}": torch.randn(d_model),
            }
            enc_path = os.path.join(tmpdir, f"W_enc_{i}.safetensors")
            save_file(enc_dict, enc_path)

            # Decoder weights - shape depends on source layer
            dec_dict = {f"W_dec_{i}": torch.randn(d_transcoder, n_layers - i, d_model)}
            dec_path = os.path.join(tmpdir, f"W_dec_{i}.safetensors")
            save_file(dec_dict, dec_path)

        if skip_connection:
            skip_dict = {"W_skip": torch.randn(d_model, d_model)}
            skip_path = os.path.join(tmpdir, "W_skip.safetensors")
            save_file(skip_dict, skip_path)

        return tmpdir

    return _create_files


# === Attribution Tests ===


@pytest.mark.parametrize("skip_connection", [False, True])
def test_compute_attribution_components(create_test_clt_files, skip_connection):
    """Test the main attribution functionality of CLT."""
    clt_path = create_test_clt_files(skip_connection=skip_connection)
    clt = load_clt(
        clt_path,
        device=torch.device("cpu"),
        lazy_encoder=True,
        lazy_decoder=True,
    )

    # Create test input
    n_pos = 10
    inputs = torch.randn(clt.n_layers, n_pos, clt.d_model, dtype=clt.b_enc.dtype)

    # Compute attribution components
    components = clt.compute_attribution_components(inputs, zero_positions=slice(0, 1))

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
    assert act_matrix.shape == (clt.n_layers, n_pos, clt.d_transcoder)

    # Check reconstruction (only positions 1 and beyond)
    reconstruction = components["reconstruction"]
    assert reconstruction.shape == (clt.n_layers, n_pos, clt.d_model)
    assert torch.allclose(reconstruction[:, 1:], clt(inputs)[:, 1:])

    # Check encoder/decoder vectors have consistent counts
    n_active_encoders = act_matrix._nnz()
    assert components["encoder_vecs"].shape[0] == n_active_encoders

    # Decoder count should be >= encoder count due to cross-layer writing
    assert components["decoder_vecs"].shape[0] >= n_active_encoders
    assert components["decoder_vecs"].shape[1] == clt.d_model

    # Check decoder locations
    decoder_locs = components["decoder_locations"]
    assert decoder_locs.shape[0] == 2


def test_encode_sparse_with_lazy_encoder(create_test_clt_files):
    """Test sparse encoding with lazy encoder loading."""
    clt_path = create_test_clt_files()

    # Test with lazy encoder
    lazy_clt = load_clt(
        clt_path,
        device=torch.device("cpu"),
        lazy_encoder=True,
        lazy_decoder=False,
    )

    # Test with eager encoder for comparison
    eager_clt = load_clt(
        clt_path,
        device=torch.device("cpu"),
        lazy_encoder=False,
        lazy_decoder=False,
    )

    # Create test input
    n_pos = 10
    inputs = torch.randn(lazy_clt.n_layers, n_pos, lazy_clt.d_model, dtype=eager_clt.b_enc.dtype)

    # Encode sparse with both
    lazy_sparse, lazy_encoders = lazy_clt.encode_sparse(inputs)
    eager_sparse, eager_encoders = eager_clt.encode_sparse(inputs)

    # Results should be identical
    assert torch.allclose(lazy_sparse.to_dense(), eager_sparse.to_dense())
    assert torch.allclose(lazy_encoders, eager_encoders)

    # Check that first position is zeroed
    assert lazy_sparse.to_dense()[:, 0].sum() == 0

    # Check shapes
    assert lazy_sparse.shape == (lazy_clt.n_layers, n_pos, lazy_clt.d_transcoder)
    assert lazy_encoders.shape[0] == lazy_sparse._nnz()
    assert lazy_encoders.shape[1] == lazy_clt.d_model


def test_decoder_slice_access(create_test_clt_files):
    """Test _get_decoder_vectors with lazy and eager decoder."""
    clt_path = create_test_clt_files()

    # Test with different decoder loading modes
    lazy_clt = load_clt(
        clt_path,
        device=torch.device("cpu"),
        lazy_encoder=False,
        lazy_decoder=True,
    )

    eager_clt = load_clt(
        clt_path,
        device=torch.device("cpu"),
        lazy_encoder=False,
        lazy_decoder=False,
    )

    # Test decoder access for layer 0 (writes to all layers)
    layer_id = 0

    # Full access
    lazy_full = lazy_clt._get_decoder_vectors(layer_id)
    eager_full = eager_clt._get_decoder_vectors(layer_id)
    assert torch.allclose(lazy_full, eager_full)
    assert lazy_full.shape == (
        lazy_clt.d_transcoder,
        lazy_clt.n_layers,
        lazy_clt.d_model,
    )

    # Slice access
    feat_ids = torch.tensor([10, 50, 100, 200])
    lazy_slice = lazy_clt._get_decoder_vectors(layer_id, feat_ids.numpy())
    eager_slice = eager_clt._get_decoder_vectors(layer_id, feat_ids.numpy())
    assert torch.allclose(lazy_slice, eager_slice)
    assert lazy_slice.shape == (len(feat_ids), lazy_clt.n_layers, lazy_clt.d_model)

    # Test decoder for last layer (writes to only itself)
    layer_id = lazy_clt.n_layers - 1
    lazy_last = lazy_clt._get_decoder_vectors(layer_id)
    eager_last = eager_clt._get_decoder_vectors(layer_id)
    assert torch.allclose(lazy_last, eager_last)
    assert lazy_last.shape == (lazy_clt.d_transcoder, 1, lazy_clt.d_model)


def test_encode_operations_with_lazy_loading(create_test_clt_files):
    """Test encode operations with different lazy loading configurations."""
    clt_path = create_test_clt_files()

    # Create test input based on first CLT's dimensions
    test_clt = load_clt(clt_path, device=torch.device("cpu"))
    n_pos = 10
    inputs = torch.randn(test_clt.n_layers, n_pos, test_clt.d_model, dtype=test_clt.b_enc.dtype)

    # Test all lazy/eager combinations
    configs = [
        (False, False),  # Both eager
        (True, False),  # Lazy encoder
        (False, True),  # Lazy decoder
        (True, True),  # Both lazy
    ]

    outputs = []
    for lazy_enc, lazy_dec in configs:
        clt = load_clt(
            clt_path,
            device=torch.device("cpu"),
            lazy_encoder=lazy_enc,
            lazy_decoder=lazy_dec,
        )

        # Test regular encode
        encoded = clt.encode(inputs)
        assert encoded.shape == (clt.n_layers, n_pos, clt.d_transcoder)

        # Test single layer encode
        layer_encoded = clt.encode_layer(inputs[0], 0)
        assert layer_encoded.shape == (n_pos, clt.d_transcoder)

        outputs.append((encoded, layer_encoded))

    # All outputs should be identical regardless of lazy loading
    for i in range(1, len(outputs)):
        assert torch.allclose(outputs[0][0], outputs[i][0])
        assert torch.allclose(outputs[0][1], outputs[i][1])


def test_cross_layer_decoder_structure(create_test_clt_files):
    """Test CLT-specific cross-layer decoder structure."""
    clt_path = create_test_clt_files()
    clt = load_clt(
        clt_path,
        device=torch.device("cpu"),
        lazy_encoder=True,
        lazy_decoder=True,
    )

    # Create test input with known sparsity pattern
    n_pos = 5
    inputs = torch.randn(clt.n_layers, n_pos, clt.d_model, dtype=clt.b_enc.dtype)

    # Get sparse features
    features, _ = clt.encode_sparse(inputs)

    # Test decoder vector selection
    pos_ids, layer_ids, feat_ids, decoder_vectors, encoder_mapping = clt.select_decoder_vectors(
        features
    )

    # Verify cross-layer structure
    # Features from layer i should write to layers i through n_layers-1
    enc_layers, _, _ = features.indices()
    unique_enc_layers = enc_layers.unique()

    for source_layer in unique_enc_layers:
        # Find decoder entries from this source layer
        source_mask = enc_layers == source_layer
        n_source_features = source_mask.sum()

        # Each source feature should write to (n_layers - source_layer) output layers
        expected_decoder_count = n_source_features * (clt.n_layers - source_layer)

        # Count actual decoder entries
        source_positions = torch.where(source_mask)[0]
        decoder_from_source = sum(encoder_mapping == pos for pos in source_positions).sum()  # type: ignore

        assert decoder_from_source == expected_decoder_count


def test_forward_pass_consistency(create_test_clt_files):
    """Test that forward pass produces consistent results across lazy modes."""
    clt_path = create_test_clt_files(n_layers=3, d_model=64, d_transcoder=256)

    # Create CLTs with different configurations
    eager_clt = load_clt(
        clt_path, device=torch.device("cpu"), lazy_encoder=False, lazy_decoder=False
    )
    lazy_clt = load_clt(clt_path, device=torch.device("cpu"), lazy_encoder=True, lazy_decoder=True)

    # Test input
    n_pos = 8
    inputs = torch.randn(eager_clt.n_layers, n_pos, eager_clt.d_model, dtype=eager_clt.b_enc.dtype)

    # Forward pass
    eager_output = eager_clt(inputs)
    lazy_output = lazy_clt(inputs)

    # Outputs should be identical
    assert torch.allclose(eager_output, lazy_output, rtol=1e-5)
    assert eager_output.shape == (eager_clt.n_layers, n_pos, eager_clt.d_model)
