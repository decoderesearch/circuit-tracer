"""Tests for TopK cross-layer transcoders (CLT) and checkpoint/config loading."""

from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from circuit_tracer.transcoder.activation_functions import TopK
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder, load_clt
from circuit_tracer.utils.caching import load_transcoders_from_cache, save_transcoders_to_cache
from circuit_tracer.utils.hf_utils import load_transcoders


def _make_topk_clt(
    n_layers: int = 2,
    d_transcoder: int = 4,
    d_model: int = 3,
    k: int | list[int] = 2,
    skip_connection: bool = False,
    dtype: torch.dtype = torch.float32,
) -> CrossLayerTranscoder:
    torch.manual_seed(0)
    clt = CrossLayerTranscoder(
        n_layers=n_layers,
        d_transcoder=d_transcoder,
        d_model=d_model,
        activation_function="topk",
        k=k,
        skip_connection=skip_connection,
        lazy_decoder=False,
        lazy_encoder=False,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    with torch.no_grad():
        clt.W_enc.copy_(torch.randn(n_layers, d_transcoder, d_model, dtype=dtype))
        clt.b_enc.copy_(torch.randn(n_layers, d_transcoder, dtype=dtype))
        clt.b_dec.copy_(torch.randn(n_layers, d_model, dtype=dtype))
        assert clt.W_dec is not None
        for layer_idx, dec_weight in enumerate(clt.W_dec):
            dec_weight.copy_(torch.randn(d_transcoder, n_layers - layer_idx, d_model, dtype=dtype))
        if skip_connection:
            assert clt.W_skip is not None
            clt.W_skip.copy_(torch.randn(n_layers, d_model, d_model, dtype=dtype))
    return clt


@torch.no_grad()
def test_topk_clt_encode_and_attribution_sparsity():
    clt = _make_topk_clt(n_layers=2, d_transcoder=5, d_model=3, k=2)
    assert isinstance(clt.activation_function, TopK)
    assert clt.activation_function.k == 2

    inputs = torch.randn(2, 4, 3, dtype=torch.float32)
    dense_acts = clt.encode(inputs)
    assert dense_acts.shape == (2, 4, 5)

    for layer_idx in range(2):
        layer_acts = clt.encode_layer(inputs[layer_idx], layer_idx)
        torch.testing.assert_close(layer_acts, dense_acts[layer_idx])
        assert torch.all((dense_acts[layer_idx] != 0).sum(dim=-1) <= 2)

    sparse_acts, encoder_vecs = clt.encode_sparse(inputs, zero_positions=slice(0, 1))
    dense_from_sparse = sparse_acts.to_dense()
    assert torch.all(dense_from_sparse[:, 0] == 0)
    torch.testing.assert_close(dense_from_sparse[:, 1:], dense_acts[:, 1:])
    assert encoder_vecs.shape == (sparse_acts._nnz(), 3)

    components = clt.compute_attribution_components(inputs, zero_positions=slice(0, 1))
    assert components["activation_matrix"]._nnz() == sparse_acts._nnz()
    assert components["reconstruction"].shape == (2, 4, 3)


@pytest.mark.parametrize("lazy_encoder", [False, True])
@pytest.mark.parametrize("lazy_decoder", [False, True])
@pytest.mark.parametrize("explicit_topk", [False, True])
@torch.no_grad()
def test_topk_clt_safetensors_roundtrip(
    tmp_path: Path, lazy_encoder: bool, lazy_decoder: bool, explicit_topk: bool
):
    original = _make_topk_clt(n_layers=2, d_transcoder=4, d_model=3, k=2, skip_connection=True)
    inputs = torch.randn(2, 3, 3, dtype=torch.float32)
    expected_acts = original.encode(inputs)
    expected_recon = original.compute_attribution_components(inputs)["reconstruction"]

    ckpt_dir = tmp_path / "topk_clt"
    original.to_safetensors(str(ckpt_dir))

    loaded = load_clt(
        str(ckpt_dir),
        device=torch.device("cpu"),
        dtype=torch.float32,
        lazy_encoder=lazy_encoder,
        lazy_decoder=lazy_decoder,
        activation_function="topk" if explicit_topk else None,
    )
    assert isinstance(loaded.activation_function, TopK)
    assert loaded.activation_function.k == 2
    torch.testing.assert_close(loaded.encode(inputs), expected_acts)
    torch.testing.assert_close(
        loaded.compute_attribution_components(inputs)["reconstruction"], expected_recon
    )


@torch.no_grad()
def test_topk_clt_per_layer_k_and_bare_k_checkpoint_key(tmp_path: Path):
    original = _make_topk_clt(n_layers=2, d_transcoder=5, d_model=3, k=[1, 3])
    assert isinstance(original.activation_function, torch.nn.ModuleList)

    inputs = torch.randn(2, 3, 3, dtype=torch.float32)
    acts = original.encode(inputs)
    assert torch.all((acts[0] != 0).sum(dim=-1) <= 1)
    assert torch.all((acts[1] != 0).sum(dim=-1) <= 3)

    ckpt_dir = tmp_path / "per_layer_topk"
    original.to_safetensors(str(ckpt_dir))

    # Rewrite k_i -> bare "k" to verify checkpoints using SingleLayerTranscoder's key name load too
    for layer_idx in range(2):
        enc_path = ckpt_dir / f"W_enc_{layer_idx}.safetensors"
        tensors = load_file(str(enc_path))
        tensors["k"] = tensors.pop(f"k_{layer_idx}")
        save_file(tensors, str(enc_path))

    loaded = load_clt(
        str(ckpt_dir),
        device=torch.device("cpu"),
        dtype=torch.float32,
        lazy_encoder=False,
        lazy_decoder=False,
        activation_function="topk",
    )
    torch.testing.assert_close(loaded.encode(inputs), acts)


@pytest.mark.parametrize("config_key", ["activation", "activation_function"])
@torch.no_grad()
def test_topk_clt_hub_and_cache_config_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_key: str
):
    original = _make_topk_clt(n_layers=2, d_transcoder=4, d_model=3, k=2)
    hub_dir = tmp_path / "hub_repo"
    original.to_safetensors(str(hub_dir))

    # Strip k_i from safetensors to verify config.yaml's activation/k alone configures TopK
    for layer_idx in range(2):
        enc_path = hub_dir / f"W_enc_{layer_idx}.safetensors"
        tensors = load_file(str(enc_path))
        tensors.pop(f"k_{layer_idx}")
        save_file(tensors, str(enc_path))

    config_data = {
        "model_kind": "cross_layer_transcoder",
        "feature_input_hook": "hook_resid_mid",
        "feature_output_hook": "hook_mlp_out",
        config_key: "topk",
        "k": 2,
    }
    config_file = hub_dir / "config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config_data, f)

    monkeypatch.setattr(
        "circuit_tracer.utils.hf_utils.snapshot_download", lambda *a, **kw: str(hub_dir)
    )
    monkeypatch.setattr(
        "circuit_tracer.utils.caching.snapshot_download", lambda *a, **kw: str(hub_dir)
    )
    monkeypatch.setattr(
        "circuit_tracer.utils.caching.hf_hub_download", lambda *a, **kw: str(config_file)
    )

    hub_loaded = load_transcoders(
        {**config_data, "repo_id": "org/topk-clt", "scan_name": "org/topk-clt"},
        device=torch.device("cpu"),
        dtype=torch.float32,
        lazy_encoder=False,
        lazy_decoder=False,
    )
    assert isinstance(hub_loaded, CrossLayerTranscoder)
    assert isinstance(hub_loaded.activation_function, TopK)
    assert hub_loaded.activation_function.k == 2

    cache_dir = tmp_path / "cache"
    save_transcoders_to_cache(
        "org/topk-clt",
        cache_dir=cache_dir,
        device=torch.device("cpu"),
        dtype=torch.float32,
        delete_hf_cache=False,
    )
    cached_clt, _ = load_transcoders_from_cache(
        "org/topk-clt",
        cache_dir=cache_dir,
        device=torch.device("cpu"),
        dtype=torch.float32,
        lazy_encoder=True,
        lazy_decoder=True,
    )
    assert isinstance(cached_clt, CrossLayerTranscoder)
    assert isinstance(cached_clt.activation_function, TopK)
    assert cached_clt.activation_function.k == 2

    inputs = torch.randn(2, 3, 3, dtype=torch.float32)
    torch.testing.assert_close(cached_clt.encode(inputs), original.encode(inputs))


def test_topk_clt_validation_errors(tmp_path: Path):
    with pytest.raises(ValueError, match="k must be provided"):
        CrossLayerTranscoder(2, 4, 3, activation_function="topk", device=torch.device("cpu"))

    with pytest.raises(ValueError, match="TopK k must be in"):
        CrossLayerTranscoder(2, 4, 3, activation_function="topk", k=0, device=torch.device("cpu"))

    with pytest.raises(ValueError, match="TopK k must be in"):
        CrossLayerTranscoder(2, 4, 3, activation_function="topk", k=5, device=torch.device("cpu"))

    with pytest.raises(ValueError, match="TopK k must be an int"):
        CrossLayerTranscoder(
            2, 4, 3, activation_function="topk", k=True, device=torch.device("cpu")
        )

    # Inconsistent k across checkpoint layers
    ckpt_dir = tmp_path / "bad_ckpt"
    _make_topk_clt(n_layers=2, d_transcoder=4, d_model=3, k=2).to_safetensors(str(ckpt_dir))
    enc_1 = ckpt_dir / "W_enc_1.safetensors"
    tensors_1 = load_file(str(enc_1))
    tensors_1.pop("k_1")
    save_file(tensors_1, str(enc_1))

    with pytest.raises(ValueError, match="Inconsistent TopK k values"):
        load_clt(str(ckpt_dir), device=torch.device("cpu"))
