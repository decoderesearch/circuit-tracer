import gc
import os
import shutil
from pathlib import Path

import pytest
import torch

from circuit_tracer.transcoder import TranscoderSet
from circuit_tracer.utils.caching import (
    empty_cache,
    get_cached_path,
    is_cached,
    save_transcoders_to_cache,
)
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

HF_REF = "mntss/gemma-scope-transcoders"
TEST_CACHE_DIR = Path.home() / ".cache" / "circuit-tracer-test"


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


@pytest.fixture(autouse=True)
def cleanup_cache():
    yield
    empty_cache(HF_REF)


@pytest.mark.requires_disk
def test_caching_enables_lazy_loading():
    # 1. Load from hub without cache - lazy loading should not work because
    # gemma-scope transcoders use npz format which doesn't support lazy loading
    transcoder_set, _ = load_transcoder_from_hub(
        HF_REF,
        device=torch.device("cpu"),
        dtype=torch.float32,
        lazy_encoder=True,
        lazy_decoder=True,
        use_cache=False,
    )

    assert isinstance(transcoder_set, TranscoderSet)

    # Verify lazy loading is NOT enabled (npz format doesn't support it)
    for transcoder in transcoder_set:
        assert transcoder.lazy_encoder is False
        assert transcoder.lazy_decoder is False

    # 2. Save to cache with sequential=False
    save_transcoders_to_cache(
        HF_REF,
        sequential=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert is_cached(HF_REF)

    # 3. Load from cache - lazy loading should now work
    transcoder_set_cached, _ = load_transcoder_from_hub(
        HF_REF,
        device=torch.device("cpu"),
        dtype=torch.float32,
        lazy_encoder=True,
        lazy_decoder=True,
    )

    assert isinstance(transcoder_set_cached, TranscoderSet)

    # Verify lazy loading IS enabled when loaded from cache
    for transcoder in transcoder_set_cached:
        assert transcoder.lazy_encoder is True
        assert transcoder.lazy_decoder is True

    # 4. Cleanup is handled by fixture


@pytest.mark.requires_disk
def test_custom_cache_directory():
    try:
        # Ensure test cache dir doesn't exist initially
        if TEST_CACHE_DIR.exists():
            shutil.rmtree(TEST_CACHE_DIR)

        # Save to custom cache directory
        save_transcoders_to_cache(
            HF_REF,
            cache_dir=TEST_CACHE_DIR,
            sequential=False,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # Verify cached in the right location
        expected_path = get_cached_path(HF_REF, cache_dir=TEST_CACHE_DIR)
        assert expected_path.exists()
        assert TEST_CACHE_DIR in expected_path.parents
        assert is_cached(HF_REF, cache_dir=TEST_CACHE_DIR)

        # Verify config and layer files exist
        assert (expected_path / "config.yaml").exists()
        layer_files = list(expected_path.glob("layer_*.safetensors"))
        assert len(layer_files) > 0

        # Delete just this model from cache
        empty_cache(HF_REF, cache_dir=TEST_CACHE_DIR)
        assert not expected_path.exists()
        assert not is_cached(HF_REF, cache_dir=TEST_CACHE_DIR)

    finally:
        # Always clean up test cache directory
        if TEST_CACHE_DIR.exists():
            shutil.rmtree(TEST_CACHE_DIR)


@pytest.mark.requires_disk
def test_cache_directory_from_env_var():
    env_cache_dir = Path.home() / ".cache" / "circuit-tracer-env-test"
    old_env = os.environ.get("CIRCUIT_TRACER_CACHE_DIR")

    try:
        # Ensure test cache dir doesn't exist initially
        if env_cache_dir.exists():
            shutil.rmtree(env_cache_dir)

        # Set environment variable
        os.environ["CIRCUIT_TRACER_CACHE_DIR"] = str(env_cache_dir)

        # Save to cache (should use env var)
        save_transcoders_to_cache(
            HF_REF,
            sequential=False,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # Verify cached in the right location (env var path)
        expected_path = get_cached_path(HF_REF)
        assert expected_path.exists()
        assert env_cache_dir in expected_path.parents
        assert is_cached(HF_REF)

        # Verify config and layer files exist
        assert (expected_path / "config.yaml").exists()
        layer_files = list(expected_path.glob("layer_*.safetensors"))
        assert len(layer_files) > 0

        # Delete using empty_cache (should also use env var)
        empty_cache(HF_REF)
        assert not expected_path.exists()
        assert not is_cached(HF_REF)

    finally:
        # Restore original env var
        if old_env is None:
            os.environ.pop("CIRCUIT_TRACER_CACHE_DIR", None)
        else:
            os.environ["CIRCUIT_TRACER_CACHE_DIR"] = old_env

        # Always clean up test cache directory
        if env_cache_dir.exists():
            shutil.rmtree(env_cache_dir)


def _write_layers(directory: Path, n_layers: int = 2, d_model: int = 8, d_transcoder: int = 16):
    from safetensors.torch import save_file

    for i in range(n_layers):
        save_file(
            {
                "W_enc": torch.randn(d_transcoder, d_model),
                "W_dec": torch.randn(d_transcoder, d_model),
                "b_enc": torch.zeros(d_transcoder),
                "b_dec": torch.zeros(d_model),
            },
            str(directory / f"layer_{i}.safetensors"),
        )


def _write_config(directory: Path, **extra):
    import yaml

    config = {
        "model_kind": "transcoder_set",
        "feature_input_hook": "mlp.hook_in",
        "feature_output_hook": "mlp.hook_out",
        **extra,
    }
    with open(directory / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)


def test_local_folder_keeps_topk_activation(tmp_path):
    """activation/k in config.yaml must survive the cache / local-folder loading path."""
    from circuit_tracer.transcoder.activation_functions import TopK

    folder = tmp_path / "my-topk-transcoders"
    folder.mkdir()
    _write_layers(folder)
    _write_config(folder, activation="topk", k=2)

    transcoders, _ = load_transcoder_from_hub(
        "my-topk-transcoders",
        cache_dir=str(tmp_path),
        device=torch.device("cpu"),
        lazy_encoder=False,
        lazy_decoder=False,
    )

    assert isinstance(transcoders, TranscoderSet)
    for transcoder in transcoders:
        assert isinstance(transcoder.activation_function, TopK)
        assert transcoder.activation_function.k == 2
    acts = transcoders[0].encode(torch.randn(5, 8))
    assert ((acts > 0).sum(-1) <= 2).all()


def test_topk_transcoder_set_to_safetensors_round_trip(tmp_path):
    """A top-k set saved with to_safetensors, with activation and k in config.yaml, loads back as top-k with the same outputs."""
    from circuit_tracer.transcoder.activation_functions import TopK
    from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder_set

    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    _write_layers(src)
    paths = {i: str(src / f"layer_{i}.safetensors") for i in range(2)}
    original = load_transcoder_set(
        paths,
        scan_name="test",
        feature_input_hook="mlp.hook_in",
        feature_output_hook="mlp.hook_out",
        device=torch.device("cpu"),
        activation="topk",
        k=3,
        lazy_encoder=False,
        lazy_decoder=False,
    )
    original.to_safetensors(str(out))
    _write_config(out, activation="topk", k=3)  # config.yaml is the source of activation and k

    reloaded, _ = load_transcoder_from_hub(
        "out",
        cache_dir=str(tmp_path),
        device=torch.device("cpu"),
        lazy_encoder=False,
        lazy_decoder=False,
    )

    assert isinstance(reloaded, TranscoderSet)
    x = torch.randn(4, 8)
    for a, b in zip(original, reloaded):
        assert isinstance(b.activation_function, TopK) and b.activation_function.k == 3
        torch.testing.assert_close(a.encode(x), b.encode(x))


def test_save_transcoders_to_cache_keeps_topk(tmp_path, monkeypatch):
    """The Hub -> cache -> load round trip keeps a top-k set's activation (as for crv-8b-instruct)."""
    import circuit_tracer.utils.caching as caching
    import circuit_tracer.utils.hf_utils as hf_utils
    from circuit_tracer.transcoder.activation_functions import TopK

    hub = tmp_path / "hub"
    hub.mkdir()
    _write_layers(hub)
    _write_config(hub, activation="topk", k=2)
    monkeypatch.setattr(caching, "hf_hub_download", lambda **kw: str(hub / "config.yaml"))
    monkeypatch.setattr(hf_utils, "snapshot_download", lambda *a, **kw: str(hub))

    cache_dir = tmp_path / "cache"
    save_transcoders_to_cache("org/topk-test", cache_dir=str(cache_dir), delete_hf_cache=False)
    assert is_cached("org/topk-test", str(cache_dir))

    transcoders, _ = load_transcoder_from_hub(
        "org/topk-test",
        cache_dir=str(cache_dir),
        device=torch.device("cpu"),
        lazy_encoder=False,
        lazy_decoder=False,
    )
    assert isinstance(transcoders, TranscoderSet)
    for transcoder in transcoders:
        assert isinstance(transcoder.activation_function, TopK)
        assert transcoder.activation_function.k == 2
