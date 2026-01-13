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
