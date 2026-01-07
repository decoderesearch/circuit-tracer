from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Literal

import torch
import yaml
from huggingface_hub import hf_hub_download, snapshot_download

from circuit_tracer.transcoder.cross_layer_transcoder import load_clt, load_gemma_scope_2_clt
from circuit_tracer.transcoder.single_layer_transcoder import (
    load_gemma_scope_2_transcoder,
    load_gemma_scope_transcoder,
    load_relu_transcoder,
    load_transcoder_set,
)
from circuit_tracer.utils.hf_utils import (
    HfUri,
    download_hf_uri,
    download_hf_uris,
    iter_transcoder_paths,
    parse_hf_uri,
)

logger = logging.getLogger(__name__)


def get_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Get the cache directory for circuit tracer.

    Priority:
    1. Explicit cache_dir parameter
    2. CIRCUIT_TRACER_CACHE_DIR environment variable
    3. ~/.cache/circuit_tracer
    """
    if cache_dir is not None:
        return Path(cache_dir)
    env_dir = os.environ.get("CIRCUIT_TRACER_CACHE_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".cache" / "circuit_tracer"


def _normalize_hf_ref(hf_ref: str) -> str:
    """Normalize an hf_ref to a filesystem-safe path component."""
    if hf_ref == "gemma":
        hf_ref = "mwhanna/gemma-scope-transcoders"
    elif hf_ref == "llama":
        hf_ref = "mntss/transcoder-Llama-3.2-1B"

    # Handle hf:// URIs by extracting the path portion
    if hf_ref.startswith("hf://"):
        uri = parse_hf_uri(hf_ref)
        normalized = uri.repo_id
        if uri.file_path:
            normalized = f"{normalized}/{uri.file_path}"
        if uri.revision:
            normalized = f"{normalized}@{uri.revision}"
        return normalized

    return hf_ref


def get_cached_path(hf_ref: str, cache_dir: str | Path | None = None) -> Path:
    """Get the cached path for an hf_ref."""
    cache_base = get_cache_dir(cache_dir)
    normalized = _normalize_hf_ref(hf_ref)
    return cache_base / normalized


def is_cached(hf_ref: str, cache_dir: str | Path | None = None) -> bool:
    """Check if transcoders for an hf_ref are cached and complete.

    Checks for the presence of config.yaml which is only saved after
    all transcoder files have been successfully downloaded and converted.
    """
    cache_path = get_cached_path(hf_ref, cache_dir)
    config_path = cache_path / "config.yaml"
    return config_path.exists()


def empty_cache(hf_ref: str | None = None, cache_dir: str | Path | None = None):
    """Delete cached transcoders.

    Args:
        hf_ref: If provided, delete only the cache for this specific model.
                If None, delete the entire cache directory contents.
        cache_dir: Override the cache directory
    """
    cache_base = get_cache_dir(cache_dir)
    if hf_ref is not None:
        cache_path = get_cached_path(hf_ref, cache_dir)
        if cache_path.exists():
            shutil.rmtree(cache_path)
            logger.info(f"Deleted cache for {hf_ref} at {cache_path}")
    else:
        if cache_base.exists():
            shutil.rmtree(cache_base)
            logger.info(f"Deleted entire cache at {cache_base}")


def _delete_hf_cache(path: str | Path):
    """Delete a cached HuggingFace file (handles symlinks to blobs)."""
    path = Path(path)
    if path.is_symlink():
        blob_path = path.resolve()
        path.unlink()
        if blob_path.exists():
            blob_path.unlink()
    elif path.exists():
        path.unlink()


def save_transcoders_to_cache(
    hf_ref: str,
    cache_dir: str | Path | None = None,
    sequential: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    delete_hf_cache: bool = True,
) -> Path:
    """Download transcoders from HuggingFace and save them to the local cache.

    Args:
        hf_ref: HuggingFace reference (e.g., "user/repo/subfolder@revision")
        cache_dir: Override the cache directory (defaults to CIRCUIT_TRACER_CACHE_DIR or ~/.cache/circuit_tracer)
        sequential: If True, download one transcoder at a time, convert, save, then delete HF cache.
                   If False, download all transcoders first, then convert.
        device: Device to use for loading transcoders during conversion (default: CPU)
        dtype: Data type for transcoder weights
        delete_hf_cache: If True and sequential=True, delete the HF cache after saving each transcoder

    Returns:
        Path to the cache directory containing the saved transcoders
    """
    if device is None:
        device = torch.device("cpu")

    if hf_ref == "gemma":
        hf_ref = "mwhanna/gemma-scope-transcoders"
    elif hf_ref == "llama":
        hf_ref = "mntss/transcoder-Llama-3.2-1B"

    hf_uri = HfUri.from_str(hf_ref)

    # First download and parse the config
    config_path = hf_hub_download(
        repo_id=hf_uri.repo_id,
        revision=hf_uri.revision,
        filename="config.yaml",
        subfolder=hf_uri.file_path,
    )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["repo_id"] = hf_uri.repo_id
    config["revision"] = hf_uri.revision
    config["subfolder"] = hf_uri.file_path
    repo_info = (
        hf_uri.repo_id if hf_uri.file_path is None else hf_uri.repo_id + "//" + hf_uri.file_path
    )
    config["scan"] = f"{repo_info}@{hf_uri.revision}" if hf_uri.revision else repo_info

    model_kind = config["model_kind"]
    cache_path = get_cached_path(hf_ref, cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    repo_id = config.get("repo_id", "")
    if "gemma-scope-2" in repo_id and "transcoders" in config:
        special_load_fn: Literal["gemma-scope", "gemma-scope-2", None] = "gemma-scope-2"
    elif "gemma-scope" in repo_id and "transcoders" in config:
        special_load_fn = "gemma-scope"
    else:
        special_load_fn = None

    if model_kind == "transcoder_set":
        _save_transcoder_set_to_cache(
            config=config,
            cache_path=cache_path,
            sequential=sequential,
            device=device,
            dtype=dtype,
            special_load_fn=special_load_fn,
            delete_hf_cache=delete_hf_cache,
        )
    elif model_kind == "cross_layer_transcoder":
        _save_clt_to_cache(
            config=config,
            cache_path=cache_path,
            sequential=sequential,
            device=device,
            dtype=dtype,
            special_load_fn=special_load_fn,
            delete_hf_cache=delete_hf_cache,
        )
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")

    # Save config only after all transcoders have been successfully saved
    simplified_config = {
        k: v for k, v in config.items() if k not in ("transcoders", "repo_id", "subfolder")
    }
    with open(cache_path / "config.yaml", "w") as f:
        yaml.dump(simplified_config, f)

    logger.info(f"Successfully cached transcoders to {cache_path}")
    return cache_path


def _save_transcoder_set_to_cache(
    config: dict,
    cache_path: Path,
    sequential: bool,
    device: torch.device,
    dtype: torch.dtype,
    special_load_fn: Literal["gemma-scope", "gemma-scope-2", None],
    delete_hf_cache: bool,
):
    """Save a transcoder set to the cache."""
    if "transcoders" not in config:
        # Uses snapshot_download pattern - iterate through paths
        for layer_idx, local_path in iter_transcoder_paths(config):
            transcoder = load_relu_transcoder(
                local_path,
                layer_idx,
                device=device,
                dtype=dtype,
                lazy_encoder=False,
                lazy_decoder=False,
            )
            save_path = cache_path / f"layer_{layer_idx}.safetensors"
            transcoder.to_safetensors(str(save_path))
            logger.info(f"Saved layer {layer_idx} to {save_path}")
        return

    if sequential:
        for layer_idx, hf_path in enumerate(config["transcoders"]):
            if hf_path.startswith("hf://"):
                local_path = download_hf_uri(hf_path)
            else:
                local_path = hf_path

            npz_format = Path(local_path).suffix == ".npz"

            if special_load_fn == "gemma-scope" and npz_format:
                load_fn = load_gemma_scope_transcoder
            elif special_load_fn == "gemma-scope-2":
                load_fn = load_gemma_scope_2_transcoder
            else:
                load_fn = load_relu_transcoder

            transcoder = load_fn(
                local_path,
                layer_idx,
                device=device,
                dtype=dtype,
                lazy_encoder=False,
                lazy_decoder=False,
            )

            save_path = cache_path / f"layer_{layer_idx}.safetensors"
            transcoder.to_safetensors(str(save_path))
            logger.info(f"Saved layer {layer_idx} to {save_path}")

            if delete_hf_cache and hf_path.startswith("hf://"):
                _delete_hf_cache(local_path)
    else:
        # Download all at once
        hf_paths = [p for p in config["transcoders"] if p.startswith("hf://")]
        local_map = download_hf_uris(hf_paths)
        transcoder_paths: dict[int, str] = {}
        for i, path in enumerate(config["transcoders"]):
            transcoder_paths[i] = local_map.get(path) or path

        for layer_idx, local_path in transcoder_paths.items():
            npz_format = Path(local_path).suffix == ".npz"

            if special_load_fn == "gemma-scope" and npz_format:
                load_fn = load_gemma_scope_transcoder
            elif special_load_fn == "gemma-scope-2":
                load_fn = load_gemma_scope_2_transcoder
            else:
                load_fn = load_relu_transcoder

            transcoder = load_fn(
                local_path,
                layer_idx,
                device=device,
                dtype=dtype,
                lazy_encoder=False,
                lazy_decoder=False,
            )

            save_path = cache_path / f"layer_{layer_idx}.safetensors"
            transcoder.to_safetensors(str(save_path))
            logger.info(f"Saved layer {layer_idx} to {save_path}")

        if delete_hf_cache:
            for i, hf_path in enumerate(config["transcoders"]):
                if hf_path.startswith("hf://"):
                    _delete_hf_cache(transcoder_paths[i])


def _save_clt_to_cache(
    config: dict,
    cache_path: Path,
    sequential: bool,
    device: torch.device,
    dtype: torch.dtype,
    special_load_fn: Literal["gemma-scope", "gemma-scope-2", None],
    delete_hf_cache: bool,
):
    """Save a cross-layer transcoder to the cache."""
    if "gemma-scope-2" in config.get("repo_id", "") and "transcoders" in config:
        # GemmaScope2 CLT format
        paths: dict[int, str] = {}
        if sequential:
            for layer_idx, hf_path in enumerate(config["transcoders"]):
                if hf_path.startswith("hf://"):
                    local_path = download_hf_uri(hf_path)
                else:
                    local_path = hf_path
                paths[layer_idx] = local_path
        else:
            hf_paths = [p for p in config["transcoders"] if p.startswith("hf://")]
            local_map = download_hf_uris(hf_paths)
            for i, path in enumerate(config["transcoders"]):
                paths[i] = local_map.get(path) or path

        clt = load_gemma_scope_2_clt(
            paths,
            feature_input_hook=config["feature_input_hook"],
            feature_output_hook=config["feature_output_hook"],
            scan=config.get("scan"),
            device=device,
            dtype=dtype,
            lazy_decoder=False,
            lazy_encoder=False,
        )

        clt.to_safetensors(str(cache_path))

        if delete_hf_cache:
            for i, hf_path in enumerate(config["transcoders"]):
                if hf_path.startswith("hf://"):
                    _delete_hf_cache(paths[i])
    else:
        # Standard CLT format - download using snapshot_download
        subfolder = config.get("subfolder")
        if subfolder:
            allow_patterns = [f"{subfolder}/*.safetensors"]
        else:
            allow_patterns = ["*.safetensors"]

        local_path = snapshot_download(
            config["repo_id"],
            revision=config.get("revision", "main"),
            allow_patterns=allow_patterns,
        )

        if subfolder:
            local_path = os.path.join(local_path, subfolder)

        clt = load_clt(
            local_path,
            feature_input_hook=config["feature_input_hook"],
            feature_output_hook=config["feature_output_hook"],
            scan=config.get("scan"),
            device=device,
            dtype=dtype,
            lazy_decoder=False,
            lazy_encoder=False,
        )

        clt.to_safetensors(str(cache_path))


def load_transcoders_from_cache(
    hf_ref: str,
    cache_dir: str | Path | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    lazy_encoder: bool = False,
    lazy_decoder: bool = True,
):
    """Load transcoders from the local cache.

    Args:
        hf_ref: HuggingFace reference that was used to cache the transcoders
        cache_dir: Override the cache directory
        device: Device to load transcoders to
        dtype: Data type for transcoder weights
        lazy_encoder: Whether to lazy load encoder weights
        lazy_decoder: Whether to lazy load decoder weights

    Returns:
        Tuple of (transcoder, config)
    """
    cache_path = get_cached_path(hf_ref, cache_dir)
    config_path = cache_path / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Cache not found for {hf_ref} at {cache_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_kind = config["model_kind"]

    if model_kind == "transcoder_set":
        # Find all layer files
        layer_files = sorted(cache_path.glob("layer_*.safetensors"))
        transcoder_paths = {int(f.stem.split("_")[1]): str(f) for f in layer_files}

        transcoder = load_transcoder_set(
            transcoder_paths,
            scan=config.get("scan", str(cache_path)),
            feature_input_hook=config["feature_input_hook"],
            feature_output_hook=config["feature_output_hook"],
            device=device,
            dtype=dtype,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
        )
    elif model_kind == "cross_layer_transcoder":
        transcoder = load_clt(
            str(cache_path),
            feature_input_hook=config["feature_input_hook"],
            feature_output_hook=config["feature_output_hook"],
            scan=config.get("scan", str(cache_path)),
            device=device,
            dtype=dtype,
            lazy_decoder=lazy_decoder,
            lazy_encoder=lazy_encoder,
        )
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")

    return transcoder, config
