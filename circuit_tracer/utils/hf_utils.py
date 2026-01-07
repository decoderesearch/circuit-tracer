from __future__ import annotations

import glob
import logging
import os
from typing import NamedTuple
from collections.abc import Iterable
from urllib.parse import parse_qs, urlparse

import torch
import yaml
from huggingface_hub import get_token, hf_api, hf_hub_download, snapshot_download
from huggingface_hub.constants import HF_HUB_ENABLE_HF_TRANSFER
from huggingface_hub.utils.tqdm import tqdm as hf_tqdm
from tqdm.contrib.concurrent import thread_map

logger = logging.getLogger(__name__)


class HfUri(NamedTuple):
    """Structured representation of a HuggingFace URI."""

    repo_id: str
    file_path: str | None
    revision: str | None

    @classmethod
    def from_str(cls, hf_ref: str):
        if hf_ref.startswith("hf://"):
            return parse_hf_uri(hf_ref)

        parts = hf_ref.split("@", 1)
        path_part = parts[0]
        revision = parts[1] if len(parts) > 1 else None

        path_components = path_part.split("/")
        if len(path_components) >= 2:
            repo_id = "/".join(path_components[:2])
            file_path = "/".join(path_components[2:]) if len(path_components) > 2 else None
        else:
            repo_id = path_part
            file_path = None

        return cls(repo_id, file_path, revision)


def load_transcoder_from_hub(
    hf_ref: str,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    lazy_encoder: bool = False,
    lazy_decoder: bool = True,
    cache_dir: str | None = None,
    use_cache: bool = True,
):
    """Load a transcoder from a HuggingFace URI.

    If the transcoder is cached locally (via save_transcoders_to_cache), it will be
    loaded from cache instead of downloading from HuggingFace.

    Args:
        hf_ref: HuggingFace reference (e.g., "mwhanna/gemma-scope-transcoders")
        device: Device to load the transcoder to
        dtype: Data type for transcoder weights
        lazy_encoder: Whether to lazy load encoder weights
        lazy_decoder: Whether to lazy load decoder weights
        cache_dir: Override the cache directory for checking/loading cached transcoders
        use_cache: Whether to check for and use cached transcoders (default: True)

    Returns:
        Tuple of (transcoder, config)
    """
    from circuit_tracer.utils.caching import is_cached, load_transcoders_from_cache

    # Check cache first
    if use_cache and is_cached(hf_ref, cache_dir):
        logger.info(f"Loading transcoders from cache for {hf_ref}")
        return load_transcoders_from_cache(
            hf_ref,
            cache_dir=cache_dir,
            device=device,
            dtype=dtype,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
        )

    # resolve legacy references
    if hf_ref == "gemma":
        hf_ref = "mwhanna/gemma-scope-transcoders"
    elif hf_ref == "llama":
        hf_ref = "mntss/transcoder-Llama-3.2-1B"

    hf_uri = HfUri.from_str(hf_ref)
    try:
        config_path = hf_hub_download(
            repo_id=hf_uri.repo_id,
            revision=hf_uri.revision,
            filename="config.yaml",
            subfolder=hf_uri.file_path,
        )
    except Exception as e:
        config_file = (
            f"{hf_uri.file_path}/config.yaml" if hf_uri.file_path is not None else "config.yaml"
        )
        raise FileNotFoundError(f"Could not download {config_file} from {hf_uri.repo_id}") from e

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["repo_id"] = hf_uri.repo_id
    config["revision"] = hf_uri.revision
    config["subfolder"] = hf_uri.file_path
    repo_info = (
        hf_uri.repo_id if hf_uri.file_path is None else hf_uri.repo_id + "//" + hf_uri.file_path
    )
    config["scan"] = f"{repo_info}@{hf_uri.revision}" if hf_uri.revision else repo_info

    return load_transcoders(config, device, dtype, lazy_encoder, lazy_decoder), config


def load_transcoders(
    config: dict,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    lazy_encoder: bool = False,
    lazy_decoder: bool = True,
):
    """Load a transcoder from a HuggingFace URI."""

    model_kind = config["model_kind"]
    if model_kind == "transcoder_set":
        from circuit_tracer.transcoder.single_layer_transcoder import (
            load_transcoder_set,
        )

        transcoder_paths = resolve_transcoder_paths(config)

        repo_id = config.get("repo_id", "")
        # consider checking for google/gemma-scope-2
        # but this is tricky as repo_id likely points to mwhanna/... rather than google
        if "gemma-scope-2" in repo_id and "transcoders" in config:
            special_load_fn = "gemma-scope-2"
        elif "gemma-scope" in repo_id and "transcoders" in config:
            special_load_fn = "gemma-scope"
        else:
            special_load_fn = None

        return load_transcoder_set(
            transcoder_paths,
            scan=config["scan"],
            feature_input_hook=config["feature_input_hook"],
            feature_output_hook=config["feature_output_hook"],
            special_load_fn=special_load_fn,
            dtype=dtype,
            device=device,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
        )
    elif model_kind == "cross_layer_transcoder":
        from circuit_tracer.transcoder.cross_layer_transcoder import (
            load_clt,
            load_gemma_scope_2_clt,
        )

        if "gemma-scope-2" in config["repo_id"] and "transcoders" in config:
            transcoder_paths = resolve_transcoder_paths(config)
            local_path = transcoder_paths

            load_fn = load_gemma_scope_2_clt

        else:
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

            load_fn = load_clt

        return load_fn(
            local_path,  # type:ignore
            scan=config["scan"],
            feature_input_hook=config["feature_input_hook"],
            feature_output_hook=config["feature_output_hook"],
            lazy_decoder=lazy_decoder,
            lazy_encoder=lazy_encoder,
            dtype=dtype,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")


def resolve_transcoder_paths(config: dict) -> dict[int, str]:
    if "transcoders" in config:
        hf_paths = [path for path in config["transcoders"] if path.startswith("hf://")]
        local_map = download_hf_uris(hf_paths)
        transcoder_paths = {
            i: local_map.get(path, path) for i, path in enumerate(config["transcoders"])
        }
    else:
        subfolder = config.get("subfolder")
        if subfolder:
            allow_patterns = [f"{subfolder}/layer_*.safetensors"]
        else:
            allow_patterns = ["layer_*.safetensors"]

        local_path = snapshot_download(
            config["repo_id"],
            revision=config.get("revision", "main"),
            allow_patterns=allow_patterns,
        )

        if subfolder:
            local_path = os.path.join(local_path, subfolder)

        layer_files = glob.glob(os.path.join(local_path, "layer_*.safetensors"))
        transcoder_paths = {
            i: os.path.join(local_path, f"layer_{i}.safetensors") for i in range(len(layer_files))
        }
    return transcoder_paths  # type:ignore


def iter_transcoder_paths(config: dict) -> Iterable[tuple[int, str]]:
    """Lazily yield (layer_index, local_path) tuples, downloading one at a time."""
    if "transcoders" in config:
        for i, path in enumerate(config["transcoders"]):
            if path.startswith("hf://"):
                local_path = download_hf_uri(path)
            else:
                local_path = path
            yield i, local_path
    else:
        subfolder = config.get("subfolder")
        if subfolder:
            allow_patterns = [f"{subfolder}/layer_*.safetensors"]
        else:
            allow_patterns = ["layer_*.safetensors"]

        local_path = snapshot_download(
            config["repo_id"],
            revision=config.get("revision", "main"),
            allow_patterns=allow_patterns,
        )

        if subfolder:
            local_path = os.path.join(local_path, subfolder)

        layer_files = glob.glob(os.path.join(local_path, "layer_*.safetensors"))
        for i in range(len(layer_files)):
            yield i, os.path.join(local_path, f"layer_{i}.safetensors")


def parse_hf_uri(uri: str) -> HfUri:
    """Parse an HF URI into repo id, file path and revision.

    Args:
        uri: String like ``hf://org/repo/file?revision=main``.

    Returns:
        ``HfUri`` with repository id, file path and optional revision.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "hf":
        raise ValueError(f"Not a huggingface URI: {uri}")
    path = parsed.path.lstrip("/")
    repo_parts = path.split("/", 1)
    if len(repo_parts) != 2:
        raise ValueError(f"Invalid huggingface URI: {uri}")
    repo_id = f"{parsed.netloc}/{repo_parts[0]}"
    file_path = repo_parts[1]
    revision = parse_qs(parsed.query).get("revision", [None])[0] or None
    return HfUri(repo_id, file_path, revision)


def download_hf_uri(uri: str) -> str:
    """Download a file referenced by a HuggingFace URI and return the local path."""
    parsed = parse_hf_uri(uri)
    assert parsed.file_path is not None, "File path is not set"
    return hf_hub_download(
        repo_id=parsed.repo_id,
        filename=parsed.file_path,
        revision=parsed.revision,
        force_download=False,
    )


def download_hf_uris(uris: Iterable[str], max_workers: int = 8) -> dict[str, str]:
    """Download multiple HuggingFace URIs concurrently with pre-flight auth checks.

    Args:
        uris: Iterable of HF URIs.
        max_workers: Maximum number of parallel workers.

    Returns:
        Mapping from input URI to the local file path on disk.
    """
    if not uris:
        return {}

    uri_list = list(uris)
    if not uri_list:
        return {}
    parsed_map = {uri: parse_hf_uri(uri) for uri in uri_list}

    # ---  Pre-flight Check ---
    logger.info("Performing pre-flight metadata check...")
    unique_repos = {info.repo_id for info in parsed_map.values()}
    token = get_token()

    for repo_id in unique_repos:
        if hf_api.repo_info(repo_id=repo_id, token=token).gated is not False:
            if token is None:
                raise PermissionError("Cannot access a gated repo without a hf token.")

    logger.info("Pre-flight check complete. Starting downloads...")

    def _download(uri: str) -> str:
        info = parsed_map[uri]
        assert info.file_path is not None, "File path is not set"

        return hf_hub_download(
            repo_id=info.repo_id,
            filename=info.file_path,
            revision=info.revision,
            token=token,
            force_download=False,
        )

    if HF_HUB_ENABLE_HF_TRANSFER:
        # Use a simple loop for sequential download if HF_TRANSFER is enabled
        results = [_download(uri) for uri in uri_list]
        return dict(zip(uri_list, results))

    # The thread_map will attempt all downloads in parallel. If any worker thread
    # raises an exception (like GatedRepoError from _download), thread_map
    # will propagate that first exception, failing the entire process.
    results = thread_map(
        _download,
        uri_list,
        desc=f"Fetching {len(parsed_map)} files",
        max_workers=max_workers,
        tqdm_class=hf_tqdm,
    )
    return dict(zip(uri_list, results))
