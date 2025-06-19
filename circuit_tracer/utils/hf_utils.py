from __future__ import annotations

from typing import Dict, Iterable, NamedTuple, Optional
from urllib.parse import parse_qs, urlparse

from huggingface_hub import hf_hub_download, get_token, hf_api
from huggingface_hub.constants import HF_HUB_ENABLE_HF_TRANSFER
from huggingface_hub.utils.tqdm import tqdm as hf_tqdm
from huggingface_hub.utils import HfFolder, GatedRepoError, RepositoryNotFoundError
from tqdm.contrib.concurrent import thread_map


class HfUri(NamedTuple):
    """Structured representation of a HuggingFace URI."""

    repo_id: str
    file_path: str
    revision: Optional[str]


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
    return hf_hub_download(
        repo_id=parsed.repo_id,
        filename=parsed.file_path,
        revision=parsed.revision,
        force_download=False,
    )


def download_hf_uris(uris: Iterable[str], max_workers: int = 8) -> Dict[str, str]:
    """Download multiple HuggingFace URIs concurrently with pre-flight auth checks.

    Args:
        uris: Iterable of HF URIs.
        max_workers: Maximum number of parallel workers.

    Returns:
        Mapping from input URI to the local file path on disk.
    """
    if not uris:
        return {}

    # Ensure uris is a list to avoid consuming an iterator multiple times
    uri_list = list(uris)
    parsed_map = {uri: parse_hf_uri(uri) for uri in uri_list}

    # --- Pre-flight Authentication Check ---
    print("-> Performing pre-flight authentication check...")
    unique_repos = {info.repo_id for info in parsed_map.values()}
    token = get_token() # Check for token once

    for repo_id in unique_repos:
        try:
            repo_info = hf_api.repo_info(repo_id=repo_id, token=token)
            # If repo is private or gated, we MUST have a token.
            if repo_info.private or repo_info.gated:
                if token is None:
                    raise PermissionError(
                        f"Repository '{repo_id}' is private or gated, but no Hugging Face token was found. "
                        "Please log in via `huggingface-cli login` or set the `HUGGING_FACE_HUB_TOKEN` env variable."
                    )
        except RepositoryNotFoundError:
            # Let hf_hub_download handle this error later if it's a real issue.
            # Sometimes a revision points to a repo that doesn't exist yet, etc.
            print(f"--> Warning: Repository '{repo_id}' not found during pre-flight check. Proceeding with download attempt.")
            pass
        except GatedRepoError as e:
            # This is a specific error for gated repos where user doesn't have access
             raise PermissionError(
                f"You have not accepted the terms of use for the gated repository '{repo_id}'. "
                f"Please visit https://huggingface.co/{repo_id} to accept the terms."
            ) from e
            
    print("--> Authentication check passed.")
    # --- End of Pre-flight Check ---

    def _download(uri: str) -> str:
        info = parsed_map[uri]
        # We can pass the token explicitly to be 100% sure it's used
        return hf_hub_download(
            repo_id=info.repo_id,
            filename=info.file_path,
            revision=info.revision,
            token=token,
            force_download=False,
        )

    if HF_HUB_ENABLE_HF_TRANSFER:
        # This part likely doesn't use threads, so it's safer but slower
        return {uri: _download(uri) for uri in uri_list}

    # Now we can safely execute the thread map
    results = thread_map(
        _download,
        uri_list,
        desc=f"Fetching {len(parsed_map)} files",
        max_workers=max_workers,
        tqdm_class=hf_tqdm,
    )
    return dict(zip(uri_list, results))