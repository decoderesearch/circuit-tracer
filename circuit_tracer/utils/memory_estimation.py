"""Memory estimates for attribution graph construction.

The attribution pipeline currently materializes dense ``N x N`` tensors for the
graph adjacency and for graph-pruning intermediates.  This module gives users a
cheap way to estimate that footprint before launching a long-context trace.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any


DTYPE_BYTES = {
    "float64": 8,
    "fp64": 8,
    "float32": 4,
    "fp32": 4,
    "bfloat16": 2,
    "bf16": 2,
    "float16": 2,
    "fp16": 2,
}

CANONICAL_DTYPE = {
    "fp64": "float64",
    "fp32": "float32",
    "bf16": "bfloat16",
    "fp16": "float16",
}


def normalize_dtype_name(dtype: str) -> str:
    """Return the canonical dtype name supported by the estimator."""
    normalized = dtype.lower()
    if normalized not in DTYPE_BYTES:
        supported = ", ".join(sorted(DTYPE_BYTES))
        raise ValueError(f"Unsupported dtype {dtype!r}. Supported dtypes: {supported}")
    return CANONICAL_DTYPE.get(normalized, normalized)


def format_bytes(num_bytes: int) -> str:
    """Format a byte count using binary units."""
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")
    if num_bytes == 0:
        return "0 B"

    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    unit_index = min(int(math.log(num_bytes, 1024)), len(units) - 1)
    value = num_bytes / (1024**unit_index)
    return f"{value:.2f} {units[unit_index]}"


@dataclass(frozen=True)
class GraphMemoryEstimate:
    """Estimated dense memory use for one attribution graph."""

    n_tokens: int
    n_layers: int
    max_feature_nodes: int
    n_logits: int
    dtype: str
    bytes_per_value: int
    n_feature_nodes: int
    n_error_nodes: int
    n_token_nodes: int
    n_logit_nodes: int
    n_total_nodes: int
    dense_adjacency_bytes: int
    dense_bool_mask_bytes: int
    graph_metadata_bytes: int
    pruning_peak_bytes: int
    estimated_peak_bytes: int
    available_memory_bytes: int | None
    usable_memory_bytes: int | None
    fits_usable_memory: bool | None
    recommendations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": {
                "n_tokens": self.n_tokens,
                "n_layers": self.n_layers,
                "max_feature_nodes": self.max_feature_nodes,
                "n_logits": self.n_logits,
                "dtype": self.dtype,
                "bytes_per_value": self.bytes_per_value,
                "available_memory": (
                    format_bytes(self.available_memory_bytes)
                    if self.available_memory_bytes is not None
                    else None
                ),
                "usable_memory": (
                    format_bytes(self.usable_memory_bytes)
                    if self.usable_memory_bytes is not None
                    else None
                ),
            },
            "nodes": {
                "feature_nodes": self.n_feature_nodes,
                "error_nodes": self.n_error_nodes,
                "token_nodes": self.n_token_nodes,
                "logit_nodes": self.n_logit_nodes,
                "total_nodes": self.n_total_nodes,
            },
            "memory": {
                "dense_adjacency": format_bytes(self.dense_adjacency_bytes),
                "dense_bool_mask": format_bytes(self.dense_bool_mask_bytes),
                "graph_metadata": format_bytes(self.graph_metadata_bytes),
                "pruning_peak": format_bytes(self.pruning_peak_bytes),
                "estimated_peak": format_bytes(self.estimated_peak_bytes),
                "dense_adjacency_bytes": self.dense_adjacency_bytes,
                "pruning_peak_bytes": self.pruning_peak_bytes,
                "estimated_peak_bytes": self.estimated_peak_bytes,
            },
            "fits_usable_memory": self.fits_usable_memory,
            "recommendations": list(self.recommendations),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        lines = [
            "# circuit-tracer memory estimate",
            "",
            "## Inputs",
            "",
            f"- Tokens: {self.n_tokens:,}",
            f"- Layers: {self.n_layers:,}",
            f"- Max feature nodes: {self.max_feature_nodes:,}",
            f"- Logit nodes: {self.n_logits:,}",
            f"- Dtype: {self.dtype} ({self.bytes_per_value} bytes/value)",
        ]
        if self.available_memory_bytes is not None:
            lines.extend(
                [
                    f"- Available memory: {format_bytes(self.available_memory_bytes)}",
                    f"- Usable memory after safety margin: {format_bytes(self.usable_memory_bytes or 0)}",
                ]
            )

        lines.extend(
            [
                "",
                "## Estimated graph size",
                "",
                f"- Feature nodes: {self.n_feature_nodes:,}",
                f"- Error nodes: {self.n_error_nodes:,}",
                f"- Token nodes: {self.n_token_nodes:,}",
                f"- Logit nodes: {self.n_logit_nodes:,}",
                f"- Total nodes: {self.n_total_nodes:,}",
                "",
                "## Estimated dense memory",
                "",
                f"- Dense adjacency tensor: {format_bytes(self.dense_adjacency_bytes)}",
                f"- Dense boolean mask: {format_bytes(self.dense_bool_mask_bytes)}",
                f"- Feature/token metadata tensors: {format_bytes(self.graph_metadata_bytes)}",
                f"- Estimated graph-pruning peak: {format_bytes(self.pruning_peak_bytes)}",
                f"- Estimated end-to-end peak: {format_bytes(self.estimated_peak_bytes)}",
            ]
        )

        if self.fits_usable_memory is not None:
            fit_text = "yes" if self.fits_usable_memory else "no"
            lines.extend(["", f"Fits usable memory: **{fit_text}**"])

        if self.recommendations:
            lines.extend(["", "## Recommendations", ""])
            lines.extend(f"- {recommendation}" for recommendation in self.recommendations)

        return "\n".join(lines)


def estimate_graph_memory(
    *,
    n_tokens: int,
    n_layers: int,
    max_feature_nodes: int = 7500,
    n_logits: int = 10,
    dtype: str = "float32",
    available_memory_gib: float | None = None,
    safety_fraction: float = 0.8,
) -> GraphMemoryEstimate:
    """Estimate dense attribution graph memory for a prompt.

    Args:
        n_tokens: Prompt length after tokenization.
        n_layers: Number of transformer layers. Each layer contributes one
            reconstruction-error node per token.
        max_feature_nodes: Upper bound used by ``attribute(..., max_feature_nodes=...)``.
        n_logits: Number of output logit target nodes.
        dtype: Floating point dtype used for dense graph tensors.
        available_memory_gib: Optional device memory budget to compare against.
        safety_fraction: Fraction of available memory treated as usable.

    Returns:
        A ``GraphMemoryEstimate`` with formatted and raw byte counts.
    """
    if n_tokens <= 0:
        raise ValueError("n_tokens must be positive")
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if max_feature_nodes < 0:
        raise ValueError("max_feature_nodes must be non-negative")
    if n_logits <= 0:
        raise ValueError("n_logits must be positive")
    if not 0 < safety_fraction <= 1:
        raise ValueError("safety_fraction must be in (0, 1]")

    dtype = normalize_dtype_name(dtype)
    bytes_per_value = DTYPE_BYTES[dtype]

    n_feature_nodes = max_feature_nodes
    n_error_nodes = n_layers * n_tokens
    n_token_nodes = n_tokens
    n_logit_nodes = n_logits
    n_total_nodes = n_feature_nodes + n_error_nodes + n_token_nodes + n_logit_nodes

    dense_matrix_entries = n_total_nodes * n_total_nodes
    dense_adjacency_bytes = dense_matrix_entries * bytes_per_value
    dense_bool_mask_bytes = dense_matrix_entries

    # Small graph-side tensors that scale linearly with node count.  This is not
    # the full model/transcoder footprint; it captures graph metadata only.
    graph_metadata_bytes = (
        n_feature_nodes * 3 * 8  # active feature layer/position/index
        + n_feature_nodes * 8  # selected feature indices
        + n_feature_nodes * bytes_per_value  # activation values
        + n_token_nodes * 8
        + n_logit_nodes * (8 + bytes_per_value)
    )

    # prune_graph currently has the original adjacency, a pruned clone, a
    # normalized pruned matrix, edge scores, and a dense bool mask live near the
    # edge-pruning step. This is intentionally approximate and conservative.
    pruning_peak_bytes = 4 * dense_adjacency_bytes + dense_bool_mask_bytes
    estimated_peak_bytes = pruning_peak_bytes + graph_metadata_bytes

    available_memory_bytes = None
    usable_memory_bytes = None
    fits_usable_memory = None
    if available_memory_gib is not None:
        if available_memory_gib <= 0:
            raise ValueError("available_memory_gib must be positive when provided")
        available_memory_bytes = int(available_memory_gib * 1024**3)
        usable_memory_bytes = int(available_memory_bytes * safety_fraction)
        fits_usable_memory = estimated_peak_bytes <= usable_memory_bytes

    recommendations = _build_recommendations(
        n_tokens=n_tokens,
        n_total_nodes=n_total_nodes,
        dense_adjacency_bytes=dense_adjacency_bytes,
        estimated_peak_bytes=estimated_peak_bytes,
        usable_memory_bytes=usable_memory_bytes,
        fits_usable_memory=fits_usable_memory,
        dtype=dtype,
        max_feature_nodes=max_feature_nodes,
    )

    return GraphMemoryEstimate(
        n_tokens=n_tokens,
        n_layers=n_layers,
        max_feature_nodes=max_feature_nodes,
        n_logits=n_logits,
        dtype=dtype,
        bytes_per_value=bytes_per_value,
        n_feature_nodes=n_feature_nodes,
        n_error_nodes=n_error_nodes,
        n_token_nodes=n_token_nodes,
        n_logit_nodes=n_logit_nodes,
        n_total_nodes=n_total_nodes,
        dense_adjacency_bytes=dense_adjacency_bytes,
        dense_bool_mask_bytes=dense_bool_mask_bytes,
        graph_metadata_bytes=graph_metadata_bytes,
        pruning_peak_bytes=pruning_peak_bytes,
        estimated_peak_bytes=estimated_peak_bytes,
        available_memory_bytes=available_memory_bytes,
        usable_memory_bytes=usable_memory_bytes,
        fits_usable_memory=fits_usable_memory,
        recommendations=tuple(recommendations),
    )


def _build_recommendations(
    *,
    n_tokens: int,
    n_total_nodes: int,
    dense_adjacency_bytes: int,
    estimated_peak_bytes: int,
    usable_memory_bytes: int | None,
    fits_usable_memory: bool | None,
    dtype: str,
    max_feature_nodes: int,
) -> list[str]:
    recommendations: list[str] = []

    if fits_usable_memory is False:
        assert usable_memory_bytes is not None
        shortfall = estimated_peak_bytes / max(usable_memory_bytes, 1)
        recommendations.append(
            "Estimated pruning peak exceeds the usable memory budget "
            f"by {shortfall:.1f}x; reduce prompt length, layers, logits, or feature cap."
        )
    elif usable_memory_bytes is not None and estimated_peak_bytes > usable_memory_bytes * 0.7:
        recommendations.append(
            "Estimate is within memory budget but close enough that allocator overhead, "
            "model weights, and transcoders can still trigger OOM."
        )

    if dtype == "float32":
        recommendations.append(
            "Try --dtype bfloat16 or --dtype float16 when numerically acceptable; dense graph "
            "memory is roughly halved."
        )

    if n_tokens >= 4096:
        recommendations.append(
            "Long prompts create one reconstruction-error node per layer per token; this is "
            "usually the dominant node-count driver."
        )

    if max_feature_nodes >= 7500:
        recommendations.append(
            "Lower --max_feature_nodes for exploratory traces; pruning thresholds do not reduce "
            "the initial dense graph allocation."
        )

    if dense_adjacency_bytes > 16 * 1024**3:
        recommendations.append(
            "Dense adjacency alone is large. A sparse or blockwise graph backend is the right "
            "next engineering target for this configuration."
        )

    if n_total_nodes > 50_000:
        recommendations.append(
            "Graph has more than 50k nodes before pruning; local visualization JSON may also "
            "become large even if attribution succeeds."
        )

    if not recommendations:
        recommendations.append("No immediate dense-graph memory risk flagged by this estimate.")

    return recommendations
