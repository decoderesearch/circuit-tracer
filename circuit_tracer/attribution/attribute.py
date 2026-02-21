"""
Unified attribution interface that routes to the correct implementation based on the ReplacementModel backend.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch

from circuit_tracer.graph import Graph

if TYPE_CHECKING:
    from circuit_tracer.attribution.targets import TargetSpec
    from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
    from circuit_tracer.replacement_model.replacement_model_transformerlens import (
        TransformerLensReplacementModel,
    )


def attribute(
    prompt: str | torch.Tensor | list[int],
    model: "NNSightReplacementModel | TransformerLensReplacementModel",
    *,
    attribution_targets: "Sequence[str] | Sequence[TargetSpec] | torch.Tensor | None" = None,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 512,
    max_feature_nodes: int | None = None,
    offload: Literal["cpu", "disk", None] = None,
    verbose: bool = False,
    update_interval: int = 4,
) -> Graph:
    """Compute an attribution graph for *prompt*.

    This function automatically routes to the correct attribution implementation
    based on the type of ReplacementModel provided.

    Args:
        prompt: Text, token ids, or tensor - will be tokenized if str.
        model: Frozen ``ReplacementModel`` (either nnsight or transformerlens backend)
        attribution_targets: Target specification in one of four formats:
                          - None: Auto-select salient logits based on probability threshold
                          - torch.Tensor: Tensor of token indices
                          - Sequence[str]: Token strings (tokenized, auto-computes probability
                            and unembed vector)
                          - Sequence[TargetSpec]: Fully specified custom targets (CustomTarget or
                            tuple[str, float, torch.Tensor]) with arbitrary unembed directions
        max_n_logits: Max number of logit nodes (used when attribution_targets is None).
        desired_logit_prob: Keep logits until cumulative prob >= this value
                           (used when attribution_targets is None).
        batch_size: How many source nodes to process per backward pass.
        max_feature_nodes: Max number of feature nodes to include in the graph.
        offload: Method for offloading model parameters to save memory.
                 Options are "cpu" (move to CPU), "disk" (save to disk),
                 or None (no offloading).
        verbose: Whether to show progress information.
        update_interval: Number of batches to process before updating the feature ranking.

    Returns:
        Graph: Fully dense adjacency (unpruned).
    """

    if model.backend == "nnsight":
        from .attribute_nnsight import attribute as attribute_nnsight

        return attribute_nnsight(
            prompt=prompt,
            model=model,  # type: ignore[arg-type]
            attribution_targets=attribution_targets,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
            update_interval=update_interval,
        )
    else:
        from .attribute_transformerlens import attribute as attribute_transformerlens

        return attribute_transformerlens(
            prompt=prompt,
            model=model,  # type: ignore[arg-type]
            attribution_targets=attribution_targets,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
            update_interval=update_interval,
        )
