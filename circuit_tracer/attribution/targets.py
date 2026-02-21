"""Attribution target specification and processing.

This module provides the AttributionTargets container class and LogitTarget record
structure for specifying and processing attribution targets in the format required
for attribution graph computation.

Key concepts:
- AttributionTargets: High-level container that encapsulates target specifications
- LogitTarget: Low-level data transfer object (DTO) storing token metadata
- Virtual indices: Technique for representing out-of-vocabulary (OOV) tokens using
  synthetic indices >= vocab_size. Required to support arbitrary string token (or functions thereof)
  attribution functionality.
"""

from collections.abc import Sequence
from typing import NamedTuple
import logging

import torch


class LogitTarget(NamedTuple):
    """Token metadata for attribution: string representation and vocabulary index."""

    token_str: str
    vocab_idx: int


class CustomTarget(NamedTuple):
    """A fully specified custom attribution target.

    Attributes:
        token_str: Label for this target (e.g., "logit(x)-logit(y)")
        prob: Weight/probability for this target
        vec: Custom unembed direction vector (d_model,)
    """

    token_str: str
    prob: float
    vec: torch.Tensor


TargetSpec = CustomTarget | tuple[str, float, torch.Tensor]


class AttributionTargets:
    """Container for processed attribution target specifications.

    Encapsulates target identifiers, softmax probabilities, and demeaned unembedding
    vectors needed for attribution graph computation.

    Supports four input formats:
    - None: Auto-select salient logits by probability threshold
    - torch.Tensor: Specific vocabulary indices (token IDs)
    - Sequence[str]: Token strings (tokenized internally)
    - Sequence[TargetSpec]: Fully specified custom targets (CustomTarget or raw tuple[str, float, torch.Tensor])

    Attributes:
        logit_targets: List of LogitTarget records with token strings and vocab indices
        logit_probabilities: Softmax probabilities for each target (k,)
        logit_vectors: Demeaned unembedding vectors (k, d_model)
    """

    def __init__(
        self,
        attribution_targets: Sequence[str] | Sequence[TargetSpec] | torch.Tensor | None,
        logits: torch.Tensor,
        unembed_proj: torch.Tensor,
        tokenizer,
        *,
        max_n_logits: int = 10,
        desired_logit_prob: float = 0.95,
    ):
        """Build attribution targets from user specification.

        Args:
            attribution_targets: Target specification in one of four formats:
                - None: Auto-select salient logits based on probability threshold
                - torch.Tensor: Tensor of vocabulary token IDs
                - Sequence[str]: Token strings (tokenized, then auto-computes probability & vector)
                - Sequence[TargetSpec]: Fully specified custom targets (CustomTarget or
                  tuple[str, float, torch.Tensor]) with custom probability and unembed direction
                  (uses virtual index for OOV tokens)
            logits: ``(d_vocab,)`` logit vector for single position
            unembed_proj: ``(d_model, d_vocab)`` unembedding matrix
            tokenizer: Tokenizer for string→int conversion
            max_n_logits: Max targets when auto-selecting (salient mode)
            desired_logit_prob: Probability threshold for salient mode
        """
        # Store tokenizer ref for decoding vocab indices to token strings
        self.tokenizer = tokenizer
        ctor_shared = {"logits": logits, "unembed_proj": unembed_proj, "tokenizer": tokenizer}

        # Dispatch to appropriate constructor based on input type
        if attribution_targets is None:
            salient_ctor = {"max_n_logits": max_n_logits, "desired_logit_prob": desired_logit_prob}
            attr_spec = self._from_salient(**salient_ctor, **ctor_shared)
        elif isinstance(attribution_targets, torch.Tensor):
            attr_spec = self._from_indices(indices=attribution_targets, **ctor_shared)
        elif isinstance(attribution_targets, Sequence):
            if not attribution_targets:
                raise ValueError("attribution_targets sequence cannot be empty")
            first = attribution_targets[0]
            if isinstance(first, str):
                attr_spec = self._from_str(token_strs=attribution_targets, **ctor_shared)  # type: ignore[arg-type]
            elif isinstance(first, (tuple, CustomTarget)):
                attr_spec = self._from_tuple(target_tuples=attribution_targets, **ctor_shared)  # type: ignore[arg-type]
            else:
                raise TypeError(
                    f"Sequence elements must be str or TargetSpec (CustomTarget or "
                    f"tuple[str, float, Tensor]), got {type(first)}"
                )
        else:
            raise TypeError(
                f"attribution_targets must be None, torch.Tensor, Sequence[str], "
                f"or Sequence[TargetSpec], got {type(attribution_targets)}"
            )
        self.logit_targets, self.logit_probabilities, self.logit_vectors = attr_spec

    def __len__(self) -> int:
        """Number of attribution targets."""
        return len(self.logit_targets)

    def __repr__(self) -> str:
        """String representation showing key info."""
        if len(self.logit_targets) > 3:
            targets_preview = self.logit_targets[:3]
            suffix = "..."
        else:
            targets_preview = self.logit_targets
            suffix = ""
        return f"AttributionTargets(n={len(self)}, targets={targets_preview}{suffix})"

    @property
    def tokens(self) -> list[str]:
        """Get token strings for all targets.

        Returns:
            List of token strings (decoded vocab tokens or arbitrary strings)
        """
        return [target.token_str for target in self.logit_targets]

    @property
    def vocab_size(self) -> int:
        """Vocabulary size from the tokenizer.

        Returns:
            Vocabulary size for determining virtual vs real indices
        """
        return self.tokenizer.vocab_size

    @property
    def token_ids(self) -> torch.Tensor:
        """Tensor of vocabulary indices (including virtual indices >= vocab_size).

        Returns a torch.Tensor of vocab indices on the same device as other tensors,
        suitable for indexing into logit vectors or embeddings.

        Returns:
            torch.Tensor: Long tensor of vocabulary indices
        """
        return torch.tensor(
            [target.vocab_idx for target in self.logit_targets],
            dtype=torch.long,
            device=self.logit_probabilities.device,
        )

    def to(self, device: str | torch.device) -> "AttributionTargets":
        """Transfer AttributionTargets to specified device.

        Only moves torch.Tensor attributes (logit_probabilities, logit_vectors);
        logit_targets list stays unchanged.

        Args:
            device: Target device (e.g., "cuda", "cpu")

        Returns:
            Self with tensors on new device
        """
        self.logit_probabilities = self.logit_probabilities.to(device)
        self.logit_vectors = self.logit_vectors.to(device)
        return self

    @staticmethod
    def _from_salient(
        logits: torch.Tensor,
        unembed_proj: torch.Tensor,
        max_n_logits: int,
        desired_logit_prob: float,
        tokenizer,
    ) -> tuple[list[LogitTarget], torch.Tensor, torch.Tensor]:
        """Auto-select salient logits by cumulative probability.

        Picks the smallest set of logits whose cumulative probability
        exceeds the threshold, up to max_n_logits.

        Args:
            logits: ``(d_vocab,)`` logit vector
            unembed_proj: ``(d_model, d_vocab)`` unembedding matrix
            max_n_logits: Hard cap on number of logits
            desired_logit_prob: Cumulative probability threshold
            tokenizer: Tokenizer for decoding vocab indices to strings

        Returns:
            Tuple of (logit_targets, probabilities, vectors) where logit_targets
            contains LogitTarget instances with actual vocab indices
        """
        probs = torch.softmax(logits, dim=-1)
        top_p, top_idx = torch.topk(probs, max_n_logits)
        cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
        indices, probs, vecs = AttributionTargets._compute_logit_vecs(
            top_idx[:cutoff], logits, unembed_proj
        )
        logit_targets = [
            LogitTarget(token_str=tokenizer.decode(idx), vocab_idx=idx) for idx in indices.tolist()
        ]
        return logit_targets, probs, vecs

    @staticmethod
    def _from_indices(
        indices: torch.Tensor,
        logits: torch.Tensor,
        unembed_proj: torch.Tensor,
        tokenizer,
    ) -> tuple[list[LogitTarget], torch.Tensor, torch.Tensor]:
        """Construct from specific vocabulary indices.

        Args:
            indices: ``(k,)`` tensor of vocabulary indices
            logits: ``(d_vocab,)`` logit vector
            unembed_proj: ``(d_model, d_vocab)`` unembedding matrix
            tokenizer: Tokenizer for decoding vocab indices to strings

        Returns:
            Tuple of (logit_targets, probabilities, vectors) where logit_targets
            contains LogitTarget instances with actual vocab indices

        Raises:
            ValueError: If any index is out of vocabulary range
        """
        vocab_size = logits.shape[0]

        # Validate all indices are within vocab range
        if (indices < 0).any() or (indices >= vocab_size).any():
            invalid = indices[(indices < 0) | (indices >= vocab_size)]
            raise ValueError(
                f"Token indices must be in range [0, {vocab_size}), "
                f"but found invalid indices: {invalid.tolist()}"
            )

        indices_out, probs, vecs = AttributionTargets._compute_logit_vecs(
            indices, logits, unembed_proj
        )

        # Create LogitTarget instances with decoded token strings
        logit_targets = [
            LogitTarget(token_str=tokenizer.decode(idx), vocab_idx=idx)
            for idx in indices_out.tolist()
        ]
        return logit_targets, probs, vecs

    @staticmethod
    def _from_str(
        token_strs: Sequence[str],
        logits: torch.Tensor,
        unembed_proj: torch.Tensor,
        tokenizer,
    ) -> tuple[list[LogitTarget], torch.Tensor, torch.Tensor]:
        """Construct from a sequence of token strings.

        Each string is tokenized and its probability/vector auto-computed.

        Args:
            token_strs: Sequence of token strings
            logits: ``(d_vocab,)`` logit vector
            unembed_proj: Unembedding matrix
            tokenizer: Tokenizer for string→int conversion

        Returns:
            Tuple of (logit_targets, probabilities, vectors)
        """
        vocab_size = logits.shape[0]
        indices = []
        for token_str in token_strs:
            try:
                ids = tokenizer.encode(token_str, add_special_tokens=False)
            except Exception as e:
                raise ValueError(
                    f"Failed to encode string token {token_str!r} using tokenizer: {e}"
                ) from e
            if not ids:
                raise ValueError(f"String token {token_str!r} encoded to empty token sequence.")
            if len(ids) > 1:
                raise ValueError(
                    f"String token {token_str!r} encoded to {len(ids)} tokens "
                    f"(IDs: {ids}). Each string must map to exactly one token. "
                    f"Consider providing single-token strings."
                )
            token_id = ids[0]
            assert 0 <= token_id < vocab_size, (
                f"Token {token_str!r} resolved to index {token_id}, "
                f"out of vocabulary range [0, {vocab_size})"
            )
            indices.append(token_id)
        return AttributionTargets._from_indices(
            indices=torch.tensor(indices, dtype=torch.long),
            logits=logits,
            unembed_proj=unembed_proj,
            tokenizer=tokenizer,
        )

    @staticmethod
    def _validate_custom_target(
        target: TargetSpec,
    ) -> CustomTarget:
        """Validate and normalize a custom target.

        Args:
            target: A CustomTarget or raw (token_str, prob, vec) tuple

        Returns:
            Validated CustomTarget instance

        Raises:
            ValueError: If the tuple has wrong length or element types
        """
        if not isinstance(target, CustomTarget):
            if len(target) != 3:
                raise ValueError(
                    f"Tuple targets must have exactly 3 elements "
                    f"(token_str, probability, vector), got {len(target)}"
                )
            token_str, prob, vec = target
        else:
            token_str, prob, vec = target.token_str, target.prob, target.vec
        if not isinstance(token_str, str):
            raise TypeError(f"Custom target token_str must be str, got {type(token_str)}")
        if not isinstance(prob, (int, float)):
            raise TypeError(f"Custom target prob must be int or float, got {type(prob)}")
        if not isinstance(vec, torch.Tensor):
            raise TypeError(f"Custom target vec must be torch.Tensor, got {type(vec)}")
        return CustomTarget(token_str=token_str, prob=float(prob), vec=vec)

    @staticmethod
    def _from_tuple(
        target_tuples: Sequence[TargetSpec],
        logits: torch.Tensor,
        unembed_proj: torch.Tensor,
        tokenizer,
    ) -> tuple[list[LogitTarget], torch.Tensor, torch.Tensor]:
        """Construct from fully specified custom targets.

        Each target provides (token_str, prob, vec) for an arbitrary
        attribution direction that may not correspond to a vocabulary token.

        Args:
            target_tuples: Sequence of CustomTarget or raw tuple instances
            logits: ``(d_vocab,)`` logit vector (used for vocab_size)
            unembed_proj: Unembedding matrix (unused but kept for interface consistency)
            tokenizer: Tokenizer (unused but kept for interface consistency)

        Returns:
            Tuple of (logit_targets, probabilities, vectors)
        """
        vocab_size = logits.shape[0]
        logit_targets, probs, vecs = [], [], []
        for position, target in enumerate(target_tuples):
            validated = AttributionTargets._validate_custom_target(target)
            virtual_idx = vocab_size + position
            logit_targets.append(LogitTarget(token_str=validated.token_str, vocab_idx=virtual_idx))
            probs.append(validated.prob)
            vecs.append(validated.vec)
        return logit_targets, torch.tensor(probs), torch.stack(vecs)

    @staticmethod
    def _compute_logit_vecs(
        indices: torch.Tensor,
        logits: torch.Tensor,
        unembed_proj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute probabilities and demeaned vectors for indices.

        Args:
            indices: ``(k,)`` vocabulary indices to compute vectors for
            logits: ``(d_vocab,)`` logit vector for single position
            unembed_proj: ``(d_model, d_vocab)`` or ``(d_vocab, d_model)`` unembedding matrix
                         (orientation auto-detected by matching vocab dimension to logits)

        Returns:
            Tuple of:
                * indices - ``(k,)`` vocabulary ids (same as input)
                * probabilities - ``(k,)`` softmax probabilities
                * demeaned_vecs - ``(k, d_model)`` unembedding columns, demeaned
        """
        probs = torch.softmax(logits, dim=-1)
        selected_probs = probs[indices]

        # Auto-detect matrix orientation by matching against vocabulary size
        d_vocab = logits.shape[0]
        if unembed_proj.shape[0] == d_vocab:
            # Shape is (d_vocab, d_model) – first axis is vocabulary (e.g., NNSight)
            cols = unembed_proj[indices]  # (k, d_model)
            demean = unembed_proj.mean(dim=0, keepdim=True)  # (1, d_model)
            demeaned_vecs = cols - demean  # (k, d_model)
        else:
            # Shape is (d_model, d_vocab) – second axis is vocabulary (e.g., TransformerLens)
            cols = unembed_proj[:, indices]  # (d_model, k)
            demean = unembed_proj.mean(dim=-1, keepdim=True)  # (d_model, 1)
            demeaned_vecs = (cols - demean).T  # (k, d_model)

        return indices, selected_probs, demeaned_vecs


def log_attribution_target_info(
    targets: "AttributionTargets",
    attribution_targets: Sequence[str] | Sequence[TargetSpec] | torch.Tensor | None,
    logger: logging.Logger,
) -> None:
    """Log information about attribution targets.

    Args:
        targets: AttributionTargets instance with processed targets
        attribution_targets: Original attribution_targets specification
        logger: Logger to use for output
    """
    prob_sum = targets.logit_probabilities.sum().item()
    if attribution_targets is None:
        target_desc = "salient logits"
        weight_desc = "cumulative probability"
    elif (
        isinstance(attribution_targets, Sequence)
        and attribution_targets
        and isinstance(attribution_targets[0], (tuple, CustomTarget))
    ):
        target_desc = "custom attribution targets"
        weight_desc = "total weight"
    else:
        target_desc = "specified logit targets"
        weight_desc = "cumulative probability"
    logger.info(f"Using {len(targets)} {target_desc} with {weight_desc} {prob_sum:.4f}")
