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

import torch


class LogitTarget(NamedTuple):
    """Data transfer object (DTO) for logit attribution targets.

    A lightweight record structure containing token metadata for attribution.

    Attributes:
        token_str: String representation of the token (decoded from vocab or arbitrary)
        vocab_idx: Vocabulary index - either a real token ID (< vocab_size) or
                   a virtual index for OOV tokens (>= vocab_size)
    """

    token_str: str
    vocab_idx: int


class AttributionTargets:
    """Container for processed attribution target specifications.

    High-level data structure that encapsulates target identifiers, softmax probabilities,
    and demeaned unembedding vectors needed for attribution graph computation.

    Supports multiple input formats for flexible target specification:
    - None: Auto-select salient logits by probability threshold
    - torch.Tensor: Specific vocabulary indices (i.e. token_ids)
    - list: Mixed targets (tuples for OOV tokens, ints/strs for valid token_ids)

    Attributes:
        logit_targets: List of LogitTarget records with token strings and vocab indices
        logit_probabilities: Softmax probabilities for each target (k,)
        logit_vectors: Demeaned unembedding vectors (k, d_model)
    """

    def __init__(
        self,
        attribution_targets: (
            Sequence[tuple[str, float, torch.Tensor] | int | str] | torch.Tensor | None
        ),
        logits: torch.Tensor,
        unembed_proj: torch.Tensor,
        tokenizer,
        *,
        max_n_logits: int = 10,
        desired_logit_prob: float = 0.95,
    ):
        """Build attribution targets from user specification.

        Args:
            attribution_targets: Target specification in one of several formats:
                - None: Auto-select salient logits based on probability threshold
                - torch.Tensor: Tensor of vocabulary token IDs
                - list[tuple[str, float, torch.Tensor] | int | str]: List where
                  each element can be:
                    * int or str: Token ID/string (auto-computes probability & vector)
                    * tuple[str, float, torch.Tensor]: Fully specified target logit with arbitrary
                      string token (or function thereof) (may use virtual index for OOV tokens)
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
        elif isinstance(attribution_targets, list):
            if not attribution_targets:
                raise ValueError("attribution_targets list cannot be empty")
            attr_spec = self._from_list(target_list=attribution_targets, **ctor_shared)
        else:
            raise TypeError(
                f"attribution_targets must be None, torch.Tensor, or list, "
                f"got {type(attribution_targets)}"
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
    def vocab_indices(self) -> list[int]:
        """All vocabulary indices including virtual indices (>= vocab_size).
        Vocab indices are a generalization of token IDs that can represent:
        - Real vocab indices (< vocab_size) for token_ids valid in the current tokenizer vocab space
        - Virtual indices (>= vocab_size) for arbitrary string tokens (or functions thereof)

        Use has_virtual_indices to check if any virtual indices are present.
        Use token_ids to get a tensor of only real vocabulary indices.

        Returns:
            List of vocabulary indices (including virtual indices)
        """
        return [target.vocab_idx for target in self.logit_targets]

    @property
    def has_virtual_indices(self) -> bool:
        """Check if any targets use virtual indices (OOV tokens).

        Virtual indices (vocab_idx >= vocab_size) are a technique for representing
        arbitrary string tokens not in the model's vocabulary.

        Returns:
            True if virtual indices are present, False otherwise
        """
        vocab_size = self.tokenizer.vocab_size
        return any(t.vocab_idx >= vocab_size for t in self.logit_targets)

    @property
    def token_ids(self) -> torch.Tensor:
        """Tensor of valid vocabulary indices (< vocab_size only).

        Returns a torch.Tensor of vocab indices on the same device as other tensors,
        suitable for indexing into logit vectors or embeddings. This property will
        raise a ValueError if any targets use virtual indices (arbitrary strings).

        Raises:
            ValueError: If any targets have virtual indices (vocab_idx >= vocab_size)

        Returns:
            torch.Tensor: Long tensor of vocabulary indices
        """
        if self.has_virtual_indices:
            raise ValueError(
                "Cannot create token_ids tensor: some targets use virtual indices "
                "(arbitrary strings not in vocabulary). Check has_virtual_indices "
                "before accessing token_ids, or use vocab_indices to get all indices."
            )
        return torch.tensor(
            self.vocab_indices, dtype=torch.long, device=self.logit_probabilities.device
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
    def _from_list(
        target_list: Sequence[tuple[str, float, torch.Tensor] | int | str],
        logits: torch.Tensor,
        unembed_proj: torch.Tensor,
        tokenizer,
    ) -> tuple[list[LogitTarget], torch.Tensor, torch.Tensor]:
        """Construct from mixed list of targets.

        Supports heterogeneous list where each element can be:
        - int: Vocabulary index (auto-compute prob/vec)
        - str: Token string (tokenize, auto-compute)
        - tuple[str, float, Tensor]: Fully specified arbitrary string or function thereof

        Args:
            targets: List of mixed target specifications
            logits: ``(d_vocab,)`` logit vector
            unembed_proj: ``(d_model, d_vocab)`` unembedding matrix
            tokenizer: Tokenizer for string→int conversion

        Returns:
            Tuple of (logit_targets, probabilities, vectors)
        """
        return AttributionTargets._process_target_list(target_list, logits, unembed_proj, tokenizer)

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
            unembed_proj: ``(d_model, d_vocab)`` unembedding matrix

        Returns:
            Tuple of:
                * indices - ``(k,)`` vocabulary ids (same as input)
                * probabilities - ``(k,)`` softmax probabilities
                * demeaned_vecs - ``(k, d_model)`` unembedding columns, demeaned
        """
        probs = torch.softmax(logits, dim=-1)
        selected_probs = probs[indices]
        cols = unembed_proj[:, indices]
        demeaned = cols - unembed_proj.mean(dim=-1, keepdim=True)
        return indices, selected_probs, demeaned.T

    @staticmethod
    def _process_target_list(
        targets: Sequence[tuple[str, float, torch.Tensor] | int | str],
        logits: torch.Tensor,
        unembed_proj: torch.Tensor,
        tokenizer,
    ) -> tuple[list[LogitTarget], torch.Tensor, torch.Tensor]:
        """Process mixed target list into LogitTarget instances, probabilities, vectors.

        Supports flexible mixed-mode targets where each element can be:
        - int: Token ID (computes probability and vector, uses actual vocab index)
        - str: Token string (tokenizes, computes probability and vector, uses actual token_id)
        - tuple[str, float, torch.Tensor]: Arbitrary string or function thereof with custom prob/vec
          (uses virtual index)

        Args:
            targets: List of attribution targets in any combination of the above formats
            logits: ``(d_vocab,)`` vector for computing probabilities
            unembed_proj: ``(d_model, d_vocab)`` unembedding matrix for computing vectors
            tokenizer: Tokenizer to use for string token conversion and to get vocab_size

        Returns:
            Tuple of:
                * logit_targets - List of LogitTarget instances where:
                    - For int/str tokens: vocab_idx is actual vocab index, token_str is decoded
                    - For tuple targets: vocab_idx is virtual (vocab_size + position),
                      token_str is the arbitrary string or function thereof
                * probabilities - ``(k,)`` probabilities
                * vectors - ``(k, d_model)`` demeaned vectors

        Raises:
            ValueError: If str token cannot be encoded or int token is out of vocab range
        """
        vocab_size = logits.shape[0]

        def validate_token_id(token_id: int, original_token: int | str) -> None:
            """Validate that token_id is within valid vocabulary range."""
            if not (0 <= token_id < vocab_size):
                raise ValueError(
                    f"Token {original_token!r} resolved to index {token_id}, which is "
                    f"out of vocabulary range [0, {vocab_size})"
                )

        def token_to_idx(token: int | str) -> int:
            """Convert token (int or str) to token index with validation."""
            if isinstance(token, str):
                try:
                    ids = tokenizer.encode(token, add_special_tokens=False)
                except Exception as e:
                    raise ValueError(
                        f"Failed to encode string token {token!r} using tokenizer: {e}"
                    ) from e

                if not ids:
                    raise ValueError(
                        f"String token {token!r} encoded to empty token sequence. "
                        f"Cannot determine valid token ID."
                    )

                token_id = ids[-1]
                validate_token_id(token_id, token)
                return token_id
            else:
                validate_token_id(token, token)
                return token

        logit_targets, probs, vecs = [], [], []

        for position, target in enumerate(targets):
            if isinstance(target, tuple):
                # Fully specified tuple: (str_token, probability, vector)
                # This is an arbitrary string or function of one, so we use virtual indices
                if len(target) != 3:
                    raise ValueError(
                        f"Tuple targets must have exactly 3 elements "
                        f"(token_str, probability, vector), got {len(target)}"
                    )
                token_str, prob, vec = target
                if not isinstance(token_str, str):
                    raise ValueError(
                        f"Tuple targets must have str as first element, got {type(token_str)}"
                    )

                # Use virtual index for arbitrary string/function thereof
                virtual_idx = vocab_size + position
                logit_targets.append(LogitTarget(token_str=token_str, vocab_idx=virtual_idx))
                probs.append(prob)
                vecs.append(vec)
            else:
                # Single token (int | str) - compute probability and vector, use valid token_ids
                idx = token_to_idx(target)
                idx_tensor = torch.tensor([idx], dtype=torch.long)
                _, prob_tensor, vec_tensor = AttributionTargets._compute_logit_vecs(
                    idx_tensor, logits, unembed_proj
                )

                token_str = tokenizer.decode(idx)
                logit_targets.append(LogitTarget(token_str=token_str, vocab_idx=idx))
                probs.append(prob_tensor.item())
                vecs.append(vec_tensor.squeeze(0))

        return logit_targets, torch.tensor(probs), torch.stack(vecs)
