"""Attribution context for the interp_engine backend.

Same contract as ``context_transformerlens.AttributionContext`` -- cache the residual stream on
the forward pass, then contract gradients against output vectors on the backward pass -- but
built on plain tensor hooks instead of named hook points.

That substitution is what lets this work on an unmodified HF model. TransformerLens can name a
backward hook site because it inserted one; here the equivalent site is simply *the tensor* the
model already produced there, and ``Tensor.register_hook`` attaches to it. Registration can
happen any time before ``backward``, so the forward pass runs first and the hooks go on the
tensors it left behind, rather than having to be installed in advance.
"""

import contextlib
import weakref
from functools import partial
from typing import TYPE_CHECKING, Iterator

import numpy as np
import torch
from einops import einsum

if TYPE_CHECKING:
    from circuit_tracer.replacement_model.replacement_model_interp_engine import (
        InterpEngineReplacementModel,
    )

#: A score hook gathers the gradient to [batch, active features, d_model] before contracting it.
#: Past this many elements it contracts in chunks instead: MPS refuses any tensor over INT_MAX
#: elements, and on every device a dense layer -- Qwen3-4B's transcoders reach 30k active features
#: on a four-token prompt -- would otherwise allocate gigabytes for one contraction.
SCORE_CHUNK_ELEMENTS = 2**30


class AttributionContext:
    """Manage hooks for computing attribution rows.

    This helper caches residual-stream activations **(forward pass)** and then registers backward
    hooks that populate a write-only buffer with *direct-effect rows* **(backward pass)**.

    The buffer layout concatenates rows for **feature nodes**, **error nodes** and
    **token-embedding nodes**.

    Args:
        activation_matrix (torch.sparse.Tensor):
            Sparse `(n_layers, n_pos, n_features)` tensor indicating **which** features fired at
            each layer/position.
        error_vectors (torch.Tensor):
            `(n_layers, n_pos, d_model)` - *residual* the CLT / PLT failed to reconstruct
            ("error nodes").
        token_vectors (torch.Tensor):
            `(n_pos, d_model)` - the token embeddings as they enter the residual stream.
        decoder_vecs (torch.Tensor):
            `(total_active_features, d_model)` - decoder rows **only for active features**,
            already multiplied by feature activations so they represent a_s * W^dec.
    """

    def __init__(
        self,
        activation_matrix: torch.sparse.Tensor,  # type: ignore
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor,
        decoder_vecs: torch.Tensor,
        encoder_vecs: torch.Tensor,
        encoder_to_decoder_map: torch.Tensor,
        decoder_locations: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        n_layers, n_pos, _ = activation_matrix.shape

        # Forward-pass cache. One slot per layer plus one for the final normalized residual, which
        # is where logit attributions are taken from.
        self._resid_activations: list[torch.Tensor | None] = [None] * (n_layers + 1)
        self._batch_buffer: torch.Tensor | None = None
        self.n_layers: int = n_layers

        self.logits = logits
        self.activation_matrix = activation_matrix
        self.error_vectors = error_vectors
        self.token_vectors = token_vectors
        self.decoder_vecs = decoder_vecs
        self.encoder_vecs = encoder_vecs

        self.encoder_to_decoder_map = encoder_to_decoder_map
        self.decoder_locations = decoder_locations

        total_active_feats = activation_matrix._nnz()
        self._row_size: int = total_active_feats + (n_layers + 1) * n_pos  # + logits later

        self._grad_handles: list[torch.utils.hooks.RemovableHandle] = []

    def _score_hook(
        self,
        output_vecs: torch.Tensor,
        write_index: slice | np.ndarray,
        read_index: slice | np.ndarray = np.s_[:],
    ):
        """Contract *gradients* with an **output vector set**, into an in-place buffer row.

        The hook computes A_{s->t}. A weak proxy is used so that a hook left registered on a
        tensor cannot keep the whole context alive.
        """
        proxy = weakref.proxy(self)
        pattern = "batch position d_model, position d_model -> position batch"

        def _hook_fn(grads: torch.Tensor) -> None:
            grads = grads.to(output_vecs.dtype)
            rows = output_vecs.shape[0]
            per_row = grads.shape[0] * grads.shape[-1]
            if rows * per_row <= SCORE_CHUNK_ELEMENTS or not isinstance(read_index, tuple):
                proxy._batch_buffer[write_index] += einsum(grads[read_index], output_vecs, pattern)  # type: ignore[index]
                return
            # The gather `grads[:, positions]` would be [batch, rows, d_model], one row per active
            # feature of this layer: contracted in chunks of rows instead, so it is never formed.
            positions = read_index[1]
            step = max(1, SCORE_CHUNK_ELEMENTS // per_row)
            scores = torch.empty(rows, grads.shape[0], dtype=grads.dtype, device=grads.device)
            for start in range(0, rows, step):
                stop = start + step
                scores[start:stop] = einsum(
                    grads[:, positions[start:stop]], output_vecs[start:stop], pattern
                )
            proxy._batch_buffer[write_index] += scores  # type: ignore[index]

        return _hook_fn

    def _install_score_hooks(self, model: "InterpEngineReplacementModel") -> None:
        """Attach the score hooks to the tensors the forward pass just produced."""
        n_layers, n_pos, _ = self.activation_matrix.shape
        nnz_layers, nnz_positions = self.decoder_locations

        feature_outputs = model.get_feature_output_tensors()

        def error_offset(layer: int) -> int:  # starting row for this layer
            return self.activation_matrix._nnz() + layer * n_pos

        for layer in range(n_layers):
            tensor = feature_outputs[layer]

            # Feature nodes: only the layers that actually had an active feature.
            layer_mask = nnz_layers == layer
            if layer_mask.any():
                self._grad_handles.append(
                    tensor.register_hook(
                        self._score_hook(
                            self.decoder_vecs[layer_mask],
                            write_index=self.encoder_to_decoder_map[layer_mask],  # type: ignore[arg-type]
                            read_index=np.s_[:, nnz_positions[layer_mask]],  # type: ignore[index]
                        )
                    )
                )

            # Error nodes: every layer has one per position.
            self._grad_handles.append(
                tensor.register_hook(
                    self._score_hook(
                        self.error_vectors[layer],
                        write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
                    )
                )
            )

        # Token-embedding nodes.
        tok_start = error_offset(n_layers)
        self._grad_handles.append(
            model.embed_tensor.register_hook(
                self._score_hook(
                    self.token_vectors,
                    write_index=np.s_[tok_start : tok_start + n_pos],
                )
            )
        )

    @contextlib.contextmanager
    def install_hooks(self, model: "InterpEngineReplacementModel") -> Iterator[None]:
        """Run the caller's forward pass, then wire the backward hooks onto its tensors.

        The model's own permanent hooks already cache what the transcoders read and write on every
        forward pass, so there is nothing to install beforehand; the work is all on the way out.
        """
        try:
            yield
            self._resid_activations[: self.n_layers] = model.get_feature_input_tensors()
            self._install_score_hooks(model)
        except BaseException:
            self.remove_hooks()
            raise

    def remove_hooks(self) -> None:
        for handle in self._grad_handles:
            handle.remove()
        self._grad_handles.clear()

    def compute_batch(
        self,
        layers: torch.Tensor,
        positions: torch.Tensor,
        inject_values: torch.Tensor,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        """Return attribution rows for a batch of (layer, pos) nodes.

        The routine overrides gradients at **exact** residual-stream locations, triggers one
        backward pass, and copies the rows from the internal buffer.

        Args:
            layers: 1-D tensor of layer indices *l* for the source nodes.
            positions: 1-D tensor of token positions *c* for the source nodes.
            inject_values: `(batch, d_model)` tensor with outer product a_s * W^(enc/dec) to
                inject as custom gradient.
            retain_graph: Keep the graph for the next batch.

        Returns:
            torch.Tensor: ``(batch, row_size)`` matrix - one row per node.
        """

        batch_size = self._resid_activations[0].shape[0]  # type: ignore[union-attr]
        self._batch_buffer = torch.zeros(
            self._row_size,
            batch_size,
            dtype=inject_values.dtype,
            device=inject_values.device,
        )

        # Custom gradient injection (per-layer registration)
        batch_idx = torch.arange(len(layers), device=layers.device)

        def _inject(grads, *, batch_indices, pos_indices, values):
            grads_out = grads.clone().to(values.dtype)
            grads_out.index_put_((batch_indices, pos_indices), values)
            return grads_out.to(grads.dtype)

        handles = []
        layers_in_batch = layers.unique().tolist()

        for layer in layers_in_batch:
            mask = layers == layer
            if not mask.any():
                continue
            fn = partial(
                _inject,
                batch_indices=batch_idx[mask],
                pos_indices=positions[mask],
                values=inject_values[mask],
            )
            handles.append(self._resid_activations[int(layer)].register_hook(fn))  # type: ignore[union-attr]

        try:
            last_layer = max(layers_in_batch)
            self._resid_activations[last_layer].backward(  # type: ignore[union-attr]
                gradient=torch.zeros_like(self._resid_activations[last_layer]),  # type: ignore[arg-type]
                retain_graph=retain_graph,
            )
        finally:
            for h in handles:
                h.remove()

        buf, self._batch_buffer = self._batch_buffer, None
        return buf.T[: len(layers)]
