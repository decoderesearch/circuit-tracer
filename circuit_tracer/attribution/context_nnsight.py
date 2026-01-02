"""
Attribution context for managing hooks during attribution computation.
"""

import weakref
from typing import TYPE_CHECKING

import numpy as np
import torch
from einops import einsum


if TYPE_CHECKING:
    from circuit_tracer.replacement_model.replacement_model_nnsight import (
        NNSightReplacementModel,
    )


class AttributionContext:
    """Manage hooks for computing attribution rows.

    This helper caches residual-stream activations **(forward pass)** and then
    registers backward hooks that populate a write-only buffer with
    *direct-effect rows* **(backward pass)**.

    The buffer layout concatenates rows for **feature nodes**, **error nodes**,
    **token-embedding nodes**

    Args:
        activation_matrix (torch.sparse.Tensor):
            Sparse `(n_layers, n_pos, n_features)` tensor indicating **which**
            features fired at each layer/position.
        error_vectors (torch.Tensor):
            `(n_layers, n_pos, d_model)` - *residual* the CLT / PLT failed to
            reconstruct ("error nodes").
        token_vectors (torch.Tensor):
            `(n_pos, d_model)` - embeddings of the prompt tokens.
        decoder_vectors (torch.Tensor):
            `(total_active_features, d_model)` - decoder rows **only for active
            features**, already multiplied by feature activations so they
            represent a_s * W^dec.
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

        # Forward-pass cache
        self._resid_activations: list[torch.Tensor] = []
        self._feature_output_activations: list[torch.Tensor] = []
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

    def cache_residual(self, model: "NNSightReplacementModel", tracer, barrier=None):
        """Cache the model's residual for use in the attribution context."""
        with tracer.invoke():
            for feature_input_loc in model.feature_input_locs:
                self._resid_activations.append(feature_input_loc.output)  # type: ignore

            self._resid_activations.append(model.pre_logit_location.output.last_hidden_state)  # type: ignore

        with tracer.invoke():
            self._feature_output_activations.append(model.embed_location.output)  # type: ignore
            for feature_output_loc_ in model.feature_output_locs:
                if barrier:
                    barrier()

                self._feature_output_activations.append(feature_output_loc_.output)  # type: ignore

    def compute_score(
        self,
        grads: torch.Tensor,
        output_vecs: torch.Tensor,
        write_index: slice,
        read_index: slice | np.ndarray = np.s_[:],
    ) -> None:
        """
        Factory that contracts *gradients* with an **output vector set**.
        The hook computes A_{s->t} and accumulates the result into an in-place buffer row.
        """

        proxy = weakref.proxy(self)
        proxy._batch_buffer[write_index] += einsum(
            grads.to(output_vecs.dtype)[read_index],
            output_vecs,
            "batch position d_model, position d_model -> position batch",
        )

    def compute_feature_attributions(self, layer, grads):
        nnz_layers, nnz_positions = self.decoder_locations

        # Feature nodes - use decoder_locations to find decoders that write to this layer
        layer_mask = nnz_layers == layer
        if layer_mask.any():
            self.compute_score(
                grads,
                self.decoder_vecs[layer_mask],
                write_index=self.encoder_to_decoder_map[layer_mask],  # type: ignore
                read_index=np.s_[:, nnz_positions[layer_mask]],  # type: ignore
            )

    def compute_error_attributions(self, layer, grads):
        _, n_pos, _ = self.activation_matrix.shape

        # Error nodes
        def error_offset(layer: int) -> int:  # starting row for this layer
            return self.activation_matrix._nnz() + layer * n_pos

        self.compute_score(
            grads,
            self.error_vectors[layer],
            write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
        )

    def compute_token_attributions(self, grads):
        n_layers, n_pos, _ = self.activation_matrix.shape

        # Token-embedding nodes
        def error_offset(layer: int) -> int:  # starting row for this layer
            return self.activation_matrix._nnz() + layer * n_pos

        tok_start = error_offset(n_layers)
        self.compute_score(
            grads,
            self.token_vectors,
            write_index=np.s_[tok_start : tok_start + n_pos],
        )

    def compute_batch(
        self,
        layers: torch.Tensor,
        positions: torch.Tensor,
        inject_values: torch.Tensor,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        """Return attribution rows for a batch of (layer, pos) nodes.

        The routine overrides gradients at **exact** residual-stream locations
        triggers one backward pass, and copies the rows from the internal buffer.

        Args:
            layers: 1-D tensor of layer indices *l* for the source nodes.
            positions: 1-D tensor of token positions *c* for the source nodes.
            inject_values: `(batch, d_model)` tensor with outer product
                a_s * W^(enc/dec) to inject as custom gradient.

        Returns:
            torch.Tensor: ``(batch, row_size)`` matrix - one row per node.
        """

        batch_size = self._resid_activations[0].shape[0]
        self._batch_buffer = torch.zeros(
            self._row_size,
            batch_size,
            dtype=inject_values.dtype,
            device=inject_values.device,
        )

        # Custom gradient injection (per-layer registration)
        batch_idx = torch.arange(len(layers), device=layers.device)

        def _inject(grad_point, *, batch_indices, pos_indices, values):
            grads_out = grad_point.grad.clone()
            grads_out.index_put_((batch_indices, pos_indices), values.to(grads_out.dtype))
            grad_point.grad = grads_out

        layers_in_batch = sorted(layers.unique().tolist(), reverse=True)

        last_layer = max(layers_in_batch)
        with self._resid_activations[last_layer].backward(
            gradient=torch.zeros_like(self._resid_activations[last_layer]),
            retain_graph=retain_graph,
        ):
            for layer in reversed(range(last_layer + 1)):
                if layer != last_layer:
                    grad = self._feature_output_activations[layer + 1].grad.clone()  # type:ignore
                    self.compute_feature_attributions(layer, grad)
                    self.compute_error_attributions(layer, grad)

                mask = layers == layer
                if mask.any():
                    _inject(
                        grad_point=self._resid_activations[layer],
                        batch_indices=batch_idx[mask],
                        pos_indices=positions[mask],
                        values=inject_values[mask],
                    )

            self.compute_token_attributions(self._feature_output_activations[0].grad)

        buf, self._batch_buffer = self._batch_buffer, None
        return buf.T[: len(layers)]
