import glob
import os

import numpy as np
import torch
from safetensors import safe_open
from torch.nn import functional as F

from circuit_tracer.transcoder.activation_functions import JumpReLU
from circuit_tracer.utils import get_default_device


class CrossLayerTranscoder(torch.nn.Module):
    """
    A cross-layer transcoder (CLT) where features read from one layer and write to all
    subsequent layers.

    Cross-layer transcoders are the core architecture enabling the circuit tracing methodology.
    Unlike per-layer transcoders, CLT features can "bridge over" multiple MLP layers, allowing
    a single feature to represent computation that spans the entire forward pass. This dramatically
    shortens paths in attribution graphs by collapsing amplification chains into single features.

    Each CLT feature has:
    - One encoder that reads from the residual stream at a specific layer
    - Multiple decoders that can write to all subsequent MLP outputs
    - The ability to represent cross-layer superposition where related computation
    is distributed across multiple transformer layers

    A single CLT provides an alternative to using multiple per-layer transcoders (managed by
    TranscoderSet) for feature-based model interpretation and replacement.

    Attributes:
        n_layers: Number of transformer layers the CLT spans
        d_transcoder: Number of features per layer
        d_model: Dimension of transformer residual stream
        W_enc: Encoder weights for each layer [n_layers, d_transcoder, d_model]
        W_dec: Decoder weights (lazily loaded) for cross-layer outputs
        b_enc: Encoder biases [n_layers, d_transcoder]
        b_dec: Decoder biases [n_layers, d_model]
        activation_function: Sparsity-inducing nonlinearity (default: ReLU)
        lazy_decoder: Whether to load decoder weights on-demand to save memory
        feature_input_hook: Hook point where features read from (e.g., "hook_resid_mid")
        feature_output_hook: Hook point where features write to (e.g., "hook_mlp_out")
        scan: Optional identifier for feature visualization
    """

    def __init__(
        self,
        n_layers: int,
        d_transcoder: int,
        d_model: int,
        activation_function: str = "relu",
        lazy_decoder=True,
        lazy_encoder=False,
        feature_input_hook: str = "hook_resid_mid",
        feature_output_hook: str = "hook_mlp_out",
        scan: str | list[str] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        clt_path: str | None = None,
    ):
        super().__init__()

        if device is None:
            device = get_default_device()

        self.n_layers = n_layers
        self.d_transcoder = d_transcoder
        self.d_model = d_model
        self.lazy_decoder = lazy_decoder
        self.lazy_encoder = lazy_encoder
        self.clt_path = clt_path

        self.feature_input_hook = feature_input_hook
        self.feature_output_hook = feature_output_hook
        self.skip_connection = False
        self.scan = scan

        if activation_function == "jump_relu":
            self.activation_function = JumpReLU(
                torch.zeros(n_layers, 1, d_transcoder, device=device, dtype=dtype)
            )
        elif activation_function == "relu":
            self.activation_function = F.relu
        else:
            raise ValueError(f"Invalid activation function: {activation_function}")

        if not lazy_encoder:
            self.W_enc = torch.nn.Parameter(
                torch.zeros(n_layers, d_transcoder, d_model, device=device, dtype=dtype)
            )

        self.b_dec = torch.nn.Parameter(torch.zeros(n_layers, d_model, device=device, dtype=dtype))
        self.b_enc = torch.nn.Parameter(
            torch.zeros(n_layers, d_transcoder, device=device, dtype=dtype)
        )

        if not lazy_decoder:
            self.W_dec = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.zeros(d_transcoder, n_layers - i, d_model, device=device, dtype=dtype)
                    )
                    for i in range(n_layers)
                ]
            )
        else:
            self.W_dec = None

    @property
    def device(self):
        """Get the device of the module's parameters."""
        return self.b_enc.device

    @property
    def dtype(self):
        """Get the dtype of the module's parameters."""
        return self.b_enc.dtype

    def _get_encoder_weights(self, layer_id=None):
        """Get encoder weights, loading from disk if lazy."""
        if not self.lazy_encoder:
            return self.W_enc if layer_id is None else self.W_enc[layer_id]

        assert self.clt_path is not None, "CLT path is not set"
        if layer_id is not None:
            # Load single layer encoder
            enc_file = os.path.join(self.clt_path, f"W_enc_{layer_id}.safetensors")
            with safe_open(enc_file, framework="pt", device=self.device.type) as f:
                return f.get_tensor(f"W_enc_{layer_id}").to(self.dtype)

        # Load all encoder weights
        W_enc = torch.zeros(
            self.n_layers, self.d_transcoder, self.d_model, device=self.device, dtype=self.dtype
        )
        for i in range(self.n_layers):
            enc_file = os.path.join(self.clt_path, f"W_enc_{i}.safetensors")
            with safe_open(enc_file, framework="pt", device=self.device.type) as f:
                W_enc[i] = f.get_tensor(f"W_enc_{i}").to(self.dtype)
        return W_enc

    def encode(self, x):
        W_enc = self._get_encoder_weights()
        features = torch.einsum("lbd,lfd->lbf", x, W_enc) + self.b_enc[:, None]
        return self.activation_function(features)

    def apply_activation_function(self, layer_id, features):
        if isinstance(self.activation_function, JumpReLU):
            mask = features > self.activation_function.threshold[layer_id]
            features = features * mask
        else:
            features = self.activation_function(features)
        return features

    def encode_layer(self, x, layer_id, apply_activation_function=True):
        W_enc_layer = self._get_encoder_weights(layer_id)
        features = torch.einsum("...d,fd->...f", x, W_enc_layer) + self.b_enc[layer_id]
        if not apply_activation_function:
            return features

        return self.apply_activation_function(layer_id, features)

    def encode_sparse(self, x, zero_first_pos=True):
        """Encode input to sparse activations, processing one layer at a time for memory efficiency.

        This method processes layers sequentially and converts to sparse format immediately
        to minimize peak memory usage, especially beneficial for large cross-layer transcoders.

        Args:
            x: Input tensor of shape (n_layers, n_pos, d_model)
            zero_first_pos: Whether to zero out position 0

        Returns:
            sparse_features: Sparse tensor of shape (n_layers, n_pos, d_transcoder)
            active_encoders: Encoder vectors for active features only
        """
        sparse_layers = []
        encoder_vectors = []

        for layer_id in range(self.n_layers):
            W_enc_layer = self._get_encoder_weights(layer_id)
            layer_features = (
                torch.einsum("bd,fd->bf", x[layer_id], W_enc_layer) + self.b_enc[layer_id]
            )

            layer_features = self.apply_activation_function(layer_id, layer_features)

            if zero_first_pos:
                layer_features[0] = 0

            sparse_layer = layer_features.to_sparse()
            sparse_layers.append(sparse_layer)

            _, feat_idx = sparse_layer.indices()
            encoder_vectors.append(W_enc_layer[feat_idx])

        sparse_features = torch.stack(sparse_layers).coalesce()
        active_encoders = torch.cat(encoder_vectors, dim=0)
        return sparse_features, active_encoders

    def _get_decoder_vectors(self, layer_id, feat_ids=None):
        to_read = feat_ids if feat_ids is not None else np.s_[:]

        if not self.lazy_decoder:
            assert self.W_dec is not None, "Decoder weights are not set"
            return self.W_dec[layer_id][to_read].to(self.dtype)

        assert self.clt_path is not None, "CLT path is not set"
        path = os.path.join(self.clt_path, f"W_dec_{layer_id}.safetensors")
        if isinstance(to_read, torch.Tensor):
            to_read = to_read.cpu()
        with safe_open(path, framework="pt", device=self.device.type) as f:
            return f.get_slice(f"W_dec_{layer_id}")[to_read].to(self.dtype)

    def select_decoder_vectors(self, features):
        if not features.is_sparse:
            features = features.to_sparse()
        layer_idx, pos_idx, feat_idx = features.indices()
        activations = features.values()
        n_layers = features.shape[0]
        device = features.device

        pos_ids = []
        layer_ids = []
        feat_ids = []

        decoder_vectors = []
        encoder_mapping = []
        st = 0

        for layer_id in range(n_layers):
            current_layer = layer_idx == layer_id
            if not current_layer.any():
                continue

            current_layer_features = feat_idx[current_layer]
            unique_feats, inv = current_layer_features.unique(return_inverse=True)

            unique_decoders = self._get_decoder_vectors(layer_id, unique_feats.cpu())
            scaled_decoders = unique_decoders[inv] * activations[current_layer, None, None]
            decoder_vectors.append(scaled_decoders.reshape(-1, self.d_model))

            n_output_layers = self.n_layers - layer_id
            pos_ids.append(pos_idx[current_layer].repeat_interleave(n_output_layers))
            feat_ids.append(current_layer_features.repeat_interleave(n_output_layers))
            layer_ids.append(
                torch.arange(layer_id, self.n_layers, device=device).repeat(
                    len(current_layer_features)
                )
            )

            source_ids = torch.arange(len(current_layer_features), device=device) + st
            st += len(current_layer_features)
            encoder_mapping.append(torch.repeat_interleave(source_ids, n_output_layers))

        pos_ids = torch.cat(pos_ids, dim=0)
        layer_ids = torch.cat(layer_ids, dim=0)
        feat_ids = torch.cat(feat_ids, dim=0)
        decoder_vectors = torch.cat(decoder_vectors, dim=0)
        encoder_mapping = torch.cat(encoder_mapping, dim=0)

        return pos_ids, layer_ids, feat_ids, decoder_vectors, encoder_mapping

    def compute_reconstruction(self, pos_ids, layer_ids, decoder_vectors):
        n_pos = pos_ids.max() + 1
        flat_idx = layer_ids * n_pos + pos_ids
        recon = torch.zeros(
            n_pos * self.n_layers,
            self.d_model,
            device=decoder_vectors.device,
            dtype=decoder_vectors.dtype,
        ).index_add_(0, flat_idx, decoder_vectors)
        return recon.reshape(self.n_layers, n_pos, self.d_model) + self.b_dec[:, None]

    def decode(self, features):
        pos_ids, layer_ids, feat_ids, decoder_vectors, _ = self.select_decoder_vectors(features)
        return self.compute_reconstruction(pos_ids, layer_ids, decoder_vectors)

    def forward(self, x):
        features = self.encode(x).to_sparse()
        return self.decode(features)

    def compute_attribution_components(self, inputs):
        """Extract active features and their encoder/decoder vectors for attribution.

        Args:
            inputs: Input tensor to encode

        Returns:
            Dict containing all components needed for AttributionContext:
                - activation_matrix: Sparse activation matrix
                - reconstruction: Reconstructed outputs
                - encoder_vecs: Concatenated encoder vectors for active features
                - decoder_vecs: Concatenated decoder vectors (scaled by activations)
                - encoder_to_decoder_map: Mapping from encoder to decoder indices
        """
        features, encoder_vectors = self.encode_sparse(inputs, zero_first_pos=True)
        pos_ids, layer_ids, feat_ids, decoder_vectors, encoder_to_decoder_map = (
            self.select_decoder_vectors(features)
        )
        reconstruction = self.compute_reconstruction(pos_ids, layer_ids, decoder_vectors)

        return {
            "activation_matrix": features,
            "reconstruction": reconstruction,
            "encoder_vecs": encoder_vectors,
            "decoder_vecs": decoder_vectors,
            "encoder_to_decoder_map": encoder_to_decoder_map,
            "decoder_locations": torch.stack((layer_ids, pos_ids)),
        }


def load_clt(
    clt_path: str,
    feature_input_hook: str = "hook_resid_mid",
    feature_output_hook: str = "hook_mlp_out",
    scan: str | list[str] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    lazy_decoder: bool = True,
    lazy_encoder: bool = False,
) -> CrossLayerTranscoder:
    """Load a cross-layer transcoder from safetensors files.

    Args:
        clt_path: Path to directory containing W_enc_*.safetensors and W_dec_*.safetensors files
        dtype: Data type for loaded tensors
        lazy_decoder: Whether to load decoder weights on-demand
        lazy_encoder: Whether to load encoder weights on-demand
        feature_input_hook: Hook point where features read from
        feature_output_hook: Hook point where features write to
        scan: Optional identifier for feature visualization
        device: Device to load tensors to (defaults to auto-detected)

    Returns:
        CrossLayerTranscoder: Loaded transcoder instance
    """
    if device is None:
        device = get_default_device()

    state_dict = _load_state_dict(clt_path, lazy_decoder, lazy_encoder, device, dtype)

    # Infer dimensions from loaded tensors
    n_layers = state_dict["b_dec"].shape[0]
    d_transcoder = state_dict["b_enc"].shape[1]
    d_model = state_dict["b_dec"].shape[1]

    act_fn = "jump_relu" if "activation_function.threshold" in state_dict else "relu"

    # Create instance and load state dict
    with torch.device("meta"):
        instance = CrossLayerTranscoder(
            n_layers,
            d_transcoder,
            d_model,
            activation_function=act_fn,
            lazy_decoder=lazy_decoder,
            lazy_encoder=lazy_encoder,
            feature_input_hook=feature_input_hook,
            feature_output_hook=feature_output_hook,
            scan=scan,
            dtype=dtype,
            clt_path=clt_path,
        )

    instance.load_state_dict(state_dict, assign=True)

    return instance


def _load_state_dict(
    clt_path, lazy_decoder=True, lazy_encoder=False, device=None, dtype=torch.bfloat16
):
    if device is None:
        device = get_default_device()

    enc_files = glob.glob(os.path.join(clt_path, "W_enc_*.safetensors"))
    n_layers = len(enc_files)

    # Get dimensions from first file
    dec_file = "W_enc_0.safetensors"
    with safe_open(os.path.join(clt_path, dec_file), framework="pt", device=device.type) as f:
        d_transcoder, d_model = f.get_slice("W_enc_0").get_shape()
        has_threshold = "threshold_0" in f.keys()

    # Preallocate tensors
    b_dec = torch.zeros(n_layers, d_model, device=device, dtype=dtype)
    b_enc = torch.zeros(n_layers, d_transcoder, device=device, dtype=dtype)

    state_dict = {"b_dec": b_dec, "b_enc": b_enc}

    if has_threshold:
        state_dict["activation_function.threshold"] = torch.zeros(
            n_layers, 1, d_transcoder, device=device, dtype=dtype
        )

    # Only create W_enc if not lazy
    if not lazy_encoder:
        W_enc = torch.zeros(n_layers, d_transcoder, d_model, device=device, dtype=dtype)
        state_dict["W_enc"] = W_enc

    # Load all layers
    for i in range(n_layers):
        enc_file = f"W_enc_{i}.safetensors"
        with safe_open(os.path.join(clt_path, enc_file), framework="pt", device=device.type) as f:
            b_dec[i] = f.get_tensor(f"b_dec_{i}").to(dtype)
            b_enc[i] = f.get_tensor(f"b_enc_{i}").to(dtype)

            # Only load W_enc if not lazy
            if not lazy_encoder:
                W_enc[i] = f.get_tensor(f"W_enc_{i}").to(dtype)

            if has_threshold:
                threshold = f.get_tensor(f"threshold_{i}").to(dtype)
                state_dict["activation_function.threshold"][i] = threshold.unsqueeze(0)

        # Load W_dec for this layer if not lazy
        if not lazy_decoder:
            dec_file = os.path.join(clt_path, f"W_dec_{i}.safetensors")
            with safe_open(dec_file, framework="pt", device=device.type) as f:
                state_dict[f"W_dec.{i}"] = f.get_tensor(f"W_dec_{i}").to(dtype)

    return state_dict
