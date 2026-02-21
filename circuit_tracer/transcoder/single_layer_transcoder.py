import os
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from circuit_tracer.transcoder.activation_functions import JumpReLU
from circuit_tracer.utils import get_default_device


class SingleLayerTranscoder(nn.Module):
    """
    A per-layer transcoder (PLT) that replaces MLP computation with interpretable features.

    Per-layer transcoders decompose the output of a single MLP layer into sparsely active
    features that often correspond to interpretable concepts. Unlike cross-layer transcoders,
    each PLT operates independently on its assigned layer, which can result in longer paths
    through attribution graphs when features amplify across multiple layers.

    Attributes:
        d_model: Dimension of the transformer's residual stream
        d_transcoder: Number of learned features (typically >> d_model for superposition)
        layer_idx: Which transformer layer this transcoder replaces
        W_enc: Encoder weights mapping residual stream to feature space
        W_dec: Decoder weights mapping features back to residual stream
        b_enc: Encoder bias terms
        b_dec: Decoder bias terms (reconstruction baseline)
        W_skip: Optional skip connection weights (https://arxiv.org/abs/2501.18823)
        activation_function: Sparsity-inducing nonlinearity (e.g., ReLU, JumpReLU)
    """

    def __init__(
        self,
        d_model: int,
        d_transcoder: int,
        activation_function,
        layer_idx: int,
        skip_connection: bool = False,
        transcoder_path: str | None = None,
        lazy_encoder: bool = False,
        lazy_decoder: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        if device is None:
            device = get_default_device()

        self.d_model = d_model
        self.d_transcoder = d_transcoder
        self.layer_idx = layer_idx
        self.transcoder_path = transcoder_path
        self.lazy_encoder = lazy_encoder
        self.lazy_decoder = lazy_decoder

        if lazy_encoder or lazy_decoder:
            assert self.transcoder_path is not None, "Transcoder path must be set for lazy loading"

        if not lazy_encoder:
            self.W_enc = nn.Parameter(
                torch.zeros(d_transcoder, d_model, device=device, dtype=dtype)
            )

        if not lazy_decoder:
            self.W_dec = nn.Parameter(
                torch.zeros(d_transcoder, d_model, device=device, dtype=dtype)
            )

        self.b_enc = nn.Parameter(torch.zeros(d_transcoder, device=device, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))

        if skip_connection:
            self.W_skip = nn.Parameter(torch.zeros(d_model, d_model, device=device, dtype=dtype))
        else:
            self.W_skip = None

        self.activation_function = activation_function

    @property
    def device(self):
        """Get the device of the module's parameters."""
        return next(self.parameters()).device

    @property
    def dtype(self):
        """Get the dtype of the module's parameters."""
        return self.b_enc.dtype

    def __getattr__(self, name):
        """Dynamically load weights when accessed if lazy loading is enabled."""

        if name == "W_enc" and self.lazy_encoder and self.transcoder_path is not None:
            with safe_open(self.transcoder_path, framework="pt", device=str(self.device)) as f:
                return f.get_tensor("W_enc").to(self.dtype)
        elif name == "W_dec" and self.lazy_decoder and self.transcoder_path is not None:
            with safe_open(self.transcoder_path, framework="pt", device=str(self.device)) as f:
                return f.get_tensor("W_dec").to(self.dtype)

        return super().__getattr__(name)

    def _get_decoder_vectors(self, feat_ids=None):
        to_read = feat_ids if feat_ids is not None else np.s_[:]
        if not self.lazy_decoder:
            return self.W_dec[to_read].to(self.dtype)

        if isinstance(to_read, torch.Tensor):
            to_read = to_read.cpu()
        with safe_open(self.transcoder_path, framework="pt", device=str(self.device)) as f:
            return f.get_slice("W_dec")[to_read].to(self.dtype)

    def encode(self, input_acts, apply_activation_function: bool = True):
        W_enc = self.W_enc
        pre_acts = F.linear(input_acts.to(W_enc.dtype), W_enc, self.b_enc)
        if not apply_activation_function:
            return pre_acts
        return self.activation_function(pre_acts)

    def decode(self, acts, input_acts: torch.Tensor | None = None):
        W_dec = self.W_dec
        reconstruction = acts @ W_dec + self.b_dec
        if self.W_skip is not None:
            assert input_acts is not None, (
                "Transcoder has skip connection but no input_acts were provided"
            )
            reconstruction = reconstruction + self.compute_skip(input_acts)
        return reconstruction

    def compute_skip(self, input_acts):
        if self.W_skip is not None:
            return input_acts @ self.W_skip.T
        else:
            raise ValueError("Transcoder has no skip connection")

    def forward(self, input_acts):
        transcoder_acts = self.encode(input_acts)
        decoded = self.decode(transcoder_acts, input_acts)
        # decoded = decoded.detach()
        # decoded.requires_grad = True

        return decoded

    def encode_sparse(self, input_acts, zero_positions: slice = slice(0, 1)):
        """Encode and return sparse activations with active encoder vectors.

        Args:
            input_acts: Input activations
            zero_positions: slice representing the positions to zero out

        Returns:
            sparse_acts: Sparse tensor of activations
            active_encoders: Encoder vectors for active features only
        """
        W_enc = self.W_enc
        pre_acts = F.linear(input_acts.to(W_enc.dtype), W_enc, self.b_enc)
        acts = self.activation_function(pre_acts)

        acts[zero_positions] = 0

        sparse_acts = acts.to_sparse()
        _, feat_idx = sparse_acts.indices()
        active_encoders = W_enc[feat_idx]

        return sparse_acts, active_encoders

    def decode_sparse(self, sparse_acts, input_acts: torch.Tensor | None = None):
        """Decode sparse activations and return reconstruction with scaled decoder vectors.

        Returns:
            reconstruction: Decoded output
            scaled_decoders: Decoder vectors scaled by activation values
        """
        pos_idx, feat_idx = sparse_acts.indices()
        values = sparse_acts.values()

        # Get decoder vectors for active features only
        W_dec = self._get_decoder_vectors(feat_idx.cpu())
        scaled_decoders = W_dec * values[:, None]

        # Reconstruct using index_add
        n_pos = sparse_acts.shape[0]
        reconstruction = torch.zeros(
            n_pos, self.d_model, device=sparse_acts.device, dtype=sparse_acts.dtype
        )
        reconstruction = reconstruction.index_add_(0, pos_idx, scaled_decoders)
        if self.W_skip is not None:
            assert input_acts is not None, (
                "Transcoder has skip connection but no input_acts were provided"
            )
            reconstruction = reconstruction + self.compute_skip(input_acts)
        reconstruction = reconstruction + self.b_dec

        return reconstruction, scaled_decoders

    def to_safetensors(self, save_path: str):
        """Save transcoder to safetensors format compatible with lazy loading.

        Saves the transcoder state dict to a single safetensors file with keys:
        W_enc, W_dec, b_enc, b_dec, and optionally activation_function.threshold and W_skip.

        Args:
            save_path: Path to the safetensors file to save
        """
        state_dict = {
            "W_enc": self.W_enc.cpu(),
            "W_dec": self.W_dec.cpu(),
            "b_enc": self.b_enc.cpu(),
            "b_dec": self.b_dec.cpu(),
        }

        if isinstance(self.activation_function, JumpReLU):
            state_dict["activation_function.threshold"] = self.activation_function.threshold.cpu()

        if self.W_skip is not None:
            state_dict["W_skip"] = self.W_skip.cpu()

        save_file(state_dict, save_path)


class TranscoderSet(nn.Module):
    """
    A collection of per-layer transcoders that enable construction of a replacement model.

    TranscoderSet manages the collection of SingleLayerTranscoders needed for this substitution,
    where each transcoder replaces the MLP computation at its corresponding layer.

    Attributes:
        transcoders: ModuleList of SingleLayerTranscoder instances, one per layer
        n_layers: Total number of layers covered
        d_transcoder: Common feature dimension across all transcoders
        feature_input_hook: Hook point where features read from (e.g., "hook_resid_mid")
        feature_output_hook: Hook point where features write to (e.g., "hook_mlp_out")
        scan: Optional identifier to identify corresponding feature visualization
        skip_connection: Whether transcoders include learned skip connections
    """

    def __init__(
        self,
        transcoders: dict[int, SingleLayerTranscoder],
        feature_input_hook: str,
        feature_output_hook: str,
        scan: str | list[str] | None = None,
    ):
        super().__init__()
        # Validate that we have continuous layers from 0 to max
        assert set(transcoders.keys()) == set(range(max(transcoders.keys()) + 1)), (
            f"Each layer should have a transcoder, but got transcoders for layers "
            f"{set(transcoders.keys())}"
        )

        self.transcoders = nn.ModuleList([transcoders[i] for i in range(len(transcoders))])
        self.n_layers = len(self.transcoders)
        self.d_transcoder = self.transcoders[0].d_transcoder

        # Verify all transcoders have the same d_transcoder
        for transcoder in self.transcoders:
            assert transcoder.d_transcoder == self.d_transcoder, (
                f"All transcoders must have the same d_transcoder, but got "
                f"{transcoder.d_transcoder} != {self.d_transcoder}"
            )

        # Store hook configuration
        self.feature_input_hook = feature_input_hook
        self.feature_output_hook = feature_output_hook
        self.scan = scan
        self.skip_connection = self.transcoders[0].W_skip is not None

    def __len__(self):
        return self.n_layers

    def __getitem__(self, idx: int) -> SingleLayerTranscoder:
        return self.transcoders[idx]  # type: ignore

    def __iter__(self) -> Iterator[SingleLayerTranscoder]:
        return iter(self.transcoders)  # type: ignore

    def apply_activation_function(self, layer_id, features):
        return self.transcoders[layer_id].activation_function(features)  # type: ignore

    def compute_skip(self, layer_id: int, inputs):
        return self.transcoders[layer_id].compute_skip(inputs)  # type: ignore

    def encode(self, input_acts):
        return torch.stack(
            [transcoder.encode(input_acts[i]) for i, transcoder in enumerate(self.transcoders)],  # type: ignore
            dim=0,
        )

    def _get_decoder_vectors(self, layer_id, features):
        return self.transcoders[layer_id]._get_decoder_vectors(features)  # type: ignore

    def select_decoder_vectors(self, features):
        if not features.is_sparse:
            features = features.to_sparse()

        all_layer_idx, all_pos_idx, all_feat_idx = features.indices()
        all_activations = features.values()
        all_scaled_decoder_vectors = []
        for unique_layer in all_layer_idx.unique():
            layer_mask = all_layer_idx == unique_layer
            feat_idx = all_feat_idx[layer_mask]
            activations = all_activations[layer_mask]

            decoder_vectors = self._get_decoder_vectors(unique_layer.item(), feat_idx)

            # Multiply each activation by its corresponding decoder vector
            scaled_decoder_vectors = activations.unsqueeze(-1) * decoder_vectors
            all_scaled_decoder_vectors.append(scaled_decoder_vectors)

        all_scaled_decoder_vectors = torch.cat(all_scaled_decoder_vectors)
        encoder_mapping = torch.arange(features._nnz(), device=features.device)

        return (
            all_pos_idx,
            all_layer_idx,
            all_feat_idx,
            all_scaled_decoder_vectors,
            encoder_mapping,
        )

    def decode(self, acts, input_acts: torch.Tensor | None):
        return torch.stack(
            [
                transcoder.decode(acts[i], None if input_acts is None else input_acts[i])
                for i, transcoder in enumerate[SingleLayerTranscoder](self.transcoders)  # type: ignore
            ],
            dim=0,
        )

    def compute_attribution_components(
        self, mlp_inputs: torch.Tensor, zero_positions: slice = slice(0, 1)
    ) -> dict[str, torch.Tensor]:
        """Extract active features and their encoder/decoder vectors for attribution.

        Args:
            mlp_inputs: (n_layers, n_pos, d_model) tensor of MLP inputs
            zero_positions: (slice) slice indicating which positions to zero out

        Returns:
            Dict containing all components needed for AttributionContext:
                - activation_matrix: Sparse (n_layers, n_pos, d_transcoder) activations
                - reconstruction: (n_layers, n_pos, d_model) reconstructed outputs
                - encoder_vecs: Concatenated encoder vectors for active features
                - decoder_vecs: Concatenated decoder vectors (scaled by activations)
                - encoder_to_decoder_map: Mapping from encoder to decoder indices
        """
        device = mlp_inputs.device

        reconstruction = torch.zeros_like(mlp_inputs)
        encoder_vectors = []
        decoder_vectors = []
        sparse_acts_list = []

        for layer, transcoder in enumerate[SingleLayerTranscoder](self.transcoders):  # type: ignore
            sparse_acts, active_encoders = transcoder.encode_sparse(
                mlp_inputs[layer], zero_positions=zero_positions
            )
            reconstruction[layer], active_decoders = transcoder.decode_sparse(
                sparse_acts, mlp_inputs[layer]
            )
            encoder_vectors.append(active_encoders)
            decoder_vectors.append(active_decoders)
            sparse_acts_list.append(sparse_acts)

        activation_matrix = torch.stack(sparse_acts_list).coalesce()
        encoder_to_decoder_map = torch.arange(activation_matrix._nnz(), device=device)

        return {
            "activation_matrix": activation_matrix,
            "reconstruction": reconstruction,
            "encoder_vecs": torch.cat(encoder_vectors, dim=0),
            "decoder_vecs": torch.cat(decoder_vectors, dim=0),
            "encoder_to_decoder_map": encoder_to_decoder_map,
            "decoder_locations": activation_matrix.indices()[:2],
        }

    def encode_layer(self, x, layer_id, apply_activation_function=True):
        return self.transcoders[layer_id].encode(
            x, apply_activation_function=apply_activation_function
        )  # type: ignore

    def to_safetensors(self, save_dir: str):
        """Save all transcoders in the set to safetensors files.

        Saves each transcoder as layer_{i}.safetensors in the specified directory.

        Args:
            save_dir: Directory path where the safetensors files will be saved
        """
        os.makedirs(save_dir, exist_ok=True)

        for i, transcoder in enumerate(self.transcoders):
            save_path = os.path.join(save_dir, f"layer_{i}.safetensors")
            transcoder.to_safetensors(save_path)  # type: ignore


def load_gemma_scope_transcoder(
    path: str,
    layer: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    revision: str | None = None,
    **kwargs,
) -> SingleLayerTranscoder:
    if device is None:
        device = get_default_device()
    if os.path.isfile(path):
        path_to_params = path
    else:
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-2b-pt-transcoders",
            filename=path,
            revision=revision,
            force_download=False,
        )

    # load the parameters, have to rename the threshold key,
    # as ours is nested inside the activation_function module
    param_dict = np.load(path_to_params)
    param_dict = {k: torch.tensor(v, device=device, dtype=dtype) for k, v in param_dict.items()}
    param_dict["activation_function.threshold"] = param_dict["threshold"]
    param_dict["W_enc"] = param_dict["W_enc"].T.contiguous()
    del param_dict["threshold"]

    # create the transcoders
    d_transcoder, d_model = param_dict["W_enc"].shape

    # JumpReLU; will get loaded via load_state_dict
    activation_function = JumpReLU(param_dict["activation_function.threshold"], 0.1)
    with torch.device("meta"):
        transcoder = SingleLayerTranscoder(d_model, d_transcoder, activation_function, layer)
    transcoder.load_state_dict(param_dict, assign=True)
    return transcoder


def load_relu_transcoder(
    path: str,
    layer: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    lazy_encoder: bool = True,
    lazy_decoder: bool = True,
):
    if device is None:
        device = get_default_device()

    param_dict = {}
    with safe_open(path, framework="pt", device=str(device)) as f:
        for k in f.keys():
            if lazy_encoder and k == "W_enc":
                continue
            if lazy_decoder and k == "W_dec":
                continue
            param_dict[k] = f.get_tensor(k)

    d_sae = param_dict["b_enc"].shape[0]
    d_model = param_dict["b_dec"].shape[0]

    assert param_dict.get("log_thresholds") is None
    activation_function = (
        JumpReLU(param_dict["activation_function.threshold"], 0.1)
        if "activation_function.threshold" in param_dict
        else F.relu
    )
    with torch.device("meta"):
        transcoder = SingleLayerTranscoder(
            d_model,
            d_sae,
            activation_function,
            layer,
            skip_connection=param_dict.get("W_skip") is not None,
            transcoder_path=path,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
        )
    transcoder.load_state_dict(param_dict, assign=True)
    return transcoder.to(dtype)


def load_gemma_scope_2_transcoder(
    path: str,
    layer: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    lazy_encoder: bool = False,
    lazy_decoder: bool = False,
) -> SingleLayerTranscoder:
    """Load a SingleLayerTranscoder from a GemmaScope2 JumpReLUSAE checkpoint.

    Args:
        path: Path to the checkpoint file
        layer: Layer index for the transcoder
        device: Device to load to
        dtype: Data type to use
        lazy_encoder: Whether to use lazy loading for encoder weights (not supported for GemmaScope2 format)
        lazy_decoder: Whether to use lazy loading for decoder weights (not supported for GemmaScope2 format)

    Returns:
        SingleLayerTranscoder: The loaded transcoder
    """
    if device is None:
        device = get_default_device()

    if lazy_encoder or lazy_decoder:
        warnings.warn(
            "Lazy loading is not supported for GemmaScope2 format due to different key naming conventions. "
            "Setting lazy_encoder=False and lazy_decoder=False.",
            UserWarning,
        )
        lazy_encoder = False
        lazy_decoder = False

    with safe_open(path, framework="pt", device=device.type) as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}

    param_dict = {
        "W_enc": state_dict["w_enc"].T.contiguous().to(device=device, dtype=dtype),
        "W_dec": state_dict["w_dec"].to(device=device, dtype=dtype),
        "b_enc": state_dict["b_enc"].to(device=device, dtype=dtype),
        "b_dec": state_dict["b_dec"].to(device=device, dtype=dtype),
        "activation_function.threshold": state_dict["threshold"].to(device=device, dtype=dtype),
    }

    if "affine_skip_connection" in state_dict:
        param_dict["W_skip"] = (
            state_dict["affine_skip_connection"].T.contiguous().to(device=device, dtype=dtype)
        )

    d_transcoder = param_dict["b_enc"].shape[0]
    d_model = param_dict["b_dec"].shape[0]

    activation_function = JumpReLU(param_dict["activation_function.threshold"], 0.1)

    with torch.device("meta"):
        transcoder = SingleLayerTranscoder(
            d_model,
            d_transcoder,
            activation_function,
            layer,
            skip_connection="W_skip" in param_dict,
        )

    transcoder.load_state_dict(param_dict, assign=True)
    return transcoder


def load_transcoder_set(
    transcoder_paths: dict,
    scan: str,
    feature_input_hook: str,
    feature_output_hook: str,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    special_load_fn: Literal["gemma-scope", "gemma-scope-2", None] = None,
    lazy_encoder: bool = True,
    lazy_decoder: bool = True,
) -> TranscoderSet:
    if device is None:
        device = get_default_device()
    """Loads either a preset set of transcoders, or a set specified by a file.

    Args:
        transcoder_paths: Dictionary mapping layer indices to transcoder paths
        scan: Scan identifier
        feature_input_hook: Hook point where features read from
        feature_output_hook: Hook point where features write to
        device (torch.device | None, optional): Device to load to
        dtype (torch.dtype, optional): Data type to use
        special_load_fn: Which special loading function to use
        lazy_encoder: Whether to use lazy loading for encoder weights
        lazy_decoder: Whether to use lazy loading for decoder weights

    Returns:
        TranscoderSet: The loaded transcoder set with all configuration
    """

    transcoders = {}
    for layer in range(len(transcoder_paths)):
        npz_format = Path(transcoder_paths[layer]).suffix == ".npz"

        if special_load_fn == "gemma-scope" and npz_format:
            load_fn = load_gemma_scope_transcoder
        elif special_load_fn == "gemma-scope-2":
            load_fn = load_gemma_scope_2_transcoder
        else:
            load_fn = load_relu_transcoder

        transcoders[layer] = load_fn(
            transcoder_paths[layer],
            layer,
            device=device,
            dtype=dtype,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
        )
    # we don't know how many layers the model has, but we need all layers from 0 to max covered
    assert set(transcoders.keys()) == set(range(max(transcoders.keys()) + 1)), (
        f"Each layer should have a transcoder, but got transcoders for layers "
        f"{set(transcoders.keys())}"
    )

    return TranscoderSet(
        transcoders,
        feature_input_hook=feature_input_hook,
        feature_output_hook=feature_output_hook,
        scan=scan,
    )
