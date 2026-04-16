import warnings
from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from functools import partial
from typing import Callable, Iterator, Literal, cast

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from nnsight.intervention.tracing.tracer import Barrier
from nnsight import LanguageModel, Envoy, save, CONFIG as NNSIGHT_CONFIG

from circuit_tracer.attribution.context_nnsight import AttributionContext
from circuit_tracer.transcoder import TranscoderSet
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
from circuit_tracer.utils import get_default_device
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
from circuit_tracer.utils.tl_nnsight_mapping import (
    get_mapping,
    convert_nnsight_config_to_transformerlens,
)

NNSIGHT_CONFIG.APP.PYMOUNT = False
NNSIGHT_CONFIG.APP.CROSS_INVOKER = False
NNSIGHT_CONFIG.APP.TRACE_CACHING = True

# Type definition for an intervention tuple (layer, position, feature_idx, value)
Intervention = tuple[
    int | torch.Tensor,
    int | slice | torch.Tensor,
    int | torch.Tensor,
    int | float | torch.Tensor,
]


class EnvoyWrapper:
    def __init__(self, envoy, input_output: Literal["input", "output"]):
        self.envoy = envoy
        self.input_output = input_output

    @property
    def output(self):
        return getattr(self.envoy, self.input_output)

    @output.setter
    def output(self, value):
        setattr(self.envoy, self.input_output, value)


class NNSightReplacementModel(LanguageModel):
    d_transcoder: int
    transcoders: TranscoderSet | CrossLayerTranscoder
    feature_input_locs: list[nn.Module]  # type: ignore
    feature_output_locs: list[nn.Module]  # type: ignore
    attention_locs: list[nn.Module]  # type: ignore
    layernorm_scale_locs: list[nn.Module]  # type: ignore
    pre_logit_location: nn.Module  # type: ignore
    embed_loc: nn.Module
    unembed_loc: nn.Module
    skip_transcoder: bool
    scan: str | list[str] | None
    backend: Literal["nnsight"]

    @classmethod
    def from_config(
        cls,
        config: AutoConfig,
        transcoders: TranscoderSet | CrossLayerTranscoder,  # Accept both
        **kwargs,
    ) -> "NNSightReplacementModel":
        """Create a NNSightReplacementModel from a given AutoConfig and TranscoderSet

        Args:
            config (AutoConfig): the config of the HuggingFace transformer
            transcoders (TranscoderSet): The transcoder set with configuration

        Returns:
            NNSightReplacementModel: The loaded NNSightReplacementModel
        """
        config._attn_implementation = "eager"  # type: ignore
        hf_model = AutoModelForCausalLM.from_config(config)
        hf_tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)  # type: ignore

        model = cls(hf_model, tokenizer=hf_tokenizer, dispatch=True, **kwargs)
        model.config = config  # type: ignore
        model._configure_replacement_model(transcoders)
        return model

    @classmethod
    def from_pretrained_and_transcoders(
        cls,
        model_name: str,
        transcoders: TranscoderSet | CrossLayerTranscoder,
        device: torch.device | str = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> "NNSightReplacementModel":
        """Create a NNSightReplacementModel from the name of HookedTransformer and TranscoderSet

        Args:
            model_name (str): the name of the pretrained HookedTransformer
            transcoders (TranscoderSet): The transcoder set with configuration

        Returns:
            NNSightReplacementModel: The loaded NNSightReplacementModel
        """
        # The goal is to build a ReplacementModel instance *using* the parent
        # LanguageModel.__init__.  Since we are in a `@classmethod`, we don't yet have
        # an object (`self`) to pass to `super().__init__`.  We create an _uninitialised_
        # instance with `__new__`, then run the parent initialiser on it.

        # 1. Allocate the instance without initialising it.
        model = cls.__new__(cls)
        # 2. Call the parent (LanguageModel) initializer on this instance.

        # Convert ``torch.device`` to a HF-compatible device map
        if isinstance(device, torch.device):
            if device.type == "cuda":
                dev_entry = device.index if device.index is not None else 0
            else:
                dev_entry = device.type  # e.g. "cpu"
        else:
            # string inputs such as "cuda:1" or "cpu".
            dev_str = str(device)
            if dev_str.startswith("cuda"):
                # "cuda" or "cuda:1"  → extract index or default to 0
                parts = dev_str.split(":")
                dev_entry = int(parts[1]) if len(parts) > 1 else 0
            else:
                dev_entry = dev_str  # "cpu" or other accelerator names

        device_map = {"": dev_entry}

        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, "quantization_config"):
            config.quantization_config["dequantize"] = True

        super(cls, model).__init__(
            model_name,
            config=config,
            device_map=device_map,
            dispatch=True,
            dtype=dtype,
            attn_implementation="eager",
        )

        model._configure_replacement_model(transcoders)
        return model

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        transcoder_set: str,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> "NNSightReplacementModel":
        """Create a NNSightReplacementModel from model name and transcoder config

        Args:
            model_name (str): the name of the pretrained HookedTransformer
            transcoder_set (str): Either a predefined transcoder set name, or a config file

        Returns:
            NNSightReplacementModel: The loaded NNSightReplacementModel
        """
        if device is None:
            device = get_default_device()

        transcoders, _ = load_transcoder_from_hub(transcoder_set, device=device, dtype=dtype)  # type: ignore

        return cls.from_pretrained_and_transcoders(
            model_name,
            transcoders,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    @staticmethod
    def _resolve_attr(root: object, attr_path: str):
        """Resolves a dotted attribute path that can additionally contain Python-style
        list indices, e.g. "model.layers[3].mlp".

        Args:
            root (object): The object from which to start attribute resolution.
            attr_path (str): Dotted path, optionally containing one level of
                ``[idx]`` list/ModuleList access.

        Returns:
            object: The resolved attribute.
        """
        current = root
        # Split on dots – each token may still contain an index expression.
        for token in attr_path.split("."):
            if not token:
                continue  # Guard against accidental empty tokens
            if "[" in token and token.endswith("]"):
                # e.g. "layers[3]"
                attr_name, idx_str = token.split("[", 1)
                idx = int(idx_str[:-1])  # strip trailing ]
                current = getattr(current, attr_name)[idx]
            else:
                current = getattr(current, token)
        return current

    def _configure_replacement_model(
        self,
        transcoder_set: TranscoderSet | CrossLayerTranscoder,
    ):
        self.backend = "nnsight"
        self.eval()
        self.cfg = convert_nnsight_config_to_transformerlens(self.config)

        # special case to zero out <bos><start_of_turn>user\n for gemmascope 2 (-it) transcoders
        gemma_3_it = "gemma-3" in self.cfg.model_name and self.cfg.model_name.endswith("-it")
        self.zero_positions = slice(0, 4) if gemma_3_it else slice(0, 1)

        transcoder_set.to(self.device, self.dtype)
        self.transcoders = transcoder_set
        self.skip_transcoder = transcoder_set.skip_connection

        # ------------------------------------------------------------------
        # Instead of eagerly resolving hook locations here (which can fail
        # outside of a `self.trace` context when multiple `.source`s exist),
        # we cache the *patterns* needed to resolve them and provide dynamic
        # property accessors which resolve the hooks on-demand inside the
        # appropriate trace context.
        # ------------------------------------------------------------------
        nnsight_config = get_mapping(self.config.architectures[0])  # type: ignore

        self._feature_input_pattern, self._feature_input_io = nnsight_config.feature_hook_mapping[
            transcoder_set.feature_input_hook
        ]
        self._feature_output_pattern, _ = nnsight_config.feature_hook_mapping[
            transcoder_set.feature_output_hook
        ]

        self._attention_pattern = nnsight_config.attention_location_pattern
        # Ensure we consistently store LayerNorm scale patterns as a list.
        self._layernorm_scale_patterns = nnsight_config.layernorm_scale_location_patterns
        self._pre_logit_location = nnsight_config.pre_logit_location
        self._embed_location = nnsight_config.embed_location

        # these are real weights, not envoys
        self.embed_weight = cast(
            torch.Tensor, self._resolve_attr(self, nnsight_config.embed_weight)
        )
        self.unembed_weight = cast(
            torch.Tensor, self._resolve_attr(self, nnsight_config.unembed_weight)
        )
        self.scan = transcoder_set.scan

        # Make sure the replacement model is entirely frozen by default.
        for param in self.parameters():
            param.requires_grad = False

    def configure_gradient_flow(self, tracer):
        with tracer.invoke():
            self.embed_location.output.requires_grad = True  # type: ignore

        with tracer.invoke():
            for freeze_loc in self.attention_locs:
                freeze_loc.output = freeze_loc.output.detach()  # type: ignore

        for layernorm_scale_locs_list in self.layernorm_scale_locs:
            with tracer.invoke():
                for freeze_loc in layernorm_scale_locs_list:
                    freeze_loc.output = freeze_loc.output.detach()  # type: ignore

    def configure_skip_connection(self, tracer, barrier=None):
        transcoders = (
            self.transcoders._module if isinstance(self.transcoders, Envoy) else self.transcoders
        )

        with tracer.invoke():
            for layer, (feature_input_loc, feature_output_loc) in enumerate(
                zip(self.feature_input_locs, self.feature_output_locs)
            ):
                if transcoders.skip_connection:  # type: ignore
                    skip = transcoders.compute_skip(layer, feature_input_loc.output)  # type: ignore
                else:
                    skip = 0 * feature_input_loc.output.sum()  # type: ignore
                feature_output_loc.output = skip + (feature_output_loc.output - skip).detach()  # type: ignore
                if barrier:
                    barrier()

    def get_activation_fn(
        self,
        sparse: bool = False,
        apply_activation_function: bool = True,
        append: bool = False,
    ) -> tuple[
        list[torch.Tensor],
        Callable[
            [Barrier | None, set[int], Iterator[int] | None], tuple[torch.Tensor, torch.Tensor]
        ],
    ]:
        activation_matrix = (
            [[] for _ in range(self.cfg.n_layers)] if append else [None] * self.cfg.n_layers
        )

        def fetch_activations(
            barrier: Barrier | None = None,
            barrier_layers: set[int] | None = None,
            activation_layers: Iterator[int] | None = None,
        ):
            # special case to zero out <bos><start_of_turn>user\n for gemmascope 2 (-it) transcoders
            layers = range(self.cfg.n_layers) if activation_layers is None else activation_layers
            for layer in layers:
                feature_input_loc = self.get_feature_input_loc(layer)
                transcoder_acts = (
                    self.transcoders._module.encode_layer(  # type: ignore
                        feature_input_loc.output,
                        layer,
                        apply_activation_function=apply_activation_function,
                    )
                    .detach()
                    .squeeze(0)
                )

                if not (append and len(activation_matrix[layer]) > 0):  # type:ignore
                    transcoder_acts[self.zero_positions] = 0

                if sparse:
                    transcoder_acts = transcoder_acts.to_sparse()

                if append:
                    activation_matrix[layer].append(transcoder_acts)  # type: ignore
                else:
                    activation_matrix[layer] = transcoder_acts  # type: ignore

                if barrier is not None and barrier_layers is not None and layer in barrier_layers:
                    barrier()

            logits = save(self.output.logits)

            # activation_layers is None means that we only need the acts for those layers, during this forward pass
            # So we don't bother creating / saving the whole cache

            if activation_layers is not None:
                activation_cache = None
            else:
                if append:
                    activation_cache = torch.stack(
                        [torch.cat(acts, dim=0) for acts in activation_matrix]
                    )
                else:
                    activation_cache = torch.stack(activation_matrix)  # type: ignore

                if sparse:
                    activation_cache = activation_cache.coalesce()

            return logits, activation_cache

        return activation_matrix, fetch_activations  # type: ignore

    def get_activations(
        self,
        inputs: str | torch.Tensor,
        sparse: bool = False,
        apply_activation_function: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the transcoder activations for a given prompt

        Args:
            inputs (str | torch.Tensor): The inputs you want to get activations over
            sparse (bool, optional): Whether to return a sparse tensor of activations.
                Useful if d_transcoder is large. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: the model logits on the inputs and the
                associated activation cache
        """
        _, fetch_activations = self.get_activation_fn(
            sparse=sparse, apply_activation_function=apply_activation_function
        )
        with torch.inference_mode(), self.trace(inputs):
            logits, activation_cache = fetch_activations()  # type:ignore
            logits = save(logits)  # type: ignore
            activation_cache = save(activation_cache)  # type: ignore

        return logits, activation_cache

    @contextmanager
    def zero_softcap(self):
        if hasattr(self.config, "final_logit_softcapping"):
            current_softcap = self.config.final_logit_softcapping  # type: ignore
            try:
                self.config.final_logit_softcapping = None  # type: ignore
                yield
            finally:
                self.config.final_logit_softcapping = current_softcap  # type: ignore
        elif hasattr(self.config, "text_config") and hasattr(
            self.config.text_config, "final_logit_softcapping"
        ):
            current_softcap = self.config.text_config.final_logit_softcapping  # type: ignore
            try:
                self.config.text_config.final_logit_softcapping = None  # type: ignore
                yield
            finally:
                self.config.text_config.final_logit_softcapping = current_softcap  # type: ignore
        else:
            yield

    def ensure_tokenized(self, prompt: str | torch.Tensor | list[int]) -> torch.Tensor:
        """Convert prompt to 1-D tensor of token ids with proper special token handling.

        This method ensures that a special token (BOS/PAD) is prepended to the input sequence.
        The first token position in transformer models typically exhibits unusually high norm
        and an excessive number of active features due to how models process the beginning of
        sequences. By prepending a special token, we ensure that actual content tokens have
        more consistent and interpretable feature activations, avoiding the artifacts present
        at position 0. This prepended token is later ignored during attribution analysis.

        Args:
            prompt: String, tensor, or list of token ids representing a single sequence

        Returns:
            1-D tensor of token ids with BOS/PAD token at the beginning

        Raises:
            TypeError: If prompt is not str, tensor, or list
            ValueError: If tensor has wrong shape (must be 1-D or 2-D with batch size 1)
        """

        if isinstance(prompt, str):
            tokens = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids.squeeze(0)
        elif isinstance(prompt, torch.Tensor):
            tokens = prompt.squeeze()
        elif isinstance(prompt, list):
            tokens = torch.tensor(prompt, dtype=torch.long).squeeze()
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")

        if tokens.ndim > 1:
            raise ValueError(f"Tensor must be 1-D, got shape {tokens.shape}")

        tokens = tokens.to(self.device)

        gemma_3_it = "gemma-3" in self.cfg.model_name and self.cfg.model_name.endswith("-it")
        if gemma_3_it:
            ignore_prefix = torch.tensor(
                [2, 105, 2364, 107], dtype=tokens.dtype, device=tokens.device
            )
            tokenization_error = (
                "Input tokens should start with <bos><start_of_turn>user\n, but got {tokens}"
            )
            assert tokens.size(0) >= 4 and torch.all(tokens[:4] == ignore_prefix), (
                tokenization_error.format(tokens=self.tokenizer.decode(tokens.cpu().tolist()))
            )
            return tokens

        # Check if a special token is already present at the beginning
        if tokens[0] in self.tokenizer.all_special_ids:
            return tokens

        # Prepend a special token to avoid artifacts at position 0
        candidate_bos_token_ids = [
            self.tokenizer.bos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        ]
        candidate_bos_token_ids += self.tokenizer.all_special_ids

        dummy_bos_token_id = next(filter(None, candidate_bos_token_ids))
        if dummy_bos_token_id is None:
            warnings.warn(
                "No suitable special token found for BOS token replacement. The first token will be ignored."
            )
        else:
            tokens = torch.cat([torch.tensor([dummy_bos_token_id], device=tokens.device), tokens])

        return tokens.to(self.device)

    @torch.no_grad()
    def setup_attribution(self, inputs: str | torch.Tensor):
        """Precomputes the transcoder activations and error vectors, saving them and the
        token embeddings.

        Args:
            inputs (str): the inputs to attribute - hard coded to be a single string (no
                batching) for now
        """

        if isinstance(inputs, str):
            tokens = self.ensure_tokenized(inputs)
        else:
            tokens = inputs.squeeze()

        assert isinstance(tokens, torch.Tensor), "Tokens must be a tensor"
        assert tokens.ndim == 1, "Tokens must be a 1D tensor"

        mlp_in_cache = [None] * self.cfg.n_layers
        mlp_out_cache = [None] * self.cfg.n_layers

        transcoders = self.transcoders

        with self.trace(tokens):
            mlp_in_cache, mlp_out_cache = [], []
            for feature_input_loc, feature_output_loc in zip(
                self.feature_input_locs, self.feature_output_locs
            ):
                mlp_in_cache.append(feature_input_loc.output)

                # we expect a dummy dimension 0, but GPT-OSS doesn't have one, so we add it.
                y = feature_output_loc.output
                if y.ndim == 2:
                    y = y.unsqueeze(0)  # type: ignore
                mlp_out_cache.append(y)

            mlp_in_cache = save(torch.cat(mlp_in_cache, dim=0))  # type: ignore
            mlp_out_cache = save(torch.cat(mlp_out_cache, dim=0))  # type: ignore
            logits = save(self.output.logits)

        attribution_data = transcoders.compute_attribution_components(
            mlp_in_cache, self.zero_positions
        )  # type: ignore

        # Compute error vectors
        error_vectors = mlp_out_cache - attribution_data["reconstruction"]

        error_vectors[:, self.zero_positions] = 0
        token_vectors = self.embed_weight[  # type: ignore
            tokens
        ].detach()  # (n_pos, d_model)  # type: ignore

        return AttributionContext(
            activation_matrix=attribution_data["activation_matrix"],
            logits=logits,
            error_vectors=error_vectors,
            token_vectors=token_vectors,
            decoder_vecs=attribution_data["decoder_vecs"],
            encoder_vecs=attribution_data["encoder_vecs"],
            encoder_to_decoder_map=attribution_data["encoder_to_decoder_map"],
            decoder_locations=attribution_data["decoder_locations"],
        )

    def setup_intervention_with_freeze(
        self, inputs: str | torch.Tensor, constrained_layers: range | None = None
    ) -> tuple[torch.Tensor, list[Callable]]:
        """Sets up an intervention with either frozen attention + LayerNorm(default) or frozen
        attention, LayerNorm, and MLPs, for constrained layers

        Args:
            inputs (str | torch.Tensor): The inputs to intervene on
            constrained_layers (range | None): whether to apply interventions only to a certain range.
                Mostly applicable to CLTs. If the given range includes all model layers, we also freeze
                layernorm denominators, computing direct effects. None means no constraints (iterative patching)

        Returns:
            tuple[torch.Tensor, list[Callable]]: The freeze hooks needed to run the desired intervention.
        """

        def get_locs_to_freeze():
            # this needs to go in a function that is called only in a trace context! otherwise you can't get the .source twice
            locs_to_freeze = {"attention": self.attention_locs}
            if constrained_layers:
                if set(range(self.cfg.n_layers)).issubset(set(constrained_layers)):  # type: ignore
                    for i, layernorm_freeze_loc in enumerate(self.layernorm_scale_locs):
                        locs_to_freeze[f"layernorm-{i}"] = layernorm_freeze_loc
                if self.skip_transcoder:
                    locs_to_freeze["feature_input"] = self.feature_input_locs
                locs_to_freeze["feature_output"] = self.feature_output_locs
            return locs_to_freeze

        activation_matrix, activation_fn = self.get_activation_fn()
        cache = {}

        # somehow, self is getting corrupted / changed somehow into type `EnvoyWrapper`, which causes issues.
        # This gets around it.
        transcoders = self.transcoders
        skip_transcoder = self.skip_transcoder

        # get transcoder activations and values to freeze to
        with self.trace() as tracer:
            with tracer.invoke(inputs):
                activation_fn()  # type:ignore
            dict_to_freeze = save(get_locs_to_freeze())  # type: ignore
            for freeze_loc_name, loc_type_to_freeze in get_locs_to_freeze().items():
                with tracer.invoke():
                    for layer, loc_to_freeze in enumerate(loc_type_to_freeze):
                        freeze_loc_output = loc_to_freeze.output
                        if freeze_loc_name != "feature_input":
                            freeze_loc_output = freeze_loc_output.detach()  # type:ignore
                        cache[freeze_loc_name, layer] = save(freeze_loc_output)  # type: ignore

        skip_diffs = {}

        def freeze_fn(freeze_loc_name, loc_type_to_freeze, direct_effects_barrier=None):
            for layer, loc_to_freeze in enumerate(loc_type_to_freeze):
                if freeze_loc_name == "feature_input":
                    # The MLP hook out freeze hook sets the value of the MLP to the value it
                    # had when run on the inputs normally. We subtract out the skip that
                    # corresponds to such a run, and add in the skip with direct effects.
                    frozen_skip = transcoders.compute_skip(  # type: ignore
                        layer, cache["feature_input", layer]
                    )
                    normal_skip = transcoders.compute_skip(layer, loc_to_freeze.output)  # type: ignore

                    skip_diffs[layer] = normal_skip - frozen_skip

                else:
                    if freeze_loc_name == "feature_output":
                        if layer not in constrained_layers:  # type: ignore
                            continue

                    original_outputs = loc_to_freeze.output
                    cached_values = cache[freeze_loc_name, layer]

                    if isinstance(original_outputs, tuple):
                        assert isinstance(cached_values, tuple)
                        assert len(original_outputs) == len(cached_values)
                        for orig, cached in zip(original_outputs, cached_values):
                            assert orig.shape == cached.shape, (
                                f"Activations shape {orig.shape} does not match cached values"
                                f" shape {cached.shape} at hook {loc_to_freeze.name}"
                            )
                    else:
                        assert original_outputs.shape == cached_values.shape, (
                            f"Activations shape {original_outputs.shape} does not match cached values"
                            f" shape {cached_values.shape} at hook {loc_to_freeze.name}"
                        )

                    if freeze_loc_name == "feature_output" and skip_transcoder:
                        loc_to_freeze.output = cached_values + skip_diffs[layer]
                    else:
                        loc_to_freeze.output = cached_values

                    if (
                        freeze_loc_name == "feature_output"
                        and direct_effects_barrier
                        and (constrained_layers is None or layer in constrained_layers)
                    ):
                        direct_effects_barrier()

        return torch.stack(activation_matrix), [
            partial(
                freeze_fn,
                freeze_loc_name=freeze_loc_name,
                loc_type_to_freeze=loc_type_to_freeze,
            )
            for freeze_loc_name, loc_type_to_freeze in dict_to_freeze.items()
        ]

    @torch.no_grad
    def _perform_feature_intervention(
        self,
        inputs,
        interventions: Sequence[Intervention],
        activation_matrix: torch.Tensor,
        original_activations: torch.Tensor | None,
        activation_barrier,
        direct_effects_barrier,
        constrained_layers: range | None = None,
        using_past_kv_cache_idx: int | None = None,
        apply_activation_function: bool = True,
    ):
        interventions_by_layer = defaultdict(list)
        for layer, pos, feature_idx, value in interventions:
            layer = layer.item() if isinstance(layer, torch.Tensor) else layer
            interventions_by_layer[layer].append((pos, feature_idx, value))

        if using_past_kv_cache_idx is not None and using_past_kv_cache_idx > 0:
            # We're generating one token at a time
            n_pos = 1
        elif original_activations is not None:
            n_pos = original_activations.size(1)
        else:
            n_pos = len(self.tokenizer(inputs).input_ids)

        layer_deltas = torch.zeros(
            [self.cfg.n_layers, n_pos, self.cfg.d_model],
            dtype=self.dtype,
            device=self.device,
        )
        for layer in range(self.cfg.n_layers):
            if interventions_by_layer[layer]:
                if constrained_layers:
                    # base deltas on original activations; don't let effects propagate
                    transcoder_activations = original_activations[layer].clone()  # type: ignore
                else:
                    activation_barrier()
                    # recompute deltas based on current activations
                    transcoder_activations = (
                        activation_matrix[layer][-1]
                        if using_past_kv_cache_idx is not None
                        else activation_matrix[layer]
                    )
                    if transcoder_activations.is_sparse:
                        transcoder_activations = transcoder_activations.to_dense()

                    if not apply_activation_function:
                        transcoder_activations = self.transcoders.apply_activation_function(
                            layer, transcoder_activations.unsqueeze(0)
                        ).squeeze(0)

                activation_deltas = torch.zeros_like(transcoder_activations)
                for pos, feature_idx, value in interventions_by_layer[layer]:
                    activation_deltas[pos, feature_idx] = (
                        value - transcoder_activations[pos, feature_idx]
                    )

                poss, feature_idxs = activation_deltas.nonzero(as_tuple=True)
                new_values = activation_deltas[poss, feature_idxs]

                decoder_vectors = self.transcoders._module._get_decoder_vectors(  # type: ignore
                    layer, feature_idxs
                )

                # Handle both 2D [n_feature_idxs, d_model] and 3D [n_feature_idxs, n_remaining_layers, d_model] cases
                if decoder_vectors.ndim == 2:
                    # Single-layer transcoder case: [n_feature_idxs, d_model]
                    decoder_vectors = decoder_vectors * new_values.unsqueeze(1)
                    layer_deltas[layer].index_add_(0, poss, decoder_vectors)
                else:
                    # Cross-layer transcoder case: [n_feature_idxs, n_remaining_layers, d_model]
                    decoder_vectors = decoder_vectors * new_values.unsqueeze(-1).unsqueeze(-1)

                    # Transpose to [n_remaining_layers, n_feature_idxs, d_model]
                    decoder_vectors = decoder_vectors.transpose(0, 1)

                    # Distribute decoder vectors across layers
                    n_remaining_layers = decoder_vectors.shape[0]
                    layer_deltas[-n_remaining_layers:].index_add_(1, poss, decoder_vectors)

            if constrained_layers is None or layer in constrained_layers:
                if direct_effects_barrier:
                    direct_effects_barrier()
                transcoder_output = self.get_feature_output_loc(layer).output  # type: ignore
                transcoder_output[:] = transcoder_output + layer_deltas[layer]  # type: ignore
                layer_deltas[layer] *= 0

        return save(self.output.logits)

    @torch.no_grad
    def feature_intervention(
        self,
        inputs: str | torch.Tensor,
        interventions: Sequence[Intervention],
        constrained_layers: range | None = None,
        freeze_attention: bool = True,
        apply_activation_function: bool = True,
        sparse: bool = False,
        return_activations: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Given the input, and a dictionary of features to intervene on, performs the
        intervention, allowing all effects to propagate (optionally allowing its effects to
        propagate through transcoders)

        Args:
            input (_type_): the input prompt to intervene on
            intervention_dict (Sequence[Intervention]): A list of interventions to perform, formatted as
                a list of (layer, position, feature_idx, value)
            constrained_layers (range | None): whether to apply interventions only to a certain range, freezing
                all MLPs within the layer range before doing so. This is mostly applicable to CLTs. If the given
                range includes all model layers, we also freeze layernorm denominators, computing direct effects.
                None means no constraints (iterative patching)
            apply_activation_function (bool): whether to apply the activation function when
                recording the activations to be returned. This is useful to set to False for
                testing purposes, as attribution predicts the change in pre-activation
                feature values.
            sparse (bool): whether to sparsify the activations in the returned cache. Setting
                this to True will take up less memory, at the expense of slower interventions.
            return_activations (bool): Whether to compute and return feature activations. If False,
                activation computation is skipped for layers not being intervened on (when
                constrained_layers is not set), saving time. Activations are not returned.
                Defaults to True.
        """
        activation_matrix, activation_fn = self.get_activation_fn(
            apply_activation_function=apply_activation_function, sparse=sparse
        )

        if (freeze_attention or constrained_layers) and interventions:
            original_activations, freeze_fns = self.setup_intervention_with_freeze(
                inputs, constrained_layers=constrained_layers
            )
        else:
            original_activations, freeze_fns = None, []

        intervention_layers = set()
        for layer, _, _, _ in interventions:
            if isinstance(layer, torch.Tensor):
                layer = layer.item()
            intervention_layers.add(layer)

        activation_layers = None if return_activations else sorted(list(intervention_layers))  # type:ignore

        with self.trace() as tracer:
            activation_barrier = None if constrained_layers else tracer.barrier(2)
            direct_effects_barrier = tracer.barrier(2) if constrained_layers else None

            with tracer.invoke(inputs):
                _, activation_cache = activation_fn(
                    barrier=activation_barrier,  # type:ignore
                    barrier_layers=intervention_layers,
                    activation_layers=activation_layers,
                )
                activation_cache = save(activation_cache)  # type:ignore

            for freeze_fn in freeze_fns:
                with tracer.invoke():
                    freeze_fn(direct_effects_barrier=direct_effects_barrier)

            with tracer.invoke():
                cached_logits = self._perform_feature_intervention(
                    inputs,
                    interventions,
                    activation_matrix,  # type: ignore
                    original_activations,
                    activation_barrier,
                    direct_effects_barrier,
                    constrained_layers,
                    using_past_kv_cache_idx=None,
                    apply_activation_function=apply_activation_function,
                )

        return cached_logits, activation_cache if return_activations else None

    def _convert_open_ended_interventions(
        self,
        interventions: Sequence[Intervention],
    ) -> Sequence[Intervention]:
        """Convert open-ended interventions into position-0 equivalents.

        An intervention is *open-ended* if its position component is a ``slice`` whose
        ``stop`` attribute is ``None`` (e.g. ``slice(1, None)``). Such interventions will
        also apply to tokens generated in an open-ended generation loop. In such cases,
        when use_past_kv_cache=True, the model only runs the most recent token
        (and there is thus only 1 position).
        """
        converted = []
        for layer, pos, feature_idx, value in interventions:
            if isinstance(pos, slice) and pos.stop is None:
                converted.append((layer, 0, feature_idx, value))
        return converted

    @torch.no_grad
    def feature_intervention_generate(
        self,
        inputs: str | torch.Tensor,
        interventions: Sequence[Intervention],
        constrained_layers: range | None = None,
        freeze_attention: bool = True,
        apply_activation_function: bool = True,
        sparse: bool = False,
        return_activations: bool = True,
        **kwargs,
    ) -> tuple[str, torch.Tensor, torch.Tensor | None]:
        """Given the input, and a dictionary of features to intervene on, performs the
        intervention, and generates a continuation, along with the logits and activations at each generation position.
        This function accepts all kwargs valid for HookedTransformer.generate(). Note that freeze_attention applies
        only to the first token generated.

        Note that if kv_cache is True (default), generation will be faster, as the model will cache the KVs, and only
        process the one new token per step; if it is False, the model will generate by doing a full forward pass across
        all tokens. Note that due to numerical precision issues, you are only guaranteed that the logits / activations of
        model.feature_intervention_generate(s, ...) are equivalent to model.feature_intervention(s, ...) if kv_cache is False.

        Args:
            input (_type_): the input prompt to intervene on
            interventions (list[tuple[int, Union[int, slice, torch.Tensor]], int,
                int | torch.Tensor]): A list of interventions to perform, formatted as
                a list of (layer, position, feature_idx, value)
            constrained_layers: (range | None = None): whether to freeze all MLPs/transcoders /
                attn patterns / layernorm denominators. This will only apply to the very first token generated. If
            freeze_attention (bool): whether to freeze all attention patterns. Applies only to first token generated
            apply_activation_function (bool): whether to apply the activation function when
                recording the activations to be returned. This is useful to set to False for
                testing purposes, as attribution predicts the change in pre-activation
                feature values.
            sparse (bool): whether to sparsify the activations in the returned cache. Setting
                this to True will take up less memory, at the expense of slower interventions.
            return_activations (bool): Whether to compute and return feature activations. If False,
                activation computation is skipped for layers not being intervened on (when
                constrained_layers is not set), saving time. Returns None for activations.
                Defaults to True.
        """

        # remove verbose kwarg, which is valid for TL models but not NNsight ones.
        kwargs.pop("verbose", None)

        tokenizer = self.tokenizer
        converted_interventions = self._convert_open_ended_interventions(interventions)

        activation_matrix, activation_fn = self.get_activation_fn(
            apply_activation_function=apply_activation_function,
            append=True,
            sparse=sparse,
        )

        if (freeze_attention or constrained_layers) and interventions:
            original_activations, freeze_fns = self.setup_intervention_with_freeze(
                inputs, constrained_layers=constrained_layers
            )
        else:
            original_activations, freeze_fns = None, []

        intervention_layers = set()
        for layer, _, _, _ in interventions:
            if isinstance(layer, torch.Tensor):
                layer = layer.item()
            intervention_layers.add(layer)

        converted_intervention_layers = set()
        for layer, _, _, _ in converted_interventions:
            if isinstance(layer, torch.Tensor):
                layer = layer.item()
            converted_intervention_layers.add(layer)

        activation_cache = [None]

        with self.generate(**kwargs) as tracer:
            activation_barrier = tracer.barrier(2)
            direct_effects_barrier = tracer.barrier(2) if constrained_layers else None

            with tracer.invoke(inputs):
                with tracer.iter[:] as act_idx:
                    current_intervention_layers = (
                        intervention_layers if act_idx == 0 else converted_intervention_layers
                    )
                    activation_layers = (
                        None
                        if return_activations
                        else list(sorted(list(current_intervention_layers)))
                    )  # type:ignore
                    current_act_barrier = (
                        None if constrained_layers and act_idx == 0 else activation_barrier
                    )

                    _, iter_activation_cache = activation_fn(
                        barrier=current_act_barrier,  # type:ignore
                        barrier_layers=current_intervention_layers,
                        activation_layers=activation_layers,
                    )
                    activation_cache[0] = save(iter_activation_cache)

            for freeze_fn in freeze_fns:
                with tracer.invoke():
                    with tracer.iter[:1]:
                        freeze_fn(direct_effects_barrier=direct_effects_barrier)

            all_logits = save(list())  # type: ignore
            with tracer.invoke():
                with tracer.iter[:] as idx:
                    logits = self._perform_feature_intervention(
                        inputs=inputs,
                        interventions=(interventions if idx == 0 else converted_interventions),
                        activation_matrix=activation_matrix,  # type: ignore
                        original_activations=original_activations,
                        activation_barrier=activation_barrier,
                        direct_effects_barrier=(direct_effects_barrier if idx == 0 else None),
                        constrained_layers=constrained_layers if idx == 0 else None,
                        using_past_kv_cache_idx=idx,  # type: ignore
                        apply_activation_function=apply_activation_function,
                    )
                    all_logits.append(logits.squeeze(0))

            with tracer.invoke():
                out = save(self.generator.output)
        return (
            tokenizer.decode(out.squeeze(0)),
            torch.cat(all_logits, dim=0),
            (activation_cache[0] if return_activations else None),
        )

    # ------------------------------------------------------------------
    # Dynamic hook location properties
    # ------------------------------------------------------------------

    def get_feature_input_loc(self, layer: int):
        """
        Returns a feature input loc wrapped in an EnvoyWrapper. This is necessary because some feature inputs need .input, and
        some need .output. An EnvoyWrapper just wraps them such that .output always returns the relevant value.
        """
        return EnvoyWrapper(
            self._resolve_attr(self, self._feature_input_pattern.format(layer=layer)),
            self._feature_input_io,  # type: ignore
        )

    @property
    def feature_input_locs(self) -> Iterator[nn.Module]:
        """Dynamically resolve the MLP input hook locations for every layer."""
        for layer in range(self.cfg.n_layers):  # type: ignore
            yield self.get_feature_input_loc(layer)  # type: ignore

    def get_feature_output_loc(self, layer: int):
        return self._resolve_attr(self, self._feature_output_pattern.format(layer=layer))

    @property
    def feature_output_locs(self) -> Iterator[nn.Module]:
        """Dynamically resolve the MLP output hook locations for every layer."""
        for layer in range(self.cfg.n_layers):  # type: ignore
            yield self.get_feature_output_loc(layer)  # type: ignore

    @property
    def attention_locs(self) -> Iterator[nn.Module]:
        """Dynamically resolve the attention pattern hook locations for every layer."""
        for layer in range(self.cfg.n_layers):  # type: ignore
            yield self._resolve_attr(self, self._attention_pattern.format(layer=layer))  # type: ignore

    @property
    def layernorm_scale_locs(self) -> list[Iterator[nn.Module]]:
        """Dynamically resolve the LayerNorm scale hook locations (can be per-layer or shared)."""
        locs = []
        for pattern in self._layernorm_scale_patterns:
            if "{layer}" in pattern:

                def layer_iterator(p=pattern):
                    for layer in range(self.cfg.n_layers):  # type: ignore
                        yield self._resolve_attr(self, p.format(layer=layer))

                locs.append(layer_iterator())
            else:

                def single_iterator(p=pattern):
                    yield self._resolve_attr(self, p)

                locs.append(single_iterator())
        return locs

    @property
    def pre_logit_location(self) -> nn.Module:
        """Dynamically resolve the pre-logit hook location."""
        return self._resolve_attr(self, self._pre_logit_location)  # type: ignore

    @property
    def embed_location(self) -> nn.Module:
        """Dynamically resolve the embed hook location."""
        return self._resolve_attr(self, self._embed_location)  # type: ignore
