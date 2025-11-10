import warnings
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from circuit_tracer.transcoder import TranscoderSet
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
from circuit_tracer.utils import get_default_device
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

from .attribution.context import AttributionContext

# Type definition for an intervention tuple (layer, position, feature_idx, value)
Intervention = tuple[
    int | torch.Tensor, int | slice | torch.Tensor, int | torch.Tensor, int | torch.Tensor
]


class ReplacementMLP(nn.Module):
    """Wrapper for a TransformerLens MLP layer that adds in extra hooks"""

    def __init__(self, old_mlp: nn.Module):
        super().__init__()
        self.old_mlp = old_mlp
        self.hook_in = HookPoint()
        self.hook_out = HookPoint()

    def forward(self, x):
        x = self.hook_in(x)
        mlp_out = self.old_mlp(x)
        return self.hook_out(mlp_out)


class ReplacementUnembed(nn.Module):
    """Wrapper for a TransformerLens Unembed layer that adds in extra hooks"""

    def __init__(self, old_unembed: nn.Module):
        super().__init__()
        self.old_unembed = old_unembed
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    @property
    def W_U(self):
        return self.old_unembed.W_U

    @property
    def b_U(self):
        return self.old_unembed.b_U

    def forward(self, x):
        x = self.hook_pre(x)
        x = self.old_unembed(x)
        return self.hook_post(x)


class ReplacementModel(HookedTransformer):
    transcoders: TranscoderSet | CrossLayerTranscoder  # Support both types
    feature_input_hook: str
    feature_output_hook: str
    skip_transcoder: bool
    scan: str | list[str] | None
    tokenizer: PreTrainedTokenizerBase

    @classmethod
    def from_config(
        cls,
        config: HookedTransformerConfig,
        transcoders: TranscoderSet | CrossLayerTranscoder,  # Accept both
        **kwargs,
    ) -> "ReplacementModel":
        """Create a ReplacementModel from a given HookedTransformerConfig and TranscoderSet

        Args:
            config (HookedTransformerConfig): the config of the HookedTransformer
            transcoders (TranscoderSet): The transcoder set with configuration

        Returns:
            ReplacementModel: The loaded ReplacementModel
        """
        model = cls(config, **kwargs)
        model._configure_replacement_model(transcoders)
        return model

    @classmethod
    def from_pretrained_and_transcoders(
        cls,
        model_name: str,
        transcoders: TranscoderSet | CrossLayerTranscoder,  # Accept both
        **kwargs,
    ) -> "ReplacementModel":
        """Create a ReplacementModel from the name of HookedTransformer and TranscoderSet

        Args:
            model_name (str): the name of the pretrained HookedTransformer
            transcoders (TranscoderSet): The transcoder set with configuration

        Returns:
            ReplacementModel: The loaded ReplacementModel
        """
        model = super().from_pretrained(
            model_name,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            **kwargs,
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
    ) -> "ReplacementModel":
        """Create a ReplacementModel from model name and transcoder config

        Args:
            model_name (str): the name of the pretrained HookedTransformer
            transcoder_set (str): Either a predefined transcoder set name, or a config file

        Returns:
            ReplacementModel: The loaded ReplacementModel
        """
        if device is None:
            device = get_default_device()

        transcoders, _ = load_transcoder_from_hub(transcoder_set, device=device, dtype=dtype)

        return cls.from_pretrained_and_transcoders(
            model_name,
            transcoders,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    def _configure_replacement_model(self, transcoder_set: TranscoderSet | CrossLayerTranscoder):
        transcoder_set.to(self.cfg.device, self.cfg.dtype)

        self.transcoders = transcoder_set
        self.feature_input_hook = transcoder_set.feature_input_hook
        self.original_feature_output_hook = transcoder_set.feature_output_hook
        self.feature_output_hook = transcoder_set.feature_output_hook + ".hook_out_grad"
        self.skip_transcoder = transcoder_set.skip_connection
        self.scan = transcoder_set.scan

        for block in self.blocks:
            block.mlp = ReplacementMLP(block.mlp)  # type: ignore

        self.unembed = ReplacementUnembed(self.unembed)

        self._configure_gradient_flow()
        self._deduplicate_attention_buffers()
        self.setup()

    def _configure_gradient_flow(self):
        if isinstance(self.transcoders, TranscoderSet):
            for layer, transcoder in enumerate(self.transcoders):
                self._configure_skip_connection(self.blocks[layer], transcoder)
        else:
            for layer in range(self.cfg.n_layers):
                self._configure_skip_connection(self.blocks[layer], self.transcoders)

        def stop_gradient(acts, hook):
            return acts.detach()

        for block in self.blocks:
            block.attn.hook_pattern.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            block.ln1.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            block.ln2.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            if hasattr(block, "ln1_post"):
                block.ln1_post.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            if hasattr(block, "ln2_post"):
                block.ln2_post.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore
            self.ln_final.hook_scale.add_hook(stop_gradient, is_permanent=True)  # type: ignore

        for param in self.parameters():
            param.requires_grad = False

        def enable_gradient(tensor, hook):
            tensor.requires_grad = True
            return tensor

        self.hook_embed.add_hook(enable_gradient, is_permanent=True)

    def _configure_skip_connection(self, block, transcoder):
        cached = {}

        def cache_activations(acts, hook):
            cached["acts"] = acts

        def add_skip_connection(acts: torch.Tensor, hook: HookPoint, grad_hook: HookPoint):
            # We add grad_hook because we need a way to hook into the gradients of the output
            # of this function. If we put the backwards hook here at hook, the grads will be 0
            # because we detached acts.
            skip_input_activation = cached.pop("acts")
            if hasattr(transcoder, "W_skip") and transcoder.W_skip is not None:
                skip = transcoder.compute_skip(skip_input_activation)
            else:
                skip = skip_input_activation * 0
            return grad_hook(skip + (acts - skip).detach())

        # add feature input hook
        output_hook_parts = self.feature_input_hook.split(".")
        subblock = block
        for part in output_hook_parts:
            subblock = getattr(subblock, part)
        subblock.add_hook(cache_activations, is_permanent=True)

        # add feature output hook and special grad hook
        output_hook_parts = self.original_feature_output_hook.split(".")
        subblock = block
        for part in output_hook_parts:
            subblock = getattr(subblock, part)
        subblock.hook_out_grad = HookPoint()
        subblock.add_hook(
            partial(add_skip_connection, grad_hook=subblock.hook_out_grad),
            is_permanent=True,
        )

    def _deduplicate_attention_buffers(self):
        """
        Share attention buffers across layers to save memory.

        TransformerLens makes separate copies of the same masks and RoPE
        embeddings for each layer - This just keeps one copy
        of each and shares it across all layers.
        """

        attn_masks = {}

        for block in self.blocks:
            attn_masks[block.attn.attn_type] = block.attn.mask  # type: ignore
            if hasattr(block.attn, "rotary_sin"):
                attn_masks["rotary_sin"] = block.attn.rotary_sin  # type: ignore
                attn_masks["rotary_cos"] = block.attn.rotary_cos  # type: ignore

        for block in self.blocks:
            block.attn.mask = attn_masks[block.attn.attn_type]  # type: ignore
            if hasattr(block.attn, "rotary_sin"):
                block.attn.rotary_sin = attn_masks["rotary_sin"]  # type: ignore
                block.attn.rotary_cos = attn_masks["rotary_cos"]  # type: ignore

    def _get_activation_caching_hooks(
        self,
        sparse: bool = False,
        apply_activation_function: bool = True,
        append: bool = False,
    ) -> tuple[list[torch.Tensor], list[tuple[str, Callable]]]:
        activation_matrix = (
            [[] for _ in range(self.cfg.n_layers)] if append else [None] * self.cfg.n_layers
        )

        def cache_activations(acts, hook, layer):
            transcoder_acts = (
                self.transcoders.encode_layer(
                    acts, layer, apply_activation_function=apply_activation_function
                )
                .detach()
                .squeeze(0)
            )
            if sparse:
                transcoder_acts = transcoder_acts.to_sparse()

            if append:
                activation_matrix[layer].append(transcoder_acts)
            else:
                activation_matrix[layer] = transcoder_acts  # type: ignore

        activation_hooks = [
            (
                f"blocks.{layer}.{self.feature_input_hook}",
                partial(cache_activations, layer=layer),
            )
            for layer in range(self.cfg.n_layers)
        ]
        return activation_matrix, activation_hooks  # type: ignore

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

        activation_cache, activation_hooks = self._get_activation_caching_hooks(
            sparse=sparse,
            apply_activation_function=apply_activation_function,
        )
        with torch.inference_mode(), self.hooks(activation_hooks):  # type: ignore
            logits = self(inputs)
        activation_cache = torch.stack(activation_cache)
        if sparse:
            activation_cache = activation_cache.coalesce()
        return logits, activation_cache

    @contextmanager
    def zero_softcap(self):
        current_softcap = self.cfg.output_logits_soft_cap
        try:
            self.cfg.output_logits_soft_cap = 0.0
            yield
        finally:
            self.cfg.output_logits_soft_cap = current_softcap

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
            tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0)
        elif isinstance(prompt, torch.Tensor):
            tokens = prompt.squeeze()
        elif isinstance(prompt, list):
            tokens = torch.tensor(prompt, dtype=torch.long).squeeze()
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")

        if tokens.ndim > 1:
            raise ValueError(f"Tensor must be 1-D, got shape {tokens.shape}")

        # Check if a special token is already present at the beginning
        if tokens[0] in self.tokenizer.all_special_ids:
            return tokens.to(self.cfg.device)

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
                "No suitable special token found for BOS token replacement. "
                "The first token will be ignored."
            )
        else:
            tokens = torch.cat([torch.tensor([dummy_bos_token_id], device=tokens.device), tokens])

        return tokens.to(self.cfg.device)

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

        mlp_in_cache, mlp_in_caching_hooks, _ = self.get_caching_hooks(
            lambda name: self.feature_input_hook in name
        )

        mlp_out_cache, mlp_out_caching_hooks, _ = self.get_caching_hooks(
            lambda name: self.feature_output_hook in name
        )
        logits = self.run_with_hooks(tokens, fwd_hooks=mlp_in_caching_hooks + mlp_out_caching_hooks)

        mlp_in_cache = torch.cat(list(mlp_in_cache.values()), dim=0)
        mlp_out_cache = torch.cat(list(mlp_out_cache.values()), dim=0)

        attribution_data = self.transcoders.compute_attribution_components(mlp_in_cache)

        # Compute error vectors
        error_vectors = mlp_out_cache - attribution_data["reconstruction"]

        error_vectors[:, 0] = 0
        token_vectors = self.W_E[tokens].detach()  # (n_pos, d_model)

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
    ) -> tuple[torch.Tensor, list[tuple[str, Callable]]]:
        """Sets up an intervention with either frozen attention + LayerNorm(default) or frozen
        attention, LayerNorm, and MLPs, for constrained layers

        Args:
            inputs (Union[str, torch.Tensor]): The inputs to intervene on
            constrained_layers (range | None): whether to apply interventions only to a certain
                range. Mostly applicable to CLTs. If the given range includes all model layers,
                we also freeze layernorm denominators, computing direct effects. None means no
                constraints (iterative patching)

        Returns:
            list[tuple[str, Callable]]: The freeze hooks needed to run the desired intervention.
        """

        hookpoints_to_freeze = ["hook_pattern"]
        if constrained_layers:
            if set(range(self.cfg.n_layers)).issubset(set(constrained_layers)):
                hookpoints_to_freeze.append("hook_scale")
            hookpoints_to_freeze.append(self.feature_output_hook)
            if self.skip_transcoder:
                hookpoints_to_freeze.append(self.feature_input_hook)

        # only freeze outputs in constrained range
        selected_hook_points = []
        for hook_point, hook_obj in self.hook_dict.items():
            if any(
                hookpoint_to_freeze in hook_point for hookpoint_to_freeze in hookpoints_to_freeze
            ):
                # don't freeze feature outputs if the layer is not in the constrained range
                if (
                    self.feature_output_hook in hook_point
                    and constrained_layers
                    and hook_obj.layer() not in constrained_layers
                ):
                    continue
                selected_hook_points.append(hook_point)

        freeze_cache, cache_hooks, _ = self.get_caching_hooks(names_filter=selected_hook_points)

        original_activations, activation_caching_hooks = self._get_activation_caching_hooks()
        self.run_with_hooks(inputs, fwd_hooks=cache_hooks + activation_caching_hooks)

        def freeze_hook(activations, hook):
            cached_values = freeze_cache[hook.name]

            assert activations.shape == cached_values.shape, (
                f"Activations shape {activations.shape} does not match cached values"
                f" shape {cached_values.shape} at hook {hook.name}"
            )
            return cached_values

        fwd_hooks = [
            (hookpoint, freeze_hook)
            for hookpoint in freeze_cache.keys()
            if self.feature_input_hook not in hookpoint
        ]

        if not (constrained_layers and self.skip_transcoder):
            return torch.stack(original_activations), fwd_hooks

        skip_diffs = {}

        def diff_hook(activations, hook, layer: int):
            # The MLP hook out freeze hook sets the value of the MLP to the value it
            # had when run on the inputs normally. We subtract out the skip that
            # corresponds to such a run, and add in the skip with direct effects.
            assert not isinstance(self.transcoders, CrossLayerTranscoder), "Skip CLTs forbidden"
            frozen_skip = self.transcoders[layer].compute_skip(freeze_cache[hook.name])
            normal_skip = self.transcoders[layer].compute_skip(activations)

            skip_diffs[layer] = normal_skip - frozen_skip

        def add_diff_hook(activations, hook, layer: int):
            # open-ended generation case
            return activations + skip_diffs[layer]

        fwd_hooks += [
            (f"blocks.{layer}.{self.feature_input_hook}", partial(diff_hook, layer=layer))
            for layer in constrained_layers
        ]
        fwd_hooks += [
            (f"blocks.{layer}.{self.feature_output_hook}", partial(add_diff_hook, layer=layer))
            for layer in constrained_layers
        ]
        return torch.stack(original_activations), fwd_hooks

    def _get_feature_intervention_hooks(
        self,
        inputs: str | torch.Tensor,
        interventions: list[Intervention],
        constrained_layers: range | None = None,
        freeze_attention: bool = True,
        apply_activation_function: bool = True,
        sparse: bool = False,
        using_past_kv_cache: bool = False,
    ):
        """Given the input, and a dictionary of features to intervene on, performs the
        intervention, allowing all effects to propagate (optionally allowing its effects to
        propagate through transcoders)

        Args:
            input (_type_): the input prompt to intervene on
            intervention_dict (List[Intervention]): A list of interventions to perform, formatted
                as a list of (layer, position, feature_idx, value)
            constrained_layers (range | None): whether to apply interventions only to a certain
                range, freezing all MLPs within the layer range before doing so. This is mostly
                applicable to CLTs. If the given range includes all model layers, we also freeze
                layernorm denominators, computing direct effects.nNone means no constraints
                (iterative patching)
            apply_activation_function (bool): whether to apply the activation function when
                recording the activations to be returned. This is useful to set to False for
                testing purposes, as attribution predicts the change in pre-activation
                feature values.
            sparse (bool): whether to sparsify the activations in the returned cache. Setting
                this to True will take up less memory, at the expense of slower interventions.
            using_past_kv_cache (bool): whether we are generating with past_kv_cache, meaning that
                n_pos is 1, and we must append onto the existing logit / activation cache if the
                hooks are run multiple times. Defaults to False
        """

        interventions_by_layer = defaultdict(list)
        for layer, pos, feature_idx, value in interventions:
            interventions_by_layer[layer].append((pos, feature_idx, value))

        if using_past_kv_cache:
            # We're generating one token at a time
            original_activations, freeze_hooks = [], []
            n_pos = 1
        elif (freeze_attention or constrained_layers) and interventions:
            original_activations, freeze_hooks = self.setup_intervention_with_freeze(
                inputs, constrained_layers=constrained_layers
            )
            n_pos = original_activations.size(1)
        else:
            original_activations, freeze_hooks = [], []
            if isinstance(inputs, torch.Tensor):
                n_pos = inputs.size(0)
            else:
                n_pos = len(self.tokenizer(inputs).input_ids)

        layer_deltas = torch.zeros(
            [self.cfg.n_layers, n_pos, self.cfg.d_model],
            dtype=self.cfg.dtype,
            device=self.cfg.device,
        )

        # This activation cache will fill up during our forward intervention pass
        activation_cache, activation_hooks = self._get_activation_caching_hooks(
            apply_activation_function=apply_activation_function,
            sparse=sparse,
            append=using_past_kv_cache,
        )

        def calculate_delta_hook(activations, hook, layer: int, layer_interventions):
            if constrained_layers:
                # base deltas on original activations; don't let effects propagate
                transcoder_activations = original_activations[layer]
            else:
                # recompute deltas based on current activations
                transcoder_activations = (
                    activation_cache[layer][-1] if using_past_kv_cache else activation_cache[layer]
                )
                if transcoder_activations.is_sparse:
                    transcoder_activations = transcoder_activations.to_dense()

                if not apply_activation_function:
                    transcoder_activations = self.transcoders.apply_activation_function(
                        layer, transcoder_activations.unsqueeze(0)
                    ).squeeze(0)

            activation_deltas = torch.zeros_like(transcoder_activations)
            for pos, feature_idx, value in layer_interventions:
                activation_deltas[pos, feature_idx] = (
                    value - transcoder_activations[pos, feature_idx]
                )

            poss, feature_idxs = activation_deltas.nonzero(as_tuple=True)
            new_values = activation_deltas[poss, feature_idxs]

            decoder_vectors = self.transcoders._get_decoder_vectors(layer, feature_idxs)

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

        def intervention_hook(activations, hook, layer: int):
            new_acts = activations
            if layer in intervention_range:
                new_acts = new_acts + layer_deltas[layer]
            layer_deltas[layer] *= 0  # clearing this is important for multi-token generation
            return new_acts

        delta_hooks = [
            (
                f"blocks.{layer}.{self.feature_output_hook}",
                partial(calculate_delta_hook, layer=layer, layer_interventions=layer_interventions),
            )
            for layer, layer_interventions in interventions_by_layer.items()
        ]

        intervention_range = constrained_layers if constrained_layers else range(self.cfg.n_layers)
        intervention_hooks = [
            (f"blocks.{layer}.{self.feature_output_hook}", partial(intervention_hook, layer=layer))
            for layer in range(self.cfg.n_layers)
        ]

        all_hooks = freeze_hooks + activation_hooks + delta_hooks + intervention_hooks
        cached_logits = [] if using_past_kv_cache else [None]

        def logit_cache_hook(activations, hook):
            # we need to manually apply the softcap (if used by the model), as it comes post-hook
            if self.cfg.output_logits_soft_cap > 0.0:
                logits = self.cfg.output_logits_soft_cap * F.tanh(
                    activations / self.cfg.output_logits_soft_cap
                )
            else:
                logits = activations.clone()
            if using_past_kv_cache:
                cached_logits.append(logits)
            else:
                cached_logits[0] = logits

        all_hooks.append(("unembed.hook_post", logit_cache_hook))

        return all_hooks, cached_logits, activation_cache

    @torch.no_grad
    def feature_intervention(
        self,
        inputs: str | torch.Tensor,
        interventions: list[Intervention],
        constrained_layers: range | None = None,
        freeze_attention: bool = True,
        apply_activation_function: bool = True,
        sparse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given the input, and a dictionary of features to intervene on, performs the
        intervention, and returns the logits and feature activations. If freeze_attention or
        constrained_layers is True, attention patterns will be frozen, along with MLPs and
        LayerNorms. If constrained_layers is set, the effects of intervention will not propagate
        through the constrained layers, and CLTs will write only to those layers. Otherwise, the
        effects of the intervention will propagate through transcoders / LayerNorms

        Args:
            input (_type_): the input prompt to intervene on
            interventions (list[tuple[int, Union[int, slice, torch.Tensor]], int,
                Union[int, torch.Tensor]]): A list of interventions to perform, formatted as
                a list of (layer, position, feature_idx, value)
            constrained_layers (range | None): whether to apply interventions only to a certain
                range. Mostly applicable to CLTs. If the given range includes all model layers,
                we also freeze layernorm denominators, computing direct effects. None means no
                constraints (iterative patching)
            freeze_attention (bool): whether to freeze all attention patterns an layernorms
            apply_activation_function (bool): whether to apply the activation function when
                recording the activations to be returned. This is useful to set to False for
                testing purposes, as attribution predicts the change in pre-activation
                feature values.
            sparse (bool): whether to sparsify the activations in the returned cache. Setting
                this to True will take up less memory, at the expense of slower interventions.
        """

        hooks, _, activation_cache = self._get_feature_intervention_hooks(
            inputs,
            interventions,
            constrained_layers=constrained_layers,
            freeze_attention=freeze_attention,
            apply_activation_function=apply_activation_function,
            sparse=sparse,
        )

        with self.hooks(hooks):  # type: ignore
            logits = self(inputs)

        activation_cache = torch.stack(activation_cache)

        return logits, activation_cache

    def _convert_open_ended_interventions(
        self,
        interventions: list[Intervention],
    ) -> list[Intervention]:
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
        interventions: list[Intervention],
        constrained_layers: range | None = None,
        freeze_attention: bool = True,
        apply_activation_function: bool = True,
        sparse: bool = False,
        **kwargs,
    ) -> tuple[str, torch.Tensor, torch.Tensor]:
        """Given the input, and a dictionary of features to intervene on, performs the
        intervention, and generates a continuation, along with the logits and activations at
        each generation position.
        This function accepts all kwargs valid for HookedTransformer.generate(). Note that
        freeze_attention applies only to the first token generated.

        This function accepts all kwargs valid for HookedTransformer.generate(). Note that
        direct_effects and freeze_attention apply only to the first token generated.

        Note that if kv_cache is True (default), generation will be faster, as the model
        will cache the KVs, and only process the one new token per step; if it is False,
        the model will generate by doing a full forward pass across all tokens. Note that
        due to numerical precision issues, you are only guaranteed that the logits /
        activations of model.feature_intervention_generate(s, ...) are equivalent to
        model.feature_intervention(s, ...) if kv_cache is False.

        Args:
            input (_type_): the input prompt to intervene on
            interventions (list[tuple[int, Union[int, slice, torch.Tensor]], int,
                Union[int, torch.Tensor]]): A list of interventions to perform, formatted as
                a list of (layer, position, feature_idx, value)
            constrained_layers: (range | None = None): whether to freeze all MLPs/transcoders /
                attn patterns / layernorm denominators. This will only apply to the very first
                token generated. If all layers are constrained, also freezes layernorm, computing
                direct effects.
            freeze_attention (bool): whether to freeze all attention patterns. Applies only to
                the first token generated
            apply_activation_function (bool): whether to apply the activation function when
                recording the activations to be returned. This is useful to set to False for
                testing purposes, as attribution predicts the change in pre-activation
                feature values.
            sparse (bool): whether to sparsify the activations in the returned cache. Setting
                this to True will take up less memory, at the expense of slower interventions.
        """

        feature_intervention_hook_output = self._get_feature_intervention_hooks(
            inputs,
            interventions,
            constrained_layers=constrained_layers,
            freeze_attention=freeze_attention,
            apply_activation_function=apply_activation_function,
            sparse=sparse,
        )

        hooks, logit_cache, activation_cache = feature_intervention_hook_output

        assert kwargs.get("use_past_kv_cache", True), (
            "Generation is only possible with use_past_kv_cache=True"
        )
        # Next, convert any open-ended interventions so they target position `0` (the
        # only token present during the incremental forward passes performed by
        # `generate`) and build the corresponding hooks.
        open_ended_interventions = self._convert_open_ended_interventions(interventions)

        # get new hooks that will target pos 0 / append logits / acts to the cache (not overwrite)
        open_ended_hooks, open_ended_logits, open_ended_activations = (
            self._get_feature_intervention_hooks(
                inputs,
                open_ended_interventions,
                constrained_layers=None,
                freeze_attention=False,
                apply_activation_function=apply_activation_function,
                sparse=sparse,
                using_past_kv_cache=True,
            )
        )

        # at the end of the model, clear original hooks and add open-ended hooks
        def clear_and_add_hooks(tensor, hook):
            self.reset_hooks()
            for open_ended_name, open_ended_hook in open_ended_hooks:
                self.add_hook(open_ended_name, open_ended_hook)

        for name, hook in hooks:
            self.add_hook(name, hook)

        self.add_hook("unembed.hook_post", clear_and_add_hooks)

        generation: str = self.generate(inputs, **kwargs)  # type:ignore
        self.reset_hooks()

        logits = torch.cat((logit_cache[0], *open_ended_logits), dim=1)  # type:ignore
        open_ended_activations = torch.stack(
            [torch.cat(acts, dim=0) for acts in open_ended_activations],  # type:ignore
            dim=0,
        )
        activation_cache = torch.stack(activation_cache)
        activations = torch.cat((activation_cache, open_ended_activations), dim=1)
        if sparse:
            activations = activations.coalesce()

        return generation, logits, activations

    def __del__(self):
        # Prevent memory leaks
        self.reset_hooks(including_permanent=True)
