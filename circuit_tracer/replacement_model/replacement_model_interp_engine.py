"""ReplacementModel backed by `interp-engine <https://github.com/decoderesearch/interp-engine>`_.

The other two backends reach their hook points through a framework that has already restructured
the model: TransformerLens rewrites the checkpoint into ``HookedTransformer``'s own module tree,
and nnsight traces into the HF tree using per-architecture accessor strings (see
``utils/tl_nnsight_mapping.py``, where ``Gemma2ForCausalLM``'s attention pattern is spelled
``model.layers[{layer}].self_attn.source.attention_interface_0.source.nn_functional_dropout_0``).
Both work; both need a hand-written table entry per architecture before a model can be traced.

This backend runs the unmodified HF model and resolves the same structural roles by *inspecting*
it, via interp-engine's ``EagerModel.resolve_point``. So the two things a replacement model needs
are derived rather than tabulated:

* **Where the transcoders read and write.** ``feature_input_hook``/``feature_output_hook`` arrive
  as TransformerLens names and are translated by ``interp_engine.mappers.tlens_hook_to_point``,
  which knows that a block-level ``hook_mlp_out`` means the *residual contribution* and therefore
  lands after the post-MLP norm on a sandwich-norm architecture like Gemma-2, but on the MLP
  itself elsewhere. That distinction is the one the nnsight table encodes by hand per family.
* **Which gradients to freeze.** Attribution needs a *local replacement model*: linear in the
  residual stream, with attention patterns and normalization scales held constant. Both freezes
  are implemented against structure rather than names -- see :func:`_frozen_eager_attention` and
  :meth:`InterpEngineReplacementModel._install_norm_scale_freeze`.

Interventions need a second kind of freeze, built on the same two places. Attribution *detaches*:
the forward value stays and the gradient stops. An intervention *replays*: the perturbed run is
made to see the attention pattern and the normalization denominator that the clean run produced,
so that what reaches the logits is the intervention's direct effect rather than its effect on the
pattern. Both freezes are expressed through the same tap attributes -- see :class:`_Replay`.

What this costs: the freezes are verified numerically at load (:meth:`_verify_freezes`) rather
than asserted, because "generic" is a claim about architectures nobody has run yet.
"""

import sys
from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator, Literal, Protocol, cast

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
from transformers import PreTrainedModel
from transformers.generation import GenerateDecoderOnlyOutput, GenerationMixin

from circuit_tracer.attribution.context_interp_engine import AttributionContext
from circuit_tracer.transcoder import TranscoderSet
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
from circuit_tracer.utils import get_default_device
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
from circuit_tracer.utils.interventions import Intervention, convert_open_ended_interventions
from circuit_tracer.utils.tl_nnsight_mapping import convert_nnsight_config_to_transformerlens
from circuit_tracer.utils.tokenization import ensure_tokenized, zero_positions

if TYPE_CHECKING:
    # Type-check-only, so the optional dependency stays optional at runtime while the engine's
    # signatures stay visible here. Reaching it through a helper instead erases them: a function
    # returning a module is typed `ModuleType`, and every call on that is `Any`. That is not
    # hypothetical -- it is why `tlens_hook_to_point` changing from a `(name, layer)` pair to an
    # `Address` type-checked clean and failed at load.
    from interp_engine.model import EagerModel

BACKEND = "interp_engine"

# Attributes the hooked code looks for, to let a caller see or replace a tensor that is a local
# variable inside some module's forward rather than any module's output. A module hook cannot reach
# these, which is the whole reason the other two backends need a framework to get at them.
_PATTERN_TAP = "_ct_pattern_tap"
_SCALE_TAP = "_ct_scale_tap"
_INPUT_SCALE_TAP = "_ct_input_scale_tap"

# Name we register the gradient-frozen attention under. transformers dispatches attention through
# a global registry keyed by ``config._attn_implementation``, so this has to be a name no other
# implementation uses; the model's config then selects it per-model rather than globally.
FROZEN_ATTN_IMPL = "circuit_tracer_frozen"

_INSTALL_ERROR = (
    "The interp_engine backend requires the interp-engine package:\n"
    "    pip install 'circuit-tracer[interp-engine]'\n"
    "It is a separate library (https://github.com/decoderesearch/interp-engine); circuit-tracer "
    "does not depend on it, so that the transformerlens and nnsight backends keep working "
    "without it.\n"
    f"If that install reported success, check your Python version: you are on {sys.version_info.major}."
    f"{sys.version_info.minor}, and the extra is marker-gated to the 3.11-3.13 the engine "
    "publishes for, so on anything else it resolves to nothing rather than failing."
)


def _import_eager_model() -> "type[EagerModel]":
    """Import interp-engine's model class, turning its absence into an actionable message.

    Returns the class rather than the package: a module-typed return would be `ModuleType`, and
    everything reached through it `Any`. Other symbols are imported at their use site, guarded the
    same way, for the same reason.
    """
    try:
        from interp_engine.model import EagerModel
    except ImportError as e:  # pragma: no cover - depends on the environment, not on us
        raise ImportError(_INSTALL_ERROR) from e
    return EagerModel


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand grouped-query key/value heads to one per query head.

    Same expansion transformers does; reimplemented rather than imported because which module
    exports ``repeat_kv`` moves between releases and between families, and getting it wrong here
    would silently produce plausible numbers.
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return expanded.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def _frozen_eager_attention(module, query, key, value, attention_mask, **kwargs):
    """Eager attention whose probabilities are constants with respect to the graph.

    Attribution requires the attention pattern to be frozen: a feature's effect should be
    measured through the values it moves, not through the pattern it induces. TransformerLens
    gets this from a ``hook_pattern`` hook point and nnsight from a per-architecture path into its
    trace; neither is available on a plain HF module, because the probabilities are a local
    variable inside the attention forward rather than any module's output.

    They *are*, however, the second return value of the attention interface transformers
    dispatches through. So this calls the family's own eager implementation -- keeping its
    scaling, causal and sliding-window masking, logit softcapping and attention sinks exactly as
    written -- and then redoes only the final ``probs @ value`` contraction with the
    probabilities detached. That contraction is the entire freeze, and it is the one line of
    attention math that does not vary between families.

    Verified bit-exact against the family's own output in :meth:`_verify_freezes`.
    """
    family = sys.modules[type(module).__module__]
    eager = getattr(family, "eager_attention_forward", None)
    if eager is None:
        raise NotImplementedError(
            f"{type(module).__name__} lives in {family.__name__}, which does not define "
            "`eager_attention_forward`, so there is no family implementation to freeze. Use the "
            "transformerlens or nnsight backend for this architecture."
        )

    attn_output, attn_weights = eager(module, query, key, value, attention_mask, **kwargs)
    if attn_weights is None:
        raise NotImplementedError(
            f"{family.__name__}'s eager attention did not return attention probabilities, so they "
            "cannot be frozen. Use the transformerlens or nnsight backend for this architecture."
        )

    # An intervention may be replaying the clean run's pattern here; see `_Replay`.
    tap = getattr(module, _PATTERN_TAP, None)
    replayed = tap(attn_weights) if tap is not None else None
    if replayed is not None:
        attn_weights = replayed

    n_rep = getattr(module, "num_key_value_groups", None)
    if n_rep is None:
        n_rep = query.shape[1] // value.shape[1]
    frozen = torch.matmul(attn_weights.detach(), _repeat_kv(value, n_rep))
    frozen = frozen.transpose(1, 2).contiguous()

    # Hand the discrepancy to the audit rather than checking it on every forward pass. The
    # attribute is set on the attention module by the model that owns it, so two models loaded in
    # one process do not audit each other. With a replayed pattern the two outputs are meant to
    # differ, so there is nothing to compare.
    audit = getattr(module, "_ct_freeze_audit", None)
    if audit is not None and replayed is None:
        audit.record_attention(attn_output.detach(), frozen.detach())

    return frozen, attn_weights


def _register_frozen_attention() -> None:
    """Register the frozen attention implementation with transformers, once per process."""
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface

    if FROZEN_ATTN_IMPL in ALL_ATTENTION_FUNCTIONS:
        return
    AttentionInterface.register(FROZEN_ATTN_IMPL, _frozen_eager_attention)
    # transformers looks the mask builder up under the same key as the attention implementation.
    # Without this the model runs with no causal mask at all -- and still produces numbers.
    ALL_MASK_ATTENTION_FUNCTIONS.register(FROZEN_ATTN_IMPL, ALL_MASK_ATTENTION_FUNCTIONS["eager"])


def _select_frozen_attention(hf_model: nn.Module) -> None:
    """Point an already-loaded model at the frozen attention implementation.

    The cast is because interp-engine types ``EagerModel.hf_model`` as ``nn.Module``, which is the
    honest annotation for something that also loads multimodal text stacks, while
    ``set_attn_implementation`` is defined on ``PreTrainedModel``. Anything reaching this backend
    came from ``AutoModelForCausalLM`` and so is one.
    """
    cast(PreTrainedModel, hf_model).set_attn_implementation(FROZEN_ATTN_IMPL)


def _generate(
    hf_model: nn.Module, input_ids: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Generate a continuation: the whole token sequence, and the logits behind each new token.

    The cast is for the same reason as in :func:`_select_frozen_attention`: interp-engine types
    ``hf_model`` as ``nn.Module``, and ``generate`` comes from ``GenerationMixin``, which
    ``PreTrainedModel`` itself does not inherit -- only the concrete causal-LM classes do. Anything
    reaching this backend came from ``AutoModelForCausalLM`` and so is one.
    """
    output = cast(GenerationMixin, hf_model).generate(
        input_ids, return_dict_in_generate=True, output_logits=True, **kwargs
    )
    assert isinstance(output, GenerateDecoderOnlyOutput) and output.logits is not None, (
        f"asked generate() for per-step logits and it returned {type(output).__name__}"
    )
    return output.sequences, output.logits


# TransformerLens hooks that name a tensor *inside* a norm module rather than any module's output.
# `RMSNorm.forward` fires `hook_normalized` on ``x / scale``, before multiplying by the learned
# weight, so it is the unweighted normalized residual -- not the following sublayer's input, which
# is that same tensor after the weight. interp-engine deliberately refuses to map these (there is
# no HF module whose output it is), so they are resolved here, to the norm in front of the named
# sublayer. The Gemma Scope transcoders read from `ln2.hook_normalized`, so this is not a corner.
_NORMALIZED_HOOKS: dict[str, str] = {
    "ln1.hook_normalized": "attn",
    "ln2.hook_normalized": "mlp",
}


def _input_tensor(args: tuple, kwargs: dict) -> tuple[torch.Tensor, object]:
    """The hidden-state argument of a module call, and a key for substituting it back.

    Necessary because transformers calls its submodules inconsistently: a decoder layer invokes
    ``self.mlp(hidden_states)`` positionally but ``self.self_attn(hidden_states=...)`` by keyword,
    so a pre-hook that only reads ``args[0]`` works on one and raises on the other.
    """
    if args and isinstance(args[0], torch.Tensor):
        return args[0], 0
    for name in ("hidden_states", "x", "input", "inputs_embeds"):
        value = kwargs.get(name)
        if isinstance(value, torch.Tensor):
            return value, name
    raise TypeError(
        f"Cannot find the hidden-state argument of this module call: positional "
        f"{[type(a).__name__ for a in args]}, keyword {sorted(kwargs)}"
    )


def _substitute_input(args: tuple, kwargs: dict, key: object, new: torch.Tensor) -> tuple:
    """Rebuild ``(args, kwargs)`` with the hidden state at ``key`` replaced."""
    if key == 0:
        return ((new,) + tuple(args[1:]), kwargs)
    return (args, {**kwargs, str(key): new})


class _Removable(Protocol):
    """What :meth:`InterpEngineReplacementModel.hooks` needs to undo an installation.

    Satisfied by ``RemovableHandle`` and by :class:`_Restore`, so module hooks and the tap
    attributes can be installed and removed as one list.
    """

    def remove(self) -> None: ...


class _Restore:
    """A ``RemovableHandle``-alike for an installation that is not a module hook."""

    def __init__(self, undo: Callable[[], None]) -> None:
        self._undo = undo

    def remove(self) -> None:
        self._undo()


# One modification to install for the duration of one intervention. Deferred rather than installed
# eagerly so that a freeze can be described once and installed twice: once to record the clean
# run's values, and once to force them.
InterventionHook = Callable[[], _Removable]


def _tap_hook(module: nn.Module, attribute: str, fn: Callable) -> InterventionHook:
    """Install ``fn`` as the tap the hooked code inside ``module``'s forward consults."""

    def install() -> _Removable:
        setattr(module, attribute, fn)
        return _Restore(lambda: setattr(module, attribute, None))

    return install


def _listener_hook(bucket: list, fn: Callable) -> InterventionHook:
    """Append ``fn`` to one of the ordered lists the permanent feature hooks walk.

    Interventions extend the forward pass through these lists rather than by registering more
    module hooks, because their order relative to the input cache and the output splice has to be
    definite: an earlier version of this backend produced a plausible but wrong attribution graph
    because two forward hooks on one module happened to run in the wrong order.
    """

    def install() -> _Removable:
        bucket.append(fn)
        return _Restore(lambda: bucket.remove(fn))

    return install


class _Replay:
    """One tensor, recorded on a clean forward pass and forced on a later one.

    ``record`` returns ``None``, which every tap reads as "observed, carry on"; ``replay`` returns
    the recorded tensor, which every tap reads as "use this instead". So the same object installed
    two different ways is the whole freeze.

    ``active`` exists for generation, where the freeze applies only to the pass that consumes the
    prompt: on later passes the tensor has one position instead of many and there is nothing
    sensible to force, which is also what the TransformerLens backend does by swapping its hooks
    out after the first pass.
    """

    def __init__(self, description: str, active: Callable[[], bool] | None = None) -> None:
        self.description = description
        self.active = active
        self.value: torch.Tensor | None = None

    def record(self, tensor: torch.Tensor) -> None:
        self.value = tensor.detach()

    def replay(self, tensor: torch.Tensor) -> torch.Tensor | None:
        if self.active is not None and not self.active():
            return None
        assert self.value is not None, f"{self.description} was never recorded"
        assert tensor.shape == self.value.shape, (
            f"{self.description} has shape {tuple(tensor.shape)} on this pass but was recorded "
            f"with shape {tuple(self.value.shape)}"
        )
        return self.value


@dataclass(frozen=True)
class _Freeze:
    """One tensor held at its clean-run value: how to record it, and how to force it."""

    record: InterventionHook
    replay: InterventionHook

    @staticmethod
    def tap(module: nn.Module, attribute: str, replay: _Replay) -> "_Freeze":
        return _Freeze(
            record=_tap_hook(module, attribute, replay.record),
            replay=_tap_hook(module, attribute, replay.replay),
        )

    @staticmethod
    def listener(bucket: list, replay: _Replay) -> "_Freeze":
        def record(tensor: torch.Tensor) -> torch.Tensor:
            replay.record(tensor)
            return tensor

        def force(tensor: torch.Tensor) -> torch.Tensor:
            replayed = replay.replay(tensor)
            return tensor if replayed is None else replayed

        return _Freeze(record=_listener_hook(bucket, record), replay=_listener_hook(bucket, force))


def _as_int(value: int | torch.Tensor) -> int:
    """A layer index, whether the caller wrote a number or a scalar tensor."""
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


class _Deltas:
    """What one forward pass has to add to the residual stream to realize the interventions.

    An intervention names a feature and a value; what the model has to be handed is that feature's
    decoder vector, scaled by how far the value is from what the feature actually did. Kept as one
    buffer across layers because a cross-layer transcoder's decoder writes into layers the
    intervened layer has not reached yet, so a layer's addition may be owed by an earlier one.

    Sized per pass rather than once, because generation runs the prompt and then one token at a
    time.
    """

    def __init__(self, n_layers: int, d_model: int, dtype: torch.dtype, device: torch.device):
        self.n_layers = n_layers
        self.d_model = d_model
        self.dtype = dtype
        self.device = device
        self.buffer: torch.Tensor | None = None

    def begin(self, _module, args: tuple, kwargs: dict) -> None:
        ids = kwargs.get("input_ids")
        if ids is None:
            ids = args[0]
        self.buffer = torch.zeros(
            [self.n_layers, ids.shape[-1], self.d_model], dtype=self.dtype, device=self.device
        )

    def add(
        self,
        layer: int,
        baseline: torch.Tensor,
        layer_interventions: list,
        transcoders: TranscoderSet | CrossLayerTranscoder,
    ) -> None:
        """Record the additions the interventions at ``layer`` imply, against ``baseline``."""
        assert self.buffer is not None, "no forward pass is running"
        activation_deltas = torch.zeros_like(baseline)
        for pos, feature_idx, value in layer_interventions:
            activation_deltas[pos, feature_idx] = value - baseline[pos, feature_idx]

        positions, feature_idxs = activation_deltas.nonzero(as_tuple=True)
        new_values = activation_deltas[positions, feature_idxs]
        decoder_vectors = transcoders._get_decoder_vectors(layer, feature_idxs)

        if decoder_vectors.ndim == 2:
            scaled = decoder_vectors * new_values.unsqueeze(1)
            self.buffer[layer].index_add_(0, positions, scaled.to(self.dtype))
        else:
            # Cross-layer: [features, remaining layers, d_model], landing on the tail of the stack.
            scaled = (decoder_vectors * new_values.unsqueeze(-1).unsqueeze(-1)).transpose(0, 1)
            self.buffer[-scaled.shape[0] :].index_add_(1, positions, scaled.to(self.dtype))

    def take(self, layer: int, tensor: torch.Tensor, add: bool) -> torch.Tensor:
        """Add ``layer``'s pending delta to ``tensor``, and clear it either way."""
        assert self.buffer is not None, "no forward pass is running"
        if add:
            tensor = tensor + self.buffer[layer]
        self.buffer[layer] *= 0
        return tensor


def _stack_activations(rows: list[list[torch.Tensor]], sparse: bool) -> torch.Tensor:
    """One tensor from the per-layer, per-pass activations a run collected.

    A single forward leaves one entry per layer. A generation leaves the prompt's positions and
    then one per generated token, which join into a single position axis.
    """
    per_layer = [row[0] if len(row) == 1 else torch.cat(row, dim=0) for row in rows]
    stacked = torch.stack(per_layer)
    return stacked.coalesce() if sparse else stacked


@dataclass(frozen=True)
class _Location:
    """A hookable tensor: a module plus which of its tensors is meant.

    ``input``/``output`` are what interp-engine's ``resolve_point`` returns. ``normalized`` is
    this backend's own, for the tensor inside a norm that :data:`_NORMALIZED_HOOKS` names.
    """

    module: nn.Module
    side: str


def _norm_modules(root: nn.Module) -> list[nn.Module]:
    """Every normalization module under ``root``.

    Found by class name because that is the one thing the families agree on: ``LlamaRMSNorm``,
    ``Gemma2RMSNorm``, ``GptOssRMSNorm``, ``nn.LayerNorm``. Collecting all of them, rather than
    the specific ones a table names, is what makes the freeze cover the norms a new architecture
    adds -- Gemma-3's ``q_norm``/``k_norm``, a sandwich norm, a final norm -- without an edit.
    """
    return [
        module
        for module in root.modules()
        if isinstance(module, nn.LayerNorm) or "norm" in type(module).__name__.lower()
    ]


def _norm_eps(norm: nn.Module) -> float:
    """The epsilon inside a norm's denominator, whatever the family calls it."""
    for attr in ("eps", "variance_epsilon", "epsilon"):
        value = getattr(norm, attr, None)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _subtracts_mean(norm: nn.Module) -> bool:
    """Whether the norm centers its input before scaling (LayerNorm) or not (RMSNorm).

    Asked of interp-engine, which owns the question. The rules it applies are structural rather
    than name-based, because the name is wrong on a family transformers ships: ``T5LayerNorm``
    (T5, mT5, UMT5, Flan-T5) subtracts no mean despite what it is called, and centering there feeds
    the transcoder a tensor the model never computed -- finite and right-shaped, so nothing
    downstream notices.

    This reimplemented those rules until the engine floor here allowed importing them. The local
    name survives the delegation because both call sites want the centering polarity, which is the
    negation of the one the engine answers in.
    """
    try:
        from interp_engine.capture import is_rms_norm
    except ImportError as e:  # pragma: no cover - depends on the environment, not on us
        raise ImportError(_INSTALL_ERROR) from e
    return not is_rms_norm(norm)


def _weight_of(module: nn.Module) -> torch.Tensor:
    """A module's ``weight``, narrowed.

    ``nn.Module.__getattr__`` types every attribute it does not declare as ``Tensor | Module``, so
    an embedding or unembedding matrix arrives here as a union rather than as the tensor it is.
    """
    weight = module.weight
    if not isinstance(weight, torch.Tensor):
        raise TypeError(
            f"{type(module).__name__}.weight is a {type(weight).__name__}, not a tensor"
        )
    return weight


class InterpEngineReplacementModel(nn.Module):
    """A local replacement model over an unmodified HF model, hooked via interp-engine.

    Exposes the surface the attribution code needs: ``ensure_tokenized``, ``get_activations``,
    ``setup_attribution``, and the module/weight accessors used for offloading.
    """

    backend: Literal["interp_engine"] = BACKEND  # type: ignore[assignment]

    def __init__(self, engine: "EagerModel", transcoders: TranscoderSet | CrossLayerTranscoder):
        super().__init__()
        self.engine = engine
        # The engine types this `nn.Module`, under which `.config` resolves through
        # `nn.Module.__getattr__` to `Tensor | Module`. Every loader that reaches here produces an
        # HF model, so narrow it once rather than at each `.config` read.
        self.hf_model: PreTrainedModel = cast(PreTrainedModel, engine.hf_model)
        self.tokenizer = engine.tokenizer
        self.arch = engine.arch
        self.cfg = convert_nnsight_config_to_transformerlens(engine.hf_model.config)

        self.transcoders = transcoders
        self.feature_input_hook = transcoders.feature_input_hook
        self.feature_output_hook = transcoders.feature_output_hook
        self.skip_transcoder = transcoders.skip_connection
        self.scan_name = transcoders.scan_name
        self.zero_positions = zero_positions(str(self.cfg.model_name))

        self.n_layers = self.arch.n_layers
        self.device = next(engine.hf_model.parameters()).device
        self.dtype = next(engine.hf_model.parameters()).dtype

        # Populated afresh on every forward pass by the permanent hooks below, and read by
        # AttributionContext after the pass. Kept on the model rather than passed around because
        # the splice hook and the caching hook have to see the same tensor.
        self._feature_input_tensors: list[torch.Tensor | None] = [None] * self.n_layers
        self._feature_output_tensors: list[torch.Tensor | None] = [None] * self.n_layers
        self._feature_output_raw: list[torch.Tensor | None] = [None] * self.n_layers
        self._embed_tensor: torch.Tensor | None = None

        # Where interventions extend the forward pass, walked by the permanent hooks above at the
        # points the transcoders read and write. See `_listener_hook` for why they are lists the
        # hooks consult rather than more hooks.
        self._input_listeners: list[list[Callable[[torch.Tensor], None]]] = [
            [] for _ in range(self.n_layers)
        ]
        self._output_transforms: list[list[Callable[[torch.Tensor], torch.Tensor]]] = [
            [] for _ in range(self.n_layers)
        ]

        self._audit = _FreezeAudit()
        self._handles: list[RemovableHandle] = []
        self._configure_replacement_model()

    # --- construction ---------------------------------------------------------

    @classmethod
    def from_pretrained_and_transcoders(
        cls,
        model_name: str,
        transcoders: TranscoderSet | CrossLayerTranscoder,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> "InterpEngineReplacementModel":
        """Load ``model_name`` through interp-engine and wrap it with ``transcoders``."""
        EagerModel = _import_eager_model()
        if device is None:
            device = get_default_device()

        _register_frozen_attention()
        engine = EagerModel(
            model_name,
            device=str(device),
            dtype=dtype,
            # Loaded eager so that the family routes through the attention interface at all, then
            # switched to the frozen implementation below.
            attn_implementation="eager",
            requires_grad=True,
            **kwargs,
        )
        _select_frozen_attention(engine.hf_model)
        transcoders.to(device, dtype)
        return cls(engine, transcoders)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        transcoder_set: str,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> "InterpEngineReplacementModel":
        """Load a model and a named transcoder set together."""
        if device is None:
            device = get_default_device()
        transcoders, _ = load_transcoder_from_hub(transcoder_set, device=device, dtype=dtype)  # type: ignore
        return cls.from_pretrained_and_transcoders(
            model_name, transcoders, device=device, dtype=dtype, **kwargs
        )

    @classmethod
    def from_config(cls, config, transcoders, **kwargs) -> "InterpEngineReplacementModel":
        """Build an untrained model from an HF config. Used by tests, to avoid a download."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        EagerModel = _import_eager_model()
        _register_frozen_attention()
        config._attn_implementation = "eager"
        hf_model = AutoModelForCausalLM.from_config(config)
        engine = EagerModel(
            config.name_or_path,
            hf_model=hf_model,
            tokenizer=AutoTokenizer.from_pretrained(config.name_or_path),
            requires_grad=True,
            **kwargs,
        )
        _select_frozen_attention(engine.hf_model)
        return cls(engine, transcoders)

    def _configure_replacement_model(self) -> None:
        """Resolve the transcoder hook points and install every permanent hook."""
        self.hf_model.eval()
        # Everything, transcoders included: gradients are wanted with respect to activations only.
        for param in self.parameters():
            param.requires_grad = False

        self._pre_sublayer_norms = self._find_pre_sublayer_norms()
        self._input_locs = self._resolve_feature_hook(self.feature_input_hook)
        self._output_locs = self._resolve_feature_hook(self.feature_output_hook)

        # Registration order is load-bearing, because on a sandwich-norm architecture both of the
        # next two put a forward hook on the *same* module: Gemma-2's transcoders write at
        # `hook_mlp_out`, which is its post-MLP norm, and that norm is also a scale-freeze target.
        # PyTorch runs forward hooks in registration order and each one that returns a value feeds
        # the next, so the splice has to be last. Reversed, the freeze would re-derive its term
        # from the norm's input and reconnect the very gradient path the splice just cut -- leaving
        # a model that runs, produces the same logits, and attributes through the MLP anyway.
        self._install_norm_scale_freeze()
        self._install_feature_hooks()
        self._install_embed_gradient()
        self._clear_taps()
        self._verify_freezes()

    def _clear_taps(self) -> None:
        """Give every tappable module its tap attribute, empty.

        So that the hooked code's ``getattr`` finds the attribute and reads ``None`` rather than
        falling into ``nn.Module.__getattr__`` and raising, on every norm of every forward pass.
        """
        for layer in range(self.n_layers):
            setattr(self.arch.attn_module(layer), _PATTERN_TAP, None)
        for norm in _norm_modules(self.hf_model):
            setattr(norm, _SCALE_TAP, None)
            setattr(norm, _INPUT_SCALE_TAP, None)

    def _find_pre_sublayer_norms(self) -> list[dict[str, nn.Module]]:
        """Which norm module feeds each sublayer, per layer, found by watching one forward pass.

        Needed because ``ln1``/``ln2`` name a *position* in the block, and the module that occupies
        it is called something different in every family: Gemma-2's pre-MLP norm is
        ``pre_feedforward_layernorm`` while Llama's is ``post_attention_layernorm`` -- a name that
        on Gemma-2 belongs to a different norm entirely. Rather than encode that, this records
        which norm's output tensor *is* the tensor each sublayer received, by identity. A name
        cannot be wrong about that.
        """
        # Strong references on purpose, matched with `is` afterwards. Keying on ``id()`` would be
        # wrong twice over: the probe runs under ``no_grad``, so intermediates are freed the moment
        # they are consumed, and CPython then hands the same address to a later tensor -- so ids
        # both go stale and collide across layers.
        produced: list[tuple[torch.Tensor, int, nn.Module]] = []
        consumed: list[dict[str, torch.Tensor]] = [{} for _ in range(self.n_layers)]
        handles = []

        def watch_norm(layer: int, norm: nn.Module):
            def hook(_m, _a, output):
                produced.append((_hidden(output), layer, norm))

            return hook

        def watch_sublayer(layer: int, kind: str):
            def hook(_m, args, kwargs):
                consumed[layer][kind] = _input_tensor(args, kwargs)[0]

            return hook

        try:
            for layer, block in enumerate(self.arch.decoder_layers):
                for norm in _norm_modules(block):
                    handles.append(norm.register_forward_hook(watch_norm(layer, norm)))
                for kind, module in (
                    ("attn", self.arch.attn_module(layer)),
                    ("mlp", self.arch.mlp_module(layer)),
                ):
                    handles.append(
                        module.register_forward_pre_hook(
                            watch_sublayer(layer, kind), with_kwargs=True
                        )
                    )

            ids = torch.tensor([[self.tokenizer.bos_token_id or 0, 1, 2]], device=self.device)
            with torch.no_grad():
                self.hf_model(ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        found: list[dict[str, nn.Module]] = []
        for layer in range(self.n_layers):
            layer_norms: dict[str, nn.Module] = {}
            for kind, tensor in consumed[layer].items():
                # A sublayer whose input came from no norm in its own block -- a parallel block
                # that shares one norm, say -- simply gets no entry, and asking for it raises.
                for candidate, produced_layer, norm in produced:
                    if candidate is tensor and produced_layer == layer:
                        layer_norms[kind] = norm
                        break
            found.append(layer_norms)
        return found

    def _resolve_feature_hook(self, hook: str) -> list["_Location"]:
        """Turn one TransformerLens hook name into a per-layer list of hook locations."""
        try:
            from interp_engine.mappers import UnmappedHook, tlens_hook_to_point
        except ImportError as e:  # pragma: no cover - depends on the environment, not on us
            raise ImportError(_INSTALL_ERROR) from e

        if hook in _NORMALIZED_HOOKS:
            kind = _NORMALIZED_HOOKS[hook]
            locations = []
            for layer in range(self.n_layers):
                norm = self._pre_sublayer_norms[layer].get(kind)
                if norm is None:
                    raise NotImplementedError(
                        f"Could not find the norm feeding the {kind} sublayer of layer {layer} on "
                        f"{self.arch.architecture}, so {hook!r} cannot be resolved. Use the "
                        "transformerlens or nnsight backend for these transcoders."
                    )
                locations.append(_Location(norm, "normalized"))
            return locations

        # Everything else is a point interp-engine already resolves. Asked per layer because
        # tlens_hook_to_point needs a layer to parse and can answer differently per architecture.
        try:
            addresses = [
                tlens_hook_to_point(f"blocks.{layer}.{hook}", self.engine)
                for layer in range(self.n_layers)
            ]
        except UnmappedHook as e:
            raise NotImplementedError(
                f"interp-engine has no canonical point for the TransformerLens hook {hook!r}, "
                f"which this transcoder set reads or writes at. {e} Use the transformerlens or "
                "nnsight backend for these transcoders."
            ) from e
        return [
            _Location(*self.engine.resolve_point(a.name, a.layer, stream=a.stream))
            for a in addresses
        ]

    # --- the local replacement model ------------------------------------------

    def _install_feature_hooks(self) -> None:
        """Cache what the transcoders read, and linearize what they write.

        The output splice is the *local replacement model*: replacing the sublayer's contribution
        with ``skip + (out - skip).detach()`` leaves the forward value untouched while routing all
        gradient through the transcoder's linear skip path. With no skip connection the term is a
        connected zero, which cuts the gradient but keeps the tensor in the graph -- the
        attribution reads its incoming gradient.
        """
        for layer, (read_at, write_at) in enumerate(zip(self._input_locs, self._output_locs)):
            if write_at.side != "output":
                raise NotImplementedError(
                    f"The transcoders write at {self.feature_output_hook!r}, which resolves to the "
                    f"*{write_at.side}* of a module; only module outputs can be spliced."
                )

            self._handles.append(self._read_hook(read_at, self._cache_input(layer)))
            self._handles.append(write_at.module.register_forward_hook(self._splice_output(layer)))

    def _read_hook(self, location: "_Location", fn: Callable) -> RemovableHandle:
        """Register a hook that hands ``fn`` the tensor ``location`` names."""
        module, side = location.module, location.side
        if side == "input":
            return module.register_forward_pre_hook(
                lambda _m, args, kwargs: fn(_input_tensor(args, kwargs)[0]), with_kwargs=True
            )
        if side == "output":
            return module.register_forward_hook(lambda _m, _a, output: fn(_hidden(output)))

        assert side == "normalized"
        # The unweighted normalized residual is not any module's output, so it cannot be read --
        # it has to be *introduced*. Substituting it as the norm's input does that: the norm is
        # scale-invariant, so running it on ``x / scale`` instead of ``x`` leaves the output alone
        # while putting the tensor the transcoder wants on the forward path, where a gradient
        # injected into it reaches the rest of the network. The scale is detached, which is also
        # exactly the norm-scale freeze this location would otherwise need separately.
        eps = _norm_eps(module)
        centered = _subtracts_mean(module)

        def pre_hook(_m, args, kwargs):
            x, key = _input_tensor(args, kwargs)
            xf = x.float()
            if centered:
                xf = xf - xf.mean(-1, keepdim=True)
            scale = torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + eps).detach()
            # This denominator gets its own tap, separate from the one `_freeze_scale` puts on the
            # same module, because it is the one that matters and the other one is not it: by the
            # time `_freeze_scale` runs, this pre-hook has already divided the input through, so
            # what it recomputes is a residual denominator of about 1. The real denominator is
            # here, and it is what TransformerLens's `hook_scale` names -- freezing it is what
            # makes the transcoder read the clean run's normalized residual.
            tap = getattr(module, _INPUT_SCALE_TAP, None)
            replayed = tap(scale) if tap is not None else None
            if replayed is not None:
                scale = replayed
            normalized = (x / scale.to(x.dtype)).to(x.dtype)
            fn(normalized)
            return _substitute_input(args, kwargs, key, normalized)

        return module.register_forward_pre_hook(pre_hook, with_kwargs=True)

    def _cache_input(self, layer: int) -> Callable:
        def cache(tensor: torch.Tensor):
            self._feature_input_tensors[layer] = tensor
            for listen in self._input_listeners[layer]:
                listen(tensor)

        return cache

    def _splice_output(self, layer: int) -> Callable:
        def splice(_module, _args, output):
            acts = _hidden(output)
            self._feature_output_raw[layer] = acts.detach()
            read = self._feature_input_tensors[layer]
            assert read is not None, f"layer {layer} feature input was not cached before its output"
            if self.skip_transcoder:
                skip = self.transcoders.compute_skip(layer, read)
            else:
                skip = read * 0
            spliced = skip + (acts - skip).detach()
            self._feature_output_tensors[layer] = spliced
            for transform in self._output_transforms[layer]:
                spliced = transform(spliced)
            return _replace_hidden(output, spliced)

        return splice

    def _install_norm_scale_freeze(self) -> None:
        """Hold every normalization denominator constant.

        For a norm ``f(x) = w * (x / s(x)) + b``, freezing the scale means computing
        ``w * (x / s(x).detach()) + b``. That is ``(f(x) - b) * (s / s.detach()) + b``, so the
        real module can run untouched and only the scale's gradient path needs dividing out. What
        makes this worth doing the algebraic way is what it *avoids* needing to know: the weight
        convention differs across families (Gemma's RMSNorm scales by ``1 + weight``, Llama's by
        ``weight``) and reimplementing the norm would have to get that right per family, whereas
        the scale formula only depends on whether the mean is subtracted.

        The forward value is unchanged; only gradients differ. Checked in :meth:`_verify_freezes`.
        """
        for norm in _norm_modules(self.hf_model):
            self._handles.append(norm.register_forward_hook(_freeze_scale(norm, self._audit)))

    def _install_embed_gradient(self) -> None:
        """Make the token embeddings a gradient source, and keep the tensor for attribution."""
        embed, _ = self.engine.resolve_point("embeddings")

        def hook(_module, _args, output):
            tensor = _hidden(output)
            # Every parameter is frozen, so this tensor is a graph leaf and can be turned into
            # one. It is where token-embedding attributions are read from.
            if not tensor.requires_grad:
                tensor.requires_grad_(True)
            self._embed_tensor = tensor
            return output

        self._handles.append(embed.register_forward_hook(hook))

    def _verify_freezes(self) -> None:
        """Check on real tensors that both freezes do what they claim.

        A short forward pass is cheap next to loading the model, and a freeze that silently does
        not hold produces an attribution graph that looks entirely reasonable.
        """
        attn_modules = [self.arch.attn_module(layer) for layer in range(self.n_layers)]
        for module in attn_modules:
            module._ct_freeze_audit = self._audit  # type: ignore[assignment]

        ids = torch.tensor([[self.tokenizer.bos_token_id or 0, 1, 2]], device=self.device)
        with self._audit.recording() as audit, torch.no_grad():
            self.hf_model(ids, use_cache=False)

        if not audit.attention:
            raise RuntimeError(
                "The frozen attention implementation was never reached, so attention patterns are "
                "not frozen. The model reports attn_implementation="
                f"{self.hf_model.config._attn_implementation!r}."
            )
        # Reconstructing `probs @ value` should be exact, not merely close: it is the same
        # contraction the family just did, on the same tensors. Anything above float noise means
        # the family does something after the softmax that this does not reproduce -- an output
        # gate, a renormalization -- and the pattern freeze cannot be trusted.
        worst = max(audit.attention)
        tolerance = 1e-2 if self.dtype in (torch.float16, torch.bfloat16) else 1e-5
        if worst > tolerance:
            raise NotImplementedError(
                f"Recomputing `probs @ value` for {self.arch.architecture} differs from the "
                f"model's own attention output by {worst:.3e}, so its attention does something "
                "after the softmax that the pattern freeze does not reproduce. Use the "
                "transformerlens or nnsight backend for this architecture."
            )

        if not audit.norms:
            raise RuntimeError("No normalization module was hooked, so norm scales are not frozen.")
        # The scale freeze must be value-neutral by construction, so this is exact rather than
        # tolerant: a nonzero ratio here means the norm was misread and the *forward pass* has
        # been perturbed, not just the gradient.
        if max(audit.norms) != 0.0:
            raise RuntimeError(
                f"The norm scale freeze changed a forward value (ratio off 1 by "
                f"{max(audit.norms):.3e}), which means a norm module was misread."
            )

    # --- accessors the attribution code needs ---------------------------------

    @property
    def embed_weight(self) -> torch.Tensor:
        return _weight_of(self.engine.resolve_point("embeddings")[0])

    @property
    def unembed_weight(self) -> torch.Tensor:
        return _weight_of(self.arch.lm_head)

    @property
    def blocks(self) -> list[nn.Module]:
        return self.arch.decoder_layers

    @property
    def mlp_modules(self) -> list[nn.Module]:
        return [self.arch.mlp_module(layer) for layer in range(self.n_layers)]

    def get_feature_output_tensors(self) -> list[torch.Tensor]:
        tensors = self._feature_output_tensors
        assert all(t is not None for t in tensors), "no forward pass has run"
        return tensors  # type: ignore[return-value]

    def get_feature_input_tensors(self) -> list[torch.Tensor]:
        tensors = self._feature_input_tensors
        assert all(t is not None for t in tensors), "no forward pass has run"
        return tensors  # type: ignore[return-value]

    @property
    def embed_tensor(self) -> torch.Tensor:
        assert self._embed_tensor is not None, "no forward pass has run"
        return self._embed_tensor

    def ensure_tokenized(self, prompt: str | torch.Tensor | list[int]) -> torch.Tensor:
        return ensure_tokenized(prompt, self.tokenizer, self.device, str(self.cfg.model_name))

    @contextmanager
    def zero_softcap(self) -> Iterator[None]:
        """Disable Gemma-style final-logit softcapping.

        Present for parity with the other backends, which need it when reading raw logits.
        """
        config = self.hf_model.config
        original = getattr(config, "final_logit_softcapping", None)
        try:
            if original is not None:
                config.final_logit_softcapping = None
            yield
        finally:
            if original is not None:
                config.final_logit_softcapping = original

    # --- forward paths --------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, **kwargs):
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        return self.hf_model(input_ids, use_cache=False, **kwargs).logits

    def forward_trunk(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run everything except the unembedding, returning the normalized final residual.

        Attribution runs a wide batch (one row per source node) and never needs logits from it,
        so materializing ``[batch, pos, vocab]`` would dominate memory for no purpose. The
        trunk's ``last_hidden_state`` is already past the final norm, which is what the logit
        attributions are taken against.
        """
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        return self.arch.trunk(input_ids, use_cache=False).last_hidden_state

    def get_activations(
        self,
        inputs: str | torch.Tensor,
        sparse: bool = False,
        apply_activation_function: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the logits and the transcoder activations for ``inputs``."""
        tokens = self.ensure_tokenized(inputs)
        with torch.no_grad():
            logits = self.forward(tokens)

        acts = []
        for layer, read in enumerate(self.get_feature_input_tensors()):
            encoded = (
                self.transcoders.encode_layer(
                    read, layer, apply_activation_function=apply_activation_function
                )
                .detach()
                .squeeze(0)
            )
            encoded[self.zero_positions] = 0
            acts.append(encoded.to_sparse() if sparse else encoded)

        activation_cache = torch.stack(acts)
        if sparse:
            activation_cache = activation_cache.coalesce()
        return logits, activation_cache

    @torch.no_grad()
    def setup_attribution(self, inputs: str | torch.Tensor) -> AttributionContext:
        """Precompute the transcoder activations, error vectors and token embeddings."""
        tokens = self.ensure_tokenized(inputs) if isinstance(inputs, str) else inputs.squeeze()
        assert isinstance(tokens, torch.Tensor), "Tokens must be a tensor"
        assert tokens.ndim == 1, "Tokens must be a 1D tensor"

        logits = self.forward(tokens)
        feature_in = torch.cat([t for t in self.get_feature_input_tensors()], dim=0)
        feature_out = torch.cat([t for t in self._feature_output_raw], dim=0)  # type: ignore[misc]

        attribution_data = self.transcoders.compute_attribution_components(
            feature_in, self.zero_positions
        )

        error_vectors = feature_out - attribution_data["reconstruction"]
        error_vectors[:, self.zero_positions] = 0
        # Raw embedding rows, matching the NNSight backend. On Gemma these are sqrt(d_model) times
        # smaller than TransformerLens reports, because TL folds that normalizer into `W_E` while HF
        # applies it as a separate multiply after `embed_tokens`. The gradient is read on the
        # pre-multiply tensor, so the factor cancels and token attributions still agree with TL to
        # float precision; only this public vector differs, and it differs the same way NNSight's
        # already does.
        token_vectors = self.embed_weight[tokens].detach()

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

    # --- interventions --------------------------------------------------------

    @contextmanager
    def hooks(self, installers: Sequence[InterventionHook]) -> Iterator[None]:
        """Install ``installers`` for the duration of the block, in the order given.

        The interp_engine counterpart of ``HookedTransformer.hooks``. Order is part of the
        argument, not an accident of it: several of these compose at the same point in the forward
        pass, and a freeze that runs after the thing it is meant to constrain does nothing.
        """
        handles = [install() for install in installers]
        try:
            yield
        finally:
            for handle in reversed(handles):
                handle.remove()

    def _freezes(
        self, constrained_layers: range | None, active: Callable[[], bool] | None = None
    ) -> list[_Freeze]:
        """The tensors to hold at their clean-run values, for the kind of intervention requested.

        Mirrors what the TransformerLens backend freezes, hook point for hook point: always the
        attention patterns; inside a constrained layer range, also the sublayer output the
        transcoders write at; and, only when that range is every layer -- which is what makes the
        measured effect a *direct* effect rather than a partly propagated one -- every
        normalization denominator.

        The denominators are found the same way the gradient freeze finds them, by collecting the
        norm modules, so a family with norms TransformerLens has no hook point for (Gemma-3's
        per-head ``q_norm``) gets those frozen too.
        """
        freezes = [
            _Freeze.tap(
                self.arch.attn_module(layer),
                _PATTERN_TAP,
                _Replay(f"layer {layer} attention pattern", active),
            )
            for layer in range(self.n_layers)
        ]
        if not constrained_layers:
            return freezes

        freezes += [
            _Freeze.listener(
                self._output_transforms[layer],
                _Replay(f"layer {layer} {self.feature_output_hook}", active),
            )
            for layer in constrained_layers
        ]

        if not set(range(self.n_layers)).issubset(set(constrained_layers)):
            return freezes

        for index, norm in enumerate(_norm_modules(self.hf_model)):
            freezes.append(
                _Freeze.tap(norm, _SCALE_TAP, _Replay(f"norm {index} denominator", active))
            )
        for layer, read_at in enumerate(self._input_locs):
            if read_at.side == "normalized":
                freezes.append(
                    _Freeze.tap(
                        read_at.module,
                        _INPUT_SCALE_TAP,
                        _Replay(f"layer {layer} {self.feature_input_hook} denominator", active),
                    )
                )
        return freezes

    def _skip_diff_hooks(
        self,
        constrained_layers: range,
        clean_inputs: list[torch.Tensor],
        active: Callable[[], bool] | None,
    ) -> list[InterventionHook]:
        """Restore the direct part of a skip connection that the output freeze just removed.

        Freezing the sublayer output pins the transcoder's skip term along with everything else,
        but the skip is linear in the sublayer's input and so is exactly the part of the sublayer
        that *should* follow the intervention. These hooks subtract the skip the clean run had and
        add the one this run earns.
        """
        skip_diffs: dict[int, torch.Tensor] = {}

        def measure(layer: int) -> Callable[[torch.Tensor], None]:
            def listen(tensor: torch.Tensor) -> None:
                if active is not None and not active():
                    skip_diffs.pop(layer, None)
                    return
                frozen = self.transcoders.compute_skip(layer, clean_inputs[layer])
                skip_diffs[layer] = self.transcoders.compute_skip(layer, tensor) - frozen

            return listen

        def apply(layer: int) -> Callable[[torch.Tensor], torch.Tensor]:
            def transform(tensor: torch.Tensor) -> torch.Tensor:
                diff = skip_diffs.get(layer)
                return tensor if diff is None else tensor + diff

            return transform

        hooks = [
            _listener_hook(self._input_listeners[layer], measure(layer))
            for layer in constrained_layers
        ]
        hooks += [
            _listener_hook(self._output_transforms[layer], apply(layer))
            for layer in constrained_layers
        ]
        return hooks

    @torch.no_grad()
    def setup_intervention_with_freeze(
        self,
        inputs: str | torch.Tensor,
        constrained_layers: range | None = None,
    ) -> tuple[torch.Tensor, list[InterventionHook]]:
        """Record a clean run, and return hooks that force its values on a later one.

        Args:
            inputs: The prompt to intervene on.
            constrained_layers: Freeze not just the attention patterns but also the sublayer
                outputs in this layer range, and -- if it covers every layer -- the normalization
                denominators, which together make the intervention's measured effect a direct one.
                ``None`` freezes only attention (iterative patching).

        Returns:
            The clean run's transcoder activations, and the hooks to pass to :meth:`hooks`.
        """
        return self._setup_intervention_with_freeze(inputs, constrained_layers, None)

    def _setup_intervention_with_freeze(
        self,
        inputs: str | torch.Tensor,
        constrained_layers: range | None,
        active: Callable[[], bool] | None,
    ) -> tuple[torch.Tensor, list[InterventionHook]]:
        freezes = self._freezes(constrained_layers, active)
        with self.hooks([freeze.record for freeze in freezes]):
            _, original_activations = self.get_activations(inputs)

        hooks = [freeze.replay for freeze in freezes]
        if constrained_layers and self.skip_transcoder:
            # The clean run left its own normalized residuals on the model, which is what the skip
            # diffs need, so no second recording pass is required.
            clean_inputs = [tensor for tensor in self.get_feature_input_tensors()]
            hooks += self._skip_diff_hooks(constrained_layers, clean_inputs, active)
        return original_activations, hooks

    def _intervention_hooks(
        self,
        inputs: str | torch.Tensor,
        interventions: Sequence[Intervention],
        constrained_layers: range | None,
        freeze_attention: bool,
        apply_activation_function: bool,
        sparse: bool,
        return_activations: bool,
        pass_index: list[int],
    ) -> tuple[list[InterventionHook], list[list[torch.Tensor]]]:
        """Build everything one intervention run installs, and the list its activations land in.

        ``pass_index`` is a one-element list holding which forward pass of a generation is running;
        the hooks read it so that one set of them can serve both the pass that consumes the prompt
        and the single-token passes after it. TransformerLens instead builds two sets and swaps
        them mid-generation; reading a counter avoids mutating hook registrations from inside a
        hook, and makes "the freeze applies only to the first token" one condition in one place.
        """
        first_pass = lambda: pass_index[0] == 0  # noqa: E731

        by_layer: dict[int, list] = defaultdict(list)
        for layer, pos, feature_idx, value in interventions:
            by_layer[_as_int(layer)].append((pos, feature_idx, value))

        open_ended_by_layer: dict[int, list] = defaultdict(list)
        for layer, pos, feature_idx, value in convert_open_ended_interventions(interventions):
            open_ended_by_layer[_as_int(layer)].append((pos, feature_idx, value))

        hooks: list[InterventionHook] = []
        original_activations: torch.Tensor | None = None
        if interventions and (freeze_attention or constrained_layers):
            original_activations, freeze_hooks = self._setup_intervention_with_freeze(
                inputs, constrained_layers, first_pass
            )
            hooks += freeze_hooks

        deltas = _Deltas(self.n_layers, self.cfg.d_model, self.dtype, self.device)
        hooks.append(
            lambda: self.hf_model.register_forward_pre_hook(deltas.begin, with_kwargs=True)
        )

        activation_rows: list[list[torch.Tensor]] = [[] for _ in range(self.n_layers)]

        def cache_activations(layer: int) -> Callable[[torch.Tensor], None]:
            def listen(tensor: torch.Tensor) -> None:
                acts = (
                    self.transcoders.encode_layer(
                        tensor, layer, apply_activation_function=apply_activation_function
                    )
                    .detach()
                    .squeeze(0)
                )
                if first_pass():
                    acts[self.zero_positions] = 0
                activation_rows[layer].append(acts.to_sparse() if sparse else acts)

            return listen

        if return_activations:
            hooks += [
                _listener_hook(self._input_listeners[layer], cache_activations(layer))
                for layer in range(self.n_layers)
            ]

        def measure_deltas(layer: int) -> Callable[[torch.Tensor], None]:
            def listen(tensor: torch.Tensor) -> None:
                layer_interventions = (by_layer if first_pass() else open_ended_by_layer)[layer]
                if not layer_interventions:
                    return
                if constrained_layers and first_pass():
                    # Base the deltas on the clean run, so that an intervention's effect on an
                    # earlier layer's features does not feed the delta at this one.
                    assert original_activations is not None
                    baseline = original_activations[layer]
                else:
                    baseline = (
                        self.transcoders.encode_layer(tensor, layer, apply_activation_function=True)
                        .detach()
                        .squeeze(0)
                    )
                deltas.add(layer, baseline, layer_interventions, self.transcoders)

            return listen

        hooks += [
            _listener_hook(self._input_listeners[layer], measure_deltas(layer))
            for layer in set(by_layer) | set(open_ended_by_layer)
        ]

        def write_deltas(layer: int) -> Callable[[torch.Tensor], torch.Tensor]:
            def transform(tensor: torch.Tensor) -> torch.Tensor:
                # Every layer gets this, including layers no intervention writes to, because the
                # per-pass buffer has to be cleared either way: generation runs these hooks again
                # for the next token.
                in_range = not (constrained_layers and first_pass()) or layer in constrained_layers
                return deltas.take(layer, tensor, add=in_range)

            return transform

        hooks += [
            _listener_hook(self._output_transforms[layer], write_deltas(layer))
            for layer in range(self.n_layers)
        ]

        def count_pass(_module, _args, _output) -> None:
            pass_index[0] += 1

        hooks.append(lambda: self.hf_model.register_forward_hook(count_pass))
        return hooks, activation_rows

    @torch.no_grad()
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
        """Set features to given values and run the model, letting the effects propagate.

        Args:
            inputs: The prompt to intervene on.
            interventions: ``(layer, position, feature index, value)`` tuples.
            constrained_layers: Freeze the sublayer outputs in this range, and the normalization
                denominators if it covers every layer, so that what the logits show is the
                intervention's direct effect. ``None`` lets effects propagate through the
                transcoders and norms (iterative patching).
            freeze_attention: Freeze the attention patterns. Implied by ``constrained_layers``.
            apply_activation_function: Whether the *returned* activations have the transcoder's
                activation function applied. False is useful for testing, since attribution
                predicts the change in pre-activation values.
            sparse: Return the activations as a sparse tensor: less memory, slower.
            return_activations: Skip computing the activations entirely, and return ``None``.

        Returns:
            The logits ``(batch, position, vocabulary)`` and the feature activations
            ``(layer, position, feature)``, or ``None`` if they were not requested.
        """
        tokens = self.ensure_tokenized(inputs)
        hooks, activation_rows = self._intervention_hooks(
            tokens,
            interventions,
            constrained_layers,
            freeze_attention,
            apply_activation_function,
            sparse,
            return_activations,
            pass_index=[0],
        )
        with self.hooks(hooks):
            logits = self.forward(tokens)

        if not return_activations:
            return logits, None
        return logits, _stack_activations(activation_rows, sparse)

    @torch.no_grad()
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
        """Intervene, then generate a continuation, reporting what happened at each step.

        Accepts the generation keywords of ``transformers``' ``generate`` (``do_sample``,
        ``temperature``, ``max_new_tokens``, ...). ``max_new_tokens`` defaults to 10, which is
        ``HookedTransformer.generate``'s default rather than ``generate``'s, so that a notebook
        that omits it gets the same length from every backend.

        The freezes apply only to the pass that consumes the prompt, as they do on the other
        backends: after that there is a single position per pass and nothing to hold constant.
        Interventions whose position is an open-ended slice carry on into the generated tokens;
        interventions at a concrete position do not, having already been applied.

        Returns:
            The generated text including the prompt, the logits that produced each generated token
            ``(generated position, vocabulary)``, and the feature activations across the prompt and
            the continuation, or ``None`` if they were not requested.
        """
        # Accepted and ignored for signature compatibility with the TransformerLens backend, whose
        # `generate` has them and `transformers`' does not.
        kwargs.pop("verbose", None)
        assert kwargs.pop("use_past_kv_cache", True), (
            "Generation is only possible with use_past_kv_cache=True"
        )
        kwargs.setdefault("max_new_tokens", 10)

        tokens = self.ensure_tokenized(inputs)
        hooks, activation_rows = self._intervention_hooks(
            tokens,
            interventions,
            constrained_layers,
            freeze_attention,
            apply_activation_function,
            sparse,
            return_activations,
            pass_index=[0],
        )

        input_ids = tokens.unsqueeze(0)
        with self.hooks(hooks):
            sequences, step_logits = _generate(
                self.hf_model,
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                use_cache=True,
                **kwargs,
            )

        # Without position 0, which `ensure_tokenized` prepended rather than the caller writing it.
        # The TransformerLens backend drops it the same way, and it is the difference between a
        # notebook printing the prompt back and printing `<bos>` in front of it.
        generation = self.tokenizer.decode(sequences[0][1:])
        # One entry per generated token, each the logits that chose it -- so the first comes from
        # the prompt's last position and the rest from the incremental passes. Softcapping, where
        # the family has it, is already applied: these are what the model returned.
        logits = torch.cat(step_logits, dim=0)

        if not return_activations:
            return generation, logits, None
        return generation, logits, _stack_activations(activation_rows, sparse)

    def __del__(self):
        for handle in getattr(self, "_handles", []):
            handle.remove()


def _hidden(output: object) -> torch.Tensor:
    """The tensor from a module output that may be a bare tensor or a tuple."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"Cannot read a hidden-state tensor out of {type(output).__name__}")


def _replace_hidden(output: object, new: torch.Tensor) -> object:
    if isinstance(output, torch.Tensor):
        return new
    if isinstance(output, tuple):
        return (new,) + tuple(output[1:])
    if isinstance(output, list):
        return [new] + list(output[1:])
    raise TypeError(f"Cannot substitute a hidden-state tensor into {type(output).__name__}")


def _freeze_scale(norm: nn.Module, audit: "_FreezeAudit") -> Callable:
    """Forward hook holding a norm's denominator constant, in either sense of constant.

    For a norm ``f(x) = w * (x / s(x)) + b``, replacing the denominator with some ``t`` gives
    ``w * (x / t) + b``, which is ``(f(x) - b) * (s / t) + b``. So the real module runs untouched
    and only the ratio ``s / t`` has to be applied afterwards -- and the same expression covers
    both freezes this backend needs, differing only in what ``t`` is:

    * Attribution wants ``t = s.detach()``, which changes no value and cuts the gradient.
    * An intervention wants ``t`` = the denominator the clean run had, which is the point: the
      value changes, and the change is what makes the effect a direct one.

    Written as ``y + (y - b) * (ratio - 1)`` rather than ``(y - b) * ratio + b`` because in the
    attribution case ``ratio`` is exactly ``1.0`` in floating point, so this form leaves the
    forward value *bit*-identical. The obvious form would round ``(y - b) + b`` and quietly perturb
    every norm in the model.

    ``eps`` matters only to the gradient, never to the value, which is why it is read off the
    module rather than dismissed as negligible. The bias has to come out before scaling because
    only the ``w * (x / s)`` term carries the scale.
    """
    eps = _norm_eps(norm)
    centered = _subtracts_mean(norm)
    bias = getattr(norm, "bias", None)

    def hook(_module, args, output):
        x = args[0].float()
        if centered:
            x = x - x.mean(-1, keepdim=True)
        scale = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        tensor = _hidden(output)

        target = scale.detach()
        tap = getattr(norm, _SCALE_TAP, None)
        replayed = tap(target) if tap is not None else None
        if replayed is not None:
            target = replayed

        ratio = (scale / target).to(tensor.dtype)
        scalable = tensor if bias is None else tensor - bias
        if replayed is None:
            audit.record_norm(ratio)
        return _replace_hidden(output, tensor + scalable * (ratio - 1))

    return hook


class _FreezeAudit:
    """Collects, during one forward pass, evidence that the freezes hold.

    Both freezes are claims about architectures in general, so they are measured on real tensors
    instead of asserted. Off by default: this runs once at load, not on every forward pass.
    """

    def __init__(self) -> None:
        self.active = False
        self.attention: list[float] = []
        self.norms: list[float] = []

    def record_attention(self, own: torch.Tensor, frozen: torch.Tensor) -> None:
        if self.active:
            self.attention.append((own - frozen).abs().max().item())

    def record_norm(self, ratio: torch.Tensor) -> None:
        if self.active:
            self.norms.append((ratio - 1).abs().max().item())

    @contextmanager
    def recording(self) -> Iterator["_FreezeAudit"]:
        self.active = True
        self.attention.clear()
        self.norms.clear()
        try:
            yield self
        finally:
            self.active = False
