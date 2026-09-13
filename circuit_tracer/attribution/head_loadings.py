"""Splitting an attribution-graph edge across the attention heads that carried it.

An attribution graph says that a source node influenced a target feature. It does not say which
attention head moved the signal, because the edge is a single number summed over every path. For a
reader trying to understand a circuit that is often the missing half: the graph shows that a name
token reached a later feature, and says nothing about the head that fetched it.

Under the constraints a graph is built with, the answer is exact rather than approximate. Attention
patterns and normalisation scales are frozen, and transcoder activations are frozen, so an MLP's
output does not move when its input does. Every linear path from a perturbation of the residual
stream to a later feature's pre-activation therefore runs through attention. At any attention layer
the incoming residual splits into one part per head plus a part that bypasses attention, those
parts propagate forward independently, and the target reads their sum.

The split is per attention layer. Paths compose, so a signal may pass through heads at several
layers on the way and there is no single head to credit for the whole journey. Asking which heads
at one layer carried an edge is a well-posed question with an exact answer; asking for one joint
attribution across all layers is not, and this module does not offer one.

Example:

    from circuit_tracer.attribution.head_loadings import FrozenRun, head_loadings

    run = FrozenRun.from_model(model, graph.input_tokens)
    loadings = head_loadings(model, graph, run, target_node=412, source_node=57, attention_layer=14)
    heads, values = loadings.ranked()
    print(f"head {heads[0].item()} carried {values[0].item():.3f} of {loadings.total.item():.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor

from circuit_tracer.graph import Graph
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder


#: Hook points around a block's ``ln2`` that a transcoder may read, which do not hold the same thing:
#: ``hook_mlp_in`` is the residual before the norm, ``ln2.hook_normalized`` is divided by the frozen
#: scale but not yet multiplied by the gain, and ``mlp.hook_in`` is the MLP's input after both. A
#: transcoder reading anywhere else is refused rather than silently misread.
SUPPORTED_FEATURE_INPUT_HOOKS = frozenset({"mlp.hook_in", "hook_mlp_in", "ln2.hook_normalized"})


class UnsupportedForLoadings(RuntimeError):
    """Raised when a model or graph breaks the linearity the split relies on."""


class NodeKind(Enum):
    """Which region of the adjacency ordering a node index falls in."""

    FEATURE = "feature"
    ERROR = "error"
    TOKEN = "token"
    LOGIT = "logit"


@dataclass(frozen=True)
class NodeLayout:
    """Index boundaries of the four node regions of an adjacency matrix.

    The adjacency ordering is ``[features] [MLP errors] [token embeddings] [logit targets]``, with
    error nodes laid out layer-major and position-minor.
    """

    n_features: int
    n_layers: int
    n_pos: int
    n_logits: int

    @classmethod
    def from_graph(cls, graph: Graph) -> NodeLayout:
        """Derive the layout from a graph, checked against its adjacency matrix."""
        n_features = len(graph.selected_features)
        n_pos = len(graph.input_tokens)
        n_logits = len(graph.logit_targets)
        total = int(graph.adjacency_matrix.shape[0])
        if n_pos == 0:
            raise ValueError("n_pos must be positive; an empty prompt has no graph")
        remainder = total - n_features - n_pos - n_logits
        if remainder < 0 or remainder % n_pos != 0:
            raise ValueError(
                "adjacency size is inconsistent with the graph: "
                f"total={total}, n_features={n_features}, n_pos={n_pos}, n_logits={n_logits} "
                f"leaves {remainder} error nodes, which is not a multiple of n_pos"
            )
        return cls(n_features, remainder // n_pos, n_pos, n_logits)

    @property
    def error_end(self) -> int:
        return self.n_features + self.n_layers * self.n_pos

    @property
    def token_end(self) -> int:
        return self.error_end + self.n_pos

    @property
    def total(self) -> int:
        return self.token_end + self.n_logits

    def kind(self, index: int) -> NodeKind:
        """Return which region ``index`` falls in."""
        if not 0 <= index < self.total:
            raise IndexError(f"node index {index} out of range for {self.total} nodes")
        if index < self.n_features:
            return NodeKind.FEATURE
        if index < self.error_end:
            return NodeKind.ERROR
        if index < self.token_end:
            return NodeKind.TOKEN
        return NodeKind.LOGIT

    def decode_error(self, index: int) -> tuple[int, int]:
        """Return ``(layer, position)`` for an error node."""
        if self.kind(index) is not NodeKind.ERROR:
            raise ValueError(f"node {index} is not an error node")
        return divmod(index - self.n_features, self.n_pos)

    def decode_token(self, index: int) -> int:
        """Return the sequence position of a token-embedding node."""
        if self.kind(index) is not NodeKind.TOKEN:
            raise ValueError(f"node {index} is not a token node")
        return index - self.error_end


@dataclass(frozen=True)
class HeadLoadings:
    """How one edge's effect divides across the heads of a single attention layer."""

    per_head: Tensor
    bypass: Tensor
    attention_layer: int

    @property
    def total(self) -> Tensor:
        """The whole edge effect, which the parts sum to."""
        return self.per_head.sum() + self.bypass

    def ranked(self) -> tuple[Tensor, Tensor]:
        """Head indices ordered by contribution magnitude, with their signed values."""
        order = self.per_head.abs().argsort(descending=True)
        return order, self.per_head[order]


class FrozenRun:
    """The frozen attention patterns and normalisation scales of one clean forward pass.

    Computing loadings for many edges of the same graph reuses these, so they are gathered once.
    """

    def __init__(
        self,
        patterns: list[Tensor],
        ln1_scales: list[Tensor],
        ln2_scales: list[Tensor],
        ln1_post_scales: list[Tensor] | None = None,
    ):
        self.patterns = patterns
        self.ln1_scales = ln1_scales
        self.ln2_scales = ln2_scales
        # Only models that normalise attention's output before the residual add, such as Gemma-2.
        self.ln1_post_scales = ln1_post_scales

    @property
    def n_pos(self) -> int:
        return int(self.patterns[0].shape[-1])

    @classmethod
    def from_model(cls, model, tokens: Tensor | str) -> FrozenRun:
        """Run the model once and keep what the split holds fixed.

        Args:
            model: A ``ReplacementModel`` on the TransformerLens backend.
            tokens: The same prompt the graph was attributed on.
        """
        wanted = ("attn.hook_pattern", "ln1.hook_scale", "ln2.hook_scale", "ln1_post.hook_scale")
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda name: any(name.endswith(tail) for tail in wanted)
        )

        def squeeze(value: Tensor) -> Tensor:
            if value.ndim not in (3, 4):
                return value
            if value.shape[0] != 1:
                raise ValueError(
                    f"expected one prompt but the cache holds a batch of {value.shape[0]}; "
                    "a graph describes a single prompt"
                )
            return value[0]

        n_layers = model.cfg.n_layers
        has_post = "blocks.0.ln1_post.hook_scale" in cache
        return cls(
            patterns=[squeeze(cache[f"blocks.{i}.attn.hook_pattern"]) for i in range(n_layers)],
            ln1_scales=[squeeze(cache[f"blocks.{i}.ln1.hook_scale"]) for i in range(n_layers)],
            ln2_scales=[squeeze(cache[f"blocks.{i}.ln2.hook_scale"]) for i in range(n_layers)],
            ln1_post_scales=[
                squeeze(cache[f"blocks.{i}.ln1_post.hook_scale"]) for i in range(n_layers)
            ]
            if has_post
            else None,
        )


def _require_supported(model) -> None:
    """Refuse models whose MLPs still pass a perturbation along, or whose norms have a bias."""
    transcoders = getattr(model, "transcoders", None)
    if isinstance(transcoders, CrossLayerTranscoder):
        raise UnsupportedForLoadings(
            "a cross-layer transcoder feature writes into every later layer, so a source node has "
            "no single point of entry; only per-layer transcoders are handled here"
        )
    if getattr(transcoders, "skip_connection", False):
        raise UnsupportedForLoadings(
            "these transcoders have a skip connection, so an MLP's output moves when its input "
            "does even with activations frozen; the split assumes it does not"
        )
    input_hook = getattr(transcoders, "feature_input_hook", "mlp.hook_in")
    if input_hook not in SUPPORTED_FEATURE_INPUT_HOOKS:
        raise UnsupportedForLoadings(
            f"features read {input_hook!r}; the readout here models only the hooks around ln2, "
            f"one of {sorted(SUPPORTED_FEATURE_INPUT_HOOKS)}"
        )
    for layer in range(model.cfg.n_layers):
        for name in ("ln1", "ln2", "ln1_post"):
            norm = getattr(model.blocks[layer], name, None)
            if getattr(norm, "b", None) is not None:
                raise UnsupportedForLoadings(
                    f"blocks.{layer}.{name} has a bias, which does not act on a perturbation; "
                    "RMSNorm is assumed here"
                )


def _normalised(model, layer: int, delta: Tensor, scale: Tensor, which: str) -> Tensor:
    """Apply a layer norm to a perturbation, with its scale frozen."""
    norm = getattr(model.blocks[layer], which)
    return delta * norm.w.to(delta.dtype) / scale.to(delta.dtype)


def _attention_step(
    model, layer: int, delta: Tensor, run: FrozenRun, *, per_head: bool = False
) -> Tensor:
    """Return what attention at ``layer`` writes back, given a perturbation of its input.

    Returns ``(n_pos, d_model)``, or ``(n_heads, n_pos, d_model)`` when ``per_head``.
    """
    attn = model.blocks[layer].attn
    normalised = _normalised(model, layer, delta, run.ln1_scales[layer], "ln1")
    # W_V and W_O are materialised per query head, so grouped-query attention needs no mapping.
    values = torch.einsum("pd,hde->hpe", normalised, attn.W_V.to(normalised.dtype))
    moved = torch.einsum("hqp,hpe->hqe", run.patterns[layer].to(values.dtype), values)
    written = torch.einsum("hqe,hed->hqd", moved, attn.W_O.to(moved.dtype))
    post = getattr(model.blocks[layer], "ln1_post", None)
    if post is not None:
        # The norm on attention's output is linear once its scale is frozen, so applying it to
        # each head's write keeps the per-head split exact.
        if run.ln1_post_scales is None:
            raise UnsupportedForLoadings(
                f"blocks.{layer} normalises attention's output (ln1_post) but the FrozenRun holds "
                "no ln1_post scales; build it with FrozenRun.from_model on this model"
            )
        written = written * post.w.to(written.dtype) / run.ln1_post_scales[layer].to(written.dtype)
    return written if per_head else written.sum(dim=0)


def _propagate(model, delta: Tensor, run: FrozenRun, from_layer: int, to_layer: int) -> Tensor:
    """Carry a perturbation from the input of ``from_layer`` to the input of ``to_layer``."""
    state = delta
    for layer in range(from_layer, to_layer):
        state = state + _attention_step(model, layer, state, run)
    return state


def _as_feature_input(model, layer: int, mid: Tensor, run: FrozenRun) -> Tensor:
    """Turn a perturbation of the residual after attention into what the transcoder's hook holds."""
    input_hook = getattr(model.transcoders, "feature_input_hook", "mlp.hook_in")
    if input_hook == "hook_mlp_in":
        return mid
    scaled = mid / run.ln2_scales[layer].to(mid.dtype)
    if input_hook == "ln2.hook_normalized":
        return scaled
    return scaled * model.blocks[layer].ln2.w.to(mid.dtype)


def _to_feature_input(model, delta: Tensor, run: FrozenRun, layer: int) -> Tensor:
    """Carry a perturbation of a block's input to what that block's transcoder reads."""
    mid = delta + _attention_step(model, layer, delta, run)
    return _as_feature_input(model, layer, mid, run)


def _seed(source: Tensor, position: int, n_pos: int) -> Tensor:
    """Place a source direction at one position of an otherwise empty residual stream."""
    if source.ndim != 1:
        raise ValueError(f"source must be 1D (d_model,), got {tuple(source.shape)}")
    if not 0 <= position < n_pos:
        raise IndexError(f"position {position} out of range for {n_pos} positions")
    delta = torch.zeros(n_pos, source.shape[0], dtype=source.dtype, device=source.device)
    delta[position] = source
    return delta


def _feature_at(graph: Graph, node: int) -> tuple[int, int, int, float]:
    """Return ``(layer, position, feature index, activation)`` for a feature node."""
    selected = int(graph.selected_features[node])
    layer, position, feature = (int(x) for x in graph.active_features[selected])
    # activation_values is aligned with active_features, not with selected_features. On graphs
    # where selection prunes nothing the two coincide, which hides the difference.
    return layer, position, feature, float(graph.activation_values[selected])


def source_vector(model, graph: Graph, node: int) -> tuple[Tensor, int, int]:
    """Return a source node's residual-stream write, with where and when it is written.

    Returns:
        ``(vector, position, layer)``, where ``layer`` is the block whose output carries the
        vector. A token embedding is written before block 0 and so reports ``layer = -1``.
    """
    layout = NodeLayout.from_graph(graph)
    kind = layout.kind(node)
    if kind is NodeKind.FEATURE:
        layer, position, feature, activation = _feature_at(graph, node)
        vector = model.transcoders[layer].W_dec[feature] * activation
        return vector, position, layer
    if kind is NodeKind.ERROR:
        raise UnsupportedForLoadings(
            "error nodes need the error vectors from the attribution run, which a graph does not "
            "carry; pass a feature or token node as the source"
        )
    if kind is NodeKind.TOKEN:
        position = layout.decode_token(node)
        return model.W_E[int(graph.input_tokens[position])], position, -1
    raise ValueError("a logit node is a target, not a source")


def reader_vector(model, graph: Graph, node: int) -> tuple[Tensor, int, int]:
    """Return a target feature's encoder row, with its position and layer."""
    layout = NodeLayout.from_graph(graph)
    if layout.kind(node) is not NodeKind.FEATURE:
        raise ValueError(f"target must be a feature node, got a {layout.kind(node).value} node")
    layer, position, feature, _ = _feature_at(graph, node)
    return model.transcoders[layer].W_enc[feature], position, layer


def edge_effect(model, graph: Graph, run: FrozenRun, target_node: int, source_node: int) -> Tensor:
    """Return the linear effect of one source node on one target feature's pre-activation.

    This is the quantity :func:`head_loadings` partitions, and it is what the graph's own
    adjacency entry for that edge measures.
    """
    _require_supported(model)
    source, source_position, source_layer = source_vector(model, graph, source_node)
    reader, target_position, target_layer = reader_vector(model, graph, target_node)
    delta = _seed(source, source_position, run.n_pos)
    arriving = _propagate(model, delta, run, source_layer + 1, target_layer)
    return _to_feature_input(model, arriving, run, target_layer)[target_position] @ reader


def head_loadings(
    model,
    graph: Graph,
    run: FrozenRun,
    target_node: int,
    source_node: int,
    attention_layer: int,
) -> HeadLoadings:
    """Split one edge's effect across the heads of a single attention layer.

    Args:
        model: A ``ReplacementModel`` on the TransformerLens backend.
        graph: The attribution graph the node indices refer to.
        run: A :class:`FrozenRun` over the same prompt the graph was attributed on.
        target_node: Adjacency index of the target, which must be a feature node.
        source_node: Adjacency index of the source, which may be a feature or token node.
        attention_layer: The layer whose heads to split over. Must lie after the source's layer
            and at or before the target's, since the target reads after its own block's attention.

    Returns:
        A :class:`HeadLoadings` whose parts sum to :func:`edge_effect` for the same edge.
    """
    _require_supported(model)
    source, source_position, source_layer = source_vector(model, graph, source_node)
    reader, target_position, target_layer = reader_vector(model, graph, target_node)
    if not source_layer < attention_layer <= target_layer:
        raise ValueError(
            f"attention_layer must lie in ({source_layer}, {target_layer}], got {attention_layer}"
        )

    delta = _seed(source, source_position, run.n_pos)
    if attention_layer == target_layer:
        # The target reads after its own block's attention, so the split happens at the readout:
        # the parts are each head's write plus the residual that reached the block untouched.
        arriving = _propagate(model, delta, run, source_layer + 1, target_layer)
        written = _attention_step(model, target_layer, arriving, run, per_head=True)
        parts = torch.cat([written, arriving.unsqueeze(0)], dim=0)
    else:
        arriving = _propagate(model, delta, run, source_layer + 1, attention_layer)
        written = _attention_step(model, attention_layer, arriving, run, per_head=True)
        split = torch.cat([written, arriving.unsqueeze(0)], dim=0)
        # Each part now travels to the target's block on its own, and the target reads after that
        # block's own attention, so one more step is applied before the readout.
        carried = [
            _propagate(model, piece, run, attention_layer + 1, target_layer) for piece in split
        ]
        parts = torch.stack(
            [part + _attention_step(model, target_layer, part, run) for part in carried]
        )
    readouts = torch.stack(
        [
            _as_feature_input(model, target_layer, part, run)[target_position] @ reader
            for part in parts
        ]
    )
    n_heads = model.cfg.n_heads
    return HeadLoadings(
        per_head=readouts[:n_heads], bypass=readouts[n_heads], attention_layer=attention_layer
    )


@dataclass(frozen=True)
class PathHeadLoadings:
    """Every attention layer's split of one edge, computed together.

    Row ``i`` of ``per_head`` and entry ``i`` of ``bypass`` belong to ``layers[i]``. Each row is its
    own partition of ``total``; the rows are not parts of one joint decomposition.
    """

    layers: list[int]
    per_head: Tensor
    bypass: Tensor
    total: Tensor

    def at(self, layer: int) -> HeadLoadings:
        """The split at one attention layer, in the form :func:`head_loadings` returns."""
        if layer not in self.layers:
            raise ValueError(f"layer {layer} is not on this path, which covers {self.layers}")
        row = self.layers.index(layer)
        return HeadLoadings(
            per_head=self.per_head[row], bypass=self.bypass[row], attention_layer=layer
        )


def path_head_loadings(
    model,
    graph: Graph,
    run: FrozenRun,
    target_node: int,
    source_node: int,
) -> PathHeadLoadings:
    """Split one edge at every attention layer on its path, in a single sweep each way.

    Calling :func:`head_loadings` once per layer re-propagates every head's part from that layer
    to the target, on the order of ``L**2 * n_heads`` attention steps for a path ``L`` layers long.
    The map is linear, so the readout is a fixed linear functional of the perturbation entering
    each layer: one forward sweep gives the perturbation, one backward sweep gives the functional,
    and each head's loading is a dot product.

    Returns:
        A :class:`PathHeadLoadings` over every layer from just after the source to the target
        inclusive, each row agreeing with :func:`head_loadings` for that layer.
    """
    _require_supported(model)
    source, source_position, source_layer = source_vector(model, graph, source_node)
    reader, target_position, target_layer = reader_vector(model, graph, target_node)
    if target_layer <= source_layer:
        raise ValueError(
            f"target_layer must come after source_layer, got {source_layer} and {target_layer}"
        )
    first = source_layer + 1

    # autograd.grad rather than backward, so nothing accumulates on the model's parameters, and grad
    # mode is restored for callers that turned it off.
    with torch.enable_grad():
        seed = _seed(source.detach(), source_position, run.n_pos).requires_grad_(True)
        states = [seed]
        for layer in range(first, target_layer):
            states.append(states[-1] + _attention_step(model, layer, states[-1], run))
        readout = (
            _to_feature_input(model, states[-1], run, target_layer)[target_position]
            @ reader.detach()
        )
        sensitivities = torch.autograd.grad(readout, states)

    with torch.no_grad():
        per_head, bypass = [], []
        for offset, layer in enumerate(range(first, target_layer)):
            arriving = states[offset].detach()
            written = _attention_step(model, layer, arriving, run, per_head=True)
            onward = sensitivities[offset + 1]
            per_head.append(torch.einsum("hpd,pd->h", written, onward))
            bypass.append((arriving * onward).sum())

        # At the target's own block the split happens at the readout, as in head_loadings.
        arriving = states[-1].detach()
        written = _attention_step(model, target_layer, arriving, run, per_head=True)
        reader = reader.detach()
        per_head.append(
            _as_feature_input(model, target_layer, written, run)[:, target_position] @ reader
        )
        bypass.append(
            _as_feature_input(model, target_layer, arriving, run)[target_position] @ reader
        )
        return PathHeadLoadings(
            layers=list(range(first, target_layer + 1)),
            per_head=torch.stack(per_head),
            bypass=torch.stack(bypass),
            total=readout.detach(),
        )
