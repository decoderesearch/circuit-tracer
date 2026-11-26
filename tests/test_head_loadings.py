"""Tests for splitting an attribution-graph edge across attention heads.

These use a stub standing in for a ReplacementModel, so they need no weights and no GPU. The
property that matters is a partition: the per-head parts plus the bypass must reproduce the edge
effect exactly, for every attention layer between the source and the target.
"""

from types import SimpleNamespace

import pytest
import torch
from transformer_lens import HookedTransformerConfig

from circuit_tracer.attribution.head_loadings import (
    FrozenRun,
    HeadLoadings,
    NodeKind,
    NodeLayout,
    UnsupportedForLoadings,
    edge_effect,
    head_loadings,
    reader_vector,
    source_vector,
)
from circuit_tracer.attribution.targets import LogitTarget
from circuit_tracer.graph import Graph

N_LAYERS = 4
N_HEADS = 3
D_MODEL = 8
D_HEAD = 4
N_POS = 5
D_TRANSCODER = 6


class StubTranscoders(list):
    """Stands in for a TranscoderSet, which is indexable by layer."""

    def __init__(self, items, *, skip_connection: bool = False):
        super().__init__(items)
        self.skip_connection = skip_connection
        self.feature_input_hook = "mlp.hook_in"


def make_model(
    *, skip: bool = False, ln_bias: bool = False, post_norm: bool = False
) -> SimpleNamespace:
    """A stub with the attributes the split reads, and random weights throughout.

    ``post_norm`` adds a normalisation on attention's output before it reaches the residual
    stream, as Gemma-2 has.
    """
    generator = torch.Generator().manual_seed(0)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    blocks = []
    for _ in range(N_LAYERS):
        attn = SimpleNamespace(
            W_V=randn(N_HEADS, D_MODEL, D_HEAD), W_O=randn(N_HEADS, D_HEAD, D_MODEL)
        )
        norm = lambda: SimpleNamespace(w=randn(D_MODEL), b=randn(D_MODEL) if ln_bias else None)  # noqa: E731
        block = SimpleNamespace(attn=attn, ln1=norm(), ln2=norm())
        if post_norm:
            block.ln1_post = norm()
        blocks.append(block)

    transcoders = StubTranscoders(
        [
            SimpleNamespace(
                W_enc=randn(D_TRANSCODER, D_MODEL),
                W_dec=randn(D_TRANSCODER, D_MODEL),
                W_skip=randn(D_MODEL, D_MODEL) if skip else None,
            )
            for _ in range(N_LAYERS)
        ],
        skip_connection=skip,
    )
    return SimpleNamespace(
        cfg=SimpleNamespace(n_layers=N_LAYERS, n_heads=N_HEADS, d_model=D_MODEL),
        blocks=blocks,
        transcoders=transcoders,
        W_E=randn(16, D_MODEL),
    )


def make_run(*, patterns: torch.Tensor | None = None, post_norm: bool = False) -> FrozenRun:
    """Frozen patterns and scales, causal and normalised as a real run's would be."""
    generator = torch.Generator().manual_seed(1)
    if patterns is None:
        raw = torch.rand(N_LAYERS, N_HEADS, N_POS, N_POS, generator=generator)
        mask = torch.tril(torch.ones(N_POS, N_POS))
        raw = raw * mask
        patterns = raw / raw.sum(dim=-1, keepdim=True)
    scales = torch.rand(N_POS, 1, generator=generator) + 0.5
    post = {"ln1_post_scales": [scales.clone() + 1.5 for _ in range(N_LAYERS)]}
    return FrozenRun(
        patterns=list(patterns),
        ln1_scales=[scales.clone() for _ in range(N_LAYERS)],
        ln2_scales=[scales.clone() + 0.25 for _ in range(N_LAYERS)],
        **(post if post_norm else {}),
    )


def reference_edge_effect(
    model,
    run: FrozenRun,
    source,
    source_position,
    source_layer,
    reader,
    target_position,
    target_layer,
) -> torch.Tensor:
    """The edge effect written out as a plain forward loop, independent of the module's helpers."""
    state = torch.zeros(N_POS, D_MODEL)
    state[source_position] = source
    for layer in range(source_layer + 1, target_layer + 1):
        block = model.blocks[layer]
        normalised = state * block.ln1.w / run.ln1_scales[layer]
        out = torch.zeros(N_POS, D_MODEL)
        for head in range(N_HEADS):
            values = normalised @ block.attn.W_V[head]
            out += (run.patterns[layer][head] @ values) @ block.attn.W_O[head]
        if hasattr(block, "ln1_post"):
            assert run.ln1_post_scales is not None
            out = out * block.ln1_post.w / run.ln1_post_scales[layer]
        state = state + out
    # What the transcoder reads depends on its hook: the residual before ln2, ln2's output before
    # its gain, or the MLP's input after the gain.
    hook = model.transcoders.feature_input_hook
    final = model.blocks[target_layer].ln2
    if hook == "hook_mlp_in":
        read = state
    elif hook == "ln2.hook_normalized":
        read = state / run.ln2_scales[target_layer]
    else:
        read = state * final.w / run.ln2_scales[target_layer]
    return read[target_position] @ reader


def make_graph(
    active: list[tuple[int, int, int]],
    selected: list[int],
    activations: list[float],
) -> Graph:
    """A graph carrying only what the split reads: the feature table and the layout."""
    n_features = len(selected)
    total = n_features + N_LAYERS * N_POS + N_POS + 1
    cfg = HookedTransformerConfig(
        n_layers=N_LAYERS, d_model=D_MODEL, n_ctx=N_POS, d_head=D_HEAD, n_heads=N_HEADS, d_vocab=16
    )
    return Graph(
        input_string="stub",
        input_tokens=torch.arange(N_POS),
        active_features=torch.tensor(active),
        adjacency_matrix=torch.zeros(total, total),
        cfg=cfg,
        selected_features=torch.tensor(selected),
        activation_values=torch.tensor(activations),
        logit_targets=[LogitTarget(token_str="x", vocab_idx=0)],
        logit_probabilities=torch.tensor([1.0]),
    )


# Two features: node 0 is written by layer 0 at position 1, node 1 is read by layer 3 at position 4.
ACTIVE = [(0, 1, 2), (3, 4, 5)]
GRAPH_ARGS = (ACTIVE, [0, 1], [3.0, 7.0])


def test_the_parts_sum_to_the_edge_effect():
    """The partition property, which is the whole claim this module makes."""
    model, run, graph = make_model(), make_run(), make_graph(*GRAPH_ARGS)
    whole = edge_effect(model, graph, run, target_node=1, source_node=0)
    for attention_layer in range(1, N_LAYERS):
        parts = head_loadings(model, graph, run, 1, 0, attention_layer)
        torch.testing.assert_close(parts.total, whole)


def test_every_head_is_accounted_for():
    parts = head_loadings(make_model(), make_graph(*GRAPH_ARGS), make_run(), 1, 0, 2)
    assert parts.per_head.shape == (N_HEADS,)
    assert parts.bypass.ndim == 0
    assert parts.attention_layer == 2


def test_a_head_with_no_output_weights_carries_nothing():
    model, run, graph = make_model(), make_run(), make_graph(*GRAPH_ARGS)
    model.blocks[2].attn.W_O[1] = 0.0
    parts = head_loadings(model, graph, run, 1, 0, 2)
    assert parts.per_head[1] == 0.0
    assert parts.per_head[0] != 0.0


def test_the_bypass_carries_everything_when_no_head_writes():
    model, run, graph = make_model(), make_run(), make_graph(*GRAPH_ARGS)
    model.blocks[2].attn.W_O[:] = 0.0
    parts = head_loadings(model, graph, run, 1, 0, 2)
    torch.testing.assert_close(parts.per_head, torch.zeros(N_HEADS))
    torch.testing.assert_close(parts.bypass, parts.total)


def test_ranked_orders_heads_by_magnitude():
    parts = HeadLoadings(torch.tensor([0.1, -0.9, 0.4]), torch.tensor(0.0), 1)
    order, values = parts.ranked()
    assert order.tolist() == [1, 2, 0]
    torch.testing.assert_close(values, torch.tensor([-0.9, 0.4, 0.1]))


@pytest.mark.parametrize("attention_layer", [0, 4, -1])
def test_an_attention_layer_outside_the_path_is_rejected(attention_layer: int):
    with pytest.raises(ValueError, match="attention_layer must lie"):
        head_loadings(make_model(), make_graph(*GRAPH_ARGS), make_run(), 1, 0, attention_layer)


def test_the_target_layer_is_a_valid_split_point():
    """The target reads after its own block's attention, so its layer counts."""
    model, run, graph = make_model(), make_run(), make_graph(*GRAPH_ARGS)
    parts = head_loadings(model, graph, run, 1, 0, N_LAYERS - 1)
    torch.testing.assert_close(parts.total, edge_effect(model, graph, run, 1, 0))


def test_a_token_embedding_works_as_a_source():
    model, run, graph = make_model(), make_run(), make_graph(*GRAPH_ARGS)
    token_node = NodeLayout.from_graph(graph).error_end + 2
    vector, position, layer = source_vector(model, graph, token_node)
    assert position == 2
    assert layer == -1
    torch.testing.assert_close(vector, model.W_E[2])
    parts = head_loadings(model, graph, run, 1, token_node, 1)
    torch.testing.assert_close(parts.total, edge_effect(model, graph, run, 1, token_node))


def test_an_error_source_is_refused_rather_than_guessed():
    graph = make_graph(*GRAPH_ARGS)
    error_node = len(graph.selected_features)
    with pytest.raises(UnsupportedForLoadings, match="error nodes"):
        source_vector(make_model(), graph, error_node)


def test_a_non_feature_target_is_refused():
    graph = make_graph(*GRAPH_ARGS)
    logit_node = NodeLayout.from_graph(graph).token_end
    with pytest.raises(ValueError, match="target must be a feature node"):
        reader_vector(make_model(), graph, logit_node)


def test_skip_transcoders_are_refused():
    """With a skip connection an MLP's output moves when its input does, so no path is frozen."""
    with pytest.raises(UnsupportedForLoadings, match="skip connection"):
        edge_effect(make_model(skip=True), make_graph(*GRAPH_ARGS), make_run(), 1, 0)


def test_a_layernorm_bias_is_refused():
    with pytest.raises(UnsupportedForLoadings, match="has a bias"):
        edge_effect(make_model(ln_bias=True), make_graph(*GRAPH_ARGS), make_run(), 1, 0)


def test_a_feature_input_hook_off_the_mlp_is_refused():
    model = make_model()
    model.transcoders.feature_input_hook = "hook_resid_mid"
    with pytest.raises(UnsupportedForLoadings, match="models only the hooks around ln2"):
        edge_effect(model, make_graph(*GRAPH_ARGS), make_run(), 1, 0)


def test_activations_are_indexed_by_active_feature_not_by_selection():
    """activation_values is aligned with active_features; the two coincide only when nothing
    is pruned, which is how the mismatch stays invisible on small graphs."""
    active = [(0, 1, 2), (1, 2, 3), (3, 4, 5)]
    graph = make_graph(active, [0, 2], [3.0, 99.0, 7.0])
    model = make_model()
    vector, _, _ = source_vector(model, graph, 0)
    torch.testing.assert_close(vector, model.transcoders[0].W_dec[2] * 3.0)
    reader, position, layer = reader_vector(model, graph, 1)
    assert (position, layer) == (4, 3)
    torch.testing.assert_close(reader, model.transcoders[3].W_enc[5])


def test_node_layout_regions_are_contiguous():
    layout = NodeLayout.from_graph(make_graph(*GRAPH_ARGS))
    assert layout.kind(0) is NodeKind.FEATURE
    assert layout.kind(layout.n_features) is NodeKind.ERROR
    assert layout.kind(layout.error_end) is NodeKind.TOKEN
    assert layout.kind(layout.token_end) is NodeKind.LOGIT
    assert layout.decode_error(layout.n_features + N_POS + 3) == (1, 3)
    assert layout.decode_token(layout.error_end + 4) == 4


def test_an_out_of_range_node_raises():
    layout = NodeLayout.from_graph(make_graph(*GRAPH_ARGS))
    with pytest.raises(IndexError, match="out of range"):
        layout.kind(layout.total)


def test_an_inconsistent_adjacency_matrix_is_reported():
    graph = make_graph(*GRAPH_ARGS)
    graph.adjacency_matrix = torch.zeros(7, 7)
    with pytest.raises(ValueError, match="inconsistent with the graph"):
        NodeLayout.from_graph(graph)


def test_a_batched_cache_is_refused():
    """A graph describes one prompt, so a batched run would silently misalign every position."""

    class BatchedModel:
        cfg = SimpleNamespace(n_layers=1, n_heads=N_HEADS)

        def run_with_cache(self, tokens, names_filter=None):
            return None, {
                "blocks.0.attn.hook_pattern": torch.zeros(2, N_HEADS, N_POS, N_POS),
                "blocks.0.ln1.hook_scale": torch.ones(2, N_POS, 1),
                "blocks.0.ln2.hook_scale": torch.ones(2, N_POS, 1),
            }

    with pytest.raises(ValueError, match="a batch of 2"):
        FrozenRun.from_model(BatchedModel(), torch.zeros(2, N_POS, dtype=torch.long))


def test_path_head_loadings_match_splitting_one_layer_at_a_time():
    """The single sweep is only worth having if it reproduces the per-layer split exactly."""
    from circuit_tracer.attribution.head_loadings import path_head_loadings

    model, run, graph = make_model(), make_run(), make_graph(*GRAPH_ARGS)
    swept = path_head_loadings(model, graph, run, target_node=1, source_node=0)
    assert swept.layers == list(range(1, N_LAYERS))
    for layer in swept.layers:
        expected = head_loadings(model, graph, run, 1, 0, layer)
        torch.testing.assert_close(
            swept.at(layer).per_head, expected.per_head, rtol=1e-4, atol=1e-6
        )
        torch.testing.assert_close(swept.at(layer).bypass, expected.bypass, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(swept.total, edge_effect(model, graph, run, 1, 0))


def test_path_head_loadings_from_a_token_start_at_block_zero():
    from circuit_tracer.attribution.head_loadings import path_head_loadings

    model, run, graph = make_model(), make_run(), make_graph(*GRAPH_ARGS)
    token_node = NodeLayout.from_graph(graph).error_end + 2
    swept = path_head_loadings(model, graph, run, target_node=1, source_node=token_node)
    assert swept.layers == list(range(0, N_LAYERS))
    expected = head_loadings(model, graph, run, 1, token_node, 0)
    torch.testing.assert_close(swept.at(0).per_head, expected.per_head, rtol=1e-4, atol=1e-6)


def test_path_head_loadings_leave_gradients_disabled():
    from circuit_tracer.attribution.head_loadings import path_head_loadings

    model, run, graph = make_model(), make_run(), make_graph(*GRAPH_ARGS)
    with torch.no_grad():
        swept = path_head_loadings(model, graph, run, target_node=1, source_node=0)
        assert not torch.is_grad_enabled()
    assert not swept.per_head.requires_grad


def test_path_head_loadings_reject_a_layer_off_the_path():
    from circuit_tracer.attribution.head_loadings import path_head_loadings

    swept = path_head_loadings(make_model(), make_graph(*GRAPH_ARGS), make_run(), 1, 0)
    with pytest.raises(ValueError, match="not on this path"):
        swept.at(0)


INPUT_HOOKS = ["mlp.hook_in", "ln2.hook_normalized", "hook_mlp_in"]


@pytest.mark.parametrize("input_hook", INPUT_HOOKS)
@pytest.mark.parametrize("post_norm", [False, True])
def test_the_edge_effect_matches_a_plain_forward_loop(post_norm: bool, input_hook: str):
    """Checked against an independent loop, since the partition alone cannot catch a step that is
    left out consistently everywhere, such as a norm on attention's output or a gain the
    transcoder never sees."""
    model, graph = make_model(post_norm=post_norm), make_graph(*GRAPH_ARGS)
    model.transcoders.feature_input_hook = input_hook
    run = make_run(post_norm=post_norm)
    source, source_position, source_layer = source_vector(model, graph, 0)
    reader, target_position, target_layer = reader_vector(model, graph, 1)
    expected = reference_edge_effect(
        model, run, source, source_position, source_layer, reader, target_position, target_layer
    )
    torch.testing.assert_close(edge_effect(model, graph, run, 1, 0), expected)


@pytest.mark.parametrize("input_hook", INPUT_HOOKS)
def test_every_readout_keeps_the_partition_exact(input_hook: str):
    """head_loadings and the single sweep each read at the target on their own, so every input hook
    is checked at every split point, the target layer included."""
    from circuit_tracer.attribution.head_loadings import path_head_loadings

    model, graph, run = make_model(), make_graph(*GRAPH_ARGS), make_run()
    model.transcoders.feature_input_hook = input_hook
    whole = edge_effect(model, graph, run, 1, 0)
    source, source_position, source_layer = source_vector(model, graph, 0)
    reader, target_position, target_layer = reader_vector(model, graph, 1)
    torch.testing.assert_close(
        whole,
        reference_edge_effect(
            model, run, source, source_position, source_layer, reader, target_position, target_layer
        ),
    )
    swept = path_head_loadings(model, graph, run, 1, 0)
    for attention_layer in range(1, N_LAYERS):
        parts = head_loadings(model, graph, run, 1, 0, attention_layer)
        torch.testing.assert_close(parts.total, whole)
        torch.testing.assert_close(
            swept.at(attention_layer).per_head, parts.per_head, rtol=1e-4, atol=1e-6
        )


def test_a_post_attention_norm_keeps_the_partition_exact():
    from circuit_tracer.attribution.head_loadings import path_head_loadings

    model, graph = make_model(post_norm=True), make_graph(*GRAPH_ARGS)
    run = make_run(post_norm=True)
    whole = edge_effect(model, graph, run, 1, 0)
    swept = path_head_loadings(model, graph, run, 1, 0)
    for attention_layer in range(1, N_LAYERS):
        parts = head_loadings(model, graph, run, 1, 0, attention_layer)
        torch.testing.assert_close(parts.total, whole)
        torch.testing.assert_close(
            swept.at(attention_layer).per_head, parts.per_head, rtol=1e-4, atol=1e-6
        )


def test_a_post_attention_norm_without_its_frozen_scales_is_refused():
    """Gemma-2 normalises attention's output; without that scale the split would be silently off."""
    with pytest.raises(UnsupportedForLoadings, match="ln1_post"):
        edge_effect(make_model(post_norm=True), make_graph(*GRAPH_ARGS), make_run(), 1, 0)


def test_the_frozen_run_keeps_post_attention_scales_when_the_model_has_them():
    class PostNormModel:
        cfg = SimpleNamespace(n_layers=1, n_heads=N_HEADS)

        def run_with_cache(self, tokens, names_filter=None):
            return None, {
                "blocks.0.attn.hook_pattern": torch.zeros(1, N_HEADS, N_POS, N_POS),
                "blocks.0.ln1.hook_scale": torch.ones(1, N_POS, 1),
                "blocks.0.ln2.hook_scale": torch.ones(1, N_POS, 1),
                "blocks.0.ln1_post.hook_scale": torch.full((1, N_POS, 1), 2.0),
            }

    run = FrozenRun.from_model(PostNormModel(), torch.zeros(1, N_POS, dtype=torch.long))
    assert run.ln1_post_scales is not None
    torch.testing.assert_close(run.ln1_post_scales[0], torch.full((N_POS, 1), 2.0))
