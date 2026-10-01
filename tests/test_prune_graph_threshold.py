"""Regression tests for ``prune_graph`` / ``find_threshold`` at threshold 1.0.

``prune_graph`` accepts ``edge_threshold`` in the closed interval [0, 1] and documents it as
"keep the minimum number of edges whose cumulative influence is >= threshold".  At 1.0 that
is "keep every edge that carries influence": all non-zero entries of the adjacency matrix
between kept nodes, and nothing else.

``find_threshold`` normalises the cumulative scores by ``torch.sum`` while the running total
comes from ``torch.cumsum``.  The two reductions round differently, so the cumulative score at
the last non-zero entry can come out strictly below 1.0.  ``searchsorted`` then runs past every
non-zero score and the clamp picks a score of 0.0, and ``edge_scores >= 0.0`` marks every pair
of nodes, including pairs with no edge at all and pairs involving nodes that node pruning
removed.  Downstream, ``create_used_nodes_and_edges`` writes one weight-0.0 link per kept pair
to the graph JSON.

The inputs below were chosen so that every float32 summation order of the non-zero scores
gives the same ``torch.sum`` result, and that result is strictly above the float32 rounding
of the exact sum that ``torch.cumsum`` (double accumulation on CPU) reaches.  The failure
therefore does not depend on the SIMD width of the machine running the test.  All tensors are
built on the CPU on purpose.
"""

import torch
from transformer_lens import HookedTransformerConfig

from circuit_tracer.attribution.targets import LogitTarget
from circuit_tracer.graph import Graph, find_threshold, prune_graph


def test_find_threshold_at_one_never_selects_a_zero_score():
    # Three non-zero scores whose float32 sum is order-independent and one ulp above the
    # correctly rounded exact sum, followed by zero scores (a flattened edge-score matrix is
    # almost entirely zeros).
    scores = torch.tensor(
        [0.91275554895401, 0.8132702112197876, 0.016527635976672173, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )
    smallest_nonzero = scores[scores > 0].min()

    threshold = find_threshold(scores, 1.0)

    # Keeping 100% of the influence means keeping every non-zero score, so the cut-off must be
    # the smallest non-zero score, never 0.0 (which would also admit the zero entries).
    assert threshold > 0, f"find_threshold(scores, 1.0) returned {threshold.item()}, expected > 0"
    assert torch.isclose(threshold, smallest_nonzero)


def _two_feature_chain_graph() -> Graph:
    """A 2-layer, 1-position graph: tok -> f1 -> f2, and f1, f2 -> logit.

    Node order is [f1, f2, err(l0), err(l1), tok, logit].  The logit edge weights are the
    ones that make the flattened edge scores sum order-independently one ulp above the
    correctly rounded exact sum (see module docstring).
    """
    w1, w2 = 0.43596285581588745, 0.7308765649795532
    n_layers, n_pos, n_features, n_logits = 2, 1, 2, 1
    n_nodes = n_features + n_layers * n_pos + n_pos + n_logits
    adjacency = torch.zeros(n_nodes, n_nodes)
    tok, logit = n_features + n_layers * n_pos, n_nodes - 1
    adjacency[0, tok] = 1.0  # f1 <- tok
    adjacency[1, 0] = 1.0  # f2 <- f1
    adjacency[logit, 0] = w1  # logit <- f1
    adjacency[logit, 1] = w2  # logit <- f2

    cfg = HookedTransformerConfig.from_dict(
        {
            "n_layers": n_layers,
            "d_model": 8,
            "n_ctx": 32,
            "d_head": 4,
            "n_heads": 2,
            "d_mlp": 16,
            "act_fn": "gelu",
            "d_vocab": 100,
            "model_name": "test-model",
            "device": "cpu",
        }
    )
    return Graph(
        input_string="a",
        input_tokens=torch.arange(n_pos),
        active_features=torch.tensor([[0, 0, 0], [1, 0, 1]]),
        adjacency_matrix=adjacency,
        cfg=cfg,
        selected_features=torch.arange(n_features),
        activation_values=torch.ones(n_features),
        logit_targets=[LogitTarget(token_str="x", vocab_idx=1)],
        logit_probabilities=torch.tensor([1.0]),
        scan_name="test-scan",
    )


def test_prune_graph_edge_threshold_one_keeps_only_real_edges():
    graph = _two_feature_chain_graph()
    real_edges = graph.adjacency_matrix != 0

    node_mask, edge_mask, _ = prune_graph(graph, node_threshold=0.8, edge_threshold=1.0)

    kept = node_mask[:, None] & node_mask[None, :]
    spurious = edge_mask & ~real_edges
    assert not spurious.any(), (
        f"{int(spurious.sum())} zero-weight (non-)edges were kept at edge_threshold=1.0; "
        f"the graph only has {int(real_edges.sum())} edges"
    )
    assert not (edge_mask & ~kept).any(), "kept an edge touching a pruned node"
    # 100% of the influence == every real edge between kept nodes, nothing more.
    assert torch.equal(edge_mask, real_edges & kept)
