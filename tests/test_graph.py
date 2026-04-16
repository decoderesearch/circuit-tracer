import gc

import numpy as np
import pytest
import torch
from transformer_lens import HookedTransformerConfig

from circuit_tracer.attribution.targets import LogitTarget
from circuit_tracer.graph import Graph, compute_edge_influence, compute_node_influence
from circuit_tracer.utils import get_default_device


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


def test_small_graph():
    value = 10
    edge_matrix = torch.zeros([12, 12])
    for node in [1, 3, 6, 8]:
        edge_matrix[11, node] = 1 / 4

    for node in [0, 1, 6]:
        edge_matrix[3, node] = 1 / 12

    for node in [9, 10]:
        edge_matrix[1, node] = 1 / 6

    edge_matrix[0, 9] = 1 / 12

    adjacency_matrix = (edge_matrix > 0).float() * value

    # These get pruned during node pruning with a 0.8 threshold
    pruned_adjacency_matrix = adjacency_matrix.clone()
    pruned_adjacency_matrix[0, 9] = 0
    pruned_adjacency_matrix[3, 0] = 0

    post_pruning_edge_matrix = torch.zeros([12, 12])
    for node in [1, 3, 6, 8]:
        post_pruning_edge_matrix[11, node] = 1 / 4

    for node in [1, 6]:
        post_pruning_edge_matrix[3, node] = 1 / 8

    for node in [9, 10]:
        post_pruning_edge_matrix[1, node] = 3 / 16

    # This is our dummy model config; it doesn't really matter besides n_layers
    gemma_small_cfg = {
        "n_layers": 2,
        "d_model": 8,
        "n_ctx": 8192,
        "d_head": 4,
        "model_name": "gemma-2-2b",
        "n_heads": 2,
        "d_mlp": 16,
        "act_fn": "gelu_pytorch_tanh",
        "d_vocab": 16,
        "eps": 1e-06,
        "use_attn_result": False,
        "use_attn_scale": True,
        "attn_scale": np.float64(16.0),
        "use_split_qkv_input": False,
        "use_hook_mlp_in": False,
        "use_attn_in": False,
        "use_local_attn": True,
        "ungroup_grouped_query_attention": False,
        "original_architecture": "Gemma2ForCausalLM",
        "from_checkpoint": False,
        "checkpoint_index": None,
        "checkpoint_label_type": None,
        "checkpoint_value": None,
        "tokenizer_name": "google/gemma-2-2b",
        "window_size": 4096,
        "attn_types": ["global", "local"],
        "init_mode": "gpt2",
        "normalization_type": "RMSPre",
        "device": get_default_device(),
        "n_devices": 1,
        "attention_dir": "causal",
        "attn_only": False,
        "seed": None,
        "initializer_range": 0.02,
        "init_weights": False,
        "scale_attn_by_inverse_layer_idx": False,
        "positional_embedding_type": "rotary",
        "final_rms": True,
        "d_vocab_out": 16,
        "parallel_attn_mlp": False,
        "rotary_dim": 4,
        "n_params": 2146959360,
        "use_hook_tokens": False,
        "gated_mlp": True,
        "default_prepend_bos": True,
        "dtype": torch.float32,
        "tokenizer_prepends_bos": True,
        "n_key_value_heads": 2,
        "post_embedding_ln": False,
        "rotary_base": 10000.0,
        "trust_remote_code": False,
        "rotary_adjacent_pairs": False,
        "load_in_4bit": False,
        "num_experts": None,
        "experts_per_token": None,
        "relative_attention_max_distance": None,
        "relative_attention_num_buckets": None,
        "decoder_start_token_id": None,
        "tie_word_embeddings": False,
        "use_normalization_before_and_after": True,
        "attn_scores_soft_cap": 50.0,
        "output_logits_soft_cap": 0.0,
        "use_NTK_by_parts_rope": False,
        "NTK_by_parts_low_freq_factor": 1.0,
        "NTK_by_parts_high_freq_factor": 4.0,
        "NTK_by_parts_factor": 8.0,
    }
    cfg = HookedTransformerConfig.from_dict(gemma_small_cfg)
    test_graph = Graph(
        input_string="ab",
        input_tokens=torch.tensor([0, 1]),
        active_features=torch.tensor([1, 2, 3, 4, 5]),
        adjacency_matrix=adjacency_matrix,
        cfg=cfg,
        logit_targets=[LogitTarget(token_str="tok_0", vocab_idx=0)],
        logit_probabilities=torch.tensor([1.0]),
        selected_features=torch.tensor([1, 2, 3, 4, 5]),
        activation_values=torch.tensor([1, 2, 3, 4, 5]) * 2,
    )
    test_graph.cfg.n_layers = 2

    logit_weights = torch.zeros(adjacency_matrix.size(0))
    logit_weights[-1] = 1.0

    node_influence_on_logits = compute_node_influence(test_graph.adjacency_matrix, logit_weights)
    influence_tensor = torch.tensor(
        [1 / 12, 1 / 3, 0, 1 / 4, 0, 0, 1 / 3, 0, 1 / 4, 1 / 4, 1 / 6, 0]
    )
    assert torch.allclose(node_influence_on_logits, influence_tensor)

    edge_influence_on_logits = compute_edge_influence(pruned_adjacency_matrix, logit_weights)
    assert torch.allclose(edge_influence_on_logits, post_pruning_edge_matrix)


def test_graph_with_tensor_logit_targets():
    """Test that Graph accepts LogitTarget list format and from_pt handles legacy tensor format."""
    cfg = HookedTransformerConfig.from_dict(
        {
            "n_layers": 2,
            "d_model": 8,
            "n_ctx": 32,
            "d_head": 4,
            "n_heads": 2,
            "d_mlp": 16,
            "act_fn": "gelu",
            "d_vocab": 50257,  # GPT-2 vocab size
            "model_name": "test-model",
            "device": get_default_device(),
        }
    )

    adjacency_matrix = torch.zeros([10, 10])
    adjacency_matrix[9, 5] = 1.0

    # Test with LogitTarget list using empty token strings (simulates legacy conversion)
    graph_tensor = Graph(
        input_string="test",
        input_tokens=torch.tensor([1, 2, 3]),
        active_features=torch.tensor([[0, 0, 5]]),
        adjacency_matrix=adjacency_matrix,
        cfg=cfg,
        logit_targets=[
            LogitTarget(token_str="", vocab_idx=262),
            LogitTarget(token_str="", vocab_idx=290),
            LogitTarget(token_str="", vocab_idx=314),
        ],
        logit_probabilities=torch.tensor([0.5, 0.3, 0.2]),
        selected_features=torch.tensor([0]),
        activation_values=torch.tensor([1.5]),
    )

    # Verify LogitTarget list format
    assert len(graph_tensor.logit_targets) == 3
    assert graph_tensor.logit_targets[0].vocab_idx == 262
    assert graph_tensor.logit_targets[1].vocab_idx == 290
    assert graph_tensor.logit_targets[2].vocab_idx == 314
    # Token strings are empty when constructed from legacy tensor
    assert graph_tensor.logit_targets[0].token_str == ""
    assert graph_tensor.logit_targets[1].token_str == ""
    assert graph_tensor.logit_targets[2].token_str == ""

    # Verify properties work
    assert graph_tensor.logit_token_ids.tolist() == [262, 290, 314]
    assert torch.equal(graph_tensor.logit_token_ids, torch.tensor([262, 290, 314]))

    # Test with LogitTarget list format (current)
    graph_list = Graph(
        input_string="test",
        input_tokens=torch.tensor([1, 2, 3]),
        active_features=torch.tensor([[0, 0, 5]]),
        adjacency_matrix=adjacency_matrix,
        cfg=cfg,
        logit_targets=[
            LogitTarget(token_str=" the", vocab_idx=262),
            LogitTarget(token_str=" a", vocab_idx=290),
            LogitTarget(token_str=" and", vocab_idx=314),
        ],
        logit_probabilities=torch.tensor([0.5, 0.3, 0.2]),
        selected_features=torch.tensor([0]),
        activation_values=torch.tensor([1.5]),
    )

    # Verify both formats produce same logit_token_ids
    assert torch.equal(graph_tensor.logit_token_ids, graph_list.logit_token_ids)
    assert graph_tensor.vocab_size == graph_list.vocab_size


@pytest.mark.parametrize(
    "logit_targets_input,expected_token_strs",
    [
        pytest.param(
            [
                LogitTarget(token_str="", vocab_idx=262),
                LogitTarget(token_str="", vocab_idx=290),
                LogitTarget(token_str="", vocab_idx=314),
            ],
            ["", "", ""],
            id="empty_token_str_format",
        ),
        pytest.param(
            [
                LogitTarget(token_str=" the", vocab_idx=262),
                LogitTarget(token_str=" a", vocab_idx=290),
                LogitTarget(token_str=" and", vocab_idx=314),
            ],
            [" the", " a", " and"],
            id="logit_target_format",
        ),
    ],
)
def test_graph_serialization_with_logit_targets(logit_targets_input, expected_token_strs):
    """Test that Graph serialization works with both tensor and LogitTarget formats."""
    import tempfile
    import os

    cfg = HookedTransformerConfig.from_dict(
        {
            "n_layers": 2,
            "d_model": 8,
            "n_ctx": 32,
            "d_head": 4,
            "n_heads": 2,
            "d_mlp": 16,
            "act_fn": "gelu",
            "d_vocab": 50257,
            "model_name": "test-model",
            "device": get_default_device(),
        }
    )

    adjacency_matrix = torch.zeros([10, 10])
    adjacency_matrix[9, 5] = 1.0

    # Create graph with parameterized format
    original_graph = Graph(
        input_string="test",
        input_tokens=torch.tensor([1, 2, 3]),
        active_features=torch.tensor([[0, 0, 5]]),
        adjacency_matrix=adjacency_matrix,
        cfg=cfg,
        logit_targets=logit_targets_input,
        logit_probabilities=torch.tensor([0.5, 0.3, 0.2]),
        selected_features=torch.tensor([0]),
        activation_values=torch.tensor([1.5]),
        vocab_size=50257,
    )

    # Save and load
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp_path = tmp.name

    try:
        original_graph.to_pt(tmp_path)
        loaded_graph = Graph.from_pt(tmp_path)

        # Verify loaded graph has correct data
        assert loaded_graph.logit_token_ids.tolist() == [262, 290, 314]
        assert loaded_graph.vocab_size == 50257
        assert torch.equal(loaded_graph.logit_token_ids, torch.tensor([262, 290, 314]))
        assert torch.equal(loaded_graph.logit_probabilities, torch.tensor([0.5, 0.3, 0.2]))

        # Verify LogitTarget objects were preserved with expected token strings
        assert len(loaded_graph.logit_targets) == 3
        assert all(isinstance(lt, LogitTarget) for lt in loaded_graph.logit_targets)
        assert loaded_graph.logit_targets[0].token_str == expected_token_strs[0]
        assert loaded_graph.logit_targets[1].token_str == expected_token_strs[1]
        assert loaded_graph.logit_targets[2].token_str == expected_token_strs[2]

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_graph_from_pt_legacy_tensor_format():
    """Test that Graph.from_pt correctly handles legacy serialized graphs with tensor logit_targets."""
    import tempfile
    import os

    cfg = HookedTransformerConfig.from_dict(
        {
            "n_layers": 2,
            "d_model": 8,
            "n_ctx": 32,
            "d_head": 4,
            "n_heads": 2,
            "d_mlp": 16,
            "act_fn": "gelu",
            "d_vocab": 50257,
            "model_name": "test-model",
            "device": get_default_device(),
        }
    )

    # Simulate a legacy .pt file with tensor logit_targets
    legacy_data = {
        "input_string": "test",
        "adjacency_matrix": torch.zeros([10, 10]),
        "cfg": cfg,
        "active_features": torch.tensor([[0, 0, 5]]),
        "logit_targets": torch.tensor([262, 290, 314]),  # Legacy tensor format
        "logit_probabilities": torch.tensor([0.5, 0.3, 0.2]),
        "vocab_size": 50257,
        "input_tokens": torch.tensor([1, 2, 3]),
        "selected_features": torch.tensor([0]),
        "activation_values": torch.tensor([1.5]),
        "scan": None,
    }

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp_path = tmp.name

    try:
        torch.save(legacy_data, tmp_path)
        loaded_graph = Graph.from_pt(tmp_path)

        # Verify from_pt converted tensor to LogitTarget list
        assert len(loaded_graph.logit_targets) == 3
        assert all(isinstance(lt, LogitTarget) for lt in loaded_graph.logit_targets)
        assert loaded_graph.logit_targets[0].vocab_idx == 262
        assert loaded_graph.logit_targets[1].vocab_idx == 290
        assert loaded_graph.logit_targets[2].vocab_idx == 314
        assert loaded_graph.logit_targets[0].token_str == ""
        assert loaded_graph.logit_token_ids.tolist() == [262, 290, 314]
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
