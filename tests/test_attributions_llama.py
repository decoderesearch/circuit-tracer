import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
from circuit_tracer.transcoder.activation_functions import TopK
from circuit_tracer.utils import get_default_device
from tests._comparison.attribution.attribute import attribute as legacy_attribute
from tests._comparison.replacement_model import ReplacementModel as LegacyReplacementModel
from tests.test_attributions_gemma import verify_feature_edges, verify_token_and_error_edges


def load_dummy_llama_model(cfg: LlamaConfig, k: int):
    transcoders = {
        layer_idx: SingleLayerTranscoder(
            cfg.hidden_size, cfg.hidden_size * 4, TopK(k), layer_idx, skip_connection=True
        )
        for layer_idx in range(cfg.num_hidden_layers)
    }
    for transcoder in transcoders.values():
        for _, param in transcoder.named_parameters():
            nn.init.uniform_(param, a=-1, b=1)

    transcoder_set = TranscoderSet(
        transcoders, feature_input_hook="mlp.hook_in", feature_output_hook="mlp.hook_out"
    )

    hf_model = LlamaForCausalLM(cfg).to(get_default_device())  # type: ignore[attr-defined]
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # to avoid gated repos
    model = ReplacementModel.from_hf_model(hf_model, tokenizer, transcoder_set)

    ids = model.tokenizer.all_special_ids
    type(model.tokenizer).all_special_ids = property(lambda self: [0] + ids)  # type: ignore
    for _, param in model.named_parameters():
        nn.init.uniform_(param, a=-1, b=1)

    return model


def verify_small_llama_model(s: torch.Tensor):
    # Create a small Llama config for testing
    llama_small_cfg = LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.017677669529663688,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=500000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
    )
    k = 4
    model = load_dummy_llama_model(llama_small_cfg, k)
    graph = attribute(s, model)

    verify_token_and_error_edges(model, graph)
    verify_feature_edges(model, graph)


def verify_large_llama_model(s: torch.Tensor):
    # Create a larger Llama config for testing
    llama_large_cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.017677669529663688,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=500000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
    )
    k = 16
    model = load_dummy_llama_model(llama_large_cfg, k)
    graph = attribute(s, model)

    verify_token_and_error_edges(model, graph)
    verify_feature_edges(model, graph)


def verify_llama_3_2_1b(s: str):
    model = ReplacementModel.boot_transformers("meta-llama/Llama-3.2-1B", "llama")
    graph = attribute(s, model, batch_size=128)

    verify_token_and_error_edges(model, graph)
    verify_feature_edges(model, graph)


def test_small_llama_model():
    s = torch.tensor([0, 3, 4, 3, 2, 5, 3, 8])
    verify_small_llama_model(s)


def test_large_llama_model():
    s = torch.tensor([0, 113, 24, 53, 27])
    verify_large_llama_model(s)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_llama_3_2_1b():
    s = "The National Digital Analytics Group (ND"
    verify_llama_3_2_1b(s)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bridge_vs_legacy_llama_3_2_1b():
    """Test Llama 3.2 1B attribution comparing bridge vs legacy implementations."""
    prompt = "The National Digital Analytics Group (ND"

    # Load bridge model (new implementation)
    bridge_model = ReplacementModel.boot_transformers("meta-llama/Llama-3.2-1B", "llama")

    # Load legacy model (old implementation)
    legacy_model = LegacyReplacementModel.from_pretrained("meta-llama/Llama-3.2-1B", "llama")

    # Run attribution on both
    bridge_graph = attribute(prompt, bridge_model, batch_size=128)
    legacy_graph = legacy_attribute(prompt, legacy_model, batch_size=128)

    # Check if active features match
    assert torch.allclose(bridge_graph.active_features, legacy_graph.active_features), (
        "Active features differ!"
    )

    # Check if activation values match
    assert torch.allclose(
        bridge_graph.activation_values, legacy_graph.activation_values, atol=1e-3, rtol=1e-4
    ), "Activation values differ!"

    # Check if adjacency matrices match
    diff = (bridge_graph.adjacency_matrix - legacy_graph.adjacency_matrix).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()

    # Allow for numerical precision differences due to different computation paths
    assert torch.allclose(
        bridge_graph.adjacency_matrix, legacy_graph.adjacency_matrix, atol=1e-3, rtol=1e-4
    ), f"Adjacency matrices differ! Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}"

    # Verify feature edges for both
    n_active = len(bridge_graph.active_features)
    n_samples = min(100, n_active)

    verify_feature_edges(
        legacy_model,  # type: ignore
        legacy_graph,
        n_samples=n_samples,
    )
    verify_feature_edges(
        bridge_model,
        bridge_graph,
        n_samples=n_samples,
    )
