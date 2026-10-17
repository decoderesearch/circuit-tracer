"""Tests for `attribute(..., freeze_attention=..., freeze_layernorm=...)`.

With both flags False, graph edges are the exact first-order (Jacobian) attributions of the
replacement model: no stop-gradient on attention patterns or normalization scales.
"""

import gc
from typing import cast

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformer_lens import HookedTransformerConfig

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.replacement_model.replacement_model_transformerlens import (
    TransformerLensReplacementModel,
)
from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
from circuit_tracer.transcoder.activation_functions import TopK
from circuit_tracer.utils import get_default_device
from tests.test_attributions_gemma import patch_tokenizer_special_ids


@pytest.fixture(autouse=True)
def cleanup():
    yield
    torch.cuda.empty_cache()
    gc.collect()


def tiny_llama() -> TransformerLensReplacementModel:
    cfg = HookedTransformerConfig.unwrap(
        {
            "n_layers": 2,
            "d_model": 8,
            "n_ctx": 64,
            "d_head": 4,
            "n_heads": 2,
            "d_mlp": 16,
            "act_fn": "silu",
            "d_vocab": 16,
            "d_vocab_out": 16,
            "eps": 1e-5,
            "model_name": "Llama-3.2-1B",
            "original_architecture": "LlamaForCausalLM",
            "tokenizer_name": "gpt2",
            "normalization_type": "RMSPre",
            "final_rms": True,
            "positional_embedding_type": "rotary",
            "rotary_dim": 4,
            "rotary_base": 500000.0,
            "gated_mlp": True,
            "n_key_value_heads": 2,
            "use_attn_scale": True,
            "attn_scale": np.float64(2.0),
            "init_weights": False,
            "dtype": torch.float32,
            "device": get_default_device(),
            "default_prepend_bos": True,
        }
    )
    torch.manual_seed(0)
    # No skip connection, so a feature's direct effect is exactly its decoder write.
    transcoders = {
        layer: SingleLayerTranscoder(
            cfg.d_model, cfg.d_model * 4, TopK(6), layer, skip_connection=False
        )
        for layer in range(cfg.n_layers)
    }
    for tc in transcoders.values():
        for _, param in tc.named_parameters():
            nn.init.uniform_(param, a=-1, b=1)
    model = ReplacementModel.from_config(
        cfg,
        TranscoderSet(
            transcoders, feature_input_hook="mlp.hook_in", feature_output_hook="mlp.hook_out"
        ),
    )
    for _, param in model.named_parameters():
        nn.init.uniform_(param, a=-1, b=1)
    assert isinstance(model, TransformerLensReplacementModel)
    return model


PROMPT = torch.tensor([0, 3, 4, 3, 2, 5, 3, 8])


def run(model, **kwargs):
    with patch_tokenizer_special_ids(model, [0]):
        return attribute(PROMPT, model, batch_size=64, **kwargs)


def logit_edges(graph):
    n_feat = graph.active_features.size(0)
    return graph.adjacency_matrix[-len(graph.logit_token_ids) :, :n_feat]


def test_defaults_are_unchanged_and_restored():
    model = tiny_llama()
    base = run(model)
    explicit = run(model, freeze_attention=True, freeze_layernorm=True)
    torch.testing.assert_close(explicit.adjacency_matrix, base.adjacency_matrix)
    run(model, freeze_attention=False, freeze_layernorm=False)
    after = run(model)  # an unfrozen run must not leak into later default runs
    torch.testing.assert_close(after.adjacency_matrix, base.adjacency_matrix)
    assert model.freeze_attention and model.freeze_layernorm


def test_unfreezing_changes_edges():
    model = tiny_llama()
    frozen = logit_edges(run(model))
    free = logit_edges(run(model, freeze_attention=False, freeze_layernorm=False))
    assert not torch.allclose(frozen, free, atol=1e-4)


@pytest.mark.parametrize("eps", [1e-3])
def test_unfrozen_logit_edges_are_first_order_effects(eps):
    """Edge(feature -> logit) must equal d(demeaned logit) / d(eps) for act -> (1 + eps) act,
    with downstream MLP outputs held at their clean values and attention / norms live."""
    model = tiny_llama()
    graph = run(model, freeze_attention=False, freeze_layernorm=False)
    edges = logit_edges(graph)
    tokens = graph.input_tokens.to(get_default_device())
    names = [f"blocks.{layer}.mlp.hook_out" for layer in range(model.cfg.n_layers)]
    with torch.no_grad():
        clean_logits, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names)
        _, acts = model.get_activations(tokens)
    assert isinstance(model.transcoders, TranscoderSet)
    clean = cast(torch.Tensor, clean_logits)[0, -1]
    targets = graph.logit_token_ids.to(get_default_device())

    for col, (layer, pos, feat) in enumerate(graph.active_features.tolist()):
        direction = model.transcoders[layer].W_dec[feat] * acts[layer, pos, feat] * eps

        def frozen_mlp(out, hook, layer_=layer):
            out = cache[hook.name].clone()
            if hook.name == f"blocks.{layer_}.mlp.hook_out":
                out[0, pos] += direction
            return out

        with torch.no_grad():
            new = model.run_with_hooks(tokens, fwd_hooks=[(n, frozen_mlp) for n in names])[0, -1]
        delta = (new[targets] - new.mean()) - (clean[targets] - clean.mean())
        torch.testing.assert_close(
            delta / eps, edges[:, col].to(delta.device), rtol=2e-2, atol=2e-3
        )


def test_nnsight_backend_flags():
    import copy

    from tests.test_attributions_llama_nnsight import llama_3_2_config, load_dummy_llama_model

    cfg = copy.deepcopy(llama_3_2_config)
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 2, 8, 16
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.vocab_size = 2, 2, 16
    cfg.torch_dtype = "float32"
    torch.manual_seed(0)
    model = load_dummy_llama_model(cfg, 4)  # type:ignore
    tokenizer_class = type(model.tokenizer)
    original = tokenizer_class.all_special_ids  # type:ignore
    try:
        tokenizer_class.all_special_ids = property(lambda self: [0])  # type:ignore
        base = attribute(PROMPT, model)
        explicit = attribute(PROMPT, model, freeze_attention=True, freeze_layernorm=True)
        free = attribute(PROMPT, model, freeze_attention=False, freeze_layernorm=False)
        after = attribute(PROMPT, model)
    finally:
        tokenizer_class.all_special_ids = original  # type:ignore
    torch.testing.assert_close(explicit.adjacency_matrix, base.adjacency_matrix)
    torch.testing.assert_close(after.adjacency_matrix, base.adjacency_matrix)
    assert not torch.allclose(logit_edges(free), logit_edges(base), atol=1e-4)
