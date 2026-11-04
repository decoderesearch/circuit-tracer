"""Token nodes carry the embedding the model actually uses.

In a frozen graph every write into the residual stream is a node, so a target's incoming edges
sum to the target's value once nothing is left unattributed. Here nothing is: no positions are
zeroed, the transcoders have no decoder bias, and the target is one raw logit. Gemma 3 scales its
embeddings by ``sqrt(d_model)`` inside the embedding module, so a token node built from the raw
embedding rows while its gradient is read at the module's output would be off by that factor, and
the edges would not sum to the logit.
"""

import copy
from typing import Any

import pytest
import torch

from circuit_tracer import attribute
from circuit_tracer.attribution.targets import CustomTarget
from tests.test_attributions_gemma3_nnsight import gemma_3_config, load_dummy_gemma_model

TOKENS = torch.tensor([0, 3, 4, 3, 2, 5, 3, 8])


def test_edges_into_a_logit_sum_to_it():
    cfg = copy.deepcopy(gemma_3_config)
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 8
    cfg.intermediate_size = 16
    cfg.head_dim = 4
    cfg.vocab_size = 16
    cfg.num_attention_heads = 2
    cfg.num_key_value_heads = 2
    cfg.final_logit_softcapping = None
    cfg.torch_dtype = "float32"
    torch.manual_seed(1)
    model = load_dummy_gemma_model(cfg)  # type: ignore[arg-type]
    hf: Any = model._module
    embed = hf.get_input_embeddings()
    assert float(getattr(embed, "embed_scale", 1.0)) > 1.0, (
        "the embedding is not scaled; the test would be vacuous"
    )
    with torch.no_grad():
        transcoders: Any = model.transcoders._module
        for transcoder in transcoders:  # nothing outside the nodes: no decoder bias
            transcoder.b_dec.zero_()
    model.zero_positions = slice(0, 0)
    tokenizer_class = type(model.tokenizer)
    original = tokenizer_class.all_special_ids  # type: ignore[attr-defined]
    tokenizer_class.all_special_ids = property(lambda self: [0])  # type: ignore[attr-defined]
    try:
        unembed = model.unembed_weight.detach()
        target = CustomTarget("logit_3", 1.0, unembed[3])
        graph = attribute(TOKENS, model, attribution_targets=[target], batch_size=16)
        logits, _ = model.get_activations(TOKENS)
    finally:
        tokenizer_class.all_special_ids = original  # type: ignore[attr-defined]
    n_sources = graph.adjacency_matrix.shape[1] - 1
    edges = graph.adjacency_matrix[-1, :n_sources].double()
    token_edges = edges[-len(TOKENS) :]
    assert token_edges.abs().max() > 1e-3 * edges.abs().max(), (
        "token nodes carry nothing; the test is vacuous"
    )
    assert float(edges.sum()) == pytest.approx(float(logits[0, -1, 3]), rel=1e-4, abs=1e-4)
