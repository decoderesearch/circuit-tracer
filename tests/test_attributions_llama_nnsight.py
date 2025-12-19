import os
import sys

import torch
import torch.nn as nn
from transformers import AutoConfig
import pytest

from circuit_tracer import attribute, ReplacementModel
from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
from circuit_tracer.transcoder.activation_functions import TopK

sys.path.append(os.path.dirname(__file__))
from test_attributions_gemma_nnsight import verify_feature_edges, verify_token_and_error_edges


def load_dummy_llama_model(cfg: AutoConfig, k: int):
    transcoders = {
        layer_idx: SingleLayerTranscoder(
            cfg.hidden_size,  # type:ignore
            cfg.hidden_size * 4,  # type:ignore
            TopK(k),
            layer_idx,
            skip_connection=True,
        )
        for layer_idx in range(cfg.num_hidden_layers)  # type:ignore
    }
    for transcoder in transcoders.values():
        for _, param in transcoder.named_parameters():
            nn.init.uniform_(param, a=-1, b=1)

    transcoder_set = TranscoderSet(
        transcoders, feature_input_hook="mlp.hook_in", feature_output_hook="mlp.hook_out"
    )

    model = ReplacementModel.from_config(cfg, transcoder_set, backend="nnsight")

    model.tokenizer.pad_token = model.tokenizer.eos_token  # type:ignore

    for _, param in model.named_parameters():
        nn.init.uniform_(param, a=-1, b=1)

    return model


def test_small_llama_model():
    s = torch.tensor([0, 3, 4, 3, 2, 5, 3, 8])
    llama_small_config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
    llama_small_config.num_hidden_layers = 2
    llama_small_config.hidden_size = 8
    llama_small_config.intermediate_size = 16
    llama_small_config.num_attention_heads = 2
    llama_small_config.num_key_value_heads = 2
    llama_small_config.vocab_size = 16
    llama_small_config.max_position_embeddings = 2048
    llama_small_config.torch_dtype = "float32"

    k = 4
    model = load_dummy_llama_model(llama_small_config, k)  # type:ignore

    # Save original property to restore later
    tokenizer_class = type(model.tokenizer)
    original_all_special_ids = tokenizer_class.all_special_ids  # type:ignore
    try:
        tokenizer_class.all_special_ids = property(lambda self: [0])  # type:ignore
        graph = attribute(s, model)
        assert isinstance(model, NNSightReplacementModel)

        verify_token_and_error_edges(model, graph)
        verify_feature_edges(model, graph)
    finally:
        # Restore original property
        tokenizer_class.all_special_ids = original_all_special_ids  # type:ignore


def test_large_llama_model():
    s = torch.tensor([0, 113, 24, 53, 27])
    llama_large_config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
    llama_large_config.num_hidden_layers = 8
    llama_large_config.hidden_size = 128
    llama_large_config.intermediate_size = 512
    llama_large_config.num_attention_heads = 4
    llama_large_config.num_key_value_heads = 4
    llama_large_config.vocab_size = 256
    llama_large_config.max_position_embeddings = 2048
    llama_large_config.torch_dtype = "float32"

    k = 16
    model = load_dummy_llama_model(llama_large_config, k)  # type:ignore

    # Save original property to restore later
    tokenizer_class = type(model.tokenizer)
    original_all_special_ids = tokenizer_class.all_special_ids  # type:ignore
    try:
        tokenizer_class.all_special_ids = property(lambda self: [0])  # type:ignore
        graph = attribute(s, model)
        assert isinstance(model, NNSightReplacementModel)

        verify_token_and_error_edges(model, graph)
        verify_feature_edges(model, graph)
    finally:
        # Restore original property
        tokenizer_class.all_special_ids = original_all_special_ids  # type:ignore


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_llama_3_2_1b():
    s = "The National Digital Analytics Group (ND"
    model = ReplacementModel.from_pretrained(
        "meta-llama/Llama-3.2-1B", "llama", backend="nnsight", device=torch.device("cuda")
    )
    graph = attribute(s, model, batch_size=128)
    assert isinstance(model, NNSightReplacementModel)

    verify_token_and_error_edges(model, graph)
    verify_feature_edges(model, graph)


if __name__ == "__main__":
    torch.manual_seed(42)
    test_small_llama_model()
    test_large_llama_model()
    test_llama_3_2_1b()
