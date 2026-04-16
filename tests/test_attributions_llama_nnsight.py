import gc
import os
import sys

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, LlamaConfig

from circuit_tracer import attribute, ReplacementModel
from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
from circuit_tracer.transcoder.activation_functions import TopK
from tests.conftest import has_32gb

sys.path.append(os.path.dirname(__file__))
from test_attributions_gemma_nnsight import verify_feature_edges, verify_token_and_error_edges

llama_3_2_config_dict = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 8192,
    "max_position_embeddings": 131072,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 16,
    "num_key_value_heads": 8,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.45.0.dev0",
    "use_cache": True,
    "vocab_size": 128256,
    "_name_or_path": "openai-community/gpt2",
}

llama_3_2_config = LlamaConfig.from_dict(llama_3_2_config_dict)


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


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
    llama_small_config = llama_3_2_config
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
    llama_large_config = llama_3_2_config
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


@pytest.mark.skipif(not has_32gb, reason="Requires >=32GB VRAM")
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
