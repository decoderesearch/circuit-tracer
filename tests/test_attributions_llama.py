import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import device
from transformer_lens import HookedTransformerConfig

from circuit_tracer.attribution import attribute
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.transcoder import SingleLayerTranscoder
from circuit_tracer.transcoder.activation_functions import TopK
from circuit_tracer.utils.device import get_default_device

sys.path.append(os.path.dirname(__file__))
from test_attributions_gemma import verify_feature_edges, verify_token_and_error_edges


def load_dummy_llama_model(cfg: HookedTransformerConfig, k: int):
    transcoders = {
        layer_idx: SingleLayerTranscoder(
            cfg.d_model, cfg.d_model * 4, TopK(k), layer_idx, skip_connection=True
        )
        for layer_idx in range(cfg.n_layers)
    }
    for transcoder in transcoders.values():
        for _, param in transcoder.named_parameters():
            nn.init.uniform_(param, a=-1, b=1)

    model = ReplacementModel.from_config(cfg, transcoders)
    model.tokenizer.bos_token_id = None
    for _, param in model.named_parameters():
        nn.init.uniform_(param, a=-1, b=1)

    return model


def verify_small_llama_model(s: torch.Tensor):
    llama_small_cfg = {
        "n_layers": 2,
        "d_model": 8,
        "n_ctx": 2048,
        "d_head": 4,
        "model_name": "Llama-3.2-1B",
        "n_heads": 2,
        "d_mlp": 16,
        "act_fn": "silu",
        "d_vocab": 16,
        "eps": 1e-05,
        "use_attn_result": False,
        "use_attn_scale": True,
        "attn_scale": np.float64(8.0),
        "use_split_qkv_input": False,
        "use_hook_mlp_in": False,
        "use_attn_in": False,
        "use_local_attn": False,
        "ungroup_grouped_query_attention": False,
        "original_architecture": "LlamaForCausalLM",
        "from_checkpoint": False,
        "checkpoint_index": None,
        "checkpoint_label_type": None,
        "checkpoint_value": None,
        "tokenizer_name": "meta-llama/Llama-3.2-1B",
        "window_size": None,
        "attn_types": None,
        "init_mode": "gpt2",
        "normalization_type": "RMSPre",
        "device": get_default_device(),
        "n_devices": 1,
        "attention_dir": "causal",
        "attn_only": False,
        "seed": None,
        "initializer_range": np.float64(0.017677669529663688),
        "init_weights": False,
        "scale_attn_by_inverse_layer_idx": False,
        "positional_embedding_type": "rotary",
        "final_rms": True,
        "d_vocab_out": 16,
        "parallel_attn_mlp": False,
        "rotary_dim": 4,
        "n_params": 1073741824,
        "use_hook_tokens": False,
        "gated_mlp": True,
        "default_prepend_bos": True,
        "dtype": torch.float32,
        "tokenizer_prepends_bos": True,
        "n_key_value_heads": 2,
        "post_embedding_ln": False,
        "rotary_base": 500000.0,
        "trust_remote_code": False,
        "rotary_adjacent_pairs": False,
        "load_in_4bit": False,
        "num_experts": None,
        "experts_per_token": None,
        "relative_attention_max_distance": None,
        "relative_attention_num_buckets": None,
        "decoder_start_token_id": None,
        "tie_word_embeddings": False,
        "use_normalization_before_and_after": False,
        "attn_scores_soft_cap": -1.0,
        "output_logits_soft_cap": -1.0,
        "use_NTK_by_parts_rope": True,
        "NTK_by_parts_low_freq_factor": 1.0,
        "NTK_by_parts_high_freq_factor": 4.0,
        "NTK_by_parts_factor": 32.0,
    }

    cfg = HookedTransformerConfig.from_dict(llama_small_cfg)
    k = 4
    model = load_dummy_llama_model(cfg, k)
    graph = attribute(s, model)

    verify_token_and_error_edges(model, graph, delete_bos=False)
    verify_feature_edges(model, graph)


def verify_large_llama_model(s: torch.Tensor):
    llama_large_cfg = {
        "n_layers": 8,
        "d_model": 128,
        "n_ctx": 2048,
        "d_head": 32,
        "model_name": "Llama-3.2-1B",
        "n_heads": 4,
        "d_mlp": 512,
        "act_fn": "silu",
        "d_vocab": 128,
        "eps": 1e-05,
        "use_attn_result": False,
        "use_attn_scale": True,
        "attn_scale": np.float64(8.0),
        "use_split_qkv_input": False,
        "use_hook_mlp_in": False,
        "use_attn_in": False,
        "use_local_attn": False,
        "ungroup_grouped_query_attention": False,
        "original_architecture": "LlamaForCausalLM",
        "from_checkpoint": False,
        "checkpoint_index": None,
        "checkpoint_label_type": None,
        "checkpoint_value": None,
        "tokenizer_name": "meta-llama/Llama-3.2-1B",
        "window_size": None,
        "attn_types": None,
        "init_mode": "gpt2",
        "normalization_type": "RMSPre",
        "device": get_default_device(),
        "n_devices": 1,
        "attention_dir": "causal",
        "attn_only": False,
        "seed": None,
        "initializer_range": np.float64(0.017677669529663688),
        "init_weights": False,
        "scale_attn_by_inverse_layer_idx": False,
        "positional_embedding_type": "rotary",
        "final_rms": True,
        "d_vocab_out": 128,
        "parallel_attn_mlp": False,
        "rotary_dim": 32,
        "n_params": 1073741824,
        "use_hook_tokens": False,
        "gated_mlp": True,
        "default_prepend_bos": True,
        "dtype": torch.float32,
        "tokenizer_prepends_bos": True,
        "n_key_value_heads": 4,
        "post_embedding_ln": False,
        "rotary_base": 500000.0,
        "trust_remote_code": False,
        "rotary_adjacent_pairs": False,
        "load_in_4bit": False,
        "num_experts": None,
        "experts_per_token": None,
        "relative_attention_max_distance": None,
        "relative_attention_num_buckets": None,
        "decoder_start_token_id": None,
        "tie_word_embeddings": False,
        "use_normalization_before_and_after": False,
        "attn_scores_soft_cap": -1.0,
        "output_logits_soft_cap": -1.0,
        "use_NTK_by_parts_rope": True,
        "NTK_by_parts_low_freq_factor": 1.0,
        "NTK_by_parts_high_freq_factor": 4.0,
        "NTK_by_parts_factor": 32.0,
    }
    cfg = HookedTransformerConfig.from_dict(llama_large_cfg)
    k = 16
    model = load_dummy_llama_model(cfg, k)
    graph = attribute(s, model)

    verify_token_and_error_edges(model, graph, delete_bos=False)
    verify_feature_edges(model, graph)


def verify_llama_3_2_1b(s: str):
    model = ReplacementModel.from_pretrained("meta-llama/Llama-3.2-1B", "llama")
    graph = attribute(s, model, batch_size=128)

    verify_token_and_error_edges(model, graph, delete_bos=True)
    verify_feature_edges(model, graph)


def test_small_llama_model():
    s = torch.tensor([10, 3, 4, 3, 2, 5, 3, 8])
    verify_small_llama_model(s)


def test_large_llama_model():
    s = torch.tensor([0, 113, 24, 53, 27])
    verify_large_llama_model(s)


def test_llama_3_2_1b():
    s = "The National Digital Analytics Group (ND"
    verify_llama_3_2_1b(s)


if __name__ == "__main__":
    torch.manual_seed(42)
    test_small_llama_model()
    test_large_llama_model()
    test_llama_3_2_1b()
