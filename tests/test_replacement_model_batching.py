import gc

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformer_lens import HookedTransformerConfig

from circuit_tracer import ReplacementModel
from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
from circuit_tracer.transcoder.activation_functions import TopK


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


def load_dummy_llama_replacement_model():
    cfg = HookedTransformerConfig.from_dict(
        {
            "n_layers": 2,
            "d_model": 32,
            "n_ctx": 32,
            "d_head": 8,
            "model_name": "Llama-3.2-1B",
            "n_heads": 4,
            "d_mlp": 64,
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
            "tokenizer_name": "gpt2",
            "window_size": None,
            "attn_types": None,
            "init_mode": "gpt2",
            "normalization_type": "RMSPre",
            "device": "cpu",
            "n_devices": 1,
            "attention_dir": "causal",
            "attn_only": False,
            "seed": 42,
            "initializer_range": np.float64(0.02),
            "init_weights": True,
            "scale_attn_by_inverse_layer_idx": False,
            "positional_embedding_type": "rotary",
            "final_rms": True,
            "d_vocab_out": 128,
            "parallel_attn_mlp": False,
            "rotary_dim": 8,
            "n_params": 123456,
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
    )

    transcoders = {
        layer_idx: SingleLayerTranscoder(
            cfg.d_model, cfg.d_model * 2, TopK(8), layer_idx, skip_connection=True
        )
        for layer_idx in range(cfg.n_layers)
    }
    for transcoder in transcoders.values():
        for _, param in transcoder.named_parameters():
            nn.init.uniform_(param, a=-0.1, b=0.1)

    model = ReplacementModel.from_config(
        cfg,
        TranscoderSet(
            transcoders,
            feature_input_hook="mlp.hook_in",
            feature_output_hook="mlp.hook_out",
        ),
    )
    return model


def test_get_activations_batch_matches_single_tensor_runs():
    model = load_dummy_llama_replacement_model()
    batched_tokens = torch.tensor([[1, 2, 3, 4], [1, 5, 6, 7]], dtype=torch.long)

    batched_logits, batched_activations = model.get_activations(batched_tokens)

    for batch_idx in range(batched_tokens.size(0)):
        single_logits, single_activations = model.get_activations(
            batched_tokens[batch_idx : batch_idx + 1]
        )

        assert torch.allclose(
            batched_logits[batch_idx], single_logits[0], atol=1e-6, rtol=1e-5
        ), f"Logits diverged for sample {batch_idx}"
        assert torch.allclose(
            batched_activations[:, batch_idx],
            single_activations,
            atol=1e-6,
            rtol=1e-5,
        ), f"Activations diverged for sample {batch_idx}"
