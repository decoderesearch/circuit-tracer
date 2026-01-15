import gc
from functools import partial

import pytest
import torch
import torch.nn as nn
from nnsight import save
from tqdm import tqdm
from transformers import AutoConfig, Gemma2Config

from circuit_tracer import Graph, ReplacementModel
from circuit_tracer.attribution.attribute_nnsight import attribute
from circuit_tracer.replacement_model.replacement_model_nnsight import (
    NNSightReplacementModel,
)
from circuit_tracer.replacement_model.replacement_model_transformerlens import (
    TransformerLensReplacementModel,
)
from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
from circuit_tracer.transcoder.activation_functions import JumpReLU

gemma_2_config_dict = {
    "architectures": ["Gemma2ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_logit_softcapping": 50.0,
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "eos_token_id": 1,
    "final_logit_softcapping": 30.0,
    "head_dim": 256,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 2304,
    "initializer_range": 0.02,
    "intermediate_size": 9216,
    "max_position_embeddings": 8192,
    "model_type": "gemma2",
    "num_attention_heads": 8,
    "num_hidden_layers": 26,
    "num_key_value_heads": 4,
    "pad_token_id": 0,
    "query_pre_attn_scalar": 256,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000.0,
    "sliding_window": 4096,
    "torch_dtype": "float32",
    "transformers_version": "4.42.4",
    "use_cache": True,
    "vocab_size": 256000,
    "_name_or_path": "openai-community/gpt2",
}

gemma_2_config = Gemma2Config.from_dict(gemma_2_config_dict)


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


def verify_token_and_error_edges(
    model: NNSightReplacementModel,
    graph: Graph,
    act_atol=1e-3,
    act_rtol=1e-3,
    logit_atol=1e-5,
    logit_rtol=1e-3,
):
    s = graph.input_tokens
    adjacency_matrix = graph.adjacency_matrix.to(device=model.device, dtype=model.dtype)
    active_features = graph.active_features.to(device=model.device)
    logit_tokens = graph.logit_tokens.to(device=model.device)
    total_active_features = active_features.size(0)
    pos_start = 1  # skip first position (BOS token)

    ctx = model.setup_attribution(s)

    error_vectors = ctx.error_vectors
    token_vectors = ctx.token_vectors

    logits, activation_cache = model.get_activations(s, apply_activation_function=False)
    logits = logits.squeeze(0)

    relevant_activations = activation_cache[
        active_features[:, 0], active_features[:, 1], active_features[:, 2]
    ]
    relevant_logits = logits[-1, logit_tokens]
    demeaned_relevant_logits = relevant_logits - logits[-1].mean()

    def verify_intervention(expected_effects, intervention, target_layer):
        _, freeze_fns = model.setup_intervention_with_freeze(
            s,
            constrained_layers=range(model.cfg.n_layers),  # type: ignore
        )
        _, activation_fn = model.get_activation_fn(apply_activation_function=False)

        with model.trace() as tracer:
            with tracer.invoke(s):
                pass

            direct_effects_barrier = tracer.barrier(2)

            with tracer.invoke():
                _, new_activation_cache = activation_fn()  # type:ignore
                new_activation_cache = save(new_activation_cache)  # type: ignore

            for i, freeze_fn in enumerate(freeze_fns):
                with tracer.invoke():
                    freeze_fn(direct_effects_barrier=direct_effects_barrier)

            with tracer.invoke():
                if "embed" == target_layer:
                    intervention(model.embed_location)

                for layer, feature_output_loc in enumerate(model.feature_output_locs):
                    direct_effects_barrier()
                    if layer == target_layer:
                        intervention(feature_output_loc)

                new_logits = save(model.output.logits.squeeze(0))  # type: ignore

        new_relevant_activations = new_activation_cache[
            active_features[:, 0], active_features[:, 1], active_features[:, 2]
        ]
        new_relevant_logits = new_logits[-1, logit_tokens]
        new_demeaned_relevant_logits = new_relevant_logits - new_logits[-1].mean()

        expected_activation_difference = expected_effects[:total_active_features]
        expected_logit_difference = expected_effects[-len(logit_tokens) :]

        assert torch.allclose(
            new_relevant_activations,
            relevant_activations + expected_activation_difference,
            atol=act_atol,
            rtol=act_rtol,
        )
        assert torch.allclose(
            new_demeaned_relevant_logits,
            demeaned_relevant_logits + expected_logit_difference,
            atol=logit_atol,
            rtol=logit_rtol,
        )

    for error_node_layer in range(error_vectors.size(0)):
        for error_node_pos in range(pos_start, error_vectors.size(1)):
            error_node_index = error_node_layer * error_vectors.size(1) + error_node_pos
            expected_effects = adjacency_matrix[:, total_active_features + error_node_index]

            def error_intervention(feature_output_loc, error_node_layer, error_node_pos):
                activations = feature_output_loc.output
                steering_vector = torch.zeros_like(activations)
                steering_vector[:, error_node_pos] += error_vectors[
                    error_node_layer, error_node_pos
                ]
                feature_output_loc.output = activations + steering_vector

            intervention = partial(
                error_intervention,
                error_node_layer=error_node_layer,
                error_node_pos=error_node_pos,
            )
            verify_intervention(expected_effects, intervention, error_node_layer)

    total_error_nodes = error_vectors.size(0) * error_vectors.size(1)
    for token_pos in range(pos_start, token_vectors.size(0)):
        expected_effects = adjacency_matrix[
            :, total_active_features + total_error_nodes + token_pos
        ]

        def token_intervention(token_loc, token_pos):
            activations = token_loc.output
            steering_vector = torch.zeros_like(activations)
            steering_vector[:, token_pos] += token_vectors[token_pos]
            token_loc.output = activations + steering_vector

        intervention = partial(token_intervention, token_pos=token_pos)  # type: ignore
        verify_intervention(expected_effects, intervention, "embed")


def verify_feature_edges(
    model: NNSightReplacementModel | TransformerLensReplacementModel,
    graph: Graph,
    n_samples: int = 100,
    act_atol=5e-4,
    act_rtol=1e-5,
    logit_atol=1e-5,
    logit_rtol=1e-3,
):
    s = graph.input_tokens
    adjacency_matrix = graph.adjacency_matrix.to(device=model.device, dtype=model.dtype)  # type:ignore
    active_features = graph.active_features.to(device=model.device)  # type:ignore
    logit_tokens = graph.logit_tokens.to(device=model.device)  # type:ignore
    total_active_features = active_features.size(0)

    logits, activation_cache = model.get_activations(s, apply_activation_function=False)
    logits = logits.squeeze(0)

    relevant_activations = activation_cache[
        active_features[:, 0], active_features[:, 1], active_features[:, 2]
    ]
    relevant_logits = logits[-1, logit_tokens]
    demeaned_relevant_logits = relevant_logits - logits[-1].mean()

    def verify_intervention(
        expected_effects, layer: int, pos: int, feature_idx: int, new_activation
    ):
        new_logits, new_activation_cache = model.feature_intervention(
            s,
            [(layer, pos, feature_idx, new_activation)],
            constrained_layers=range(model.cfg.n_layers),  # type: ignore
            apply_activation_function=False,
        )
        new_logits = new_logits.squeeze(0)

        new_relevant_activations = new_activation_cache[  # type:ignore
            active_features[:, 0], active_features[:, 1], active_features[:, 2]
        ]  # type:ignore
        new_relevant_logits = new_logits[-1, logit_tokens]
        new_demeaned_relevant_logits = new_relevant_logits - new_logits[-1].mean()

        expected_activation_difference = expected_effects[:total_active_features]
        expected_logit_difference = expected_effects[-len(logit_tokens) :]

        assert torch.allclose(
            new_relevant_activations,
            relevant_activations + expected_activation_difference,
            atol=act_atol,
            rtol=act_rtol,
        )
        assert torch.allclose(
            new_demeaned_relevant_logits,
            demeaned_relevant_logits + expected_logit_difference,
            atol=logit_atol,
            rtol=logit_rtol,
        )

    random_order = torch.randperm(active_features.size(0))
    chosen_nodes = random_order[:n_samples]
    for chosen_node in tqdm(chosen_nodes):
        layer, pos, feature_idx = active_features[chosen_node]
        old_activation = activation_cache[layer, pos, feature_idx]
        new_activation = old_activation * 2
        expected_effects = adjacency_matrix[:, chosen_node]
        verify_intervention(expected_effects, layer, pos, feature_idx, new_activation)  # type: ignore


def load_dummy_gemma_model(cfg: AutoConfig) -> NNSightReplacementModel:
    transcoders = {
        layer_idx: SingleLayerTranscoder(
            cfg.hidden_size,  # type: ignore
            cfg.hidden_size * 4,  # type: ignore
            JumpReLU(torch.tensor(0.0), 0.1),
            layer_idx,
        )
        for layer_idx in range(cfg.num_hidden_layers)  # type: ignore
    }
    for transcoder in transcoders.values():
        for _, param in transcoder.named_parameters():
            nn.init.uniform_(param, a=-1, b=1)

    transcoder_set = TranscoderSet(
        transcoders, feature_input_hook="hook_resid_mid", feature_output_hook="hook_mlp_out"
    )

    model = ReplacementModel.from_config(cfg, transcoder_set, backend="nnsight")

    for _, param in model.named_parameters():
        nn.init.uniform_(param, a=-1, b=1)

    for transcoder in model.transcoders[0]:  # type: ignore
        nn.init.uniform_(transcoder.activation_function.threshold, a=0, b=1)

    model.tokenizer.pad_token = model.tokenizer.eos_token  # type:ignore

    assert isinstance(model, NNSightReplacementModel)
    return model


def test_small_gemma_model():
    s = torch.tensor([0, 3, 4, 3, 2, 5, 3, 8])
    gemma_small_config = gemma_2_config
    gemma_small_config.num_hidden_layers = 2
    gemma_small_config.hidden_size = 8
    gemma_small_config.intermediate_size = 16
    gemma_small_config.head_dim = 4
    gemma_small_config.vocab_size = 16
    gemma_small_config.num_attention_heads = 2
    gemma_small_config.num_key_value_heads = 2
    gemma_small_config.final_logit_softcapping = None  # type: ignore
    gemma_small_config.torch_dtype = "float32"
    model = load_dummy_gemma_model(gemma_small_config)  # type: ignore

    # Save original property to restore later
    tokenizer_class = type(model.tokenizer)
    original_all_special_ids = tokenizer_class.all_special_ids  # type:ignore
    try:
        tokenizer_class.all_special_ids = property(lambda self: [0])  # type:ignore
        graph = attribute(s, model)

        verify_token_and_error_edges(model, graph)
        verify_feature_edges(model, graph)
    finally:
        # Restore original property
        tokenizer_class.all_special_ids = original_all_special_ids  # type:ignore


def test_large_gemma_model():
    s = torch.tensor([0, 113, 24, 53, 27])
    gemma_large_cfg = gemma_2_config
    gemma_large_cfg.num_hidden_layers = 16
    gemma_large_cfg.hidden_size = 64
    gemma_large_cfg.intermediate_size = 128
    gemma_large_cfg.head_dim = 32
    gemma_large_cfg.vocab_size = 128
    gemma_large_cfg.num_attention_heads = 16
    gemma_large_cfg.num_key_value_heads = 16
    gemma_large_cfg.final_logit_softcapping = None  # type:ignore
    gemma_large_cfg.torch_dtype = "float64"
    model = load_dummy_gemma_model(gemma_large_cfg)  # type: ignore

    # Save original property to restore later
    tokenizer_class = type(model.tokenizer)
    original_all_special_ids = tokenizer_class.all_special_ids  # type:ignore
    try:
        tokenizer_class.all_special_ids = property(lambda self: [0])  # type:ignore
        graph = attribute(s, model)

        verify_token_and_error_edges(model, graph)
        verify_feature_edges(model, graph)
    finally:
        # Restore original property
        tokenizer_class.all_special_ids = original_all_special_ids  # type:ignore


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gemma_2_2b():
    s = "The National Digital Analytics Group (ND"
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", backend="nnsight")
    assert isinstance(model, NNSightReplacementModel)

    graph = attribute(s, model, batch_size=256)

    print("Changing logit softcap to 0, as the logits will otherwise be off.")
    with model.zero_softcap():
        verify_token_and_error_edges(model, graph)
        verify_feature_edges(model, graph)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gemma_2_2b_clt():
    s = "The National Digital Analytics Group (ND"
    model = ReplacementModel.from_pretrained(
        "google/gemma-2-2b", "mntss/clt-gemma-2-2b-426k", backend="nnsight"
    )

    assert isinstance(model, NNSightReplacementModel)

    graph = attribute(s, model, batch_size=256)

    print("Changing logit softcap to 0, as the logits will otherwise be off.")
    with model.zero_softcap():
        verify_token_and_error_edges(model, graph)
        verify_feature_edges(model, graph)


if __name__ == "__main__":
    torch.manual_seed(42)
    test_small_gemma_model()
    test_large_gemma_model()
    test_gemma_2_2b()
    test_gemma_2_2b_clt()  # This will pass, but is slow to run
