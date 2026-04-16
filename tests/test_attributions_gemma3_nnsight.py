import copy
import gc
from functools import partial

import pytest
import torch
import torch.nn as nn
from nnsight import save
from tqdm import tqdm
from transformers import AutoConfig, Gemma3TextConfig

from circuit_tracer import attribute, Graph, ReplacementModel
from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
from circuit_tracer.transcoder.activation_functions import JumpReLU
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
from tests.conftest import has_32gb

gemma_3_config_dict = {
    "_sliding_window_pattern": 6,
    "architectures": ["Gemma3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_logit_softcapping": None,
    "bos_token_id": 2,
    "eos_token_id": 1,
    "final_logit_softcapping": None,
    "head_dim": 256,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 640,
    "initializer_range": 0.02,
    "intermediate_size": 2048,
    "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
    "max_position_embeddings": 32768,
    "model_type": "gemma3_text",
    "num_attention_heads": 4,
    "num_hidden_layers": 18,
    "num_key_value_heads": 1,
    "pad_token_id": 0,
    "query_pre_attn_scalar": 256,
    "rms_norm_eps": 1e-06,
    "rope_local_base_freq": 10000.0,
    "rope_scaling": None,
    "rope_theta": 1000000.0,
    "sliding_window": 512,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.55.0.dev0",
    "use_bidirectional_attention": False,
    "use_cache": True,
    "vocab_size": 262144,
    "_name_or_path": "openai-community/gpt2",
}

gemma_3_config = Gemma3TextConfig.from_dict(gemma_3_config_dict)


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


def initialize_transcoder_weights(W_enc, W_dec, b_enc, b_dec):
    """Initialize transcoder weights with He uniform and unit norm normalization.

    Args:
        W_enc: Encoder weight tensor (any shape ending in [..., d_transcoder, d_model])
        W_dec: Decoder weight tensor or list of tensors
        b_enc: Encoder bias tensor
        b_dec: Decoder bias tensor
    """
    with torch.no_grad():
        nn.init.kaiming_uniform_(W_enc, mode="fan_in", nonlinearity="linear")
        W_enc.data = W_enc.data / W_enc.data.norm(dim=-1, keepdim=True)

        if isinstance(W_dec, torch.nn.ParameterList):
            for dec in W_dec:
                nn.init.kaiming_uniform_(dec, mode="fan_in", nonlinearity="linear")
                dec.data = dec.data / dec.data.norm(dim=0, keepdim=True)
                dec.data *= 0.01
        else:
            nn.init.kaiming_uniform_(W_dec, mode="fan_in", nonlinearity="linear")
            W_dec.data = W_dec.data / W_dec.data.norm(dim=0, keepdim=True)
            W_dec.data *= 0.01

        nn.init.uniform_(b_enc, a=-0.5, b=0.0)
        nn.init.uniform_(b_dec, a=-0.01, b=0.01)


def set_l0_via_thresholds(activations, threshold_tensor, target_l0=16):
    """Set activation function thresholds to achieve target L0 sparsity.

    Args:
        activations: Tensor of shape [n_layers, n_positions, n_features]
        threshold_tensor: Tensor to update with thresholds
        target_l0: Target number of active features per position
    """
    n_layers, n_positions, n_features = activations.shape
    device = activations.device

    for layer_idx in range(n_layers):
        layer_acts = activations[layer_idx]
        nonzero_features = (layer_acts != 0).any(dim=0).nonzero(as_tuple=True)[0]

        n_keep = min(len(nonzero_features), target_l0 * n_positions)
        chosen_indices = nonzero_features[
            torch.randperm(len(nonzero_features), device=device)[:n_keep]
        ]

        chosen_set = torch.zeros(n_features, dtype=torch.bool, device=device)
        if len(chosen_indices) > 0:
            chosen_set[chosen_indices] = True

        min_acts = layer_acts.min(dim=0)[0]
        max_acts = layer_acts.max(dim=0)[0]
        thresholds = torch.where(chosen_set, torch.clamp(min_acts - 1, min=0), max_acts + 1)

        if threshold_tensor.dim() == 1:
            threshold_tensor.data = thresholds
        elif threshold_tensor.dim() == 2:
            threshold_tensor.data[layer_idx] = thresholds
        elif threshold_tensor.dim() == 3:
            threshold_tensor.data[layer_idx, 0] = thresholds


def verify_token_and_error_edges(
    model: NNSightReplacementModel,
    graph: Graph,
    act_atol=1e-3,
    act_rtol=1e-3,
    logit_atol=1e-5,
    logit_rtol=1e-3,
    pos_start=1,
):
    s = graph.input_tokens
    adjacency_matrix = graph.adjacency_matrix.to(device=model.device, dtype=model.dtype)
    active_features = graph.active_features.to(device=model.device)
    logit_tokens = graph.logit_tokens.to(device=model.device)
    total_active_features = active_features.size(0)

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
    model: NNSightReplacementModel,
    graph: Graph,
    n_samples: int = 100,
    act_atol=1e-3,  # dummy transcoder gemma3 tests need slightly higher tolerance
    act_rtol=1e-5,
    logit_atol=1e-5,
    logit_rtol=1e-3,
):
    s = graph.input_tokens
    adjacency_matrix = graph.adjacency_matrix.to(device=model.device, dtype=model.dtype)
    active_features = graph.active_features.to(device=model.device)
    logit_tokens = graph.logit_tokens.to(device=model.device)
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

    random_order = torch.randperm(active_features.size(0))
    chosen_nodes = random_order[:n_samples]
    for chosen_node in tqdm(chosen_nodes):
        layer, pos, feature_idx = active_features[chosen_node]
        old_activation = activation_cache[layer, pos, feature_idx]
        new_activation = old_activation * 2
        expected_effects = adjacency_matrix[:, chosen_node]
        verify_intervention(expected_effects, layer, pos, feature_idx, new_activation)  # type:ignore


def load_dummy_gemma_model(cfg: AutoConfig):
    transcoders = {
        layer_idx: SingleLayerTranscoder(
            cfg.hidden_size,  # type:ignore
            cfg.hidden_size * 4,  # type:ignore
            JumpReLU(0.0, 0.1),
            layer_idx,
        )
        for layer_idx in range(cfg.num_hidden_layers)  # type:ignore
    }
    for transcoder in transcoders.values():
        for _, param in transcoder.named_parameters():
            nn.init.uniform_(param, a=-1, b=1)

    transcoder_set = TranscoderSet(
        transcoders, feature_input_hook="hook_resid_mid", feature_output_hook="hook_mlp_out"
    )

    model = ReplacementModel.from_config(cfg, transcoder_set, backend="nnsight")

    model.tokenizer.pad_token = model.tokenizer.eos_token  # type:ignore

    for _, param in model.named_parameters():
        nn.init.uniform_(param, a=-1, b=1)

    for transcoder in model.transcoders[0]:  # type:ignore
        nn.init.uniform_(transcoder.activation_function.threshold, a=0, b=1)

    return model


def load_gemma3_with_dummy_transcoders():
    cfg = AutoConfig.from_pretrained("google/gemma-3-270m")

    transcoders = {
        layer_idx: SingleLayerTranscoder(
            cfg.hidden_size,
            cfg.hidden_size * 4,
            JumpReLU(torch.zeros(cfg.hidden_size * 4), 0.1),
            layer_idx,
        )
        for layer_idx in range(cfg.num_hidden_layers)
    }
    for transcoder in transcoders.values():
        initialize_transcoder_weights(
            transcoder.W_enc, transcoder.W_dec, transcoder.b_enc, transcoder.b_dec
        )

    transcoder_set = TranscoderSet(
        transcoders, feature_input_hook="hook_resid_mid", feature_output_hook="hook_mlp_out"
    )

    model = ReplacementModel.from_pretrained_and_transcoders(
        "google/gemma-3-270m", transcoder_set, backend="nnsight"
    )

    _, activations = model.get_activations("The National Digital Analytics Group (ND")

    for layer_idx, transcoder in enumerate(model.transcoders[0]):  # type:ignore
        layer_acts = activations[layer_idx : layer_idx + 1]
        set_l0_via_thresholds(layer_acts, transcoder.activation_function.threshold, target_l0=16)

    return model


def load_gemma3_with_dummy_clt():
    cfg = gemma_3_config

    clt = CrossLayerTranscoder(
        n_layers=cfg.num_hidden_layers,
        d_transcoder=cfg.hidden_size * 4,
        d_model=cfg.hidden_size,
        activation_function="jump_relu",
        lazy_decoder=False,
        lazy_encoder=False,
        feature_input_hook="hook_resid_mid",
        feature_output_hook="hook_mlp_out",
        dtype=torch.float32,
    )

    initialize_transcoder_weights(clt.W_enc, clt.W_dec, clt.b_enc, clt.b_dec)

    model = ReplacementModel.from_pretrained_and_transcoders(
        "google/gemma-3-270m", clt, backend="nnsight"
    )

    _, activations = model.get_activations("The National Digital Analytics Group (ND")
    set_l0_via_thresholds(activations, clt.activation_function.threshold, target_l0=16)

    return model


def test_small_gemma_model():
    s = torch.tensor([0, 3, 4, 3, 2, 5, 3, 8])
    gemma_small_config = copy.deepcopy(gemma_3_config)
    gemma_small_config.num_hidden_layers = 2
    gemma_small_config.hidden_size = 8
    gemma_small_config.intermediate_size = 16
    gemma_small_config.head_dim = 4
    gemma_small_config.vocab_size = 16
    gemma_small_config.num_attention_heads = 2
    gemma_small_config.num_key_value_heads = 2
    gemma_small_config.final_logit_softcapping = None
    gemma_small_config.torch_dtype = "float32"
    model = load_dummy_gemma_model(gemma_small_config)  # type:ignore

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


def test_large_gemma_model():
    s = torch.tensor([0, 113, 24, 53, 27])
    gemma_large_cfg = copy.deepcopy(gemma_3_config)
    gemma_large_cfg.num_hidden_layers = 16
    gemma_large_cfg.hidden_size = 64
    gemma_large_cfg.intermediate_size = 128
    gemma_large_cfg.head_dim = 32
    gemma_large_cfg.vocab_size = 128
    gemma_large_cfg.num_attention_heads = 16
    gemma_large_cfg.num_key_value_heads = 16
    gemma_large_cfg.final_logit_softcapping = None
    gemma_large_cfg.torch_dtype = "float32"
    model = load_dummy_gemma_model(gemma_large_cfg)  # type: ignore

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
def test_gemma3_with_dummy_transcoders():
    s = "The National Digital Analytics Group (ND"
    model = load_gemma3_with_dummy_transcoders()
    model.to(torch.float32)  # type:ignore
    graph = attribute(s, model, batch_size=256)

    assert isinstance(model, NNSightReplacementModel)

    with model.zero_softcap():
        verify_token_and_error_edges(model, graph)
        verify_feature_edges(model, graph)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gemma3_with_dummy_clt():
    s = "The National Digital Analytics Group (ND"
    model = load_gemma3_with_dummy_clt()
    model.to(torch.float32)  # type:ignore
    graph = attribute(s, model, batch_size=256)

    assert isinstance(model, NNSightReplacementModel)

    with model.zero_softcap():
        verify_token_and_error_edges(model, graph)
        verify_feature_edges(model, graph)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gemma_3_1b():
    s = "The National Digital Analytics Group (ND"
    model = ReplacementModel.from_pretrained(
        "google/gemma-3-1b-pt",
        "mwhanna/gemma-scope-2-1b-pt/transcoder_all/width_16k_l0_small_affine",
        dtype=torch.float32,
        backend="nnsight",
    )
    graph = attribute(s, model)
    assert isinstance(model, NNSightReplacementModel)

    print("Changing logit softcap to 0, as the logits will otherwise be off.")
    with model.zero_softcap():
        verify_token_and_error_edges(model, graph)
        verify_feature_edges(model, graph)


@pytest.mark.skipif(not has_32gb, reason="Requires >=32GB VRAM")
def test_gemma_3_1b_it():
    s = "<bos><start_of_turn>user\nThe National Digital Analytics Group (ND"
    model = ReplacementModel.from_pretrained(
        "google/gemma-3-1b-it",
        "mwhanna/gemma-scope-2-1b-it/transcoder_all/width_16k_l0_small_affine",
        dtype=torch.float32,
        backend="nnsight",
    )
    graph = attribute(s, model)
    assert isinstance(model, NNSightReplacementModel)

    print("Changing logit softcap to 0, as the logits will otherwise be off.")
    with model.zero_softcap():
        verify_token_and_error_edges(model, graph, pos_start=4)
        verify_feature_edges(model, graph)


@pytest.mark.skipif(not has_32gb, reason="Requires >=32GB VRAM")
def test_gemma_3_1b_clt():
    s = "The National Digital Analytics Group (ND"
    model = ReplacementModel.from_pretrained(
        "google/gemma-3-270m",
        "mwhanna/gemma-scope-2-270m-pt/clt/width_262k_l0_medium_affine",
        dtype=torch.float32,
        backend="nnsight",
    )
    graph = attribute(s, model)
    assert isinstance(model, NNSightReplacementModel)

    print("Changing logit softcap to 0, as the logits will otherwise be off.")
    with model.zero_softcap():
        verify_token_and_error_edges(model, graph)
        verify_feature_edges(model, graph)


@pytest.mark.skipif(not has_32gb, reason="Requires >=32GB VRAM")
def test_gemma_3_4b():
    s = "The National Digital Analytics Group (ND"
    model = ReplacementModel.from_pretrained(
        "google/gemma-3-4b-pt",
        "mwhanna/gemma-scope-2-4b-pt/transcoder_all/width_16k_l0_small_affine",
        dtype=torch.float64,
        backend="nnsight",
        lazy_encoder=True,
    )
    graph = attribute(s, model)
    assert isinstance(model, NNSightReplacementModel)

    print("Changing logit softcap to 0, as the logits will otherwise be off.")
    with model.zero_softcap():
        verify_token_and_error_edges(model, graph)
        verify_feature_edges(model, graph, act_rtol=5e-4)  # we need slightly higher tolerance here


if __name__ == "__main__":
    torch.manual_seed(42)
    test_small_gemma_model()
    test_large_gemma_model()
    test_gemma3_with_dummy_transcoders()
    test_gemma3_with_dummy_clt()
    test_gemma_3_1b()
    test_gemma_3_1b_it()
    test_gemma_3_1b_clt()
    test_gemma_3_4b()
