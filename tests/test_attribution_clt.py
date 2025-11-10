import torch
import torch.nn as nn
from tqdm import tqdm
from transformer_lens import HookedTransformerConfig
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from circuit_tracer import Graph, ReplacementModel, attribute
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
from circuit_tracer.utils import get_default_device
from tests._comparison.attribution.attribute import attribute as legacy_attribute
from tests._comparison.replacement_model import ReplacementModel as LegacyReplacementModel


def create_clt_model(cfg: GPT2Config):
    """Create a CLT and ReplacementModel with random weights."""
    # Create CLT with 4x expansion
    clt = CrossLayerTranscoder(
        n_layers=cfg.n_layer,
        d_transcoder=cfg.n_embd * 4,
        d_model=cfg.n_embd,
        dtype=torch.float32,
        lazy_decoder=False,
    )

    # Initialize CLT weights
    with torch.no_grad():
        for param in clt.parameters():
            nn.init.uniform_(param, a=-0.1, b=0.1)

    # Create model
    hf_model = GPT2LMHeadModel(cfg).to(get_default_device())  # type: ignore[attr-defined]
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = ReplacementModel.from_hf_model(hf_model, tokenizer, clt)

    # Monkey patch all_special_ids if necessary
    type(model.tokenizer).all_special_ids = property(lambda self: [0])  # type: ignore

    # Initialize model weights
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                nn.init.uniform_(param, a=-0.1, b=0.1)

    return model


def verify_feature_edges(
    model: ReplacementModel,
    graph: Graph,
    n_samples: int = 100,
    act_atol=5e-4,
    act_rtol=1e-5,
    logit_atol=1e-5,
    logit_rtol=1e-3,
):
    """Verify that feature interventions produce the expected effects using feature_intervention
    method."""
    s = graph.input_tokens
    adjacency_matrix = graph.adjacency_matrix.to(get_default_device())
    active_features = graph.active_features.to(get_default_device())
    logit_tokens = graph.logit_tokens.to(get_default_device())
    total_active_features = active_features.size(0)

    logits, activation_cache = model.get_activations(s, apply_activation_function=False)
    logits = logits.squeeze(0)

    relevant_activations = activation_cache[
        active_features[:, 0], active_features[:, 1], active_features[:, 2]
    ]
    relevant_logits = logits[-1, logit_tokens]
    demeaned_relevant_logits = relevant_logits - logits[-1].mean()

    def verify_intervention(
        expected_effects,
        layer: int | torch.Tensor,
        pos: int | torch.Tensor,
        feature_idx: int | torch.Tensor,
        new_activation,
    ):
        new_logits, new_activation_cache = model.feature_intervention(
            s,
            [(layer, pos, feature_idx, new_activation)],
            constrained_layers=range(model.cfg.n_layers),
            apply_activation_function=False,
        )
        new_logits = new_logits.squeeze(0)

        assert new_activation_cache is not None
        new_relevant_activations = new_activation_cache[
            active_features[:, 0], active_features[:, 1], active_features[:, 2]
        ]
        new_relevant_logits = new_logits[-1, logit_tokens]
        new_demeaned_relevant_logits = new_relevant_logits - new_logits[-1].mean()

        expected_activation_difference = expected_effects[:total_active_features]
        expected_logit_difference = expected_effects[-len(logit_tokens) :]

        activation_diff = (
            new_relevant_activations - (relevant_activations + expected_activation_difference)
        ).abs()
        activation_rel_diff = activation_diff / (
            (relevant_activations + expected_activation_difference).abs() + 1e-8
        )

        logit_diff = (
            new_demeaned_relevant_logits - (demeaned_relevant_logits + expected_logit_difference)
        ).abs()
        logit_rel_diff = logit_diff / (
            (demeaned_relevant_logits + expected_logit_difference).abs() + 1e-8
        )

        if not torch.allclose(
            new_relevant_activations,
            relevant_activations + expected_activation_difference,
            atol=act_atol,
            rtol=act_rtol,
        ):
            max_abs_idx = activation_diff.argmax()
            max_rel_idx = activation_rel_diff.argmax()
            expected_acts = relevant_activations + expected_activation_difference

            print("Activation check failed:")
            print(
                f"  Max abs diff: {activation_diff.max():.6e} "
                f"({new_relevant_activations[max_abs_idx].item():.6e} vs "
                f"{expected_acts[max_abs_idx].item():.6e})"
            )
            print(f"  Mean abs diff: {activation_diff.mean():.6e}")
            print(
                f"  Max rel diff: {activation_rel_diff.max():.6e} "
                f"({new_relevant_activations[max_rel_idx].item():.6e} vs "
                f"{expected_acts[max_rel_idx].item():.6e})"
            )
            print(f"  Mean rel diff: {activation_rel_diff.mean():.6e}")
            print(f"  Tolerance: atol={act_atol}, rtol={act_rtol}")

        assert torch.allclose(
            new_relevant_activations,
            relevant_activations + expected_activation_difference,
            atol=act_atol,
            rtol=act_rtol,
        )

        if not torch.allclose(
            new_demeaned_relevant_logits,
            demeaned_relevant_logits + expected_logit_difference,
            atol=logit_atol,
            rtol=logit_rtol,
        ):
            max_abs_idx = logit_diff.argmax()
            max_rel_idx = logit_rel_diff.argmax()
            expected_logits = demeaned_relevant_logits + expected_logit_difference

            print("Logit check failed:")
            print(
                f"  Max abs diff: {logit_diff.max():.6e} "
                f"({new_demeaned_relevant_logits[max_abs_idx].item():.6e} vs "
                f"{expected_logits[max_abs_idx].item():.6e})"
            )
            print(f"  Mean abs diff: {logit_diff.mean():.6e}")
            print(
                f"  Max rel diff: {logit_rel_diff.max():.6e} "
                f"({new_demeaned_relevant_logits[max_rel_idx].item():.6e} vs "
                f"{expected_logits[max_rel_idx].item():.6e})"
            )
            print(f"  Mean rel diff: {logit_rel_diff.mean():.6e}")
            print(f"  Tolerance: atol={logit_atol}, rtol={logit_rtol}")

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
        verify_intervention(expected_effects, layer, pos, feature_idx, new_activation)


def test_clt_attribution():
    """Test CLT attribution and intervention mechanism."""
    # Minimal config
    cfg = GPT2Config(
        n_layer=4,
        n_embd=8,
        n_ctx=32,
        n_head=2,
        intermediate_size=32,
        activation_function="gelu",
        vocab_size=50,
        model_type="gpt2",
    )

    # Create model
    model = create_clt_model(cfg)

    # Run attribution
    prompt = torch.tensor([[0, 1, 2, 3, 4, 5]])
    graph = attribute(prompt, model, max_n_logits=5, desired_logit_prob=0.8, batch_size=32)

    # Test feature interventions
    n_active = len(graph.active_features)
    n_samples = min(100, n_active)

    verify_feature_edges(model, graph, n_samples=n_samples)


# -- verify that legacy code did work -- #


def test_clt_attribution_legacy():
    """Test CLT attribution and intervention mechanism."""
    # Minimal config
    cfg = HookedTransformerConfig.from_dict(
        {
            "n_layers": 4,
            "d_model": 8,
            "n_ctx": 32,
            "d_head": 4,
            "n_heads": 2,
            "d_mlp": 32,
            "act_fn": "gelu",
            "d_vocab": 50,
            "model_name": "test-clt",
            "device": get_default_device(),
            "tokenizer_name": "gpt2",
        }
    )

    # Create model
    model = create_legacy_clt_model(cfg)

    # Run attribution
    prompt = torch.tensor([[0, 1, 2, 3, 4, 5]])
    graph = legacy_attribute(prompt, model, max_n_logits=5, desired_logit_prob=0.8, batch_size=32)

    # Test feature interventions
    n_active = len(graph.active_features)
    n_samples = min(100, n_active)

    verify_feature_edges(model, graph, n_samples=n_samples)  # type: ignore


def create_legacy_clt_model(cfg: HookedTransformerConfig):
    """Create a CLT and ReplacementModel with random weights."""
    # Create CLT with 4x expansion
    clt = CrossLayerTranscoder(
        n_layers=cfg.n_layers,
        d_transcoder=cfg.d_model * 4,
        d_model=cfg.d_model,
        dtype=cfg.dtype,
        lazy_decoder=False,
    )

    # Initialize CLT weights
    with torch.no_grad():
        for param in clt.parameters():
            nn.init.uniform_(param, a=-0.1, b=0.1)

    # Create model
    model = LegacyReplacementModel.from_config(cfg, clt)

    # Monkey patch all_special_ids if necessary
    type(model.tokenizer).all_special_ids = property(lambda self: [0])  # type: ignore

    # Initialize model weights
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                nn.init.uniform_(param, a=-0.1, b=0.1)

    return model


def test_bridge_vs_legacy_clt_attribution_on_gpt2():
    """Test CLT attribution comparing bridge vs legacy on real GPT-2 with identical CLTs."""

    # Create two identical CLTs
    n_layers = 12
    d_model = 768
    d_transcoder = 128  # Small for faster testing

    clt1 = CrossLayerTranscoder(
        n_layers=n_layers,
        d_transcoder=d_transcoder,
        d_model=d_model,
        dtype=torch.float32,
        lazy_decoder=False,
    )
    clt2 = CrossLayerTranscoder(
        n_layers=n_layers,
        d_transcoder=d_transcoder,
        d_model=d_model,
        dtype=torch.float32,
        lazy_decoder=False,
    )

    # Share weights between the two CLTs
    with torch.no_grad():
        for param1, param2 in zip(clt1.parameters(), clt2.parameters()):
            assert param1.shape == param2.shape
            data = torch.randn_like(param1)
            param1.data = data.clone()
            param2.data = data.clone()

    # Create bridge model (new implementation)
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2", device_map="cpu")
    hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    bridge_model = ReplacementModel.from_hf_model(
        hf_model, hf_tokenizer, clt2, device=torch.device("cpu")
    )

    # Create legacy model (old implementation)
    legacy_model = LegacyReplacementModel.from_pretrained_and_transcoders(
        "gpt2", clt1, device="cpu"
    )

    # Test with a simple prompt
    prompt = torch.tensor([[0, 1, 2, 3, 4, 5]])

    # Run attribution on both
    bridge_graph = attribute(
        prompt, bridge_model, max_n_logits=5, desired_logit_prob=0.8, batch_size=32
    )
    legacy_graph = legacy_attribute(
        prompt, legacy_model, max_n_logits=5, desired_logit_prob=0.8, batch_size=32
    )

    # Check if active features match
    assert torch.allclose(bridge_graph.active_features, legacy_graph.active_features), (
        "Active features differ!"
    )

    # Check if activation values match
    assert torch.allclose(
        bridge_graph.activation_values, legacy_graph.activation_values, atol=1e-2, rtol=1e-3
    ), "Activation values differ!"

    # Check if adjacency matrices match
    diff = (bridge_graph.adjacency_matrix - legacy_graph.adjacency_matrix).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()

    # Allow for numerical precision differences due to different computation paths
    # Max relative diff is ~4.6% on small values, max absolute diff is ~0.64
    assert torch.allclose(
        bridge_graph.adjacency_matrix, legacy_graph.adjacency_matrix, atol=0.1, rtol=5e-3
    ), f"Adjacency matrices differ! Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}"

    n_active = len(bridge_graph.active_features)
    n_samples = min(100, n_active)

    verify_feature_edges(
        legacy_model,  # type: ignore
        legacy_graph,
        n_samples=n_samples,
        act_atol=1e-1,
        act_rtol=1e-2,
        logit_atol=1e-1,
        logit_rtol=1e-2,
    )
    verify_feature_edges(
        bridge_model,
        bridge_graph,
        n_samples=n_samples,
        act_atol=1e-1,
        act_rtol=1e-2,
        logit_atol=1e-1,
        logit_rtol=1e-2,
    )


if __name__ == "__main__":
    test_clt_attribution()
