import gc

import pytest
import torch

from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.attribution.attribute_nnsight import attribute as attribute_nnsight
from circuit_tracer.attribution.attribute_transformerlens import (
    attribute as attribute_transformerlens,
)


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


@pytest.fixture(scope="module")
def models():
    """Load models once for all tests."""
    model_nnsight = ReplacementModel.from_pretrained(
        "google/gemma-2-2b", "gemma", backend="nnsight", dtype=torch.float32
    )
    model_tl = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", dtype=torch.float32)
    return model_nnsight, model_tl


@pytest.fixture
def dallas_supernode_features():
    """Features from Dallas-Austin circuit supernodes."""
    return {
        "say_austin": [(23, 10, 12237)],
        "say_capital": [(21, 10, 5943), (17, 10, 7178), (7, 10, 691), (16, 10, 4298)],
        "capital": [(15, 4, 4494), (6, 4, 4662), (4, 4, 7671), (3, 4, 13984), (1, 4, 1000)],
        "texas": [
            (20, 9, 15589),
            (19, 9, 7477),
            (16, 9, 25),
            (4, 9, 13154),
            (14, 9, 2268),
            (7, 9, 6861),
        ],
        "state": [(6, 7, 4012), (0, 7, 13727)],
    }


@pytest.fixture
def oakland_supernode_features():
    """Features from Oakland-Sacramento circuit supernodes."""
    return {
        "say_sacramento": [(19, 10, 9209)],
        "california": [
            (22, 10, 4367),
            (21, 10, 2464),
            (6, 9, 13909),
            (8, 9, 14641),
            (14, 9, 12562),
        ],
    }


@pytest.fixture
def shanghai_supernode_features():
    """Features from Shanghai-Beijing circuit supernodes."""
    return {
        "china": [
            (19, 9, 12274),
            (14, 9, 12274),
            (6, 9, 6811),
            (4, 9, 11570),
            (4, 9, 4257),
            (19, 10, 12274),
            (18, 10, 7639),
        ],
    }


@pytest.fixture
def vancouver_supernode_features():
    """Features from Vancouver-Victoria circuit supernodes."""
    return {
        "say_victoria": [(21, 10, 2236)],
        "bc": [(18, 10, 1025)],
    }


@pytest.fixture
def multilingual_supernode_features():
    """Features from multilingual circuit supernodes."""
    return {
        "say_big": [(23, 8, 8683), (21, 8, 10062), (23, 8, 8488)],
        "small": [(15, 5, 5617), (14, 5, 11360), (3, 5, 6627), (3, 5, 2908), (2, 5, 5452)],
        "opposite": [(6, 2, 16184), (4, 2, 95)],
        "french": [(21, 8, 1144), (22, 8, 10566), (20, 8, 1454), (23, 8, 2592), (19, 8, 5802)],
        "chinese": [(24, 8, 2394), (22, 8, 11933), (20, 8, 12983), (21, 8, 13505), (23, 8, 13630)],
        "say_small": [(21, 8, 9082)],
        "big": [(15, 5, 5756), (6, 5, 4362), (3, 5, 2873), (2, 5, 4298)],
    }


@pytest.fixture
def dallas_austin_prompt():
    """Dallas-Austin reasoning prompt."""
    return "Fact: the capital of the state containing Dallas is"


@pytest.fixture
def oakland_sacramento_prompt():
    """Oakland-Sacramento reasoning prompt."""
    return "Fact: the capital of the state containing Oakland is"


@pytest.fixture
def small_big_prompts():
    """Multilingual opposite prompts."""
    return {
        "english": 'The opposite of "small" is "',
        "french": 'Le contraire de "petit" est "',
        "chinese": '"小"的反义词是"',
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dallas_austin_activations(models, dallas_austin_prompt):
    """Test get_activations consistency for Dallas-Austin prompt."""
    model_nnsight, model_tl = models

    logits_nnsight, acts_nnsight = model_nnsight.get_activations(dallas_austin_prompt)
    logits_tl, acts_tl = model_tl.get_activations(dallas_austin_prompt)

    max_act_diff = (acts_nnsight - acts_tl).abs().max()
    assert torch.allclose(acts_nnsight, acts_tl, atol=5e-4, rtol=1e-5), (
        f"Dallas-Austin activations differ by max {max_act_diff}"
    )

    max_logit_diff = (logits_nnsight - logits_tl).abs().max()
    assert torch.allclose(logits_nnsight, logits_tl, atol=1e-4, rtol=1e-5), (
        f"Dallas-Austin logits differ by max {max_logit_diff}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dallas_austin_attribution(models, dallas_austin_prompt):
    """Test attribution consistency for Dallas-Austin prompt."""
    model_nnsight, model_tl = models

    with model_nnsight.zero_softcap():
        graph_nnsight = attribute_nnsight(dallas_austin_prompt, model_nnsight, verbose=False)
    with model_tl.zero_softcap():
        graph_tl = attribute_transformerlens(dallas_austin_prompt, model_tl, verbose=False)

    assert (graph_nnsight.active_features == graph_tl.active_features).all(), (
        "Dallas-Austin active features don't match"
    )

    assert (graph_nnsight.selected_features == graph_tl.selected_features).all(), (
        "Dallas-Austin selected features don't match"
    )

    assert torch.allclose(
        graph_nnsight.adjacency_matrix, graph_tl.adjacency_matrix, atol=5e-4, rtol=1e-5
    ), (
        f"Dallas-Austin adjacency matrices differ by max "
        f"{(graph_nnsight.adjacency_matrix - graph_tl.adjacency_matrix).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dallas_intervention_say_capital_ablation(
    models, dallas_austin_prompt, dallas_supernode_features
):
    """Test ablating 'Say a capital' supernode (-2x)."""
    model_nnsight, model_tl = models

    # Create intervention: ablate say_capital features
    interventions = [
        (layer, pos, feat, -2.0) for layer, pos, feat in dallas_supernode_features["say_capital"]
    ]

    with model_nnsight.zero_softcap():
        logits_nnsight, acts_nnsight = model_nnsight.feature_intervention(
            dallas_austin_prompt,
            interventions,
            apply_activation_function=False,
            constrained_layers=range(model_nnsight.config.num_hidden_layers),
        )

    with model_tl.zero_softcap():
        logits_tl, acts_tl = model_tl.feature_intervention(
            dallas_austin_prompt,
            interventions,
            apply_activation_function=False,
            constrained_layers=range(model_tl.cfg.n_layers),
        )

    assert torch.allclose(logits_nnsight, logits_tl, atol=1e-4, rtol=1e-5), (
        f"Say capital ablation logits differ by max {(logits_nnsight - logits_tl).abs().max()}"
    )

    assert torch.allclose(acts_nnsight, acts_tl, atol=5e-4, rtol=1e-5), (
        f"Say capital ablation activations differ by max {(acts_nnsight - acts_tl).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dallas_intervention_capital_ablation(
    models, dallas_austin_prompt, dallas_supernode_features
):
    """Test ablating 'capital' supernode (-2x)."""
    model_nnsight, model_tl = models

    interventions = [
        (layer, pos, feat, -2.0) for layer, pos, feat in dallas_supernode_features["capital"]
    ]

    with model_nnsight.zero_softcap():
        logits_nnsight, acts_nnsight = model_nnsight.feature_intervention(
            dallas_austin_prompt,
            interventions,
            apply_activation_function=False,
            constrained_layers=range(model_nnsight.config.num_hidden_layers),
        )

    with model_tl.zero_softcap():
        logits_tl, acts_tl = model_tl.feature_intervention(
            dallas_austin_prompt,
            interventions,
            apply_activation_function=False,
            constrained_layers=range(model_tl.cfg.n_layers),
        )

    assert torch.allclose(logits_nnsight, logits_tl, atol=1e-4, rtol=1e-5), (
        f"Capital ablation logits differ by max {(logits_nnsight - logits_tl).abs().max()}"
    )

    assert torch.allclose(acts_nnsight, acts_tl, atol=5e-4, rtol=1e-5), (
        f"Capital ablation activations differ by max {(acts_nnsight - acts_tl).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dallas_intervention_texas_ablation(
    models, dallas_austin_prompt, dallas_supernode_features
):
    """Test ablating 'Texas' supernode (-2x)."""
    model_nnsight, model_tl = models

    interventions = [
        (layer, pos, feat, -2.0) for layer, pos, feat in dallas_supernode_features["texas"]
    ]

    with model_nnsight.zero_softcap():
        logits_nnsight, acts_nnsight = model_nnsight.feature_intervention(
            dallas_austin_prompt,
            interventions,
            apply_activation_function=False,
            constrained_layers=range(model_nnsight.config.num_hidden_layers),
        )

    with model_tl.zero_softcap():
        logits_tl, acts_tl = model_tl.feature_intervention(
            dallas_austin_prompt,
            interventions,
            apply_activation_function=False,
            constrained_layers=range(model_tl.cfg.n_layers),
        )

    assert torch.allclose(logits_nnsight, logits_tl, atol=1e-4, rtol=1e-5), (
        f"Texas ablation logits differ by max {(logits_nnsight - logits_tl).abs().max()}"
    )

    assert torch.allclose(acts_nnsight, acts_tl, atol=5e-4, rtol=1e-5), (
        f"Texas ablation activations differ by max {(acts_nnsight - acts_tl).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dallas_intervention_state_ablation(
    models, dallas_austin_prompt, dallas_supernode_features
):
    """Test ablating 'state' supernode (-2x)."""
    model_nnsight, model_tl = models

    interventions = [
        (layer, pos, feat, -2.0) for layer, pos, feat in dallas_supernode_features["state"]
    ]

    with model_nnsight.zero_softcap():
        logits_nnsight, acts_nnsight = model_nnsight.feature_intervention(
            dallas_austin_prompt,
            interventions,
            apply_activation_function=False,
            constrained_layers=range(model_nnsight.config.num_hidden_layers),
        )

    with model_tl.zero_softcap():
        logits_tl, acts_tl = model_tl.feature_intervention(
            dallas_austin_prompt,
            interventions,
            apply_activation_function=False,
            constrained_layers=range(model_tl.cfg.n_layers),
        )

    assert torch.allclose(logits_nnsight, logits_tl, atol=1e-4, rtol=1e-5), (
        f"State ablation logits differ by max {(logits_nnsight - logits_tl).abs().max()}"
    )

    assert torch.allclose(acts_nnsight, acts_tl, atol=5e-4, rtol=1e-5), (
        f"State ablation activations differ by max {(acts_nnsight - acts_tl).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dallas_intervention_replace_texas_with_california(
    models, dallas_austin_prompt, dallas_supernode_features, oakland_supernode_features
):
    """Test replacing Texas with California (Texas -2x, California +2x)."""
    model_nnsight, model_tl = models

    # Get activations from Oakland prompt for California features
    oakland_logits, oakland_acts = model_nnsight.get_activations(
        "Fact: the capital of the state containing Oakland is"
    )

    # Ablate Texas features
    interventions = [
        (layer, pos, feat, -2.0) for layer, pos, feat in dallas_supernode_features["texas"]
    ]

    # Add California features (using activation values from Oakland prompt)
    for layer, pos, feat in oakland_supernode_features["california"]:
        act_value = oakland_acts[layer, pos, feat].item()
        interventions.append((layer, pos, feat, 2.0 * act_value))

    with model_nnsight.zero_softcap():
        logits_nnsight, acts_nnsight = model_nnsight.feature_intervention(
            dallas_austin_prompt,
            interventions,
            apply_activation_function=False,
            constrained_layers=range(model_nnsight.config.num_hidden_layers),
        )

    with model_tl.zero_softcap():
        logits_tl, acts_tl = model_tl.feature_intervention(
            dallas_austin_prompt,
            interventions,
            apply_activation_function=False,
            constrained_layers=range(model_tl.cfg.n_layers),
        )

    assert torch.allclose(logits_nnsight, logits_tl, atol=1e-4, rtol=1e-5), (
        f"Texas->California intervention logits differ by max "
        f"{(logits_nnsight - logits_tl).abs().max()}"
    )

    assert torch.allclose(acts_nnsight, acts_tl, atol=5e-4, rtol=1e-5), (
        f"Texas->California intervention activations differ by max "
        f"{(acts_nnsight - acts_tl).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dallas_intervention_replace_texas_with_china(
    models, dallas_austin_prompt, dallas_supernode_features, shanghai_supernode_features
):
    """Test replacing Texas with China (Texas -2x, China +2x)."""
    model_nnsight, model_tl = models

    # Get activations from Shanghai prompt for China features
    shanghai_logits, shanghai_acts = model_nnsight.get_activations(
        "Fact: the capital of the country containing Shanghai is"
    )

    # Ablate Texas features
    interventions = [
        (layer, pos, feat, -2.0) for layer, pos, feat in dallas_supernode_features["texas"]
    ]

    # Add China features
    for layer, pos, feat in shanghai_supernode_features["china"]:
        act_value = shanghai_acts[layer, pos, feat].item()
        interventions.append((layer, pos, feat, 2.0 * act_value))

    with model_nnsight.zero_softcap():
        logits_nnsight, acts_nnsight = model_nnsight.feature_intervention(
            dallas_austin_prompt,
            interventions,
            apply_activation_function=False,
            constrained_layers=range(model_nnsight.config.num_hidden_layers),
        )

    with model_tl.zero_softcap():
        logits_tl, acts_tl = model_tl.feature_intervention(
            dallas_austin_prompt,
            interventions,
            apply_activation_function=False,
            constrained_layers=range(model_tl.cfg.n_layers),
        )

    assert torch.allclose(logits_nnsight, logits_tl, atol=1e-4, rtol=1e-5), (
        f"Texas->China intervention logits differ by max {(logits_nnsight - logits_tl).abs().max()}"
    )

    assert torch.allclose(acts_nnsight, acts_tl, atol=5e-4, rtol=1e-5), (
        f"Texas->China intervention activations differ by max "
        f"{(acts_nnsight - acts_tl).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dallas_intervention_replace_texas_with_bc(
    models, dallas_austin_prompt, dallas_supernode_features, vancouver_supernode_features
):
    """Test replacing Texas with British Columbia (Texas -2x, BC +2x)."""
    model_nnsight, model_tl = models

    # Get activations from Vancouver prompt for BC features
    vancouver_logits, vancouver_acts = model_nnsight.get_activations(
        "Fact: the capital of the territory containing Vancouver is"
    )

    # Ablate Texas features
    interventions = [
        (layer, pos, feat, -2.0) for layer, pos, feat in dallas_supernode_features["texas"]
    ]

    # Add BC features
    for layer, pos, feat in vancouver_supernode_features["bc"]:
        act_value = vancouver_acts[layer, pos, feat].item()
        interventions.append((layer, pos, feat, 2.0 * act_value))

    with model_nnsight.zero_softcap():
        logits_nnsight, acts_nnsight = model_nnsight.feature_intervention(
            dallas_austin_prompt,
            interventions,
            apply_activation_function=False,
            constrained_layers=range(model_nnsight.config.num_hidden_layers),
        )

    with model_tl.zero_softcap():
        logits_tl, acts_tl = model_tl.feature_intervention(
            dallas_austin_prompt,
            interventions,
            apply_activation_function=False,
            constrained_layers=range(model_tl.cfg.n_layers),
        )

    assert torch.allclose(logits_nnsight, logits_tl, atol=1e-4, rtol=1e-5), (
        f"Texas->BC intervention logits differ by max {(logits_nnsight - logits_tl).abs().max()}"
    )

    assert torch.allclose(acts_nnsight, acts_tl, atol=5e-4, rtol=1e-5), (
        f"Texas->BC intervention activations differ by max {(acts_nnsight - acts_tl).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_oakland_sacramento_activations(models, oakland_sacramento_prompt):
    """Test get_activations consistency for Oakland-Sacramento prompt."""
    model_nnsight, model_tl = models

    logits_nnsight, acts_nnsight = model_nnsight.get_activations(oakland_sacramento_prompt)
    logits_tl, acts_tl = model_tl.get_activations(oakland_sacramento_prompt)

    max_act_diff = (acts_nnsight - acts_tl).abs().max()
    assert torch.allclose(acts_nnsight, acts_tl, atol=5e-4, rtol=1e-5), (
        f"Oakland-Sacramento activations differ by max {max_act_diff}"
    )

    max_logit_diff = (logits_nnsight - logits_tl).abs().max()
    assert torch.allclose(logits_nnsight, logits_tl, atol=1e-4, rtol=1e-5), (
        f"Oakland-Sacramento logits differ by max {max_logit_diff}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_oakland_sacramento_attribution(models, oakland_sacramento_prompt):
    """Test attribution consistency for Oakland-Sacramento prompt."""
    model_nnsight, model_tl = models

    with model_nnsight.zero_softcap():
        graph_nnsight = attribute_nnsight(oakland_sacramento_prompt, model_nnsight, verbose=False)
    with model_tl.zero_softcap():
        graph_tl = attribute_transformerlens(oakland_sacramento_prompt, model_tl, verbose=False)

    assert (graph_nnsight.active_features == graph_tl.active_features).all(), (
        "Oakland-Sacramento active features don't match"
    )

    assert (graph_nnsight.selected_features == graph_tl.selected_features).all(), (
        "Oakland-Sacramento selected features don't match"
    )

    assert torch.allclose(
        graph_nnsight.adjacency_matrix, graph_tl.adjacency_matrix, atol=5e-4, rtol=1e-5
    ), (
        f"Oakland-Sacramento adjacency matrices differ by max "
        f"{(graph_nnsight.adjacency_matrix - graph_tl.adjacency_matrix).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multilingual_english_activations(models, small_big_prompts):
    """Test get_activations consistency for English opposite prompt."""
    model_nnsight, model_tl = models
    prompt = small_big_prompts["english"]

    logits_nnsight, acts_nnsight = model_nnsight.get_activations(
        prompt, apply_activation_function=False
    )
    logits_tl, acts_tl = model_tl.get_activations(prompt, apply_activation_function=False)

    max_act_diff = (acts_nnsight - acts_tl).abs().max()
    assert torch.allclose(acts_nnsight, acts_tl, atol=5e-4, rtol=1e-5), (
        f"English multilingual activations differ by max {max_act_diff}"
    )

    max_logit_diff = (logits_nnsight - logits_tl).abs().max()
    assert torch.allclose(logits_nnsight, logits_tl, atol=1e-4, rtol=1e-5), (
        f"English multilingual logits differ by max {max_logit_diff}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multilingual_french_activations(models, small_big_prompts):
    """Test get_activations consistency for French opposite prompt."""
    model_nnsight, model_tl = models
    prompt = small_big_prompts["french"]

    logits_nnsight, acts_nnsight = model_nnsight.get_activations(prompt)
    logits_tl, acts_tl = model_tl.get_activations(prompt)

    max_act_diff = (acts_nnsight - acts_tl).abs().max()
    assert torch.allclose(acts_nnsight, acts_tl, atol=5e-4, rtol=1e-5), (
        f"French multilingual activations differ by max {max_act_diff}"
    )

    max_logit_diff = (logits_nnsight - logits_tl).abs().max()
    assert torch.allclose(logits_nnsight, logits_tl, atol=1e-4, rtol=1e-5), (
        f"French multilingual logits differ by max {max_logit_diff}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multilingual_chinese_activations(models, small_big_prompts):
    """Test get_activations consistency for Chinese opposite prompt."""
    model_nnsight, model_tl = models
    prompt = small_big_prompts["chinese"]

    logits_nnsight, acts_nnsight = model_nnsight.get_activations(prompt)
    logits_tl, acts_tl = model_tl.get_activations(prompt)

    max_act_diff = (acts_nnsight - acts_tl).abs().max()
    assert torch.allclose(acts_nnsight, acts_tl, atol=5e-4, rtol=1e-5), (
        f"Chinese multilingual activations differ by max {max_act_diff}"
    )

    max_logit_diff = (logits_nnsight - logits_tl).abs().max()
    assert torch.allclose(logits_nnsight, logits_tl, atol=1e-4, rtol=1e-5), (
        f"Chinese multilingual logits differ by max {max_logit_diff}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multilingual_french_attribution(models, small_big_prompts):
    """Test attribution consistency for French opposite prompt."""
    model_nnsight, model_tl = models
    prompt = small_big_prompts["french"]

    with model_nnsight.zero_softcap():
        graph_nnsight = attribute_nnsight(prompt, model_nnsight, verbose=False)
    with model_tl.zero_softcap():
        graph_tl = attribute_transformerlens(prompt, model_tl, verbose=False)

    assert (graph_nnsight.active_features == graph_tl.active_features).all(), (
        "French multilingual active features don't match"
    )

    assert (graph_nnsight.selected_features == graph_tl.selected_features).all(), (
        "French multilingual selected features don't match"
    )

    assert torch.allclose(
        graph_nnsight.adjacency_matrix, graph_tl.adjacency_matrix, atol=5e-4, rtol=1e-5
    ), (
        f"French multilingual adjacency matrices differ by max "
        f"{(graph_nnsight.adjacency_matrix - graph_tl.adjacency_matrix).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multilingual_french_ablation(models, small_big_prompts, multilingual_supernode_features):
    """Test ablating French language features (-2x)."""
    model_nnsight, model_tl = models
    prompt = small_big_prompts["french"]

    interventions = [
        (layer, pos, feat, -2.0) for layer, pos, feat in multilingual_supernode_features["french"]
    ]

    with model_nnsight.zero_softcap():
        logits_nnsight, acts_nnsight = model_nnsight.feature_intervention(
            prompt,
            interventions,
            apply_activation_function=False,
            constrained_layers=range(model_nnsight.config.num_hidden_layers),
        )

    with model_tl.zero_softcap():
        logits_tl, acts_tl = model_tl.feature_intervention(
            prompt,
            interventions,
            apply_activation_function=False,
            constrained_layers=range(model_tl.cfg.n_layers),
        )

    assert torch.allclose(logits_nnsight, logits_tl, atol=1e-4, rtol=1e-5), (
        f"French ablation logits differ by max {(logits_nnsight - logits_tl).abs().max()}"
    )

    assert torch.allclose(acts_nnsight, acts_tl, atol=5e-4, rtol=1e-5), (
        f"French ablation activations differ by max {(acts_nnsight - acts_tl).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multilingual_french_to_chinese(models, small_big_prompts, multilingual_supernode_features):
    """Test replacing French with Chinese (French -2x, Chinese +2x)."""
    model_nnsight, model_tl = models
    prompt_fr = small_big_prompts["french"]
    prompt_zh = small_big_prompts["chinese"]

    # Get activations from Chinese prompt
    chinese_logits, chinese_acts = model_nnsight.get_activations(prompt_zh)

    # Ablate French features
    interventions = [
        (layer, pos, feat, -2.0) for layer, pos, feat in multilingual_supernode_features["french"]
    ]

    # Add Chinese features
    for layer, pos, feat in multilingual_supernode_features["chinese"]:
        act_value = chinese_acts[layer, pos, feat].item()
        interventions.append((layer, pos, feat, 2.0 * act_value))

    with model_nnsight.zero_softcap():
        logits_nnsight, acts_nnsight = model_nnsight.feature_intervention(
            prompt_fr,
            interventions,
            apply_activation_function=False,
            constrained_layers=range(model_nnsight.config.num_hidden_layers),
        )

    with model_tl.zero_softcap():
        logits_tl, acts_tl = model_tl.feature_intervention(
            prompt_fr,
            interventions,
            apply_activation_function=False,
            constrained_layers=range(model_tl.cfg.n_layers),
        )

    assert torch.allclose(logits_nnsight, logits_tl, atol=1e-4, rtol=1e-5), (
        f"French->Chinese intervention logits differ by max "
        f"{(logits_nnsight - logits_tl).abs().max()}"
    )

    assert torch.allclose(acts_nnsight, acts_tl, atol=5e-4, rtol=1e-5), (
        f"French->Chinese intervention activations differ by max "
        f"{(acts_nnsight - acts_tl).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multilingual_replace_small_with_big(
    models, small_big_prompts, multilingual_supernode_features
):
    """Test replacing small with big (small -2x, big +2x)."""
    model_nnsight, model_tl = models
    prompt_fr = small_big_prompts["french"]

    # Get activations from the reverse prompt (big->small)
    prompt_fr_rev = 'Le contraire de "grand" est "'
    big_small_logits, big_small_acts = model_nnsight.get_activations(prompt_fr_rev)

    # Ablate small features
    interventions = [
        (layer, pos, feat, -2.0) for layer, pos, feat in multilingual_supernode_features["small"]
    ]

    # Add big features
    for layer, pos, feat in multilingual_supernode_features["big"]:
        act_value = big_small_acts[layer, pos, feat].item()
        interventions.append((layer, pos, feat, 2.0 * act_value))

    with model_nnsight.zero_softcap():
        logits_nnsight, acts_nnsight = model_nnsight.feature_intervention(
            prompt_fr,
            interventions,
            apply_activation_function=False,
            constrained_layers=range(model_nnsight.config.num_hidden_layers),
        )

    with model_tl.zero_softcap():
        logits_tl, acts_tl = model_tl.feature_intervention(
            prompt_fr,
            interventions,
            apply_activation_function=False,
            constrained_layers=range(model_tl.cfg.n_layers),
        )

    assert torch.allclose(logits_nnsight, logits_tl, atol=1e-4, rtol=1e-5), (
        f"Small->Big intervention logits differ by max {(logits_nnsight - logits_tl).abs().max()}"
    )

    assert torch.allclose(acts_nnsight, acts_tl, atol=5e-4, rtol=1e-5), (
        f"Small->Big intervention activations differ by max {(acts_nnsight - acts_tl).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_setup_attribution_consistency(models, dallas_austin_prompt):
    """Test that attribution contexts are consistent between backends."""
    model_nnsight, model_tl = models

    ctx_tl = model_tl.setup_attribution(dallas_austin_prompt)
    ctx_nnsight = model_nnsight.setup_attribution(dallas_austin_prompt)

    assert torch.allclose(ctx_nnsight.error_vectors, ctx_tl.error_vectors, atol=1e-3, rtol=1e-5), (
        f"Error vectors differ by max "
        f"{(ctx_nnsight.error_vectors - ctx_tl.error_vectors).abs().max()}"
    )

    assert torch.allclose(ctx_nnsight.decoder_vecs, ctx_tl.decoder_vecs, atol=1e-4, rtol=1e-5), (
        f"Decoder vectors differ by max "
        f"{(ctx_nnsight.decoder_vecs - ctx_tl.decoder_vecs).abs().max()}"
    )

    assert torch.allclose(ctx_nnsight.encoder_vecs, ctx_tl.encoder_vecs, atol=1e-4, rtol=1e-5), (
        f"Encoder vectors differ by max "
        f"{(ctx_nnsight.encoder_vecs - ctx_tl.encoder_vecs).abs().max()}"
    )


def run_all_tests():
    """Run all tests when script is executed directly."""
    print("Loading models...")
    model_nnsight = ReplacementModel.from_pretrained(
        "google/gemma-2-2b", "gemma", backend="nnsight", dtype=torch.float32
    )
    model_tl = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", dtype=torch.float32)
    models_fixture = (model_nnsight, model_tl)

    # Prompts
    dallas_austin = "Fact: the capital of the state containing Dallas is"
    oakland_sacramento = "Fact: the capital of the state containing Oakland is"
    small_big = {
        "english": 'The opposite of "small" is "',
        "french": 'Le contraire de "petit" est "',
        "chinese": '"小"的反义词是"',
    }

    # Feature fixtures
    dallas_features = {
        "say_austin": [(23, 10, 12237)],
        "say_capital": [(21, 10, 5943), (17, 10, 7178), (7, 10, 691), (16, 10, 4298)],
        "capital": [(15, 4, 4494), (6, 4, 4662), (4, 4, 7671), (3, 4, 13984), (1, 4, 1000)],
        "texas": [
            (20, 9, 15589),
            (19, 9, 7477),
            (16, 9, 25),
            (4, 9, 13154),
            (14, 9, 2268),
            (7, 9, 6861),
        ],
        "state": [(6, 7, 4012), (0, 7, 13727)],
    }
    oakland_features = {
        "say_sacramento": [(19, 10, 9209)],
        "california": [
            (22, 10, 4367),
            (21, 10, 2464),
            (6, 9, 13909),
            (8, 9, 14641),
            (14, 9, 12562),
        ],
    }
    shanghai_features = {
        "china": [
            (19, 9, 12274),
            (14, 9, 12274),
            (6, 9, 6811),
            (4, 9, 11570),
            (4, 9, 4257),
            (19, 10, 12274),
            (18, 10, 7639),
        ],
    }
    vancouver_features = {
        "say_victoria": [(21, 10, 2236)],
        "bc": [(18, 10, 1025)],
    }
    multilingual_features = {
        "say_big": [(23, 8, 8683), (21, 8, 10062), (23, 8, 8488)],
        "small": [(15, 5, 5617), (14, 5, 11360), (3, 5, 6627), (3, 5, 2908), (2, 5, 5452)],
        "opposite": [(6, 2, 16184), (4, 2, 95)],
        "french": [(21, 8, 1144), (22, 8, 10566), (20, 8, 1454), (23, 8, 2592), (19, 8, 5802)],
        "chinese": [(24, 8, 2394), (22, 8, 11933), (20, 8, 12983), (21, 8, 13505), (23, 8, 13630)],
        "say_small": [(21, 8, 9082)],
        "big": [(15, 5, 5756), (6, 5, 4362), (3, 5, 2873), (2, 5, 4298)],
    }

    print("\n=== Testing Dallas-Austin Circuit ===")
    print("Running test_dallas_austin_activations...")
    test_dallas_austin_activations(models_fixture, dallas_austin)
    print("✓ Dallas-Austin activations consistency test passed")

    print("Running test_dallas_austin_attribution...")
    test_dallas_austin_attribution(models_fixture, dallas_austin)
    print("✓ Dallas-Austin attribution consistency test passed")

    print("\n=== Testing Dallas-Austin Interventions ===")
    print("Running test_dallas_intervention_say_capital_ablation...")
    test_dallas_intervention_say_capital_ablation(models_fixture, dallas_austin, dallas_features)
    print("✓ Dallas Say-capital ablation test passed")

    print("Running test_dallas_intervention_capital_ablation...")
    test_dallas_intervention_capital_ablation(models_fixture, dallas_austin, dallas_features)
    print("✓ Dallas capital ablation test passed")

    print("Running test_dallas_intervention_texas_ablation...")
    test_dallas_intervention_texas_ablation(models_fixture, dallas_austin, dallas_features)
    print("✓ Dallas Texas ablation test passed")

    print("Running test_dallas_intervention_state_ablation...")
    test_dallas_intervention_state_ablation(models_fixture, dallas_austin, dallas_features)
    print("✓ Dallas state ablation test passed")

    print("Running test_dallas_intervention_replace_texas_with_california...")
    test_dallas_intervention_replace_texas_with_california(
        models_fixture, dallas_austin, dallas_features, oakland_features
    )
    print("✓ Dallas Texas->California replacement test passed")

    print("Running test_dallas_intervention_replace_texas_with_china...")
    test_dallas_intervention_replace_texas_with_china(
        models_fixture, dallas_austin, dallas_features, shanghai_features
    )
    print("✓ Dallas Texas->China replacement test passed")

    print("Running test_dallas_intervention_replace_texas_with_bc...")
    test_dallas_intervention_replace_texas_with_bc(
        models_fixture, dallas_austin, dallas_features, vancouver_features
    )
    print("✓ Dallas Texas->BC replacement test passed")

    print("\n=== Testing Oakland-Sacramento Circuit ===")
    print("Running test_oakland_sacramento_activations...")
    test_oakland_sacramento_activations(models_fixture, oakland_sacramento)
    print("✓ Oakland-Sacramento activations consistency test passed")

    print("Running test_oakland_sacramento_attribution...")
    test_oakland_sacramento_attribution(models_fixture, oakland_sacramento)
    print("✓ Oakland-Sacramento attribution consistency test passed")

    print("\n=== Testing Multilingual Circuits ===")
    print("Running test_multilingual_english_activations...")
    test_multilingual_english_activations(models_fixture, small_big)
    print("✓ English multilingual activations consistency test passed")

    print("Running test_multilingual_french_activations...")
    test_multilingual_french_activations(models_fixture, small_big)
    print("✓ French multilingual activations consistency test passed")

    print("Running test_multilingual_chinese_activations...")
    test_multilingual_chinese_activations(models_fixture, small_big)
    print("✓ Chinese multilingual activations consistency test passed")

    print("Running test_multilingual_french_attribution...")
    test_multilingual_french_attribution(models_fixture, small_big)
    print("✓ French multilingual attribution consistency test passed")

    print("\n=== Testing Multilingual Interventions ===")
    print("Running test_multilingual_french_ablation...")
    test_multilingual_french_ablation(models_fixture, small_big, multilingual_features)
    print("✓ French ablation test passed")

    print("Running test_multilingual_french_to_chinese...")
    test_multilingual_french_to_chinese(models_fixture, small_big, multilingual_features)
    print("✓ French->Chinese replacement test passed")

    print("Running test_multilingual_replace_small_with_big...")
    test_multilingual_replace_small_with_big(models_fixture, small_big, multilingual_features)
    print("✓ Small->Big replacement test passed")

    print("\n=== Testing Attribution Setup ===")
    print("Running test_setup_attribution_consistency...")
    test_setup_attribution_consistency(models_fixture, dallas_austin)
    print("✓ Attribution setup consistency test passed")

    print("\n" + "=" * 70)
    print("All tutorial notebook tests passed! ✓")
    print("Total tests run: 20")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
