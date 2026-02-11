import gc

import pytest
import torch

from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.attribution.attribute_nnsight import attribute as attribute_nnsight
from circuit_tracer.attribution.attribute_transformerlens import (
    attribute as attribute_transformerlens,
)
from tests.conftest import has_32gb

# Mark all tests in this module as requiring 32GB+ VRAM
pytestmark = [pytest.mark.skipif(not has_32gb, reason="Requires >=32GB VRAM")]


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


@pytest.fixture(scope="module")
def models():
    """Load models once for all tests."""
    model_nnsight = ReplacementModel.from_pretrained(
        "google/gemma-2-2b", "mntss/clt-gemma-2-2b-426k", backend="nnsight"
    )
    model_tl = ReplacementModel.from_pretrained("google/gemma-2-2b", "mntss/clt-gemma-2-2b-426k")
    return model_nnsight, model_tl


@pytest.fixture
def test_string():
    """Test string for all tests."""
    return "The National Digital Analytics Group (ND"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_activations_consistency(models, test_string):
    """Test that nnsight and transformerlens backends produce consistent activations."""
    model_nnsight, model_tl = models

    logits_nnsight, acts_nnsight = model_nnsight.get_activations(
        test_string, apply_activation_function=False
    )
    logits_tl, acts_tl = model_tl.get_activations(test_string, apply_activation_function=False)

    # Check activations are close
    max_act_diff = (acts_nnsight - acts_tl).abs().max()
    assert torch.allclose(acts_nnsight, acts_tl, atol=5e-4, rtol=1e-5), (
        f"Activations differ by max {max_act_diff}"
    )

    # Check logits are close
    max_logit_diff = (logits_nnsight - logits_tl).abs().max()
    assert torch.allclose(logits_nnsight, logits_tl, atol=1e-4, rtol=1e-5), (
        f"Logits differ by max {max_logit_diff}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_attribution_context_consistency(models, test_string):
    """Test that attribution contexts are consistent between backends."""
    model_nnsight, model_tl = models

    ctx_tl = model_tl.setup_attribution(test_string)
    ctx_nnsight = model_nnsight.setup_attribution(test_string)

    # Check error vectors
    assert torch.allclose(ctx_nnsight.error_vectors, ctx_tl.error_vectors, atol=1e-3, rtol=1e-5), (
        f"Error vectors differ by max {(ctx_nnsight.error_vectors - ctx_tl.error_vectors).abs().max()}"
    )

    # Check token vectors
    # token_diff = (ctx_nnsight.token_vectors - ctx_tl.token_vectors).abs().max()
    # assert token_diff < 1e-4, f"Token vectors differ by max {token_diff}"

    # Check decoder vectors
    assert torch.allclose(ctx_nnsight.decoder_vecs, ctx_tl.decoder_vecs, atol=1e-4, rtol=1e-5), (
        f"Decoder vectors differ by max {(ctx_nnsight.decoder_vecs - ctx_tl.decoder_vecs).abs().max()}"
    )

    # Check encoder vectors
    assert torch.allclose(ctx_nnsight.encoder_vecs, ctx_tl.encoder_vecs, atol=1e-4, rtol=1e-5), (
        f"Encoder vectors differ by max {(ctx_nnsight.encoder_vecs - ctx_tl.encoder_vecs).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_attribution_graph_consistency(models, test_string):
    """Test that attribution graphs are consistent between backends."""
    model_nnsight, model_tl = models

    with model_nnsight.zero_softcap():
        graph_nnsight = attribute_nnsight(test_string, model_nnsight, verbose=False)
    with model_tl.zero_softcap():
        graph_tl = attribute_transformerlens(test_string, model_tl, verbose=False)

    # Check active features match
    assert (graph_nnsight.active_features == graph_tl.active_features).all(), (
        "Active features don't match between backends"
    )

    # Check selected features match
    assert (graph_nnsight.selected_features == graph_tl.selected_features).all(), (
        "Selected features don't match between backends"
    )

    # Check adjacency matrices are close
    assert torch.allclose(
        graph_nnsight.adjacency_matrix, graph_tl.adjacency_matrix, atol=5e-4, rtol=1e-5
    ), (
        f"Adjacency matrices differ by max {(graph_nnsight.adjacency_matrix - graph_tl.adjacency_matrix).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_feature_intervention_consistency(models, test_string):
    """Test that feature interventions produce consistent results."""
    model_nnsight, model_tl = models

    # Perform interventions
    with model_nnsight.zero_softcap():
        intervened_logits_nnsight, intervened_acts_nnsight = model_nnsight.feature_intervention(
            test_string,
            [(6, 3, 9865, 21.1131)],
            apply_activation_function=False,
            constrained_layers=range(model_nnsight.cfg.n_layers),
        )

    with model_tl.zero_softcap():
        intervened_logits_tl, intervened_acts_tl = model_tl.feature_intervention(
            test_string,
            [(6, 3, 9865, 21.1131)],
            apply_activation_function=False,
            constrained_layers=range(model_tl.cfg.n_layers),
        )

    # Check logits are close
    assert torch.allclose(intervened_logits_nnsight, intervened_logits_tl, atol=1e-4, rtol=1e-5), (
        f"Intervened logits differ by max {(intervened_logits_nnsight - intervened_logits_tl).abs().max()}"
    )

    # Check activations are close
    assert torch.allclose(intervened_acts_nnsight, intervened_acts_tl, atol=5e-4, rtol=1e-5), (
        f"Intervened activations differ by max {(intervened_acts_nnsight - intervened_acts_tl).abs().max()}"
    )


def run_all_tests():
    """Run all tests when script is executed directly."""
    print("Loading models...")
    # Create models directly instead of calling fixture
    model_nnsight = ReplacementModel.from_pretrained(
        "google/gemma-2-2b", "mntss/clt-gemma-2-2b-426k", backend="nnsight"
    )
    model_tl = ReplacementModel.from_pretrained("google/gemma-2-2b", "mntss/clt-gemma-2-2b-426k")
    models_fixture = (model_nnsight, model_tl)

    # Create test string directly instead of calling fixture
    test_string_fixture = "The National Digital Analytics Group (ND"

    print("Running test_get_activations_consistency...")
    test_get_activations_consistency(models_fixture, test_string_fixture)
    print("✓ Activations consistency test passed")

    print("Running test_attribution_context_consistency...")
    test_attribution_context_consistency(models_fixture, test_string_fixture)
    print("✓ Attribution context consistency test passed")

    print("Running test_attribution_graph_consistency...")
    test_attribution_graph_consistency(models_fixture, test_string_fixture)
    print("✓ Attribution graph consistency test passed")

    print("Running test_feature_intervention_consistency...")
    test_feature_intervention_consistency(models_fixture, test_string_fixture)
    print("✓ Feature intervention consistency test passed")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    run_all_tests()
