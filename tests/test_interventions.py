import pytest
import torch

from circuit_tracer import ReplacementModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_intervention_return_activations():
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma")

    s = "The National Digital Analytics Group (ND"

    interventions = [(21, 7, 5066, 0.0)]

    logits_with_activations, activations = model.feature_intervention(
        s,
        interventions,
        constrained_layers=range(model.cfg.n_layers),
        return_activations=True,
    )

    logits_without_activations, no_activations = model.feature_intervention(
        s,
        interventions,
        constrained_layers=range(model.cfg.n_layers),
        return_activations=False,
    )

    assert torch.allclose(
        logits_with_activations, logits_without_activations, atol=1e-6, rtol=1e-5
    ), "Logits should be identical regardless of return_activations setting"

    assert activations is not None, "Activations should be returned when return_activations=True"
    assert no_activations is None, "Activations should be None when return_activations=False"


if __name__ == "__main__":
    torch.manual_seed(42)
    test_intervention_return_activations()
