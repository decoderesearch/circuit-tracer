import gc
from contextlib import contextmanager

import pytest
import torch

from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.attribution.attribute_nnsight import attribute as attribute_nnsight
from circuit_tracer.attribution.attribute_transformerlens import (
    attribute as attribute_transformerlens,
)
from circuit_tracer.attribution.targets import CustomTarget
from circuit_tracer.graph import compute_node_influence
from circuit_tracer.utils.demo_utils import get_unembed_vecs
from tests.conftest import has_32gb

# decorator used to gate individual tests on available VRAM
skip32gb = pytest.mark.skipif(not has_32gb, reason="Requires >=32GB VRAM")


def _move_replacement_model(model, device):
    """Move a ReplacementModel (and its transcoders) to *device*, updating internal refs.

    Works for both NNSight and TransformerLens backends.
    """
    device = torch.device(device) if isinstance(device, str) else device

    # Move model parameters
    model.to(device)

    # Move transcoders — NNSight wraps them in an Envoy so .to() only takes device
    try:
        model.transcoders.to(device, torch.float32)
    except TypeError:
        model.transcoders.to(device)

    # Update stale tensor references left on the NNSight model instance.
    # `.to()` replaces Parameter tensors inside the module but external refs
    # (e.g. embed_weight, unembed_weight) still point at the old device.
    for attr in ("embed_weight", "unembed_weight"):
        t = getattr(model, attr, None)
        if t is not None and t.device != device:
            setattr(model, attr, t.to(device))

    # Update backend-specific device tracking
    if hasattr(model, "cfg") and hasattr(model.cfg, "device"):
        # TransformerLens backend
        model.cfg.device = device


@contextmanager
def clean_cuda(model, min_bytes: int = 1 << 20):
    """Move *model* to CUDA; on exit automatically free large transient CUDA tensors.

    Snapshots data_ptrs of all large CUDA tensors after the model moves to CUDA
    (capturing model weights as 'known'). On exit, any new large CUDA tensor not
    in the snapshot has its storage replaced via ``set_(torch.empty(0))``, freeing
    VRAM even while Python references remain alive. Then ``gc.collect()`` +
    ``empty_cache()`` flush remaining allocations before the model moves back to CPU.
    Callers do not need explicit ``del`` statements for large GPU-resident objects.
    """
    _move_replacement_model(model, "cuda")

    def _is_large_dense_cuda(t: object) -> bool:
        return (
            isinstance(t, torch.Tensor)
            and t.is_cuda
            and t.layout == torch.strided
            and t.nbytes >= min_bytes
        )

    known_ptrs: set[int] = {obj.data_ptr() for obj in gc.get_objects() if _is_large_dense_cuda(obj)}
    try:
        yield
    finally:
        freed_ptrs: set[int] = set()
        for obj in gc.get_objects():
            if (
                _is_large_dense_cuda(obj)
                and obj.data_ptr() not in known_ptrs
                and obj.data_ptr() not in freed_ptrs
            ):
                freed_ptrs.add(obj.data_ptr())
                try:
                    obj.set_(torch.empty(0))
                except Exception:
                    pass
        gc.collect()
        torch.cuda.empty_cache()
        _move_replacement_model(model, "cpu")


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


@pytest.fixture(scope="module")
def models_cpu():
    """Load both models on CPU for memory-constrained sequential backend tests.

    Tests using this fixture should wrap each backend run in ``clean_cuda``
    to move the active model to CUDA and restore it to CPU when done,
    automatically freeing transient GPU-resident objects between backend phases.
    """
    model_nnsight = ReplacementModel.from_pretrained(
        "google/gemma-2-2b",
        "gemma",
        backend="nnsight",
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    model_tl = ReplacementModel.from_pretrained(
        "google/gemma-2-2b", "gemma", dtype=torch.float32, device=torch.device("cpu")
    )
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


@skip32gb
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


@skip32gb
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


@skip32gb
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


@skip32gb
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


@skip32gb
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


@skip32gb
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


@skip32gb
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


@skip32gb
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


@skip32gb
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


@skip32gb
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


@skip32gb
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


@skip32gb
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


@skip32gb
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


@skip32gb
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


@skip32gb
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


@skip32gb
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


@skip32gb
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


@skip32gb
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
@skip32gb
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


def _build_demo_custom_target(model, prompt, token_x, token_y, backend):
    """Build a CustomTarget for logit(token_x) − logit(token_y).

    Backend-agnostic helper matching the attribution_targets_demo pattern.
    Uses ``get_unembed_vecs`` from ``demo_utils`` for unembedding extraction.
    """
    tokenizer = model.tokenizer
    idx_x = tokenizer.encode(token_x, add_special_tokens=False)[-1]
    idx_y = tokenizer.encode(token_y, add_special_tokens=False)[-1]

    input_ids = model.ensure_tokenized(prompt)
    with torch.no_grad():
        logits, _ = model.get_activations(input_ids)
    last_logits = logits.squeeze(0)[-1]

    vec_x, vec_y = get_unembed_vecs(model, [idx_x, idx_y], backend)
    diff_vec = vec_x - vec_y
    probs = torch.softmax(last_logits, dim=-1)
    diff_prob = max((probs[idx_x] - probs[idx_y]).abs().item(), 1e-6)

    return (
        CustomTarget(token_str=f"logit({token_x})-logit({token_y})", prob=diff_prob, vec=diff_vec),
        idx_x,
        idx_y,
    )


def _build_demo_semantic_target(model, prompt, group_a_tokens, group_b_tokens, label, backend):
    """Build a CustomTarget for an abstract concept direction via vector rejection.

    For each (capital, state) pair, project the capital vector onto the state
    vector and subtract that projection, leaving pure "capital-ness".

    Backend-agnostic helper matching the attribution_targets_demo pattern.
    """
    assert len(group_a_tokens) == len(group_b_tokens), (
        "Groups must have equal length for paired differences"
    )
    tokenizer = model.tokenizer
    ids_a = [tokenizer.encode(t, add_special_tokens=False)[-1] for t in group_a_tokens]
    ids_b = [tokenizer.encode(t, add_special_tokens=False)[-1] for t in group_b_tokens]

    vecs_a = get_unembed_vecs(model, ids_a, backend)
    vecs_b = get_unembed_vecs(model, ids_b, backend)

    # Vector rejection: for each pair, remove the state-direction component
    residuals = []
    for va, vb in zip(vecs_a, vecs_b):
        va_f, vb_f = va.float(), vb.float()
        proj = (va_f @ vb_f) / (vb_f @ vb_f) * vb_f
        residuals.append((va_f - proj).to(va.dtype))

    direction = torch.stack(residuals).mean(0)

    input_ids = model.ensure_tokenized(prompt)
    with torch.no_grad():
        logits, _ = model.get_activations(input_ids)
    probs = torch.softmax(logits.squeeze(0)[-1], dim=-1)
    avg_prob = max(sum(probs[i].item() for i in ids_a) / len(ids_a), 1e-6)

    return CustomTarget(token_str=label, prob=avg_prob, vec=direction)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_attribution_targets_string(models_cpu, dallas_austin_prompt):
    """Test attribution with Sequence[str] targets consistency between TL and NNSight."""
    model_nnsight, model_tl = models_cpu
    str_targets = ["▁Austin", "▁Dallas"]

    # --- NNSight backend ---
    with clean_cuda(model_nnsight):
        graph_nnsight = attribute_nnsight(
            dallas_austin_prompt,
            model_nnsight,
            attribution_targets=str_targets,
            verbose=False,
            batch_size=256,
        )
        nn_active = graph_nnsight.active_features.cpu()
        nn_selected = graph_nnsight.selected_features.cpu()
        nn_tokens = [t.token_str for t in graph_nnsight.logit_targets]
        nn_adj = graph_nnsight.adjacency_matrix.cpu()

    # --- TL backend ---
    with clean_cuda(model_tl):
        graph_tl = attribute_transformerlens(
            dallas_austin_prompt,
            model_tl,
            attribution_targets=str_targets,
            verbose=False,
            batch_size=128,
        )
        tl_active = graph_tl.active_features.cpu()
        tl_selected = graph_tl.selected_features.cpu()
        tl_tokens = [t.token_str for t in graph_tl.logit_targets]
        tl_adj = graph_tl.adjacency_matrix.cpu()

    # --- Compare CPU tensors ---
    assert (nn_active == tl_active).all(), (
        "String-target active features don't match between backends"
    )
    assert (nn_selected == tl_selected).all(), (
        "String-target selected features don't match between backends"
    )
    assert nn_tokens == tl_tokens, f"String-target logit tokens differ: {nn_tokens} vs {tl_tokens}"
    assert torch.allclose(nn_adj, tl_adj, atol=5e-4, rtol=1e-5), (
        f"String-target adjacency matrices differ by max {(nn_adj - tl_adj).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_attribution_targets_tensor(models_cpu, dallas_austin_prompt):
    """Test attribution with torch.Tensor targets consistency between TL and NNSight.

    Uses the same token IDs as the string-target test (pre-tokenized equivalent).
    """
    model_nnsight, model_tl = models_cpu
    # Resolve token IDs for Austin and Dallas (same as string-target test)
    tok = model_nnsight.tokenizer
    idx_austin = tok.encode("▁Austin", add_special_tokens=False)[-1]
    idx_dallas = tok.encode("▁Dallas", add_special_tokens=False)[-1]
    tensor_targets = torch.tensor([idx_austin, idx_dallas])

    # --- NNSight backend ---
    with clean_cuda(model_nnsight):
        graph_nnsight = attribute_nnsight(
            dallas_austin_prompt,
            model_nnsight,
            attribution_targets=tensor_targets,
            verbose=False,
            batch_size=256,
        )
        nn_active = graph_nnsight.active_features.cpu()
        nn_selected = graph_nnsight.selected_features.cpu()
        nn_tokens = [t.token_str for t in graph_nnsight.logit_targets]
        nn_adj = graph_nnsight.adjacency_matrix.cpu()

    # --- TL backend ---
    with clean_cuda(model_tl):
        graph_tl = attribute_transformerlens(
            dallas_austin_prompt,
            model_tl,
            attribution_targets=tensor_targets,
            verbose=False,
            batch_size=128,
        )
        tl_active = graph_tl.active_features.cpu()
        tl_selected = graph_tl.selected_features.cpu()
        tl_tokens = [t.token_str for t in graph_tl.logit_targets]
        tl_adj = graph_tl.adjacency_matrix.cpu()

    # --- Compare CPU tensors ---
    assert (nn_active == tl_active).all(), (
        "Tensor-target active features don't match between backends"
    )
    assert (nn_selected == tl_selected).all(), (
        "Tensor-target selected features don't match between backends"
    )
    assert nn_tokens == tl_tokens, f"Tensor-target logit tokens differ: {nn_tokens} vs {tl_tokens}"
    assert torch.allclose(nn_adj, tl_adj, atol=5e-4, rtol=1e-5), (
        f"Tensor-target adjacency matrices differ by max {(nn_adj - tl_adj).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_attribution_targets_logit_diff(models_cpu, dallas_austin_prompt):
    """Test attribution with CustomTarget consistency between TL and NNSight."""
    model_nnsight, model_tl = models_cpu

    # --- NNSight backend ---
    with clean_cuda(model_nnsight):
        custom_nnsight, _, _ = _build_demo_custom_target(
            model_nnsight, dallas_austin_prompt, "▁Austin", "▁Dallas", backend="nnsight"
        )
        graph_nnsight = attribute_nnsight(
            dallas_austin_prompt,
            model_nnsight,
            attribution_targets=[custom_nnsight],
            verbose=False,
            batch_size=256,
        )
        nn_active = graph_nnsight.active_features.cpu()
        nn_selected = graph_nnsight.selected_features.cpu()
        nn_tokens = [t.token_str for t in graph_nnsight.logit_targets]
        nn_adj = graph_nnsight.adjacency_matrix.cpu()

    # --- TL backend ---
    with clean_cuda(model_tl):
        custom_tl, _, _ = _build_demo_custom_target(
            model_tl, dallas_austin_prompt, "▁Austin", "▁Dallas", backend="transformerlens"
        )
        graph_tl = attribute_transformerlens(
            dallas_austin_prompt,
            model_tl,
            attribution_targets=[custom_tl],
            verbose=False,
            batch_size=128,
        )
        tl_active = graph_tl.active_features.cpu()
        tl_selected = graph_tl.selected_features.cpu()
        tl_tokens = [t.token_str for t in graph_tl.logit_targets]
        tl_adj = graph_tl.adjacency_matrix.cpu()

    # --- Compare CPU tensors ---
    assert (nn_active == tl_active).all(), (
        "Custom-target active features don't match between backends"
    )
    assert (nn_selected == tl_selected).all(), (
        "Custom-target selected features don't match between backends"
    )
    assert nn_tokens == tl_tokens, f"Custom-target logit tokens differ: {nn_tokens} vs {tl_tokens}"
    assert torch.allclose(nn_adj, tl_adj, atol=5e-4, rtol=1e-5), (
        f"Custom-target adjacency matrices differ by max {(nn_adj - tl_adj).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_attribution_targets_logit_diff_intervention(models_cpu, dallas_austin_prompt):
    """Test custom-target feature amplification consistency between TL and NNSight."""
    model_nnsight, model_tl = models_cpu
    n_top = 10

    def _get_top_features(graph, n):
        n_logits = len(graph.logit_targets)
        n_features = len(graph.selected_features)
        logit_weights = torch.zeros(
            graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device
        )
        logit_weights[-n_logits:] = graph.logit_probabilities
        node_influence = compute_node_influence(graph.adjacency_matrix, logit_weights)
        _, top_idx = torch.topk(node_influence[:n_features], min(n, n_features))
        return [tuple(graph.active_features[graph.selected_features[i]].tolist()) for i in top_idx]

    # --- NNSight backend ---
    with clean_cuda(model_nnsight):
        custom_nnsight, idx_x_nn, idx_y_nn = _build_demo_custom_target(
            model_nnsight, dallas_austin_prompt, "▁Austin", "▁Dallas", backend="nnsight"
        )
        graph_nnsight = attribute_nnsight(
            dallas_austin_prompt,
            model_nnsight,
            attribution_targets=[custom_nnsight],
            verbose=False,
            batch_size=256,
        )
        top_feats_nn = _get_top_features(graph_nnsight, n_top)

        input_ids_nn = model_nnsight.ensure_tokenized(dallas_austin_prompt)
        orig_logits_nn, acts_nn = model_nnsight.get_activations(input_ids_nn, sparse=True)

        interv_nn = [(ly, p, f, 10.0 * acts_nn[ly, p, f]) for (ly, p, f) in top_feats_nn]
        new_logits_nn, _ = model_nnsight.feature_intervention(input_ids_nn, interv_nn)

        orig_gap_nn = (
            (orig_logits_nn.squeeze(0)[-1, idx_x_nn] - orig_logits_nn.squeeze(0)[-1, idx_y_nn])
            .cpu()
            .item()
        )
        new_gap_nn = (
            (new_logits_nn.squeeze(0)[-1, idx_x_nn] - new_logits_nn.squeeze(0)[-1, idx_y_nn])
            .cpu()
            .item()
        )

    # --- TL backend ---
    with clean_cuda(model_tl):
        custom_tl, idx_x_tl, idx_y_tl = _build_demo_custom_target(
            model_tl, dallas_austin_prompt, "▁Austin", "▁Dallas", backend="transformerlens"
        )
        graph_tl = attribute_transformerlens(
            dallas_austin_prompt,
            model_tl,
            attribution_targets=[custom_tl],
            verbose=False,
            batch_size=128,
        )
        top_feats_tl = _get_top_features(graph_tl, n_top)

        input_ids_tl = model_tl.ensure_tokenized(dallas_austin_prompt)
        orig_logits_tl, acts_tl = model_tl.get_activations(input_ids_tl, sparse=True)

        interv_tl = [(ly, p, f, 10.0 * acts_tl[ly, p, f]) for (ly, p, f) in top_feats_tl]
        new_logits_tl, _ = model_tl.feature_intervention(input_ids_tl, interv_tl)

        orig_gap_tl = (
            (orig_logits_tl.squeeze(0)[-1, idx_x_tl] - orig_logits_tl.squeeze(0)[-1, idx_y_tl])
            .cpu()
            .item()
        )
        new_gap_tl = (
            (new_logits_tl.squeeze(0)[-1, idx_x_tl] - new_logits_tl.squeeze(0)[-1, idx_y_tl])
            .cpu()
            .item()
        )

    # --- Compare on CPU ---
    assert new_gap_nn > orig_gap_nn, (
        f"NNSight: amplification should widen gap, got {orig_gap_nn:.4f} -> {new_gap_nn:.4f}"
    )
    assert new_gap_tl > orig_gap_tl, (
        f"TL: amplification should widen gap, got {orig_gap_tl:.4f} -> {new_gap_tl:.4f}"
    )

    assert abs(new_gap_nn - new_gap_tl) < 0.5, (
        f"Post-intervention gaps differ too much: NNSight={new_gap_nn:.4f}, TL={new_gap_tl:.4f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_attribution_targets_semantic(models_cpu, dallas_austin_prompt):
    """Test attribution with semantic concept CustomTarget consistency between TL and NNSight."""
    model_nnsight, model_tl = models_cpu
    capitals = ["▁Austin", "▁Sacramento", "▁Olympia", "▁Atlanta"]
    states = ["▁Texas", "▁California", "▁Washington", "▁Georgia"]
    label = "Concept: Capitals − States"

    # --- NNSight backend ---
    with clean_cuda(model_nnsight):
        sem_nnsight = _build_demo_semantic_target(
            model_nnsight, dallas_austin_prompt, capitals, states, label, backend="nnsight"
        )
        graph_nnsight = attribute_nnsight(
            dallas_austin_prompt,
            model_nnsight,
            attribution_targets=[sem_nnsight],
            verbose=False,
            batch_size=256,
        )
        nn_active = graph_nnsight.active_features.cpu()
        nn_selected = graph_nnsight.selected_features.cpu()
        nn_tokens = [t.token_str for t in graph_nnsight.logit_targets]
        nn_adj = graph_nnsight.adjacency_matrix.cpu()

    # --- TL backend ---
    with clean_cuda(model_tl):
        sem_tl = _build_demo_semantic_target(
            model_tl, dallas_austin_prompt, capitals, states, label, backend="transformerlens"
        )
        graph_tl = attribute_transformerlens(
            dallas_austin_prompt,
            model_tl,
            attribution_targets=[sem_tl],
            verbose=False,
            batch_size=128,
        )
        tl_active = graph_tl.active_features.cpu()
        tl_selected = graph_tl.selected_features.cpu()
        tl_tokens = [t.token_str for t in graph_tl.logit_targets]
        tl_adj = graph_tl.adjacency_matrix.cpu()

    # --- Compare CPU tensors ---
    assert (nn_active == tl_active).all(), (
        "Semantic-target active features don't match between backends"
    )
    assert (nn_selected == tl_selected).all(), (
        "Semantic-target selected features don't match between backends"
    )
    assert nn_tokens == tl_tokens, (
        f"Semantic-target logit tokens differ: {nn_tokens} vs {tl_tokens}"
    )
    assert torch.allclose(nn_adj, tl_adj, atol=5e-4, rtol=1e-5), (
        f"Semantic-target adjacency matrices differ by max {(nn_adj - tl_adj).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_attribution_targets_semantic_intervention(models_cpu, dallas_austin_prompt):
    """Test semantic-target feature amplification consistency between TL and NNSight."""
    model_nnsight, model_tl = models_cpu
    n_top = 10
    capitals = ["▁Austin", "▁Sacramento", "▁Olympia", "▁Atlanta"]
    states = ["▁Texas", "▁California", "▁Washington", "▁Georgia"]
    label = "Concept: Capitals − States"

    def _get_top_features(graph, n):
        n_logits = len(graph.logit_targets)
        n_features = len(graph.selected_features)
        logit_weights = torch.zeros(
            graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device
        )
        logit_weights[-n_logits:] = graph.logit_probabilities
        node_influence = compute_node_influence(graph.adjacency_matrix, logit_weights)
        _, top_idx = torch.topk(node_influence[:n_features], min(n, n_features))
        return [tuple(graph.active_features[graph.selected_features[i]].tolist()) for i in top_idx]

    # --- NNSight backend ---
    with clean_cuda(model_nnsight):
        sem_nnsight = _build_demo_semantic_target(
            model_nnsight, dallas_austin_prompt, capitals, states, label, backend="nnsight"
        )
        idx_x_nn = model_nnsight.tokenizer.encode("▁Austin", add_special_tokens=False)[-1]
        idx_y_nn = model_nnsight.tokenizer.encode("▁Dallas", add_special_tokens=False)[-1]

        graph_nnsight = attribute_nnsight(
            dallas_austin_prompt,
            model_nnsight,
            attribution_targets=[sem_nnsight],
            verbose=False,
            batch_size=256,
        )
        top_feats_nn = _get_top_features(graph_nnsight, n_top)

        input_ids_nn = model_nnsight.ensure_tokenized(dallas_austin_prompt)
        orig_logits_nn, acts_nn = model_nnsight.get_activations(input_ids_nn, sparse=True)

        interv_nn = [(ly, p, f, 10.0 * acts_nn[ly, p, f]) for (ly, p, f) in top_feats_nn]
        new_logits_nn, _ = model_nnsight.feature_intervention(input_ids_nn, interv_nn)

        orig_gap_nn = (
            (orig_logits_nn.squeeze(0)[-1, idx_x_nn] - orig_logits_nn.squeeze(0)[-1, idx_y_nn])
            .cpu()
            .item()
        )
        new_gap_nn = (
            (new_logits_nn.squeeze(0)[-1, idx_x_nn] - new_logits_nn.squeeze(0)[-1, idx_y_nn])
            .cpu()
            .item()
        )

    # --- TL backend ---
    with clean_cuda(model_tl):
        sem_tl = _build_demo_semantic_target(
            model_tl, dallas_austin_prompt, capitals, states, label, backend="transformerlens"
        )
        idx_x_tl = model_tl.tokenizer.encode("▁Austin", add_special_tokens=False)[-1]
        idx_y_tl = model_tl.tokenizer.encode("▁Dallas", add_special_tokens=False)[-1]

        graph_tl = attribute_transformerlens(
            dallas_austin_prompt,
            model_tl,
            attribution_targets=[sem_tl],
            verbose=False,
            batch_size=128,
        )
        top_feats_tl = _get_top_features(graph_tl, n_top)

        input_ids_tl = model_tl.ensure_tokenized(dallas_austin_prompt)
        orig_logits_tl, acts_tl = model_tl.get_activations(input_ids_tl, sparse=True)

        interv_tl = [(ly, p, f, 10.0 * acts_tl[ly, p, f]) for (ly, p, f) in top_feats_tl]
        new_logits_tl, _ = model_tl.feature_intervention(input_ids_tl, interv_tl)

        orig_gap_tl = (
            (orig_logits_tl.squeeze(0)[-1, idx_x_tl] - orig_logits_tl.squeeze(0)[-1, idx_y_tl])
            .cpu()
            .item()
        )
        new_gap_tl = (
            (new_logits_tl.squeeze(0)[-1, idx_x_tl] - new_logits_tl.squeeze(0)[-1, idx_y_tl])
            .cpu()
            .item()
        )

    # --- Compare on CPU ---
    assert new_gap_nn > orig_gap_nn, (
        f"NNSight: semantic amplification should widen gap, got {orig_gap_nn:.4f} -> {new_gap_nn:.4f}"
    )
    assert new_gap_tl > orig_gap_tl, (
        f"TL: semantic amplification should widen gap, got {orig_gap_tl:.4f} -> {new_gap_tl:.4f}"
    )

    assert abs(new_gap_nn - new_gap_tl) < 0.5, (
        f"Semantic post-intervention gaps differ too much: NNSight={new_gap_nn:.4f}, TL={new_gap_tl:.4f}"
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

    print("\n=== Testing Attribution Targets Demo ===")

    print("Running test_attribution_targets_string...")
    test_attribution_targets_string(models_fixture, dallas_austin)
    print("✓ Attribution targets string test passed")

    print("Running test_attribution_targets_logit_diff...")
    test_attribution_targets_logit_diff(models_fixture, dallas_austin)
    print("✓ Attribution targets logit-diff test passed")

    print("Running test_attribution_targets_logit_diff_intervention...")
    test_attribution_targets_logit_diff_intervention(models_fixture, dallas_austin)
    print("✓ Attribution targets logit-diff intervention test passed")

    print("Running test_attribution_targets_semantic...")
    test_attribution_targets_semantic(models_fixture, dallas_austin)
    print("✓ Attribution targets semantic test passed")

    print("Running test_attribution_targets_semantic_intervention...")
    test_attribution_targets_semantic_intervention(models_fixture, dallas_austin)
    print("✓ Attribution targets semantic intervention test passed")

    print("\n" + "=" * 70)
    print("All tutorial notebook tests passed! ✓")
    print("Total tests run: 24")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
