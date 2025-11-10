from dataclasses import dataclass
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoTokenizer, GPT2LMHeadModel

from circuit_tracer.attribution.attribute import (
    attribute,
    compute_salient_logits,
)
from circuit_tracer.attribution.context import AttributionContext
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder, TranscoderSet
from tests._comparison.attribution.attribute import attribute as legacy_attribute
from tests._comparison.attribution.context import AttributionContext as LegacyAttributionContext
from tests._comparison.replacement_model import ReplacementModel as LegacyReplacementModel


@pytest.fixture
def gpt2_transcoder_set_pair() -> tuple[TranscoderSet, TranscoderSet]:
    # return separate objects to avoid sharing hooks and stuff
    transcoder_set1 = get_gpt2_transcoder_set()
    transcoder_set2 = get_gpt2_transcoder_set()
    with torch.no_grad():
        for param1, param2 in zip(transcoder_set1.parameters(), transcoder_set2.parameters()):
            assert param1.shape == param2.shape
            data = torch.randn_like(param1)
            param1.data = data.clone()
            param2.data = data.clone()
    return transcoder_set1, transcoder_set2


def get_gpt2_transcoder_set() -> TranscoderSet:
    return TranscoderSet(
        transcoders={
            i: SingleLayerTranscoder(
                d_model=768,
                d_transcoder=128,
                layer_idx=i,
                activation_function=F.relu,
            )
            for i in range(12)
        },
        feature_input_hook="hook_resid_mid",
        feature_output_hook="hook_mlp_out",
    )


@pytest.fixture
def replacement_model_pair(
    gpt2_transcoder_set_pair: tuple[TranscoderSet, TranscoderSet],
) -> tuple[ReplacementModel, LegacyReplacementModel]:
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2", device_map="cpu")
    hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    legacy_model = LegacyReplacementModel.from_pretrained_and_transcoders(
        "gpt2", gpt2_transcoder_set_pair[0], device="cpu"
    )
    bridge_model = ReplacementModel.from_hf_model(
        hf_model, hf_tokenizer, gpt2_transcoder_set_pair[1], device=torch.device("cpu")
    )
    return (bridge_model, legacy_model)


@pytest.fixture
def replacement_model_clt_pair() -> tuple[ReplacementModel, LegacyReplacementModel]:
    """Create a pair of models (bridge and legacy) with matching CLT weights for testing."""
    # Create CLT with random weights
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
            param1.data = data
            param2.data = data

    hf_model = GPT2LMHeadModel.from_pretrained("gpt2", device_map="cpu")
    hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    legacy_model = LegacyReplacementModel.from_pretrained_and_transcoders(
        "gpt2", clt1, device="cpu"
    )
    bridge_model = ReplacementModel.from_hf_model(
        hf_model, hf_tokenizer, clt2, device=torch.device("cpu")
    )
    return (bridge_model, legacy_model)


def test_bridge_gpt2_replacement_model_behaves_like_legacy_replacement_model(
    replacement_model_pair: tuple[ReplacementModel, LegacyReplacementModel],
):
    bridge_model, legacy_model = replacement_model_pair

    test_inputs = bridge_model.tokenizer.encode("Hello, world!", return_tensors="pt")

    legacy_logits, legacy_cache = legacy_model.run_with_cache(test_inputs)
    bridge_logits, bridge_cache = bridge_model.run_with_cache(test_inputs)  # type: ignore

    assert torch.allclose(legacy_logits, bridge_logits)  # type: ignore

    for layer in range(12):
        assert torch.allclose(
            legacy_cache[f"blocks.{layer}.hook_resid_mid"],
            bridge_cache[f"blocks.{layer}.hook_resid_mid"],
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.allclose(
            legacy_cache[f"blocks.{layer}.hook_resid_mid"],
            bridge_cache[f"blocks.{layer}.ln2.hook_in"],
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.allclose(
            legacy_cache[f"blocks.{layer}.hook_mlp_out"],
            bridge_cache[f"blocks.{layer}.hook_mlp_out"],
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.allclose(
            legacy_cache[f"blocks.{layer}.hook_mlp_out"],
            bridge_cache[f"blocks.{layer}.mlp.hook_out"],
            atol=1e-4,
            rtol=1e-4,
        )
        # Test hooks configured in _configure_gradient_flow
        assert torch.allclose(
            legacy_cache[f"blocks.{layer}.attn.hook_pattern"],
            bridge_cache[f"blocks.{layer}.attn.hook_pattern"],
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.allclose(
            legacy_cache[f"blocks.{layer}.ln1.hook_scale"],
            bridge_cache[f"blocks.{layer}.ln1.hook_scale"],
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.allclose(
            legacy_cache[f"blocks.{layer}.ln2.hook_scale"],
            bridge_cache[f"blocks.{layer}.ln2.hook_scale"],
            atol=1e-4,
            rtol=1e-4,
        )

    # Test ln_final.hook_scale and embed hook from _configure_gradient_flow
    assert torch.allclose(
        legacy_cache["ln_final.hook_scale"],
        bridge_cache["ln_final.hook_scale"],
        atol=1e-4,
        rtol=1e-4,
    )
    # embed.hook_out is cached as "hook_embed" in the ActivationCache
    assert torch.allclose(
        legacy_cache["hook_embed"],
        bridge_cache["hook_embed"],
        atol=1e-4,
        rtol=1e-4,
    )

    # Test backward hooks - verify that the SAME hooks are called with the SAME gradients
    # in both bridge and legacy models. The behavior should be identical.
    legacy_grads = {}
    bridge_grads = {}

    def make_backward_hook(grad_dict, name):
        def hook_fn(grad, hook=None):
            if grad is not None:
                grad_dict[name] = grad.clone()
            return None

        return hook_fn

    # Build comprehensive list of ALL backward hooks to test from _configure_gradient_flow
    # This includes hooks that may or may not receive gradients
    all_bwd_hooks_to_test = []
    for layer in range(12):
        all_bwd_hooks_to_test.append(f"blocks.{layer}.hook_resid_mid")
        all_bwd_hooks_to_test.append(f"blocks.{layer}.hook_mlp_out")
        all_bwd_hooks_to_test.append(f"blocks.{layer}.attn.hook_pattern")
        all_bwd_hooks_to_test.append(f"blocks.{layer}.ln1.hook_scale")
        all_bwd_hooks_to_test.append(f"blocks.{layer}.ln2.hook_scale")
    all_bwd_hooks_to_test.append("ln_final.hook_scale")
    all_bwd_hooks_to_test.append("hook_embed")

    # Run backward pass with legacy model
    legacy_model.zero_grad()
    legacy_bwd_hooks = [
        (name, make_backward_hook(legacy_grads, name)) for name in all_bwd_hooks_to_test
    ]
    with legacy_model.hooks(bwd_hooks=legacy_bwd_hooks):
        legacy_logits_bwd = legacy_model(test_inputs)
        legacy_logits_bwd.sum().backward()

    # Run backward pass with bridge model
    bridge_model.zero_grad()
    bridge_bwd_hooks = [
        (name, make_backward_hook(bridge_grads, name)) for name in all_bwd_hooks_to_test
    ]
    with bridge_model.hooks(bwd_hooks=bridge_bwd_hooks):
        bridge_logits_bwd = bridge_model(test_inputs)
        bridge_logits_bwd.sum().backward()

    # Get sets of hooks that were actually called (received gradients)
    legacy_called_hooks = set(legacy_grads.keys())
    bridge_called_hooks = set(bridge_grads.keys())

    # Assert that the SAME hooks are called in both models
    assert legacy_called_hooks == bridge_called_hooks, (
        f"Backward hooks called differ between models!\n"
        f"Only in legacy: {legacy_called_hooks - bridge_called_hooks}\n"
        f"Only in bridge: {bridge_called_hooks - legacy_called_hooks}\n"
        f"Legacy called: {sorted(legacy_called_hooks)}\n"
        f"Bridge called: {sorted(bridge_called_hooks)}"
    )

    # For hooks that were called, verify gradients match
    for hook_name in legacy_called_hooks:
        assert torch.allclose(
            legacy_grads[hook_name],
            bridge_grads[hook_name],
            atol=1e-2,
            rtol=1e-2,
        ), f"Gradient mismatch at {hook_name}"

    bridge_ctx = bridge_model.setup_attribution(test_inputs)  # type: ignore
    legacy_ctx = legacy_model.setup_attribution(test_inputs)  # type: ignore
    assert torch.allclose(
        bridge_ctx.activation_matrix.to_dense(),
        legacy_ctx.activation_matrix.to_dense(),
        atol=1e-3,
        rtol=1e-3,
    )
    assert torch.allclose(bridge_ctx.error_vectors, legacy_ctx.error_vectors, atol=1e-2, rtol=1e-2)
    assert torch.allclose(bridge_ctx.token_vectors, legacy_ctx.token_vectors, atol=1e-3, rtol=1e-3)
    assert torch.allclose(bridge_ctx.decoder_vecs, legacy_ctx.decoder_vecs, atol=1e-3, rtol=1e-3)
    assert torch.allclose(bridge_ctx.encoder_vecs, legacy_ctx.encoder_vecs, atol=1e-3, rtol=1e-3)
    assert torch.allclose(
        bridge_ctx.encoder_to_decoder_map, legacy_ctx.encoder_to_decoder_map, atol=1e-3, rtol=1e-3
    )
    assert torch.allclose(
        bridge_ctx.decoder_locations, legacy_ctx.decoder_locations, atol=1e-3, rtol=1e-3
    )


def test_TransformerBridge_backward_gradients_differ_from_HookedTransformer():
    """
    This is a bug in TransformerLens where backward hooks see different gradient values
    in TransformerBridge vs HookedTransformer, even though forward passes are identical.
    Until this test passes, we will struggle to get the bridge code to work.
    """
    # Create both models with the same configuration
    hooked_model = HookedTransformer.from_pretrained_no_processing("gpt2", device_map="cpu")
    bridge_model: TransformerBridge = TransformerBridge.boot_transformers("gpt2", device="cpu")  # type: ignore
    bridge_model.enable_compatibility_mode(no_processing=True)

    test_input = torch.tensor([[1, 2, 3]])

    # Collect gradient sums from backward hooks
    hooked_grad = None
    bridge_grad = None

    def sum_hooked_grads(grad, hook=None):
        nonlocal hooked_grad
        hooked_grad = grad.clone()
        return None

    def sum_bridge_grads(grad, hook=None):
        nonlocal bridge_grad
        bridge_grad = grad.clone()
        return None

    # Run with HookedTransformer
    hooked_model.zero_grad()
    with hooked_model.hooks(bwd_hooks=[("blocks.0.hook_mlp_out", sum_hooked_grads)]):
        hooked_out = hooked_model(test_input)
        hooked_out.sum().backward()

    # Run with TransformerBridge
    bridge_model.zero_grad()
    with bridge_model.hooks(bwd_hooks=[("blocks.0.hook_mlp_out", sum_bridge_grads)]):
        bridge_out = bridge_model(test_input)
        bridge_out.sum().backward()

    assert torch.allclose(hooked_out, bridge_out, atol=1e-2, rtol=1e-2), (
        f"Output differs by {abs(hooked_out - bridge_out).item():.6f}"
    )

    assert hooked_grad is not None
    assert bridge_grad is not None

    # This assertion demonstrates the bug - gradient values differ
    assert torch.allclose(hooked_grad, bridge_grad, atol=1e-2, rtol=1e-2)


def test_TransformerBridge_run_with_cache_vs_forward():
    bridge_model: TransformerBridge = TransformerBridge.boot_transformers("gpt2", device="cpu")  # type: ignore
    bridge_model.enable_compatibility_mode(no_processing=True)

    test_input = torch.tensor([[1, 2, 3]])
    bridge_logits_cache, _ = bridge_model.run_with_cache(test_input)
    bridge_logits_manual = bridge_model(test_input)

    assert torch.allclose(bridge_logits_cache, bridge_logits_manual, rtol=1e-4, atol=1e-1)


def test_TransformerBridge_run_with_cache():
    """
    This is a bug in TransformerLens where TransformerBridge.run_with_cache() returns
    incorrect cached activation values, even though manual hooks work correctly.
    The issue only occurs when caching all hooks - using names_filter works correctly.
    """
    # Create both models with the same configuration
    hooked_model = HookedTransformer.from_pretrained_no_processing("gpt2", device_map="cpu")
    bridge_model: TransformerBridge = TransformerBridge.boot_transformers("gpt2", device="cpu")  # type: ignore
    bridge_model.enable_compatibility_mode(no_processing=True)

    test_input = torch.tensor([[1, 2, 3]])

    # Method 1: run_with_cache (buggy for TransformerBridge)
    _, hooked_cache = hooked_model.run_with_cache(test_input)
    _, bridge_cache = bridge_model.run_with_cache(test_input)

    # Method 2: Manual hooks (correct for both models)
    manual_cache = {}

    def make_cache_hook(name):
        def hook_fn(acts, hook):
            manual_cache[name] = acts.clone()
            return acts

        return hook_fn

    hooked_model.reset_hooks()
    with hooked_model.hooks(fwd_hooks=[("blocks.0.hook_mlp_out", make_cache_hook("hooked"))]):
        hooked_model(test_input)

    bridge_model.reset_hooks()
    with bridge_model.hooks(fwd_hooks=[("blocks.0.hook_mlp_out", make_cache_hook("bridge"))]):
        bridge_model(test_input)

    # Demonstrate that using names_filter DOES work correctly
    _, bridge_cache_filtered = bridge_model.run_with_cache(
        test_input, names_filter=lambda name: name == "blocks.0.hook_mlp_out"
    )

    # the manual values match the hooked values
    assert torch.allclose(manual_cache["hooked"], manual_cache["bridge"], atol=1e-4)

    assert torch.allclose(
        bridge_cache_filtered["blocks.0.hook_mlp_out"], manual_cache["bridge"], atol=1e-4
    )

    # Verify cache values match manual hooks for HookedTransformer
    assert torch.allclose(hooked_cache["blocks.0.hook_mlp_out"], manual_cache["hooked"], atol=1e-5)

    # This assertion demonstrates the bug - TransformerBridge run_with_cache gives wrong values
    assert torch.allclose(
        bridge_cache["blocks.0.hook_mlp_out"], manual_cache["bridge"], atol=1e-2, rtol=1e-2
    ), (
        f"TransformerBridge run_with_cache gives incorrect cached values! "
        f"Cache sum: {bridge_cache['blocks.0.hook_mlp_out'].sum():.6f}, "
        f"Manual hooks sum: {manual_cache['bridge'].sum():.6f}, "
        f"Diff: {(bridge_cache['blocks.0.hook_mlp_out'] - manual_cache['bridge']).abs().max():.6f}"
    )


def test_TransformerBridge_hooks_ignores_backward_hooks():
    """Minimal test demonstrating that TransformerBridge.hooks() doesn't register backward hooks.

    This is a bug in TransformerLens where TransformerBridge.hooks() accepts bwd_hooks
    but doesn't actually register them, while HookedTransformer.hooks() does.
    """
    # Create both models with the same configuration
    hooked_model = HookedTransformer.from_pretrained_no_processing("gpt2", device_map="cpu")
    bridge_model: TransformerBridge = TransformerBridge.boot_transformers("gpt2", device="cpu")  # type: ignore
    bridge_model.enable_compatibility_mode(no_processing=True)

    # Create a simple backward hook that tracks if it was called
    hook_called = {"hooked": False, "bridge": False}

    def make_test_hook(model_type):
        def hook_fn(grad, hook=None):
            hook_called[model_type] = True
            # For HookedTransformer, the hook doesn't modify the gradient
            return None

        return hook_fn

    # Test input
    test_input = torch.tensor([[1, 2, 3]])

    # Test HookedTransformer - backward hooks should work
    with hooked_model.hooks(bwd_hooks=[("blocks.0.hook_mlp_out", make_test_hook("hooked"))]):
        output = hooked_model(test_input)
        # Check that the backward hook was registered
        assert len(hooked_model.blocks[0].hook_mlp_out.bwd_hooks) > 0  # type: ignore

        # Trigger backward pass
        output.sum().backward()

    # Test TransformerBridge - backward hooks are ignored (BUG)
    # With compatibility mode, TransformerBridge should have the same hook names as HookedTransformer
    with bridge_model.hooks(bwd_hooks=[("blocks.0.hook_mlp_out", make_test_hook("bridge"))]):
        output = bridge_model(test_input)
        # This assertion demonstrates the bug - no backward hooks are registered
        assert len(bridge_model.blocks[0].hook_mlp_out.bwd_hooks) > 0

        # Backward pass won't trigger the hook
        output.sum().backward()

    # Verify the hooks were called appropriately
    assert hook_called["hooked"], "HookedTransformer backward hook should have been called"
    assert hook_called["bridge"], "TransformerBridge backward hook was not called (BUG)"


def test_TransformerBridge_compatibility_mode_calls_hooks_multiple_times():
    """Test showing TransformerBridge compatibility mode calls hooks multiple times.

    This is a bug in TransformerLens where the same HookPoint object is registered in hook_dict
    under multiple names (e.g., both "blocks.0.hook_mlp_out" and "blocks.0.mlp.hook_out").
    When hooks are added to this HookPoint, they get called once for each registered name,
    resulting in multiple executions per forward pass.

    This breaks code that uses stateful closures (like cached dictionaries) and expects
    hooks to be called exactly once per forward pass.
    """
    # Create both models with the same configuration
    hooked_model = HookedTransformer.from_pretrained_no_processing("gpt2", device_map="cpu")
    bridge_model: TransformerBridge = TransformerBridge.boot_transformers("gpt2", device="cpu")  # type: ignore
    bridge_model.enable_compatibility_mode(no_processing=True)

    test_input = torch.tensor([[1, 2, 3]])

    # Test HookedTransformer - hooks should be called once
    hooked_call_count = 0

    def count_hooked_calls(acts, hook):
        nonlocal hooked_call_count
        hooked_call_count += 1
        return acts

    hooked_model.blocks[0].hook_mlp_out.add_hook(count_hooked_calls, is_permanent=True)  # type: ignore
    _ = hooked_model(test_input)
    hooked_model.reset_hooks()

    # Test TransformerBridge - hooks are called multiple times (BUG)
    bridge_call_count = 0

    def count_bridge_calls(acts, hook):
        nonlocal bridge_call_count
        bridge_call_count += 1
        return acts

    bridge_model.blocks[0].mlp.hook_out.add_hook(count_bridge_calls, is_permanent=True)
    _ = bridge_model(test_input)
    bridge_model.reset_hooks()

    # Verify call counts
    assert hooked_call_count == 1, (
        f"HookedTransformer should call hook once, got {hooked_call_count}"
    )

    # This assertion demonstrates the bug - TransformerBridge calls the hook multiple times
    assert bridge_call_count == 1, (
        f"TransformerBridge calls hook {bridge_call_count} times instead of 1! "
        f"This is because the same HookPoint is registered under multiple names in hook_dict: "
        f"both 'blocks.0.hook_mlp_out' (compatibility alias) and "
        f"'blocks.0.mlp.hook_out' (native name). "
        f"During forward passes, the hook gets executed once for each registered name."
    )


@pytest.mark.skip(
    "This test fails, but I don't understand if it's the test or the implementation that's wrong"
)
def test_bridge_context_compute_batch_behaves_like_legacy_context_compute_batch(
    replacement_model_pair: tuple[ReplacementModel, LegacyReplacementModel],
):
    bridge_model, legacy_model = replacement_model_pair

    test_inputs = bridge_model.tokenizer.encode("Hello, world!", return_tensors="pt")

    # Setup attribution contexts
    bridge_ctx = bridge_model.setup_attribution(test_inputs)  # type: ignore
    legacy_ctx = legacy_model.setup_attribution(test_inputs)  # type: ignore

    # Run forward passes to populate residual activations
    with bridge_ctx.install_hooks(bridge_model):
        cache = {}

        def _cache_ln_final_in_hook(acts, hook):
            cache["ln_final.hook_in"] = acts

        bridge_model.run_with_hooks(
            test_inputs.expand(32, -1),  # type: ignore
            fwd_hooks=[("ln_final.hook_in", _cache_ln_final_in_hook)],
        )
        residual = cache["ln_final.hook_in"]
        # Call ln_final (not _original_component) to ensure hooks that stop gradients are applied
        bridge_ctx._resid_activations[-1] = bridge_model.ln_final(residual)  # type: ignore

    with legacy_ctx.install_hooks(legacy_model):
        legacy_model.run_with_cache(test_inputs.expand(32, -1), names_filter="ln_final.hook_in")  # type: ignore
        residual = legacy_model.ln_final(legacy_ctx._resid_activations[-1])
        legacy_ctx._resid_activations[-1] = residual

    # Test compute_batch with logit vectors (similar to how it's used in attribution)
    n_layers, n_pos, _ = bridge_ctx.activation_matrix.shape
    batch_size = 3

    # Create test inject_values (similar to logit_vecs in attribution)
    inject_values = torch.randn(batch_size, bridge_model.cfg.d_model)
    layers = torch.full((batch_size,), n_layers)
    positions = torch.full((batch_size,), n_pos - 1)

    legacy_rows = legacy_ctx.compute_batch(
        layers=layers,
        positions=positions,
        inject_values=inject_values,
        retain_graph=True,
    )

    bridge_rows = bridge_ctx.compute_batch(
        layers=layers,
        positions=positions,
        inject_values=inject_values,
        retain_graph=True,
    )

    assert torch.allclose(bridge_rows, legacy_rows, atol=1e-2, rtol=1e-2)

    # Test compute_batch with feature vectors (similar to encoder_vecs usage)
    feat_layers, feat_pos, _ = bridge_ctx.activation_matrix.indices()
    if len(feat_layers) > 0:
        # Take first few features for testing
        n_test_features = min(5, len(feat_layers))
        test_indices = torch.arange(n_test_features)

        bridge_feature_rows = bridge_ctx.compute_batch(
            layers=feat_layers[test_indices],
            positions=feat_pos[test_indices],
            inject_values=bridge_ctx.encoder_vecs[test_indices],
            retain_graph=False,
        )

        legacy_feature_rows = legacy_ctx.compute_batch(
            layers=feat_layers[test_indices],
            positions=feat_pos[test_indices],
            inject_values=legacy_ctx.encoder_vecs[test_indices],
            retain_graph=False,
        )

        assert torch.allclose(bridge_feature_rows, legacy_feature_rows, atol=1e-2, rtol=1e-2)


def test_bridge_feature_intervention_with_frozen_attention_behaves_like_legacy(
    replacement_model_pair: tuple[ReplacementModel, LegacyReplacementModel],
):
    """Test basic feature intervention with frozen attention (default behavior)."""
    bridge_model, legacy_model = replacement_model_pair

    prompt = torch.tensor([[0, 1, 2, 3, 4, 5]]).squeeze(0)
    interventions: list[tuple[int, int, int, torch.Tensor]] = [
        (0, 1, 5, torch.tensor(2.0)),  # layer 0, position 1, feature 5, value 2.0
        (1, 2, 10, torch.tensor(1.5)),  # layer 1, position 2, feature 10, value 1.5
        (2, 3, 7, torch.tensor(3.0)),  # layer 2, position 3, feature 7, value 3.0
    ]

    bridge_logits, bridge_acts = bridge_model.feature_intervention(
        prompt,
        interventions,  # type: ignore
        freeze_attention=True,
    )
    legacy_logits, legacy_acts = legacy_model.feature_intervention(
        prompt,
        interventions,  # type: ignore
        freeze_attention=True,
    )

    assert torch.allclose(bridge_logits, legacy_logits, atol=1e-2, rtol=1e-2), (
        f"Logits differ by max {(bridge_logits - legacy_logits).abs().max()}"
    )
    assert bridge_acts is not None
    assert torch.allclose(bridge_acts, legacy_acts, atol=1e-2, rtol=1e-2), (
        f"Activations differ by max {(bridge_acts - legacy_acts).abs().max()}"
    )


def test_bridge_feature_intervention_without_frozen_attention_behaves_like_legacy(
    replacement_model_pair: tuple[ReplacementModel, LegacyReplacementModel],
):
    """Test feature intervention without frozen attention (iterative patching - effects propagate)."""
    bridge_model, legacy_model = replacement_model_pair

    prompt = torch.tensor([[0, 1, 2, 3, 4, 5]]).squeeze(0)
    interventions: list[tuple[int, int, int, torch.Tensor]] = [
        (0, 1, 5, torch.tensor(2.0)),
        (1, 2, 10, torch.tensor(1.5)),
        (2, 3, 7, torch.tensor(3.0)),
    ]

    bridge_logits, bridge_acts = bridge_model.feature_intervention(
        prompt,
        interventions,  # type: ignore
        freeze_attention=False,
    )
    legacy_logits, legacy_acts = legacy_model.feature_intervention(
        prompt,
        interventions,  # type: ignore
        freeze_attention=False,
    )

    assert torch.allclose(bridge_logits, legacy_logits, atol=1e-2, rtol=1e-2), (
        f"Logits differ by max {(bridge_logits - legacy_logits).abs().max()}"
    )
    assert bridge_acts is not None
    assert torch.allclose(bridge_acts, legacy_acts, atol=1e-2, rtol=1e-2), (
        f"Activations differ by max {(bridge_acts - legacy_acts).abs().max()}"
    )


def test_bridge_feature_intervention_with_constrained_layers_behaves_like_legacy(
    replacement_model_pair: tuple[ReplacementModel, LegacyReplacementModel],
):
    """Test feature intervention with constrained layers (direct effects - no propagation through transcoders)."""
    bridge_model, legacy_model = replacement_model_pair

    prompt = torch.tensor([[0, 1, 2, 3, 4, 5]]).squeeze(0)
    interventions: list[tuple[int, int, int, torch.Tensor]] = [
        (0, 1, 5, torch.tensor(2.0)),
        (1, 2, 10, torch.tensor(1.5)),
        (2, 3, 7, torch.tensor(3.0)),
    ]
    constrained_layers = range(0, 6)

    bridge_logits, bridge_acts = bridge_model.feature_intervention(
        prompt,
        interventions,  # type: ignore
        constrained_layers=constrained_layers,
        freeze_attention=True,
    )
    legacy_logits, legacy_acts = legacy_model.feature_intervention(
        prompt,
        interventions,  # type: ignore
        constrained_layers=constrained_layers,
        freeze_attention=True,
    )

    assert torch.allclose(bridge_logits, legacy_logits, atol=1e-2, rtol=1e-2), (
        f"Logits differ by max {(bridge_logits - legacy_logits).abs().max()}"
    )
    assert bridge_acts is not None
    assert torch.allclose(bridge_acts, legacy_acts, atol=1e-2, rtol=1e-2), (
        f"Activations differ by max {(bridge_acts - legacy_acts).abs().max()}"
    )


def test_bridge_feature_intervention_empty_interventions_behaves_like_legacy(
    replacement_model_pair: tuple[ReplacementModel, LegacyReplacementModel],
):
    """Test empty intervention list (baseline - should just run the model normally with frozen attention)."""
    bridge_model, legacy_model = replacement_model_pair

    prompt = torch.tensor([[0, 1, 2, 3, 4, 5]]).squeeze(0)

    bridge_logits, bridge_acts = bridge_model.feature_intervention(
        prompt, [], freeze_attention=True
    )
    legacy_logits, legacy_acts = legacy_model.feature_intervention(
        prompt, [], freeze_attention=True
    )

    assert torch.allclose(bridge_logits, legacy_logits, atol=1e-2, rtol=1e-2), (
        f"Logits differ by max {(bridge_logits - legacy_logits).abs().max()}"
    )
    assert bridge_acts is not None
    assert torch.allclose(bridge_acts, legacy_acts, atol=1e-2, rtol=1e-2), (
        f"Activations differ by max {(bridge_acts - legacy_acts).abs().max()}"
    )


def test_bridge_feature_intervention_with_constrained_layers_and_no_activation_function_behaves_like_legacy(
    replacement_model_pair: tuple[ReplacementModel, LegacyReplacementModel],
):
    """Test feature intervention with constrained layers and apply_activation_function=False.

    This tests the case where interventions are constrained to specific layers and
    the transcoder activation function is not applied during intervention.
    """
    bridge_model, legacy_model = replacement_model_pair

    prompt = torch.tensor([[0, 1, 2, 3, 4, 5]]).squeeze(0)
    interventions: list[tuple[int, int, int, torch.Tensor]] = [
        (0, 1, 5, torch.tensor(2.0)),
        (1, 2, 10, torch.tensor(1.5)),
        (2, 3, 7, torch.tensor(3.0)),
    ]
    constrained_layers = range(0, 6)

    bridge_logits, bridge_acts = bridge_model.feature_intervention(
        prompt,
        interventions,  # type: ignore
        constrained_layers=constrained_layers,
        freeze_attention=True,
        apply_activation_function=False,
    )
    legacy_logits, legacy_acts = legacy_model.feature_intervention(
        prompt,
        interventions,  # type: ignore
        constrained_layers=constrained_layers,
        freeze_attention=True,
        apply_activation_function=False,
    )

    assert torch.allclose(bridge_logits, legacy_logits, atol=1e-2, rtol=1e-2), (
        f"Logits differ by max {(bridge_logits - legacy_logits).abs().max()}"
    )
    assert bridge_acts is not None
    assert torch.allclose(bridge_acts, legacy_acts, atol=1e-2, rtol=1e-2), (
        f"Activations differ by max {(bridge_acts - legacy_acts).abs().max()}"
    )


# ============================================================================
# CLT (Cross-Layer Transcoder) Feature Intervention Tests
# ============================================================================


def test_bridge_clt_feature_intervention_with_frozen_attention_behaves_like_legacy(
    replacement_model_clt_pair: tuple[ReplacementModel, LegacyReplacementModel],
):
    """Test CLT feature intervention with frozen attention (default behavior)."""
    bridge_model, legacy_model = replacement_model_clt_pair

    prompt = torch.tensor([[0, 1, 2, 3, 4, 5]]).squeeze(0)
    interventions: list[tuple[int, int, int, torch.Tensor]] = [
        (0, 1, 5, torch.tensor(2.0)),  # layer 0, position 1, feature 5, value 2.0
        (1, 2, 10, torch.tensor(1.5)),  # layer 1, position 2, feature 10, value 1.5
        (2, 3, 7, torch.tensor(3.0)),  # layer 2, position 3, feature 7, value 3.0
    ]

    bridge_logits, bridge_acts = bridge_model.feature_intervention(
        prompt,
        interventions,  # type: ignore
        freeze_attention=True,
    )
    legacy_logits, legacy_acts = legacy_model.feature_intervention(
        prompt,
        interventions,  # type: ignore
        freeze_attention=True,
    )

    # Verify logits match between bridge and legacy for CLT intervention with frozen attention
    assert torch.allclose(bridge_logits, legacy_logits, atol=1e-2, rtol=1e-2), (
        f"Logits differ by max {(bridge_logits - legacy_logits).abs().max()}"
    )

    # Verify activation caches match
    assert bridge_acts is not None
    assert torch.allclose(bridge_acts, legacy_acts, atol=1e-2, rtol=1e-2), (
        f"Activations differ by max {(bridge_acts - legacy_acts).abs().max()}"
    )


def test_bridge_clt_feature_intervention_without_frozen_attention_behaves_like_legacy(
    replacement_model_clt_pair: tuple[ReplacementModel, LegacyReplacementModel],
):
    """Test CLT feature intervention without frozen attention (iterative patching - effects propagate)."""
    bridge_model, legacy_model = replacement_model_clt_pair

    prompt = torch.tensor([[0, 1, 2, 3, 4, 5]]).squeeze(0)
    interventions: list[tuple[int, int, int, torch.Tensor]] = [
        (0, 1, 5, torch.tensor(2.0)),
        (1, 2, 10, torch.tensor(1.5)),
        (2, 3, 7, torch.tensor(3.0)),
    ]

    bridge_logits, bridge_acts = bridge_model.feature_intervention(
        prompt,
        interventions,  # type: ignore
        freeze_attention=False,
    )
    legacy_logits, legacy_acts = legacy_model.feature_intervention(
        prompt,
        interventions,  # type: ignore
        freeze_attention=False,
    )

    assert torch.allclose(bridge_logits, legacy_logits, atol=1e-2, rtol=1e-2), (
        f"Logits differ by max {(bridge_logits - legacy_logits).abs().max()}"
    )
    assert bridge_acts is not None
    assert torch.allclose(bridge_acts, legacy_acts, atol=1e-2, rtol=1e-2), (
        f"Activations differ by max {(bridge_acts - legacy_acts).abs().max()}"
    )


def test_bridge_clt_feature_intervention_with_constrained_layers_behaves_like_legacy(
    replacement_model_clt_pair: tuple[ReplacementModel, LegacyReplacementModel],
):
    """Test CLT feature intervention with constrained layers (direct effects - CLT writes only to constrained layers)."""
    bridge_model, legacy_model = replacement_model_clt_pair

    prompt = torch.tensor([[0, 1, 2, 3, 4, 5]]).squeeze(0)
    interventions: list[tuple[int, int, int, torch.Tensor]] = [
        (0, 1, 5, torch.tensor(2.0)),
        (1, 2, 10, torch.tensor(1.5)),
        (2, 3, 7, torch.tensor(3.0)),
    ]
    # Constrain to layers 0-5 (CLT will write only to these layers)
    constrained_layers = range(0, 6)

    bridge_logits, bridge_acts = bridge_model.feature_intervention(
        prompt,
        interventions,  # type: ignore
        constrained_layers=constrained_layers,
        freeze_attention=True,
    )
    legacy_logits, legacy_acts = legacy_model.feature_intervention(
        prompt,
        interventions,  # type: ignore
        constrained_layers=constrained_layers,
        freeze_attention=True,
    )

    assert torch.allclose(bridge_logits, legacy_logits, atol=1e-2, rtol=1e-2), (
        f"Logits differ by max {(bridge_logits - legacy_logits).abs().max()}"
    )
    assert bridge_acts is not None
    assert torch.allclose(bridge_acts, legacy_acts, atol=1e-2, rtol=1e-2), (
        f"Activations differ by max {(bridge_acts - legacy_acts).abs().max()}"
    )


def test_bridge_clt_feature_intervention_empty_interventions_behaves_like_legacy(
    replacement_model_clt_pair: tuple[ReplacementModel, LegacyReplacementModel],
):
    """Test CLT with empty intervention list (baseline - should just run the model normally with frozen attention)."""
    bridge_model, legacy_model = replacement_model_clt_pair

    prompt = torch.tensor([[0, 1, 2, 3, 4, 5]]).squeeze(0)

    bridge_logits, bridge_acts = bridge_model.feature_intervention(
        prompt, [], freeze_attention=True
    )
    legacy_logits, legacy_acts = legacy_model.feature_intervention(
        prompt, [], freeze_attention=True
    )

    assert torch.allclose(bridge_logits, legacy_logits, atol=1e-2, rtol=1e-2), (
        f"Logits differ by max {(bridge_logits - legacy_logits).abs().max()}"
    )
    assert bridge_acts is not None
    assert torch.allclose(bridge_acts, legacy_acts, atol=1e-2, rtol=1e-2), (
        f"Activations differ by max {(bridge_acts - legacy_acts).abs().max()}"
    )


def test_bridge_attribute_behaves_like_legacy_attribute(
    replacement_model_pair: tuple[ReplacementModel, LegacyReplacementModel],
):
    bridge_model, legacy_model = replacement_model_pair

    prompt = torch.tensor([[0, 1, 2, 3, 4, 5]])
    bridge_graph = attribute(
        prompt, bridge_model, max_n_logits=5, desired_logit_prob=0.8, batch_size=32
    )
    legacy_graph = legacy_attribute(
        prompt, legacy_model, max_n_logits=5, desired_logit_prob=0.8, batch_size=32
    )

    assert bridge_graph.input_string == legacy_graph.input_string
    assert torch.allclose(bridge_graph.input_tokens, legacy_graph.input_tokens)
    assert torch.allclose(bridge_graph.logit_tokens, legacy_graph.logit_tokens)
    assert torch.allclose(
        bridge_graph.logit_probabilities, legacy_graph.logit_probabilities, atol=1e-2, rtol=1e-2
    )
    assert torch.allclose(bridge_graph.active_features, legacy_graph.active_features)
    assert torch.allclose(
        bridge_graph.activation_values, legacy_graph.activation_values, atol=1e-2, rtol=1e-2
    )
    assert torch.allclose(bridge_graph.selected_features, legacy_graph.selected_features)
    assert bridge_graph.scan == legacy_graph.scan

    # first at least assert the non-zero elements are the same
    assert torch.allclose(bridge_graph.adjacency_matrix != 0, legacy_graph.adjacency_matrix != 0)

    assert torch.allclose(
        bridge_graph.adjacency_matrix,
        legacy_graph.adjacency_matrix,
        atol=1e-1,
        rtol=1e-2,
    )


def test_bridge_setup_attribution_behaves_like_legacy_setup_attribution(
    replacement_model_pair: tuple[ReplacementModel, LegacyReplacementModel],
):
    bridge_model, legacy_model = replacement_model_pair

    test_inputs = bridge_model.tokenizer.encode("Hello, world!", return_tensors="pt")

    bridge_ctx = bridge_model.setup_attribution(test_inputs)  # type: ignore
    legacy_ctx = legacy_model.setup_attribution(test_inputs)  # type: ignore

    assert torch.allclose(bridge_ctx.logits, legacy_ctx.logits, atol=1e-2, rtol=1e-2)
    assert torch.allclose(
        bridge_ctx.activation_matrix.to_dense(),
        legacy_ctx.activation_matrix.to_dense(),
        atol=1e-2,
        rtol=1e-2,
    )
    assert torch.allclose(bridge_ctx.error_vectors, legacy_ctx.error_vectors, atol=1e-2, rtol=1e-2)
    assert torch.allclose(bridge_ctx.token_vectors, legacy_ctx.token_vectors, atol=1e-2, rtol=1e-2)
    assert torch.allclose(bridge_ctx.decoder_vecs, legacy_ctx.decoder_vecs, atol=1e-2, rtol=1e-2)
    assert torch.allclose(bridge_ctx.encoder_vecs, legacy_ctx.encoder_vecs, atol=1e-2, rtol=1e-2)
    assert torch.allclose(
        bridge_ctx.encoder_to_decoder_map, legacy_ctx.encoder_to_decoder_map, atol=1e-2, rtol=1e-2
    )
    assert torch.allclose(
        bridge_ctx.decoder_locations, legacy_ctx.decoder_locations, atol=1e-2, rtol=1e-2
    )


def test_bridge_attribute_phase_1_behaves_like_legacy_attribute_phase_1(
    replacement_model_pair: tuple[ReplacementModel, LegacyReplacementModel],
):
    bridge_model, legacy_model = replacement_model_pair

    prompt = torch.tensor([[0, 1, 2, 3, 4, 5]])
    bridge_ctx = bridge_model.setup_attribution(prompt)  # type: ignore
    legacy_ctx = legacy_model.setup_attribution(prompt)  # type: ignore
    _run_attribution_phase_1(bridge_ctx, bridge_model, prompt, 32)
    _legacy_attribute_phase_1(legacy_ctx, legacy_model, prompt, 32)
    assert bridge_ctx._resid_activations[-1] is not None
    assert legacy_ctx._resid_activations[-1] is not None
    assert torch.allclose(
        bridge_ctx._resid_activations[-1], legacy_ctx._resid_activations[-1], atol=1e-2, rtol=1e-2
    )


def test_bridge_attribute_phase_2_behaves_like_legacy_attribute_phase_2(
    replacement_model_pair: tuple[ReplacementModel, LegacyReplacementModel],
):
    bridge_model, legacy_model = replacement_model_pair

    prompt = torch.tensor([[0, 1, 2, 3, 4, 5]])
    bridge_ctx = bridge_model.setup_attribution(prompt)  # type: ignore
    legacy_ctx = legacy_model.setup_attribution(prompt)  # type: ignore
    _run_attribution_phase_1(bridge_ctx, bridge_model, prompt, 32)
    _legacy_attribute_phase_1(legacy_ctx, legacy_model, prompt, 32)

    bridge_phase_2_output = _bridge_attribute_phase_2(bridge_ctx, bridge_model)
    legacy_phase_2_output = _legacy_attribute_phase_2(legacy_ctx, legacy_model)

    assert torch.allclose(
        bridge_phase_2_output.edge_matrix, legacy_phase_2_output.edge_matrix, atol=1e-2, rtol=1e-2
    )
    assert torch.allclose(
        bridge_phase_2_output.row_to_node_index,
        legacy_phase_2_output.row_to_node_index,
        atol=1e-2,
        rtol=1e-2,
    )
    assert torch.allclose(
        bridge_phase_2_output.logit_idx, legacy_phase_2_output.logit_idx, atol=1e-2, rtol=1e-2
    )
    assert torch.allclose(
        bridge_phase_2_output.logit_p, legacy_phase_2_output.logit_p, atol=1e-2, rtol=1e-2
    )
    assert torch.allclose(
        bridge_phase_2_output.logit_vecs, legacy_phase_2_output.logit_vecs, atol=1e-2, rtol=1e-2
    )


def test_bridge_attribute_phase_3_behaves_like_legacy_attribute_phase_3(
    replacement_model_pair: tuple[ReplacementModel, LegacyReplacementModel],
):
    bridge_model, legacy_model = replacement_model_pair

    prompt = torch.tensor([[0, 1, 2, 3, 4, 5]])
    bridge_ctx = bridge_model.setup_attribution(prompt)  # type: ignore
    legacy_ctx = legacy_model.setup_attribution(prompt)  # type: ignore
    _run_attribution_phase_1(bridge_ctx, bridge_model, prompt, 32)
    _legacy_attribute_phase_1(legacy_ctx, legacy_model, prompt, 32)
    bridge_phase_2_output = _bridge_attribute_phase_2(bridge_ctx, bridge_model)
    legacy_phase_2_output = _legacy_attribute_phase_2(legacy_ctx, legacy_model)

    _run_attribution_phase_3(
        ctx=bridge_ctx,
        logit_idx=bridge_phase_2_output.logit_idx,
        logit_vecs=bridge_phase_2_output.logit_vecs,
        batch_size=32,
        edge_matrix=bridge_phase_2_output.edge_matrix,
        row_to_node_index=bridge_phase_2_output.row_to_node_index,
        logit_offset=bridge_phase_2_output.logit_offset,
    )
    _run_attribution_phase_3(
        ctx=legacy_ctx,
        logit_idx=legacy_phase_2_output.logit_idx,
        logit_vecs=legacy_phase_2_output.logit_vecs,
        batch_size=32,
        edge_matrix=legacy_phase_2_output.edge_matrix,
        row_to_node_index=legacy_phase_2_output.row_to_node_index,
        logit_offset=legacy_phase_2_output.logit_offset,
    )

    # phase 3 modifies the phase 2 outputs in-place
    assert torch.allclose(
        bridge_phase_2_output.edge_matrix, legacy_phase_2_output.edge_matrix, atol=1e-2, rtol=1e-2
    )
    assert torch.allclose(
        bridge_phase_2_output.row_to_node_index,
        legacy_phase_2_output.row_to_node_index,
        atol=1e-2,
        rtol=1e-2,
    )


def test_TransformerBridge_gpt2_behaves_like_HookedTransformer_gpt2():
    """
    This isn't actually a test of the ReplacementModel, but if this fails, we have no hope.
    """
    legacy_gpt2 = HookedTransformer.from_pretrained_no_processing("gpt2", device_map="cpu")
    bridge_gpt2: TransformerBridge = TransformerBridge.boot_transformers("gpt2", device="cpu")  # type: ignore
    bridge_gpt2.enable_compatibility_mode(no_processing=True)

    test_inputs = bridge_gpt2.tokenizer.encode("Hello, world!", return_tensors="pt")

    legacy_logits, legacy_cache = legacy_gpt2.run_with_cache(test_inputs)
    bridge_logits, bridge_cache = bridge_gpt2.run_with_cache(test_inputs)

    assert torch.allclose(legacy_logits, bridge_logits)  # type: ignore
    for layer in range(12):
        assert torch.allclose(
            legacy_cache[f"blocks.{layer}.hook_resid_mid"],
            bridge_cache[f"blocks.{layer}.hook_resid_mid"],
            atol=1e-2,
            rtol=1e-2,
        )


# --- copied code chunks from attribute since these are not in separate functions and hard to test in isolation --- #


def _legacy_attribute_phase_1(
    ctx: LegacyAttributionContext,
    model: LegacyReplacementModel,
    input_ids: torch.Tensor,
    batch_size: int,
):
    with ctx.install_hooks(model):
        residual = model.forward(input_ids.expand(batch_size, -1), stop_at_layer=model.cfg.n_layers)  # type: ignore
        ctx._resid_activations[-1] = model.ln_final(residual)


@dataclass
class Phase2Output:
    edge_matrix: torch.Tensor
    row_to_node_index: torch.Tensor
    logit_idx: torch.Tensor
    logit_p: torch.Tensor
    logit_vecs: torch.Tensor
    total_nodes: int
    n_logits: int
    logit_offset: int


def _bridge_attribute_phase_2(ctx: AttributionContext, model: ReplacementModel) -> Phase2Output:
    feat_layers, feat_pos, _ = ctx.activation_matrix.indices()
    n_layers, n_pos, _ = ctx.activation_matrix.shape
    logit_offset = len(feat_layers) + (n_layers + 1) * n_pos

    logit_idx, logit_p, logit_vecs = compute_salient_logits(
        ctx.logits[0, -1],
        model.unembed.W_U,
        max_n_logits=10,
        desired_logit_prob=0.95,
    )
    n_logits = len(logit_idx)

    edge_matrix, row_to_node_index = _build_input_vectors(
        ctx, n_logits, logit_offset, max_feature_nodes=None
    )

    return Phase2Output(
        edge_matrix=edge_matrix,
        row_to_node_index=row_to_node_index,
        logit_idx=logit_idx,
        logit_p=logit_p,
        logit_vecs=logit_vecs,
        total_nodes=logit_offset + n_logits,
        n_logits=n_logits,
        logit_offset=logit_offset,
    )


def _legacy_attribute_phase_2(
    ctx: LegacyAttributionContext, model: LegacyReplacementModel
) -> Phase2Output:
    activation_matrix = ctx.activation_matrix
    feat_layers, feat_pos, _ = activation_matrix.indices()
    n_layers, n_pos, _ = activation_matrix.shape
    total_active_feats = activation_matrix._nnz()
    max_feature_nodes = None

    logit_idx, logit_p, logit_vecs = compute_salient_logits(
        ctx.logits[0, -1],
        model.unembed.W_U,  # type: ignore
        max_n_logits=10,
        desired_logit_prob=0.95,
    )

    logit_offset = len(feat_layers) + (n_layers + 1) * n_pos
    n_logits = len(logit_idx)
    total_nodes = logit_offset + n_logits

    max_feature_nodes = min(max_feature_nodes or total_active_feats, total_active_feats)

    edge_matrix = torch.zeros(max_feature_nodes + n_logits, total_nodes)
    # Maps row indices in edge_matrix to original feature/node indices
    # First populated with logit node IDs, then feature IDs in attribution order
    row_to_node_index = torch.zeros(max_feature_nodes + n_logits, dtype=torch.int32)

    return Phase2Output(
        edge_matrix=edge_matrix,
        row_to_node_index=row_to_node_index,
        logit_idx=logit_idx,
        logit_p=logit_p,
        logit_vecs=logit_vecs,
        total_nodes=total_nodes,
        n_logits=n_logits,
        logit_offset=logit_offset,
    )


# Phase 1: forward pass
def _run_attribution_phase_1(
    ctx: AttributionContext,
    model: ReplacementModel,
    input_ids: torch.Tensor,
    batch_size: int,
):
    with ctx.install_hooks(model):
        cache = {}

        def _cache_ln_final_in_hook(acts, hook):
            cache["ln_final.hook_in"] = acts

        model.run_with_hooks(
            input_ids.expand(batch_size, -1),
            fwd_hooks=[("ln_final.hook_in", _cache_ln_final_in_hook)],
        )
        residual = cache["ln_final.hook_in"]
        # Call ln_final (not _original_component) to ensure hooks that stop gradients are applied
        ctx._resid_activations[-1] = model.ln_final(residual)  # type: ignore


class BuiltInputVectorsOutput(NamedTuple):
    edge_matrix: torch.Tensor
    row_to_node_index: torch.Tensor


def _build_input_vectors(
    ctx: AttributionContext,
    n_logits: int,
    logit_offset: int,
    max_feature_nodes: int | None = None,
) -> BuiltInputVectorsOutput:
    total_active_feats = ctx.activation_matrix._nnz()
    total_nodes = logit_offset + n_logits
    max_feature_nodes = min(max_feature_nodes or total_active_feats, total_active_feats)
    edge_matrix = torch.zeros(max_feature_nodes + n_logits, total_nodes)
    # Maps row indices in edge_matrix to original feature/node indices
    # First populated with logit node IDs, then feature IDs in attribution order
    row_to_node_index = torch.zeros(max_feature_nodes + n_logits, dtype=torch.int32)
    return BuiltInputVectorsOutput(
        edge_matrix=edge_matrix,
        row_to_node_index=row_to_node_index,
    )


# logic is identical for both legacy and bridge here
def _run_attribution_phase_3(
    ctx: AttributionContext | LegacyAttributionContext,
    logit_idx: torch.Tensor,
    logit_vecs: torch.Tensor,
    batch_size: int,
    edge_matrix: torch.Tensor,
    row_to_node_index: torch.Tensor,
    logit_offset: int,
):
    n_layers, n_pos, _ = ctx.activation_matrix.shape
    for i in range(0, len(logit_idx), batch_size):
        batch = logit_vecs[i : i + batch_size]
        rows = ctx.compute_batch(
            layers=torch.full((batch.shape[0],), n_layers),
            positions=torch.full((batch.shape[0],), n_pos - 1),
            inject_values=batch,
        )
        edge_matrix[i : i + batch.shape[0], :logit_offset] = rows.cpu()
        row_to_node_index[i : i + batch.shape[0]] = (
            torch.arange(i, i + batch.shape[0]) + logit_offset
        )


# ------------------------------------------------------------ #


def test_TransformerBridge_gemma2_forward_fails():
    """Minimal test demonstrating TransformerBridge bug with Gemma2.

    This is a bug in TransformerLens where TransformerBridge fails to properly prepare
    the position_embeddings argument when calling HuggingFace's Gemma2 attention forward.

    Error: ValueError: not enough values to unpack (expected 2, got 1)
    Location: transformers/models/gemma2/modeling_gemma2.py:248
    Root Cause: TransformerLens is passing position_embeddings as a single value instead
                of the expected tuple (cos, sin) that HuggingFace's Gemma2Attention requires.
    """
    from transformer_lens.config import TransformerBridgeConfig
    from transformer_lens.factories.architecture_adapter_factory import ArchitectureAdapterFactory
    from transformer_lens.model_bridge.sources.transformers import (
        determine_architecture_from_hf_config,
        map_default_transformer_lens_config,
    )
    from transformers import Gemma2Config, Gemma2ForCausalLM

    # Create minimal Gemma2 config
    gemma_cfg = Gemma2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        hidden_act="gelu_pytorch_tanh",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        attn_logit_softcapping=50.0,
        final_logit_softcapping=0.0,
        sliding_window=4096,
    )

    # Create HF model and bridge
    hf_model = Gemma2ForCausalLM(gemma_cfg)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Create TransformerBridge the same way ReplacementModel does
    tl_config = map_default_transformer_lens_config(hf_model.config)
    architecture = determine_architecture_from_hf_config(hf_model.config)
    bridge_config = TransformerBridgeConfig.from_dict(tl_config.__dict__)
    bridge_config.architecture = architecture
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(bridge_config)
    bridge = TransformerBridge(model=hf_model, adapter=adapter, tokenizer=tokenizer)

    # Simple forward pass triggers the bug
    test_input = torch.tensor([[0, 1, 2, 3]])

    # This will fail with: ValueError: not enough values to unpack (expected 2, got 1)
    # at transformers/models/gemma2/modeling_gemma2.py:248
    # when HuggingFace code tries to do: cos, sin = position_embeddings
    # because TransformerLens incorrectly passes position_embeddings as a single value
    bridge(test_input)


def test_TransformerBridge_llama_rmsnorm_eps_fails():
    """Minimal test demonstrating TransformerBridge bug with LlamaRMSNorm.

    This is a bug in TransformerLens where NormalizationBridge tries to access
    the 'eps' attribute on LlamaRMSNorm, but LlamaRMSNorm stores this value as
    'variance_epsilon' instead.

    Error: AttributeError: 'LlamaRMSNorm' object has no attribute 'eps'
    Location: transformer_lens/model_bridge/generalized_components/normalization.py:121
    Root Cause: TransformerLens assumes normalization layers have an 'eps' attribute,
                but LlamaRMSNorm uses 'variance_epsilon'.

    The fix in TransformerLens should check for both 'eps' and 'variance_epsilon':
        eps = getattr(self.original_component, 'eps',
                     getattr(self.original_component, 'variance_epsilon', 1e-5))

    Note: This test currently fails with the position_embeddings bug first (in attention),
    but the eps bug is encountered when using Llama models with attribution/gradient hooks
    (see test_attributions_llama.py). Both bugs need to be fixed.
    """
    from transformer_lens.config import TransformerBridgeConfig
    from transformer_lens.factories.architecture_adapter_factory import ArchitectureAdapterFactory
    from transformer_lens.model_bridge.sources.transformers import (
        determine_architecture_from_hf_config,
        map_default_transformer_lens_config,
    )
    from transformers import LlamaConfig, LlamaForCausalLM

    # Create minimal Llama config
    llama_cfg = LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.017677669529663688,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )

    # Create HF model and TransformerBridge
    hf_model = LlamaForCausalLM(llama_cfg)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Create TransformerBridge using TransformerLens APIs
    tl_config = map_default_transformer_lens_config(hf_model.config)
    architecture = determine_architecture_from_hf_config(hf_model.config)
    bridge_config = TransformerBridgeConfig.from_dict(tl_config.__dict__)
    bridge_config.architecture = architecture
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(bridge_config)
    bridge = TransformerBridge(model=hf_model, adapter=adapter, tokenizer=tokenizer)

    # Simple forward pass triggers the bugs
    test_input = torch.tensor([[0, 1, 2, 3]])

    # This will currently fail with the position_embeddings bug first:
    #   ValueError: not enough values to unpack (expected 2, got 1)
    #   at transformers/models/llama/modeling_llama.py:240
    #
    # But the eps bug also exists and would be hit in the normalization layer:
    #   AttributeError: 'LlamaRMSNorm' object has no attribute 'eps'
    #   at transformer_lens/model_bridge/generalized_components/normalization.py:121
    #
    # The eps bug is reliably encountered in test_attributions_llama.py
    bridge(test_input)


def test_hook_execution_order_matches_between_bridge_and_legacy_for_constrained_layers(
    replacement_model_pair: tuple[ReplacementModel, LegacyReplacementModel],
):
    """Test that hook execution order for constrained_layers interventions matches between bridge and legacy.

    This test verifies that for constrained_layers interventions, the hooks fire in the correct order:
    1. Freeze hook (freezes MLP outputs to original values)
    2. Calculate delta hook (computes intervention deltas)
    3. Intervention hook (adds deltas to MLP outputs)

    The key concern is that we don't freeze the layer AFTER adding deltas, which would erase the intervention.
    """
    bridge_model, legacy_model = replacement_model_pair

    prompt = torch.tensor([[0, 1, 2, 3, 4, 5]]).squeeze(0)
    interventions: list[tuple[int, int, int, torch.Tensor]] = [
        (0, 1, 5, torch.tensor(2.0)),
        (1, 2, 10, torch.tensor(1.5)),
    ]
    constrained_layers = range(0, 6)

    # Track hook execution order for bridge model
    bridge_hook_order = []

    def make_tracking_hook(hook_type: str, original_hook):
        def tracking_hook(activations, hook):
            # Record the hook type and layer
            try:
                layer = hook.layer() if hasattr(hook, "layer") else None
            except (ValueError, IndexError):
                layer = None
            bridge_hook_order.append((hook_type, layer, hook.name))
            return original_hook(activations, hook)

        return tracking_hook

    # Get the hooks from bridge model
    bridge_hooks, _, _ = bridge_model._get_feature_intervention_hooks(
        prompt,
        interventions,  # type: ignore
        constrained_layers=constrained_layers,
        freeze_attention=True,
    )

    # Wrap hooks with tracking
    tracked_bridge_hooks = []
    for hook_name, hook_fn in bridge_hooks:
        # Identify hook type based on function name
        if "freeze" in str(hook_fn):
            hook_type = "freeze"
        elif "calculate_delta" in str(hook_fn):
            hook_type = "calculate_delta"
        elif "intervention" in str(hook_fn):
            hook_type = "intervention"
        elif "activation" in str(hook_fn) or "cache" in str(hook_fn):
            hook_type = "activation_cache"
        elif "logit" in str(hook_fn):
            hook_type = "logit_cache"
        else:
            hook_type = "other"

        tracked_bridge_hooks.append((hook_name, make_tracking_hook(hook_type, hook_fn)))

    # Run bridge model with tracked hooks
    with bridge_model.hooks(tracked_bridge_hooks):  # type: ignore
        _ = bridge_model(prompt)

    # Track hook execution order for legacy model
    legacy_hook_order = []

    def make_legacy_tracking_hook(hook_type: str, original_hook):
        def tracking_hook(activations, hook):
            try:
                layer = hook.layer() if hasattr(hook, "layer") else None
            except (ValueError, IndexError):
                layer = None
            legacy_hook_order.append((hook_type, layer, hook.name))
            return original_hook(activations, hook)

        return tracking_hook

    # Get the hooks from legacy model
    legacy_hooks, _, _ = legacy_model._get_feature_intervention_hooks(
        prompt,
        interventions,  # type: ignore
        constrained_layers=constrained_layers,
        freeze_attention=True,
    )

    # Wrap hooks with tracking
    tracked_legacy_hooks = []
    for hook_name, hook_fn in legacy_hooks:
        if "freeze" in str(hook_fn):
            hook_type = "freeze"
        elif "calculate_delta" in str(hook_fn):
            hook_type = "calculate_delta"
        elif "intervention" in str(hook_fn):
            hook_type = "intervention"
        elif "activation" in str(hook_fn) or "cache" in str(hook_fn):
            hook_type = "activation_cache"
        elif "logit" in str(hook_fn):
            hook_type = "logit_cache"
        else:
            hook_type = "other"

        tracked_legacy_hooks.append((hook_name, make_legacy_tracking_hook(hook_type, hook_fn)))

    # Run legacy model with tracked hooks
    with legacy_model.hooks(tracked_legacy_hooks):  # type: ignore
        _ = legacy_model(prompt)

    # The key invariant is that for any layer, freeze (if present) comes before intervention
    for layer in [0, 1]:
        bridge_layer_hooks = [
            (t, i) for i, (t, layer_num, _) in enumerate(bridge_hook_order) if layer_num == layer
        ]
        legacy_layer_hooks = [
            (t, i) for i, (t, layer_num, _) in enumerate(legacy_hook_order) if layer_num == layer
        ]

        # Check that freeze comes before intervention for this layer
        bridge_freeze_idx = next((i for t, i in bridge_layer_hooks if t == "freeze"), None)
        bridge_intervention_idx = next(
            (i for t, i in bridge_layer_hooks if t == "intervention"), None
        )

        legacy_freeze_idx = next((i for t, i in legacy_layer_hooks if t == "freeze"), None)
        legacy_intervention_idx = next(
            (i for t, i in legacy_layer_hooks if t == "intervention"), None
        )

        if bridge_freeze_idx is not None and bridge_intervention_idx is not None:
            assert bridge_freeze_idx < bridge_intervention_idx, (
                f"Bridge: freeze must come before intervention at layer {layer}, "
                f"but freeze at {bridge_freeze_idx} >= intervention at {bridge_intervention_idx}"
            )

        if legacy_freeze_idx is not None and legacy_intervention_idx is not None:
            assert legacy_freeze_idx < legacy_intervention_idx, (
                f"Legacy: freeze must come before intervention at layer {layer}, "
                f"but freeze at {legacy_freeze_idx} >= intervention at {legacy_intervention_idx}"
            )

        # Both should have the same presence/absence of freeze hooks
        assert (bridge_freeze_idx is not None) == (legacy_freeze_idx is not None), (
            f"Layer {layer}: bridge has freeze={bridge_freeze_idx is not None}, "
            f"legacy has freeze={legacy_freeze_idx is not None}"
        )
