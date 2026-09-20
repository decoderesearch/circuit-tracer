"""The interp_engine backend must agree with nnsight on a *skip* transcoder set.

The gemma-2 parity test covers the ordinary path against TransformerLens. This file exists for one
reason: `mntss/gemma-scope-transcoders` has no skip connection, so nothing there reaches the branch
that adds `input_acts @ W_skip` to each layer's reconstruction, nor the skip-difference hooks the
interventions build on top of it. The gemma-scope-2 `*_small_affine` sets do carry `W_skip`.

A skip connection is worth its own test because it adds a second route from a layer's *input* to its
output, one that bypasses the features entirely. The backends read that input independently, so a
skip set is what would catch a wrong-sublayer read that a skipless set cannot see: with no skip the
input is only ever used to compute feature activations, which are compared anyway.

The reference here is nnsight rather than TransformerLens because TransformerLens does not support
gemma-3. The skip set that *is* checked against TransformerLens is the Llama one, in
`test_transformerlens_interp_engine_same_llama.py`; this file covers a different architecture and a
different input point (`mlp.hook_in` rather than `hook_resid_mid`).
"""

import gc

import pytest
import torch

from circuit_tracer.attribution.attribute import attribute
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.replacement_model.replacement_model import Backend
from tests.conftest import has_32gb

pytestmark = [pytest.mark.skipif(not has_32gb, reason="Requires >=32GB VRAM")]

MODEL = "google/gemma-3-1b-pt"
# feature_input_hook is `mlp.hook_in` and the weights carry `W_skip`.
TRANSCODERS = "mwhanna/gemma-scope-2-1b-pt/transcoder_all/width_16k_l0_small_affine"
PROMPT = "The capital of the state containing Dallas is the city of"
BACKENDS: tuple[Backend, ...] = ("nnsight", "interp_engine")
BATCH_SIZE = 128

# Ablating the strongest activations rather than a named circuit: this model has no published
# supernodes, and any large perturbation serves the purpose of making the comparison non-trivial.
N_ABLATE = 8


def _collect(backend: Backend) -> dict[str, torch.Tensor | bool]:
    # float32 rather than the checkpoint's bfloat16: these error vectors reach a magnitude of ~120,
    # where bf16's ~8e-3 relative step is far coarser than the differences this test is looking for.
    model = ReplacementModel.from_pretrained(
        MODEL, TRANSCODERS, backend=backend, dtype=torch.float32
    )
    n_layers = model.cfg.n_layers

    logits, acts = model.get_activations(PROMPT, apply_activation_function=False)
    ctx = model.setup_attribution(PROMPT)
    graph = attribute(PROMPT, model, batch_size=BATCH_SIZE, verbose=False)

    # Ablation targets come from the *post*-activation values, which are the ones that are actually
    # firing. Picking by largest pre-activation magnitude instead selects strongly negative
    # pre-activations, whose features are already off, so setting them to 0 is a no-op.
    clean_logits, active_acts = model.get_activations(PROMPT)
    top = torch.topk(active_acts.flatten(), N_ABLATE).indices
    picks = [tuple(int(i) for i in torch.unravel_index(idx, active_acts.shape)) for idx in top]
    interventions = [(layer, pos, feature, 0.0) for layer, pos, feature in picks]
    direct_logits, direct_acts = model.feature_intervention(
        PROMPT,
        interventions,
        constrained_layers=range(n_layers),
        apply_activation_function=False,
    )
    propagated_logits, _ = model.feature_intervention(
        PROMPT, interventions, apply_activation_function=False
    )

    tensors = {
        "logits": logits,
        "acts": acts,
        "error_vectors": ctx.error_vectors,
        "decoder_vecs": ctx.decoder_vecs,
        "encoder_vecs": ctx.encoder_vecs,
        "adjacency_matrix": graph.adjacency_matrix,
        "active_features": graph.active_features,
        "selected_features": graph.selected_features,
        "clean_logits": clean_logits,
        "direct_logits": direct_logits,
        "direct_acts": direct_acts,
        "propagated_logits": propagated_logits,
        "ablated": torch.tensor(picks),
    }
    out: dict[str, torch.Tensor | bool] = {
        k: v.detach().to("cpu", torch.float32 if v.is_floating_point() else v.dtype)
        for k, v in tensors.items()
    }
    out["skip_connection"] = bool(model.transcoders.skip_connection)

    del model, ctx, graph
    gc.collect()
    torch.cuda.empty_cache()
    return out


@pytest.fixture(scope="module")
def results() -> dict[str, dict[str, torch.Tensor | bool]]:
    """One model at a time; see the gemma-2 parity test for why they are not held side by side."""
    return {backend: _collect(backend) for backend in BACKENDS}


@pytest.fixture(scope="module")
def nns(results):
    return results["nnsight"]


@pytest.fixture(scope="module")
def ie(results):
    return results["interp_engine"]


# Tolerances are relative to each tensor's own scale rather than absolute, because this model's
# numbers are enormous: its transcoder pre-activations reach ~8.9e3 and its error vectors ~4.9e2,
# against ~1e0 for the gemma-2 set the other parity test uses. An absolute tolerance carried over
# from there would fail on pure float spread. Measured agreement is ~1e-6 of scale on every
# quantity, so 1e-4 leaves two orders of headroom while staying far tighter than any real error:
# feeding the skip the wrong input would move things by a fraction of `‖W_skip‖`, which is 2e2-4e3
# per layer here.
REL_TOL = 1e-4


def _assert_close(a: torch.Tensor, b: torch.Tensor, name: str, rel_tol: float = REL_TOL):
    assert a.shape == b.shape, f"{name} shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}"
    scale = b.abs().max().clamp(min=1.0)
    diff = (a - b).abs().max()
    assert diff <= rel_tol * scale, (
        f"{name} differ by max {diff} against a scale of {scale} "
        f"({diff / scale:.2e} relative, limit {rel_tol:.0e})"
    )


def test_the_transcoders_really_have_a_skip_connection(nns, ie):
    """Without this the rest of the file silently becomes a slower duplicate of the gemma-2 test."""
    for backend, result in (("nnsight", nns), ("interp_engine", ie)):
        assert result["skip_connection"], (
            f"{backend} loaded {TRANSCODERS} without a skip connection, so this file is no longer "
            "testing the skip path"
        )


def test_forward_pass_consistency(nns, ie):
    _assert_close(ie["logits"], nns["logits"], "Logits")
    _assert_close(ie["acts"], nns["acts"], "Transcoder activations")


def test_attribution_context_consistency(nns, ie):
    """The error vectors are the assertion that actually carries the skip term.

    An error vector is the residual between the true sublayer output and the transcoder's
    reconstruction, and on a skip set that reconstruction includes `input_acts @ W_skip`. A backend
    feeding the skip the wrong input shows up here even when every feature activation matches.
    """
    _assert_close(ie["error_vectors"], nns["error_vectors"], "Error vectors")
    _assert_close(ie["decoder_vecs"], nns["decoder_vecs"], "Decoder vectors")
    _assert_close(ie["encoder_vecs"], nns["encoder_vecs"], "Encoder vectors")


def test_attribution_graph_consistency(nns, ie):
    assert (ie["active_features"] == nns["active_features"]).all(), "Active features differ"
    assert (ie["selected_features"] == nns["selected_features"]).all(), "Selected features differ"
    _assert_close(ie["adjacency_matrix"], nns["adjacency_matrix"], "Adjacency")


def test_interventions_actually_perturb(nns, ie):
    assert (ie["ablated"] == nns["ablated"]).all(), (
        "The backends picked different features to ablate, so the comparison below is meaningless"
    )
    for backend, result in (("nnsight", nns), ("interp_engine", ie)):
        moved = (result["direct_logits"] - result["clean_logits"]).abs().max()
        assert moved > 0.1, f"{backend}'s ablation barely moved the logits ({moved})"


def test_intervention_consistency_with_skip(nns, ie):
    """Interventions on a skip set: the skip-difference hooks run only on this path."""
    _assert_close(ie["direct_logits"], nns["direct_logits"], "Direct-effect logits")
    _assert_close(ie["direct_acts"], nns["direct_acts"], "Direct-effect activations")
    _assert_close(ie["propagated_logits"], nns["propagated_logits"], "Propagated logits")
