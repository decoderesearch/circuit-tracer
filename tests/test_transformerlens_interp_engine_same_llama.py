"""The interp_engine backend must agree with transformerlens on Llama-3.2-1B.

This set is the awkward one, and it is the reason this file exists separately from the gemma-2 test:
`mntss/transcoder-Llama-3.2-1B` reads at `hook_resid_mid` -- the residual stream *between* the two
sublayers -- and it carries a `W_skip`. Every other set circuit-tracer ships reads at a normalized
point (`ln2.hook_normalized`) or the MLP's own input (`mlp.hook_in`), and only the gemma-3 sets
carry a skip.

So this is the only case that tests both at once, and the only one that tests a skip against
TransformerLens rather than against nnsight, TransformerLens not supporting gemma-3.

`hook_resid_mid` matters beyond being one more name. Reading it means the transcoder's input is the
raw pre-norm residual, so the sublayer's normalization sits *between* what the transcoder reads and
what it reconstructs -- unlike the normalized points, where the read happens after it. A backend
that read one sublayer too early would still produce right-shaped, plausible activations.
"""

import gc

import pytest
import torch

from circuit_tracer.attribution.attribute import attribute
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.replacement_model.replacement_model import Backend
from tests.conftest import has_32gb

pytestmark = [pytest.mark.skipif(not has_32gb, reason="Requires >=32GB VRAM")]

MODEL = "meta-llama/Llama-3.2-1B"
# The "llama" alias resolves here. feature_input_hook is `hook_resid_mid`; the weights carry W_skip.
TRANSCODERS = "llama"
PROMPT = "The capital of the state containing Dallas is the city of"
BACKENDS: tuple[Backend, ...] = ("transformerlens", "interp_engine")
# These transcoders are ~64k wide and 17 GiB for the set, so the backward pass gets a smaller batch
# than the gemma tests use. Chunking the batched backward does not change any result.
BATCH_SIZE = 64

N_ABLATE = 8


def _collect(backend: Backend) -> dict[str, torch.Tensor | bool]:
    model = ReplacementModel.from_pretrained(MODEL, TRANSCODERS, backend=backend)
    n_layers = model.cfg.n_layers

    logits, acts = model.get_activations(PROMPT, apply_activation_function=False)
    ctx = model.setup_attribution(PROMPT)
    graph = attribute(PROMPT, model, batch_size=BATCH_SIZE, verbose=False)

    # From the post-activation values, so the features picked are ones that are actually firing;
    # ablating an already-off feature is a no-op, which `test_interventions_actually_perturb` catches.
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
    out["reads_resid_mid"] = model.transcoders.feature_input_hook == "hook_resid_mid"

    del model, ctx, graph
    gc.collect()
    torch.cuda.empty_cache()
    return out


@pytest.fixture(scope="module")
def results() -> dict[str, dict[str, torch.Tensor | bool]]:
    """One model at a time: this set alone is 17 GiB, so the pair does not fit on a 32 GiB card."""
    return {backend: _collect(backend) for backend in BACKENDS}


@pytest.fixture(scope="module")
def tl(results):
    return results["transformerlens"]


@pytest.fixture(scope="module")
def ie(results):
    return results["interp_engine"]


# Relative to each tensor's own scale rather than absolute: see the gemma-3 skip test for why
# absolute tolerances do not carry between models.
REL_TOL = 1e-4


def _assert_close(a: torch.Tensor, b: torch.Tensor, name: str, rel_tol: float = REL_TOL):
    assert a.shape == b.shape, f"{name} shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}"
    scale = b.abs().max().clamp(min=1.0)
    diff = (a - b).abs().max()
    assert diff <= rel_tol * scale, (
        f"{name} differ by max {diff} against a scale of {scale} "
        f"({diff / scale:.2e} relative, limit {rel_tol:.0e})"
    )


def test_this_set_is_the_awkward_one(tl, ie):
    """Guards both properties this file exists to cover, so a swap upstream cannot quietly gut it."""
    for backend, result in (("transformerlens", tl), ("interp_engine", ie)):
        assert result["reads_resid_mid"], (
            f"{backend} loaded {TRANSCODERS} but it no longer reads at hook_resid_mid, so this "
            "file is not testing that path"
        )
        assert result["skip_connection"], (
            f"{backend} loaded {TRANSCODERS} without a skip connection, so this file is not "
            "testing the skip path"
        )


def test_forward_pass_consistency(tl, ie):
    _assert_close(ie["logits"], tl["logits"], "Logits")
    _assert_close(ie["acts"], tl["acts"], "Transcoder activations")


def test_attribution_context_consistency(tl, ie):
    """Error vectors carry both the skip term and the resid_mid read; see the gemma-3 skip test."""
    _assert_close(ie["error_vectors"], tl["error_vectors"], "Error vectors")
    _assert_close(ie["decoder_vecs"], tl["decoder_vecs"], "Decoder vectors")
    _assert_close(ie["encoder_vecs"], tl["encoder_vecs"], "Encoder vectors")


def test_attribution_graph_consistency(tl, ie):
    assert (ie["active_features"] == tl["active_features"]).all(), "Active features differ"
    assert (ie["selected_features"] == tl["selected_features"]).all(), "Selected features differ"
    _assert_close(ie["adjacency_matrix"], tl["adjacency_matrix"], "Adjacency")


def test_interventions_actually_perturb(tl, ie):
    assert (ie["ablated"] == tl["ablated"]).all(), (
        "The backends picked different features to ablate, so the comparison below is meaningless"
    )
    for backend, result in (("transformerlens", tl), ("interp_engine", ie)):
        moved = (result["direct_logits"] - result["clean_logits"]).abs().max()
        assert moved > 0.1, f"{backend}'s ablation barely moved the logits ({moved})"


def test_intervention_consistency(tl, ie):
    _assert_close(ie["direct_logits"], tl["direct_logits"], "Direct-effect logits")
    _assert_close(ie["direct_acts"], tl["direct_acts"], "Direct-effect activations")
    _assert_close(ie["propagated_logits"], tl["propagated_logits"], "Propagated logits")
