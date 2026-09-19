"""The interp_engine backend must produce the same numbers as transformerlens.

Mirrors ``test_transformerlens_nnsight_same_gemma.py``: attribution graphs, and the interventions
that check the graph was worth reading.

The two backends reach the same numbers by different routes -- transformerlens rewrites the model
into its own weight convention, while interp_engine hooks the Hugging Face module tree in place --
so agreement here is what says the second route is right, and is the only thing that catches a
freeze that silently stopped holding.
"""

import gc
import math

import pytest
import torch

from circuit_tracer.attribution.attribute import attribute
from circuit_tracer.attribution.targets import CustomTarget
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.replacement_model.replacement_model import Backend
from circuit_tracer.utils.demo_utils import get_unembed_vecs
from tests.conftest import has_32gb

pytestmark = [pytest.mark.skipif(not has_32gb, reason="Requires >=32GB VRAM")]

MODEL = "google/gemma-2-2b"
TRANSCODERS = "mntss/gemma-scope-transcoders"
PROMPT = "The capital of the state containing Dallas is the city of"
BACKENDS: tuple[Backend, ...] = ("transformerlens", "interp_engine")
# Below the default of 512, which needs more than a 32 GiB card has once the model and transcoders
# are resident. This only chunks the batched backward pass, so it does not change any result here.
BATCH_SIZE = 128

# The Dallas-Austin circuit from the demo notebooks. Ablating its "Texas" supernode is a large,
# known-meaningful perturbation, so the backends agreeing here is a stronger statement than it would
# be on arbitrary features -- and `test_interventions_actually_perturb` keeps it that way.
INTERVENTION_PROMPT = "Fact: the capital of the state containing Dallas is"
TEXAS_SUPERNODE = [
    (20, 9, 15589),
    (19, 9, 7477),
    (16, 9, 25),
    (4, 9, 13154),
    (14, 9, 2268),
    (7, 9, 6861),
]
ABLATE_TEXAS = [(layer, pos, feature, -2.0) for layer, pos, feature in TEXAS_SUPERNODE]

# The multilingual circuit's French-to-Spanish swap. Open-ended positions (`slice(6, None)`) are
# what makes this a generation test rather than a longer forward pass: they have to keep applying to
# tokens that did not exist when the intervention was written.
GENERATE_PROMPT = "Fait: Michael Jordan joue au"
FRENCH_TO_SPANISH = [(20, slice(6, None), 1454, 0.0), (20, slice(6, None), 341, 272.0)]

# The attribution_targets demo's logit-difference target. Worth its own case because it is the one
# notebook path that reads the unembedding matrix directly, and the two backends store it in
# opposite orientations -- so a wrong branch in `get_unembed_vecs` would attribute to a direction
# assembled out of the wrong rows without failing anywhere else.
TARGET_TOKENS = ("▁Austin", "▁Dallas")


def _collect(backend: Backend) -> dict[str, torch.Tensor | str]:
    """Every tensor these tests compare, on the CPU, model already freed."""
    model = ReplacementModel.from_pretrained(MODEL, TRANSCODERS, backend=backend)
    n_layers = model.cfg.n_layers

    logits, acts = model.get_activations(PROMPT, apply_activation_function=False)
    ctx = model.setup_attribution(PROMPT)
    # Gemma-2 caps its logits with a tanh, which flattens the gradient the graph is built from;
    # the other cross-backend tests drop it for the same reason.
    with model.zero_softcap():
        graph = attribute(PROMPT, model, batch_size=BATCH_SIZE, verbose=False)

    clean_logits, _ = model.get_activations(INTERVENTION_PROMPT)
    # Constrained to every layer: attention patterns, normalization denominators and sublayer
    # outputs all held at their clean values, so the logits show only the direct effect.
    direct_logits, direct_acts = model.feature_intervention(
        INTERVENTION_PROMPT,
        ABLATE_TEXAS,
        constrained_layers=range(n_layers),
        apply_activation_function=False,
    )
    # Unconstrained: only the attention patterns are frozen, and the ablation's effects propagate
    # through the transcoders and norms.
    propagated_logits, propagated_acts = model.feature_intervention(
        INTERVENTION_PROMPT, ABLATE_TEXAS, apply_activation_function=False
    )
    with model.zero_softcap():
        generation, generated_logits, generated_acts = model.feature_intervention_generate(
            GENERATE_PROMPT,
            FRENCH_TO_SPANISH,
            constrained_layers=range(n_layers),
            do_sample=False,
        )

    tokenizer = model.tokenizer
    assert tokenizer is not None
    target_ids = [tokenizer.encode(token, add_special_tokens=False)[-1] for token in TARGET_TOKENS]
    vec_x, vec_y = get_unembed_vecs(model, target_ids, backend)
    direction = vec_x - vec_y
    with model.zero_softcap():
        custom_graph = attribute(
            PROMPT,
            model,
            attribution_targets=[
                CustomTarget(token_str="logit-diff", prob=1.0, vec=direction),
            ],
            batch_size=BATCH_SIZE,
            verbose=False,
        )

    tensors = {
        "tokens": model.ensure_tokenized(PROMPT),
        "logits": logits,
        "acts": acts,
        "error_vectors": ctx.error_vectors,
        "token_vectors": ctx.token_vectors,
        "decoder_vecs": ctx.decoder_vecs,
        "encoder_vecs": ctx.encoder_vecs,
        "adjacency_matrix": graph.adjacency_matrix,
        "active_features": graph.active_features,
        "selected_features": graph.selected_features,
        "clean_logits": clean_logits,
        "direct_logits": direct_logits,
        "direct_acts": direct_acts,
        "propagated_logits": propagated_logits,
        "propagated_acts": propagated_acts,
        "generated_logits": generated_logits,
        "generated_acts": generated_acts,
        "unembed_direction": direction,
        "custom_adjacency": custom_graph.adjacency_matrix,
        "custom_active_features": custom_graph.active_features,
    }
    out: dict[str, torch.Tensor | str] = {
        k: v.detach().to("cpu", torch.float32 if v.is_floating_point() else v.dtype)
        for k, v in tensors.items()
    }
    out["generation"] = generation

    del model, ctx, graph, custom_graph
    gc.collect()
    torch.cuda.empty_cache()
    return out


@pytest.fixture(scope="module")
def results() -> dict[str, dict[str, torch.Tensor | str]]:
    """Both backends' artifacts, gathered one model at a time.

    Sequentially, not side by side: gemma-2-2b plus its 26 transcoders is ~13.5 GiB, so holding two
    backends resident leaves no room for attribution's batched backward pass on a 32 GiB card.
    """
    return {backend: _collect(backend) for backend in BACKENDS}


@pytest.fixture(scope="module")
def tl(results):
    return results["transformerlens"]


@pytest.fixture(scope="module")
def ie(results):
    return results["interp_engine"]


def _assert_close(a: torch.Tensor, b: torch.Tensor, name: str, atol: float, rtol: float = 1e-5):
    assert a.shape == b.shape, f"{name} shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}"
    assert torch.allclose(a, b, atol=atol, rtol=rtol), (
        f"{name} differ by max {(a - b).abs().max()} (atol={atol})"
    )


def test_tokenization_identical(tl, ie):
    """Nothing downstream is comparable if the backends disagree on the token sequence."""
    assert tl["tokens"].tolist() == ie["tokens"].tolist()


def test_forward_pass_consistency(tl, ie):
    _assert_close(ie["logits"], tl["logits"], "Logits", atol=1e-4)
    _assert_close(ie["acts"], tl["acts"], "Activations", atol=5e-4)


def test_attribution_context_consistency(tl, ie):
    _assert_close(ie["error_vectors"], tl["error_vectors"], "Error vectors", atol=1e-3)
    _assert_close(ie["decoder_vecs"], tl["decoder_vecs"], "Decoder vectors", atol=5e-4)
    _assert_close(ie["encoder_vecs"], tl["encoder_vecs"], "Encoder vectors", atol=1e-4)


def test_token_vectors_differ_only_by_the_embedding_normalizer(tl, ie):
    """Pins the one convention the backends genuinely do not share.

    transformerlens folds Gemma's ``sqrt(d_model)`` embedding normalizer into ``W_E``; Hugging Face
    applies it as a separate multiply after ``embed_tokens``, and both interp_engine and nnsight
    report the raw embedding row. interp_engine reads its token gradient on the pre-multiply tensor,
    so the factor cancels and token attributions still match -- which
    :func:`test_attribution_graph_consistency` checks. Asserting the exact factor here keeps that
    reasoning honest: if the normalizer ever moved, this fails loudly instead of the graph drifting.
    """
    d_model = tl["token_vectors"].shape[-1]
    _assert_close(
        ie["token_vectors"] * math.sqrt(d_model), tl["token_vectors"], "Token vectors", atol=1e-3
    )


def test_attribution_graph_consistency(tl, ie):
    assert (ie["active_features"] == tl["active_features"]).all(), (
        "Active features don't match between backends"
    )
    assert (ie["selected_features"] == tl["selected_features"]).all(), (
        "Selected features don't match between backends"
    )
    _assert_close(ie["adjacency_matrix"], tl["adjacency_matrix"], "Adjacency matrices", atol=5e-4)


def test_adjacency_edge_ranking_agrees(tl, ie):
    """Guards the thing a user actually reads off the graph: which edges are the strong ones.

    ``allclose`` on the whole matrix can pass while the largest edges are reordered, and a reordering
    is what changes the pruned graph someone ends up looking at.
    """
    a, b = tl["adjacency_matrix"].flatten(), ie["adjacency_matrix"].flatten()
    strong = a.abs() > a.abs().max() * 0.01
    corr = torch.corrcoef(torch.stack([a[strong], b[strong]]))[0, 1]
    assert corr > 0.9999, f"Top-1% edge weights only correlate at {corr}"


def test_interventions_actually_perturb(tl, ie):
    """The parity assertions below only mean something if the interventions did something.

    Two backends that both silently failed to apply an intervention would agree perfectly.
    """
    for backend, result in (("transformerlens", tl), ("interp_engine", ie)):
        for kind in ("direct", "propagated"):
            moved = (result[f"{kind}_logits"] - result["clean_logits"]).abs().max()
            assert moved > 1.0, f"{backend}'s {kind} ablation barely moved the logits ({moved})"
        # And that constraining the layers is not a no-op, so that a freeze which quietly stopped
        # replaying fails here rather than as a mysterious tolerance breach against the other
        # backend.
        spread = (result["direct_logits"] - result["propagated_logits"]).abs().max()
        assert spread > 1.0, (
            f"{backend}'s direct and propagated effects are the same to {spread}, so the freeze "
            "is not constraining anything"
        )


def test_direct_effect_intervention_consistency(tl, ie):
    """Freezing every layer: the case that exercises the value replay in all three places."""
    _assert_close(ie["direct_logits"], tl["direct_logits"], "Direct-effect logits", atol=1e-4)
    _assert_close(ie["direct_acts"], tl["direct_acts"], "Direct-effect activations", atol=5e-4)


def test_propagated_intervention_consistency(tl, ie):
    """Freezing only attention, so the ablation's effects reach the later layers' features."""
    _assert_close(ie["propagated_logits"], tl["propagated_logits"], "Propagated logits", atol=1e-4)
    _assert_close(ie["propagated_acts"], tl["propagated_acts"], "Propagated activations", atol=5e-4)


def test_generation_with_intervention_consistency(tl, ie):
    """Whether the two backends generate the same continuation under the same intervention.

    Both drive generation with a KV cache, so each step sees one position and the freezes apply
    only to the pass that consumed the prompt -- the divergence this catches is a backend that
    kept freezing, or stopped intervening, once generation started.
    """
    assert ie["generation"] == tl["generation"], (
        f"Generated text differs:\n  transformerlens: {tl['generation']!r}\n"
        f"  interp_engine:   {ie['generation']!r}"
    )
    _assert_close(ie["generated_logits"], tl["generated_logits"], "Generated logits", atol=1e-3)
    _assert_close(ie["generated_acts"], tl["generated_acts"], "Generated activations", atol=5e-3)


def test_custom_target_attribution_consistency(tl, ie):
    """Attributing to a direction read out of the unembedding matrix, as the targets demo does."""
    _assert_close(ie["unembed_direction"], tl["unembed_direction"], "Unembed direction", atol=1e-5)
    assert (ie["custom_active_features"] == tl["custom_active_features"]).all(), (
        "Custom-target active features differ"
    )
    _assert_close(
        ie["custom_adjacency"], tl["custom_adjacency"], "Custom-target adjacency", atol=1e-3
    )
