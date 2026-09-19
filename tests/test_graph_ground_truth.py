"""A graph built here agrees with the one production Neuronpedia built from the same request.

The fixtures under ``fixtures/graphs/`` are graphs ``POST https://www.neuronpedia.org/api/graph/generate``
returned for the prompt ``123``, one per transcoder set Neuronpedia serves, pruned as hard as that API
allows so they stay small. Each records the model, the transcoder set and every generation and
pruning parameter in its ``metadata``, and the test builds a graph from exactly those, on each
backend, through :func:`circuit_tracer.utils.create_graph_files.build_graph_model` -- the same path
as ``create_graph_files``. What is compared is the wire format the frontend reads, node ids and all,
against a graph the ``interp_engine`` backend built on production's CUDA card. For ``transformerlens``
and ``nnsight`` that makes it a cross-backend check on that format too.

Agreement is measured rather than asserted exact. Two bfloat16 attributions do not reproduce bit for
bit across devices or backends, and pruning turns a small difference in influence at the threshold
into a node present on one side and absent on the other. The tolerances below leave room over what
runs against production showed. On an RTX 5090, all three backends on the gemma-scope set: Jaccard
0.97 to 1.0, activations within 0.5% at the median and 2% at worst, influence within 0.006, 97% of
the links shared with weights correlated past 0.9999. On MPS, interp_engine on all three sets:
Jaccard 0.92 to 1.0, activations within 0.7% at the median and 11% at worst, influence within
0.045, edge weights correlated past 0.997.

The gemma-scope case runs wherever there is a GPU: about 13 GB of downloads, and in bfloat16 it
peaks at 9.1 GiB on transformerlens and ~7 GiB on the other two, so the CI job runs it on a 16 GB
card. The CLT and Qwen cases are ``requires_disk``: 160 GiB and 57 GiB of transcoders, and their
resident encoders alone are ~11 GiB and 28 GiB. Opt in with ``-m requires_disk`` and pick with
``-k``. The self-agreement test at the bottom runs everywhere and keeps the fixtures and the
comparison honest without weights.
"""

import gc
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from circuit_tracer.attribution.attribute import attribute
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.replacement_model.replacement_model import Backend
from circuit_tracer.utils.create_graph_files import build_graph_model

FIXTURES = Path(__file__).parent / "fixtures" / "graphs"
PROMPT = "123"
BACKENDS: tuple[Backend, ...] = ("transformerlens", "nnsight", "interp_engine")


@dataclass(frozen=True)
class Case:
    fixture: str
    #: The HF checkpoint the fixture was generated on. The transcoder set is in its metadata.
    model: str
    #: Skip on a smaller card. Below the card's nominal size, since ``total_memory`` reports less:
    #: a 16 GB T4 is 14.7 GiB, a 48 GB L40S is 44.5. The default case peaks at 9.1 GiB.
    min_vram_gib: int


CASES = {
    "gemma-2-2b-gemmascope-16k": Case(
        "123-gemma-2-2b-gemmascope-transcoder-16k.json", "google/gemma-2-2b", 12
    ),
    "gemma-2-2b-clt-2.5M": Case("123-gemma-2-2b-clt-hp.json", "google/gemma-2-2b", 20),
    "qwen3-4b-transcoders": Case("123-qwen3-4b-transcoder-hp.json", "Qwen/Qwen3-4B", 40),
}
FIXTURE_PARAMS = [pytest.param(case, id=name) for name, case in CASES.items()]
#: The default case downloads ~13 GB; the other two need a big disk and a big card, so they opt in.
CASE_PARAMS = [
    pytest.param(
        case,
        id=name,
        marks=[] if name == "gemma-2-2b-gemmascope-16k" else [pytest.mark.requires_disk],
    )
    for name, case in CASES.items()
]

#: Tolerances. See the module docstring for where they come from. Influence is in [0, 1] and is
#: compared by difference: a correlation says little on the dozen nodes a hard-pruned graph keeps.
FEATURE_JACCARD_MIN = 0.85
ACTIVATION_REL_MEDIAN_MAX = 0.02
ACTIVATION_REL_MAX = 0.2
INFLUENCE_ABS_MEDIAN_MAX = 0.02
INFLUENCE_ABS_MAX = 0.1
EDGE_PEARSON_MIN = 0.99
SHARED_LINKS_MIN_FRACTION = 0.9
LOGIT_PROB_ATOL = 0.03


def _accelerator() -> torch.device | None:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


DEVICE = _accelerator()


def _load(name: str) -> dict[str, Any]:
    with (FIXTURES / name).open(encoding="utf-8") as f:
        return json.load(f)


def _pearson(x: list[float], y: list[float]) -> float:
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / (sx * sy)


def _features(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Feature nodes by ``node_id`` (``layer_feature_ctx``), the key that is stable across generators."""
    return {
        n["node_id"]: n for n in graph["nodes"] if n["feature_type"] == "cross layer transcoder"
    }


def _logits(graph: dict[str, Any]) -> dict[str, float]:
    """Token -> probability for the logit nodes, the token read out of the node's label."""
    return {
        n["clerp"].split('"')[1]: n["token_prob"]
        for n in graph["nodes"]
        if n["feature_type"] == "logit"
    }


def _links(graph: dict[str, Any]) -> dict[tuple[str, str], float]:
    return {(link["source"], link["target"]): link["weight"] for link in graph["links"]}


def assert_graphs_agree(reference: dict[str, Any], got: dict[str, Any]) -> None:
    """The comparison, on the wire format both sides produce."""
    assert got["metadata"]["prompt_tokens"] == reference["metadata"]["prompt_tokens"]

    ref_features, got_features = _features(reference), _features(got)
    shared = set(ref_features) & set(got_features)
    jaccard = len(shared) / len(set(ref_features) | set(got_features))
    assert jaccard >= FEATURE_JACCARD_MIN, (
        f"feature nodes: {len(ref_features)} on production, {len(got_features)} here, "
        f"{len(shared)} shared (jaccard {jaccard:.3f})"
    )

    relative = [
        abs(ref_features[k]["activation"] - got_features[k]["activation"])
        / abs(ref_features[k]["activation"])
        for k in shared
        if ref_features[k]["activation"]
    ]
    assert statistics.median(relative) <= ACTIVATION_REL_MEDIAN_MAX, (
        f"activation median rel {statistics.median(relative):.4f}"
    )
    assert max(relative) <= ACTIVATION_REL_MAX, f"activation max rel {max(relative):.4f}"

    influence = [abs(ref_features[k]["influence"] - got_features[k]["influence"]) for k in shared]
    assert statistics.median(influence) <= INFLUENCE_ABS_MEDIAN_MAX, (
        f"influence median diff {statistics.median(influence):.4f}"
    )
    assert max(influence) <= INFLUENCE_ABS_MAX, f"influence max diff {max(influence):.4f}"

    ref_logits, got_logits = _logits(reference), _logits(got)
    assert max(ref_logits, key=lambda t: ref_logits[t]) == max(
        got_logits, key=lambda t: got_logits[t]
    ), f"top logit: {ref_logits} vs {got_logits}"
    for token, prob in ref_logits.items():
        assert abs(prob - got_logits.get(token, 0.0)) <= LOGIT_PROB_ATOL, (
            f"logit {token!r}: {prob} vs {got_logits.get(token)}"
        )

    ref_links, got_links = _links(reference), _links(got)
    shared_links = set(ref_links) & set(got_links)
    assert len(shared_links) >= SHARED_LINKS_MIN_FRACTION * len(ref_links), (
        f"links: {len(ref_links)} on production, {len(got_links)} here, {len(shared_links)} shared"
    )
    edges = _pearson([ref_links[k] for k in shared_links], [got_links[k] for k in shared_links])
    assert edges >= EDGE_PEARSON_MIN, f"edge weight pearson {edges:.4f}"


@pytest.mark.parametrize("case", FIXTURE_PARAMS)
def test_each_fixture_agrees_with_itself(case: Case) -> None:
    """Weight-free: the fixture parses, records what the heavy test reads, and passes its own comparison."""
    graph = _load(case.fixture)
    metadata = graph["metadata"]
    assert metadata["prompt_tokens"][1:] == list(PROMPT)
    assert metadata["info"]["transcoder_set"]
    assert set(metadata["generation_settings"]) >= {
        "max_n_logits",
        "desired_logit_prob",
        "batch_size",
        "max_feature_nodes",
    }
    assert set(metadata["pruning_settings"]) == {"node_threshold", "edge_threshold"}
    assert_graphs_agree(graph, graph)


@pytest.mark.skipif(DEVICE is None, reason="attribution on the cpu takes too long to be a test")
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", CASE_PARAMS)
def test_a_graph_built_here_agrees_with_production(case: Case, backend: Backend) -> None:
    assert DEVICE is not None
    if DEVICE.type == "cuda":
        total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        if total_gib < case.min_vram_gib:
            pytest.skip(
                f"{case.model} wants a {case.min_vram_gib} GiB card, this one has {total_gib:.0f}"
            )
        torch.cuda.reset_peak_memory_stats()

    reference = _load(case.fixture)
    metadata = reference["metadata"]
    generation, pruning = metadata["generation_settings"], metadata["pruning_settings"]

    # As Neuronpedia loads it: bfloat16, encoders resident, decoders read lazily.
    model = ReplacementModel.from_pretrained(
        case.model,
        metadata["info"]["transcoder_set"],
        backend=backend,
        device=DEVICE,
        dtype=torch.bfloat16,
        lazy_encoder=False,
        lazy_decoder=True,
    )
    try:
        graph = attribute(
            PROMPT,
            model,
            max_n_logits=generation["max_n_logits"],
            desired_logit_prob=generation["desired_logit_prob"],
            batch_size=generation["batch_size"],
            max_feature_nodes=generation["max_feature_nodes"],
            verbose=False,
        )
        got = build_graph_model(
            graph,
            metadata["slug"],
            metadata["scan"],
            pruning["node_threshold"],
            pruning["edge_threshold"],
        )
    finally:
        del model
        gc.collect()
        if DEVICE.type == "cuda":
            # Kept in the log (`pytest -rP`) so the CI card can be sized from evidence.
            print(f"{backend}: peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
            torch.cuda.empty_cache()
        elif DEVICE.type == "mps":
            torch.mps.empty_cache()

    assert_graphs_agree(reference, got.model_dump())
