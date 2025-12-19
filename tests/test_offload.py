import pytest
import torch

from circuit_tracer import ReplacementModel
from circuit_tracer.attribution.attribute_transformerlens import attribute as attribute_tl
from circuit_tracer.attribution.attribute_nnsight import attribute as attribute_nnsight
from circuit_tracer.replacement_model.replacement_model_transformerlens import (
    TransformerLensReplacementModel,
)
from circuit_tracer.replacement_model.replacement_model_nnsight import (
    NNSightReplacementModel,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_offload_tl():
    s = "The National Digital Analytics Group (ND"
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma")
    assert isinstance(model, TransformerLensReplacementModel)
    graph_none = attribute_tl(s, model, offload=None)
    graph_cpu = attribute_tl(s, model, offload="cpu")
    assert torch.allclose(
        graph_none.adjacency_matrix, graph_cpu.adjacency_matrix, atol=1e-5, rtol=1e-3
    )

    graph_disk = attribute_tl(s, model, offload="disk")
    assert torch.allclose(
        graph_none.adjacency_matrix, graph_disk.adjacency_matrix, atol=1e-5, rtol=1e-3
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_offload_nnsight():
    s = "The National Digital Analytics Group (ND"
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", backend="nnsight")
    if torch.cuda.is_available():
        model.cuda()

    assert isinstance(model, NNSightReplacementModel)
    graph_none = attribute_nnsight(s, model, offload=None)
    graph_cpu = attribute_nnsight(s, model, offload="cpu")
    assert torch.allclose(
        graph_none.adjacency_matrix, graph_cpu.adjacency_matrix, atol=1e-5, rtol=1e-3
    )

    graph_disk = attribute_nnsight(s, model, offload="disk")
    assert torch.allclose(
        graph_none.adjacency_matrix, graph_disk.adjacency_matrix, atol=1e-5, rtol=1e-3
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    test_offload_tl()
    test_offload_nnsight()
