import gc

import pytest
import torch

from circuit_tracer.transcoder.activation_functions import JumpReLU, TopK


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


def test_JumpReLU_filters_activations_below_threshold():
    threshold = torch.tensor([1.0, 0.5, 1.5, 2.0, 0.8])
    act_fn = JumpReLU(threshold=threshold, bandwidth=1.0)

    # Test input with values both above and below threshold
    x = torch.tensor([[-1.0, 0.5, 1.0, 1.5, 2.0], [2.0, 1.0, 2.0, 3.0, 0.5]])
    result = act_fn(x)

    # Values below threshold should be 0, values > threshold should be preserved
    expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 2.0], [2.0, 1.0, 2.0, 3.0, 0.0]])

    assert torch.allclose(result, expected)


def test_TopK():
    act_fn = TopK(k=2)

    x = torch.tensor([[1.0, 5.0, 3.0, 2.0], [4.0, 1.0, 6.0, 2.0]])
    result = act_fn(x)

    # Should keep the 2 largest values in each row and zero the rest
    expected = torch.tensor([[0.0, 5.0, 3.0, 0.0], [4.0, 0.0, 6.0, 0.0]])
    assert torch.allclose(result, expected)
