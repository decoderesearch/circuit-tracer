"""Hermetic tests for `_subtracts_mean` (no model/GPU/network required, but interp-engine installed).

The interp_engine backend introduces a transcoder's input tensor by substituting `x / scale` as a
norm's input, and whether that scale is computed on a *centered* x is a property of the norm family.
The test used to be the class name -- "rms" in `type(norm).__name__` -- which is wrong on the T5
lineage: `T5LayerNorm` (T5, mT5, UMT5, Flan-T5) subtracts no mean despite what it is called.

Centering there is a silent fault rather than a crash. The substituted tensor stays finite and
right-shaped, the forward pass still runs, and the transcoder simply reads activations the model
never computed -- so the only thing that catches it is a test that names the family.

`_subtracts_mean` now delegates to `interp_engine.capture.is_rms_norm` rather than applying its own
copy of those rules, which changes what these cases pin: not that this repo implements the rules
correctly, but that the engine's answer is still the one this backend needs. That is the more useful
of the two, the engine being free to revise a predicate it owns -- and it is why these stayed here
instead of being deleted along with the duplicated code.
"""

import torch
import torch.nn as nn

from circuit_tracer.replacement_model.replacement_model_interp_engine import _subtracts_mean


def test_t5s_layernorm_is_not_centered_despite_its_name():
    """The regression this file exists for."""
    from transformers.models.t5.modeling_t5 import T5LayerNorm

    norm = T5LayerNorm(8)
    assert "rms" not in type(norm).__name__.lower(), "the premise of the old name-based test"
    assert not _subtracts_mean(norm)


def test_the_rms_norms_of_the_supported_families_are_not_centered():
    from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    assert not _subtracts_mean(LlamaRMSNorm(8))
    assert not _subtracts_mean(Gemma2RMSNorm(8))
    assert not _subtracts_mean(nn.RMSNorm(8))


def test_a_real_layernorm_is_centered():
    assert _subtracts_mean(nn.LayerNorm(8))
    assert _subtracts_mean(nn.LayerNorm(8, elementwise_affine=False))


def test_a_bias_means_centered_whatever_the_class_is_called():
    class ConfusinglyNamedRMSNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(8))
            self.bias = nn.Parameter(torch.zeros(8))

    assert _subtracts_mean(ConfusinglyNamedRMSNorm())


def test_a_gain_only_norm_is_not_centered():
    class SomeFamilyNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(8))

    assert not _subtracts_mean(SomeFamilyNorm())


def test_the_t5_norm_would_have_been_centered_by_the_old_test():
    """Pins the bug itself, so a revert to a name test fails here rather than in a graph.

    Reproduces the old predicate verbatim and asserts it disagrees with the current one on T5 --
    i.e. that this file is testing something the old code got wrong, not merely something it did.
    """
    from transformers.models.t5.modeling_t5 import T5LayerNorm

    def old_predicate(norm: nn.Module) -> bool:
        return isinstance(norm, nn.LayerNorm) or "rms" not in type(norm).__name__.lower()

    norm = T5LayerNorm(8)
    assert old_predicate(norm) is True
    assert _subtracts_mean(norm) is False
