"""ensure_tokenized must prepend a special token whose id is 0.

Tokenizers such as Pythia / GPT-NeoX use id 0 for BOS and EOS, and it is their only
special token. Filtering the candidates with ``filter(None, ...)`` dropped id 0 as
falsy, so ``next()`` raised StopIteration instead of prepending it.
"""

from types import SimpleNamespace

import pytest
import torch

from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
from circuit_tracer.replacement_model.replacement_model_transformerlens import (
    TransformerLensReplacementModel,
)

BACKENDS = [TransformerLensReplacementModel, NNSightReplacementModel]


def make_model_stub(bos_token_id, pad_token_id, eos_token_id, all_special_ids):
    tokenizer = SimpleNamespace(
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        all_special_ids=all_special_ids,
    )
    cfg = SimpleNamespace(device="cpu", model_name="pythia-14m")
    return SimpleNamespace(cfg=cfg, device="cpu", tokenizer=tokenizer)


@pytest.mark.parametrize("model_cls", BACKENDS)
def test_prepends_special_token_with_id_zero(model_cls):
    # Pythia's tokenizer: bos_token_id=0, eos_token_id=0, pad_token_id=None
    model = make_model_stub(0, None, 0, [0])
    tokens = model_cls.ensure_tokenized(model, [5801, 627])
    assert tokens.tolist() == [0, 5801, 627]


@pytest.mark.parametrize("model_cls", BACKENDS)
def test_prefers_bos_with_id_zero_over_other_special_tokens(model_cls):
    model = make_model_stub(0, None, 1, [0, 1])
    tokens = model_cls.ensure_tokenized(model, [5801, 627])
    assert tokens.tolist() == [0, 5801, 627]


@pytest.mark.parametrize("model_cls", BACKENDS)
def test_warns_when_tokenizer_has_no_special_tokens(model_cls):
    model = make_model_stub(None, None, None, [])
    with pytest.warns(UserWarning, match="No suitable special token"):
        tokens = model_cls.ensure_tokenized(model, torch.tensor([5801, 627]))
    assert tokens.tolist() == [5801, 627]
