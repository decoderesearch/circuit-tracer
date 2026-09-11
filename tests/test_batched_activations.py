"""Batched vs single-prompt consistency for ReplacementModel.get_activations.

Regression tests for https://github.com/safety-research/circuit-tracer/issues/67:
the activation-caching hook assumed a batch size of 1 (`squeeze(0)`), so for
batched inputs `transcoder_acts[self.zero_positions] = 0` zeroed the leading
*sequences* of the batch instead of the leading *positions* of each sequence.
"""

import gc

import pytest
import torch
import torch.nn as nn
from transformer_lens import HookedTransformerConfig

from circuit_tracer import ReplacementModel
from circuit_tracer.replacement_model.replacement_model_transformerlens import (
    TransformerLensReplacementModel,
)
from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
from circuit_tracer.transcoder.activation_functions import TopK
from circuit_tracer.utils import get_default_device


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


def load_dummy_model() -> TransformerLensReplacementModel:
    cfg = HookedTransformerConfig.from_dict(
        {
            "n_layers": 2,
            "d_model": 8,
            "n_ctx": 64,
            "d_head": 4,
            "n_heads": 2,
            "d_mlp": 16,
            "d_vocab": 16,
            "act_fn": "silu",
            "model_name": "test-model",
            "tokenizer_name": "gpt2",  # using wrong tokenizer to avoid gated repos
            "normalization_type": "RMSPre",
            "positional_embedding_type": "rotary",
            "rotary_dim": 4,
            "final_rms": True,
            "gated_mlp": True,
            "init_weights": False,
            "device": get_default_device(),
            "dtype": torch.float32,
        }
    )

    transcoders = {
        layer_idx: SingleLayerTranscoder(
            cfg.d_model, cfg.d_model * 4, TopK(4), layer_idx, skip_connection=True
        )
        for layer_idx in range(cfg.n_layers)
    }
    for transcoder in transcoders.values():
        for _, param in transcoder.named_parameters():
            nn.init.uniform_(param, a=-1, b=1)

    transcoder_set = TranscoderSet(
        transcoders,
        feature_input_hook="mlp.hook_in",
        feature_output_hook="mlp.hook_out",
    )
    model = ReplacementModel.from_config(cfg, transcoder_set)

    for _, param in model.named_parameters():
        nn.init.uniform_(param, a=-1, b=1)

    assert isinstance(model, TransformerLensReplacementModel)
    return model


def test_get_activations_batched_matches_single():
    """A batch of equal-length prompts must produce the same activations as
    running each prompt individually (no padding involved)."""
    torch.manual_seed(0)
    model = load_dummy_model()

    seq_a = torch.tensor([0, 3, 4, 3, 2, 5, 3, 8])
    seq_b = torch.tensor([0, 7, 1, 9, 6, 2, 4, 5])

    logits_a, acts_a = model.get_activations(seq_a)
    logits_b, acts_b = model.get_activations(seq_b)
    logits_batch, acts_batch = model.get_activations(torch.stack([seq_a, seq_b]))

    # Logits: (1, pos, d_vocab) for single inputs, (2, pos, d_vocab) for the batch.
    torch.testing.assert_close(logits_batch[0], logits_a[0])
    torch.testing.assert_close(logits_batch[1], logits_b[0])

    # Activations: (n_layers, pos, d_tc) for single inputs,
    # (n_layers, batch, pos, d_tc) for the batch.
    assert acts_batch.shape == (model.cfg.n_layers, 2, seq_a.shape[0], model.cfg.d_model * 4)
    torch.testing.assert_close(acts_batch[:, 0], acts_a)
    torch.testing.assert_close(acts_batch[:, 1], acts_b)


def test_get_activations_batched_zeroes_positions_not_sequences():
    """The BOS-artifact zeroing must apply to leading positions of every
    sequence, not to leading sequences of the batch."""
    torch.manual_seed(0)
    model = load_dummy_model()

    batch = torch.tensor([[0, 3, 4, 3, 2, 5, 3, 8], [0, 7, 1, 9, 6, 2, 4, 5]])
    _, acts = model.get_activations(batch)

    # Position 0 of each sequence is zeroed (BOS artifact suppression) ...
    assert torch.all(acts[:, :, 0] == 0)
    # ... but the rest of every sequence is not.
    assert acts[:, 0, 1:].abs().sum() > 0, "first sequence in the batch was zeroed out"
    assert acts[:, 1, 1:].abs().sum() > 0, "second sequence in the batch was zeroed out"


def test_get_activations_batched_sparse():
    """Sparse mode preserves batched shapes and values."""
    torch.manual_seed(0)
    model = load_dummy_model()

    batch = torch.tensor([[0, 3, 4, 3, 2, 5, 3, 8], [0, 7, 1, 9, 6, 2, 4, 5]])
    _, acts_dense = model.get_activations(batch)
    _, acts_sparse = model.get_activations(batch, sparse=True)

    torch.testing.assert_close(acts_sparse.to_dense(), acts_dense)
