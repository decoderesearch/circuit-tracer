"""Every architecture in `VERIFIED_MODEL_TYPES` passes `_verify_freezes`.

Loading one of those architectures skips the check, so this test is what keeps the set honest. It
runs on tiny untrained models: the freezes are properties of the architecture, not the weights.
"""

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.replacement_model.replacement_model_interp_engine import VERIFIED_MODEL_TYPES
from circuit_tracer.transcoder.activation_functions import JumpReLU
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder, TranscoderSet

# The gpt2 tokenizer stands in for every family; its bos id is what `_verify_freezes` feeds the
# model, so the vocabulary has to be at least that large.
TINY = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=50304,
    _name_or_path="openai-community/gpt2",
)


def _tiny_model(model_type: str):
    config = AutoConfig.for_model(
        model_type, architectures=[MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[model_type]], **TINY
    )
    transcoders = {
        layer: SingleLayerTranscoder(
            config.hidden_size, config.hidden_size * 4, JumpReLU(0.0, 0.1), layer
        )
        for layer in range(config.num_hidden_layers)
    }
    for transcoder in transcoders.values():
        for param in transcoder.parameters():
            nn.init.uniform_(param, a=-1, b=1)
    transcoder_set = TranscoderSet(
        transcoders, feature_input_hook="hook_resid_mid", feature_output_hook="hook_mlp_out"
    )
    return ReplacementModel.from_config(config, transcoder_set, backend="interp_engine")


def test_verified_types_are_all_covered_here():
    assert set(VERIFIED_MODEL_TYPES) == {"gemma2", "gemma3_text", "llama", "qwen3"}


@pytest.mark.parametrize("model_type", sorted(VERIFIED_MODEL_TYPES))
def test_freezes_hold(model_type: str):
    torch.manual_seed(0)
    model = _tiny_model(model_type)
    model._verify_freezes()
