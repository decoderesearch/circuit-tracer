import gc

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens import HookedTransformerConfig

from circuit_tracer import ReplacementModel
from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_intervention_return_activations_tl():
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma")

    s = "The National Digital Analytics Group (ND"

    interventions = [(21, 7, 5066, 0.0)]

    logits_with_activations, activations = model.feature_intervention(
        s,
        interventions,
        constrained_layers=range(model.cfg.n_layers),  # type:ignore
        return_activations=True,
    )

    logits_without_activations, no_activations = model.feature_intervention(
        s,
        interventions,
        constrained_layers=range(model.cfg.n_layers),  # type:ignore
        return_activations=False,
    )

    assert torch.allclose(
        logits_with_activations, logits_without_activations, atol=1e-6, rtol=1e-5
    ), "Logits should be identical regardless of return_activations setting"

    assert activations is not None, "Activations should be returned when return_activations=True"
    assert no_activations is None, "Activations should be None when return_activations=False"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_intervention_return_activations_nnsight():
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", backend="nnsight")

    s = "The National Digital Analytics Group (ND"

    interventions = [(21, 7, 5066, 0.0)]

    logits_with_activations, activations = model.feature_intervention(
        s,
        interventions,
        constrained_layers=range(model.config.num_hidden_layers),  # type:ignore
        return_activations=True,
    )

    logits_without_activations, no_activations = model.feature_intervention(
        s,
        interventions,
        constrained_layers=range(model.config.num_hidden_layers),  # type:ignore
        return_activations=False,
    )

    assert torch.allclose(
        logits_with_activations, logits_without_activations, atol=1e-6, rtol=1e-5
    ), "Logits should be identical regardless of return_activations setting"

    assert activations is not None, "Activations should be returned when return_activations=True"
    assert no_activations is None, "Activations should be None when return_activations=False"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_intervention_generate_return_activations_tl():
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma")

    s = "Fait: Michael Jordan joue au"

    interventions = [(20, slice(6, None), 1454, 0), (20, slice(6, None), 341, 272)]

    with model.zero_softcap():
        str_with_acts, logits_with_activations, activations = model.feature_intervention_generate(
            s,
            interventions,
            constrained_layers=range(model.cfg.n_layers),  # type:ignore
            return_activations=True,
            do_sample=False,
        )

        str_without_acts, logits_without_activations, no_activations = (
            model.feature_intervention_generate(
                s,
                interventions,
                constrained_layers=range(model.cfg.n_layers),  # type:ignore
                return_activations=False,
                do_sample=False,
            )
        )

    assert str_with_acts == str_without_acts, (
        f"Generated strings should be identical, but got {str_with_acts}, {str_without_acts}"
    )

    assert logits_with_activations.dim() == 2, (
        "feature_intervention_generate should return 2-D logits (seq_len, vocab_size)"
    )
    assert logits_without_activations.dim() == 2, (
        "feature_intervention_generate should return 2-D logits (seq_len, vocab_size)"
    )

    assert torch.allclose(
        logits_with_activations, logits_without_activations, atol=1e-6, rtol=1e-5
    ), "Logits should be identical regardless of return_activations setting"

    assert activations is not None, "Activations should be returned when return_activations=True"
    assert no_activations is None, "Activations should be None when return_activations=False"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_intervention_generate_return_activations_nnsight():
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", backend="nnsight")

    s = "Fait: Michael Jordan joue au"

    interventions = [(20, slice(6, None), 1454, 0), (20, slice(6, None), 341, 272)]

    with model.zero_softcap():
        str_with_acts, logits_with_activations, activations = model.feature_intervention_generate(
            s,
            interventions,
            constrained_layers=range(model.config.num_hidden_layers),  # type:ignore
            return_activations=True,
            do_sample=False,
        )

        str_without_acts, logits_without_activations, no_activations = (
            model.feature_intervention_generate(
                s,
                interventions,
                constrained_layers=range(model.config.num_hidden_layers),  # type:ignore
                return_activations=False,
                do_sample=False,
            )
        )

    assert str_with_acts == str_without_acts, (
        f"Generated strings should be identical, but got {str_with_acts}, {str_without_acts}"
    )

    assert logits_with_activations.dim() == 2, (
        "feature_intervention_generate should return 2-D logits (seq_len, vocab_size)"
    )
    assert logits_without_activations.dim() == 2, (
        "feature_intervention_generate should return 2-D logits (seq_len, vocab_size)"
    )

    assert torch.allclose(
        logits_with_activations, logits_without_activations, atol=1e-6, rtol=1e-5
    ), "Logits should be identical regardless of return_activations setting"

    assert activations is not None, "Activations should be returned when return_activations=True"
    assert no_activations is None, "Activations should be None when return_activations=False"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_intervention_sparse_tl():
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma")

    s = "The National Digital Analytics Group (ND"

    interventions = [(21, 7, 5066, 0.0)]

    logits_dense, activations_dense = model.feature_intervention(
        s,
        interventions,
        constrained_layers=range(model.cfg.n_layers),  # type:ignore
        sparse=False,
        return_activations=True,
    )

    logits_sparse, activations_sparse = model.feature_intervention(
        s,
        interventions,
        constrained_layers=range(model.cfg.n_layers),  # type:ignore
        sparse=True,
        return_activations=True,
    )

    assert torch.allclose(logits_dense, logits_sparse, atol=1e-6, rtol=1e-5), (
        "Logits should be identical regardless of sparse setting"
    )

    assert torch.allclose(
        activations_dense,  # type:ignore
        activations_sparse.to_dense(),  # type:ignore
        atol=1e-6,
        rtol=1e-5,
    ), "Activations should be identical regardless of sparse setting"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_intervention_sparse_nnsight():
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", backend="nnsight")

    s = "The National Digital Analytics Group (ND"

    interventions = [(21, 7, 5066, 0.0)]

    logits_dense, activations_dense = model.feature_intervention(
        s,
        interventions,
        constrained_layers=range(model.config.num_hidden_layers),  # type:ignore
        sparse=False,
        return_activations=True,
    )

    logits_sparse, activations_sparse = model.feature_intervention(
        s,
        interventions,
        constrained_layers=range(model.config.num_hidden_layers),  # type:ignore
        sparse=True,
        return_activations=True,
    )

    assert torch.allclose(logits_dense, logits_sparse, atol=1e-6, rtol=1e-5), (
        "Logits should be identical regardless of sparse setting"
    )

    assert torch.allclose(
        activations_dense,  # type:ignore
        activations_sparse.to_dense(),  # type:ignore
        atol=1e-6,
        rtol=1e-5,
    ), "Activations should be identical regardless of sparse setting"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_intervention_generate_sparse_tl():
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma")

    s = "Fait: Michael Jordan joue au"

    interventions = [(20, slice(6, None), 1454, 0), (20, slice(6, None), 341, 272)]

    with model.zero_softcap():
        str_dense, logits_dense, activations_dense = model.feature_intervention_generate(
            s,
            interventions,
            constrained_layers=range(model.cfg.n_layers),  # type:ignore
            sparse=False,
            do_sample=False,
            return_activations=True,
        )

        str_sparse, logits_sparse, activations_sparse = model.feature_intervention_generate(
            s,
            interventions,
            constrained_layers=range(model.cfg.n_layers),  # type:ignore
            sparse=True,
            do_sample=False,
            return_activations=True,
        )

    assert str_dense == str_sparse, (
        f"Generated strings should be identical, but got {str_dense}, {str_sparse}"
    )

    assert logits_dense.dim() == 2, (
        "feature_intervention_generate should return 2-D logits (seq_len, vocab_size)"
    )

    assert torch.allclose(logits_dense, logits_sparse, atol=1e-6, rtol=1e-5), (
        "Logits should be identical regardless of sparse setting"
    )

    assert torch.allclose(
        activations_dense,  # type:ignore
        activations_sparse.to_dense(),  # type:ignore
        atol=1e-6,
        rtol=1e-5,
    ), "Activations should be identical regardless of sparse setting"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_intervention_generate_sparse_nnsight():
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", backend="nnsight")

    s = "Fait: Michael Jordan joue au"

    interventions = [(20, slice(6, None), 1454, 0), (20, slice(6, None), 341, 272)]

    with model.zero_softcap():
        str_dense, logits_dense, activations_dense = model.feature_intervention_generate(
            s,
            interventions,
            constrained_layers=range(model.config.num_hidden_layers),  # type:ignore
            sparse=False,
            do_sample=False,
            return_activations=True,
        )

        str_sparse, logits_sparse, activations_sparse = model.feature_intervention_generate(
            s,
            interventions,
            constrained_layers=range(model.config.num_hidden_layers),  # type:ignore
            sparse=True,
            do_sample=False,
            return_activations=True,
        )

    assert str_dense == str_sparse, (
        f"Generated strings should be identical, but got {str_dense}, {str_sparse}"
    )

    assert logits_dense.dim() == 2, (
        "feature_intervention_generate should return 2-D logits (seq_len, vocab_size)"
    )

    assert torch.allclose(logits_dense, logits_sparse, atol=1e-6, rtol=1e-5), (
        "Logits should be identical regardless of sparse setting"
    )

    assert torch.allclose(
        activations_dense,  # type:ignore
        activations_sparse.to_dense(),  # type:ignore
        atol=1e-6,
        rtol=1e-5,
    ), "Activations should be identical regardless of sparse setting"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_intervention_generate_logits_dim_tl():
    """Regression test: feature_intervention_generate must return 2-D logits (seq_len, vocab_size)."""
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma")

    s = "Fait: Michael Jordan joue au"
    interventions = [(20, slice(6, None), 1454, 0), (20, slice(6, None), 341, 272)]

    with model.zero_softcap():
        _, logits, _ = model.feature_intervention_generate(
            s,
            interventions,
            constrained_layers=range(model.cfg.n_layers),  # type:ignore
            do_sample=False,
        )

    assert logits.dim() == 2, (
        f"Expected 2-D logits (seq_len, vocab_size) but got {logits.dim()}-D with shape {logits.shape}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_intervention_generate_logits_dim_nnsight():
    """Regression test: feature_intervention_generate must return 2-D logits (seq_len, vocab_size)."""
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", backend="nnsight")

    s = "Fait: Michael Jordan joue au"
    interventions = [(20, slice(6, None), 1454, 0), (20, slice(6, None), 341, 272)]

    with model.zero_softcap():
        _, logits, _ = model.feature_intervention_generate(
            s,
            interventions,
            constrained_layers=range(model.config.num_hidden_layers),  # type:ignore
            do_sample=False,
        )

    assert logits.dim() == 2, (
        f"Expected 2-D logits (seq_len, vocab_size) but got {logits.dim()}-D with shape {logits.shape}"
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    test_intervention_return_activations_tl()
    test_intervention_return_activations_nnsight()
    test_intervention_generate_return_activations_tl()
    test_intervention_generate_return_activations_nnsight()

    test_intervention_sparse_tl()
    test_intervention_sparse_nnsight()
    test_intervention_generate_sparse_tl()
    test_intervention_generate_sparse_nnsight()


def load_tiny_gpt2_model():
    """Random 2-layer model with the GPT-2 tokenizer, which does not add a BOS token
    itself (TransformerLens's to_tokens prepends one)."""
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=16,
        d_head=4,
        n_heads=4,
        d_mlp=32,
        n_ctx=64,
        d_vocab=50257,
        act_fn="gelu_new",
        normalization_type="LN",
        tokenizer_name="gpt2",
        model_name="tiny-gpt2",
        original_architecture="GPT2LMHeadModel",
        default_prepend_bos=True,
        device="cpu",
    )
    transcoders = {
        layer: SingleLayerTranscoder(cfg.d_model, 64, F.relu, layer, device=torch.device("cpu"))
        for layer in range(cfg.n_layers)
    }
    for transcoder in transcoders.values():
        for _, param in transcoder.named_parameters():
            nn.init.uniform_(param, a=-1, b=1)
    transcoder_set = TranscoderSet(
        transcoders, feature_input_hook="mlp.hook_in", feature_output_hook="mlp.hook_out"
    )
    model = ReplacementModel.from_config(cfg, transcoder_set)
    for _, param in model.named_parameters():
        nn.init.uniform_(param, a=-1, b=1)
    return model


def test_intervention_without_freeze_string_input_tl():
    model = load_tiny_gpt2_model()
    s = "The capital of France is"
    interventions = [(0, 2, 3, 5.0)]

    # to_tokens prepends BOS, the bare GPT-2 tokenizer does not
    tokens = model.to_tokens(s)[0]
    assert len(tokens) == len(model.tokenizer(s).input_ids) + 1  # type: ignore

    logits_str, _ = model.feature_intervention(s, interventions, freeze_attention=False)
    logits_tok, _ = model.feature_intervention(tokens, interventions, freeze_attention=False)

    assert logits_str.shape[1] == len(tokens)
    assert torch.allclose(logits_str, logits_tok)


def test_intervention_without_freeze_2d_tensor_input_tl():
    model = load_tiny_gpt2_model()
    tokens = model.to_tokens("The capital of France is")  # (1, n_pos)
    interventions = [(0, 0, 3, 5.0)]

    logits_2d, _ = model.feature_intervention(tokens, interventions, freeze_attention=False)
    logits_1d, _ = model.feature_intervention(tokens[0], interventions, freeze_attention=False)
    clean_logits = model(tokens)

    assert torch.allclose(logits_2d, logits_1d)
    # sanity check: the intervention changes the output
    assert not torch.allclose(logits_2d[:, 0], clean_logits[:, 0])
