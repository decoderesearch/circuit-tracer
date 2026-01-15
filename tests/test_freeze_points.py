import gc

import pytest
import torch
from nnsight import save

from circuit_tracer import ReplacementModel
from circuit_tracer.replacement_model.replacement_model_transformerlens import (
    TransformerLensReplacementModel,
)
from circuit_tracer.replacement_model.replacement_model_nnsight import (
    NNSightReplacementModel,
)


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


# Model configurations: (model_name, transcoder_set_name)
# Add new models here to extend tests
MODEL_CONFIGS_TL = [
    ("google/gemma-2-2b", "gemma"),
]

MODEL_CONFIGS_NNSIGHT = [
    ("google/gemma-2-2b", "gemma"),
    # ("google/gemma-3-1b-pt", "mwhanna/gemma-scope-2-1b-pt/clt/width_262k_l0_medium_affine"),  # This requires lazy loading
    (
        "google/gemma-3-4b-pt",
        # we use width_16k here instead of 262k to avoid large download not used elsewhere in the test suite
        "mwhanna/gemma-scope-2-4b-pt/transcoder_all/width_16k_l0_small_affine",
    ),
]


def _fuzz_tokens(tokens: torch.Tensor, vocab_size: int, special_ids: set[int]) -> torch.Tensor:
    """Create a fuzzed version of tokens, keeping BOS but randomizing the rest."""
    fuzzed = tokens.clone()
    n_tokens = len(tokens)

    # Generate random non-special tokens for positions 1 onwards
    for i in range(1, n_tokens):
        while True:
            rand_token = torch.randint(0, vocab_size, (1,), device=tokens.device).item()
            if rand_token not in special_ids:
                fuzzed[i] = rand_token  # type: ignore
                break

    return fuzzed


def verify_freeze_gradient_invariance_tl(
    model: TransformerLensReplacementModel,
    inputs: str | torch.Tensor,
    atol: float = 1e-5,
):
    """Verify that freeze hooks make gradients invariant to input changes (TransformerLens).

    When all nonlinearities are frozen, the gradient of any output with respect to
    the inputs should be the same regardless of what the actual input tokens are
    (except for the BOS token which must match).
    """
    torch.manual_seed(42)
    input1 = model.ensure_tokenized(inputs)

    special_ids = set(model.tokenizer.all_special_ids)  # type: ignore
    input2 = _fuzz_tokens(input1, model.cfg.d_vocab, special_ids)

    _, freeze_hooks = model.setup_intervention_with_freeze(
        input1, constrained_layers=range(model.cfg.n_layers)
    )

    def get_gradients_with_freeze(tokens: torch.Tensor):
        mlp_out_grads: dict[int, torch.Tensor] = {}
        embed_grad: list[torch.Tensor | None] = [None]
        hook_handles: list = []

        def make_mlp_bwd_hook(layer: int):
            def grad_hook(grads: torch.Tensor):
                mlp_out_grads[layer] = grads.detach().clone()

            return grad_hook

        def embed_bwd_hook(grads: torch.Tensor):
            embed_grad[0] = grads.detach().clone()

        def make_mlp_fwd_hook(layer: int):
            def fwd_hook(activations, hook):  # type: ignore
                activations.requires_grad_(True)
                handle = activations.register_hook(make_mlp_bwd_hook(layer))
                hook_handles.append(handle)
                return activations

            return fwd_hook

        def embed_fwd_hook(activations, hook):  # type: ignore
            activations.requires_grad_(True)
            handle = activations.register_hook(embed_bwd_hook)
            hook_handles.append(handle)
            return activations

        # Gradient capture hooks run AFTER freeze hooks
        grad_capture_hooks = [
            (f"blocks.{layer}.{model.feature_output_hook}", make_mlp_fwd_hook(layer))
            for layer in range(model.cfg.n_layers)
        ]
        grad_capture_hooks.append(("hook_embed", embed_fwd_hook))

        try:
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=list(freeze_hooks) + grad_capture_hooks,  # type: ignore
            )
            max_logit = logits[0, -1].max()
            max_logit.backward()
        finally:
            for handle in hook_handles:
                handle.remove()

        assert embed_grad[0] is not None
        return mlp_out_grads, embed_grad[0]

    grads1_mlp, grads1_embed = get_gradients_with_freeze(input1)
    grads2_mlp, grads2_embed = get_gradients_with_freeze(input2)
    assert torch.allclose(grads1_embed, grads2_embed, atol=atol), "Embed gradients differ"

    for layer in range(model.cfg.n_layers):
        assert torch.allclose(grads1_mlp[layer], grads2_mlp[layer], atol=atol), (
            f"MLP output gradients differ at layer {layer}"
        )


def verify_freeze_gradient_invariance_nnsight(
    model: NNSightReplacementModel,
    inputs: str | torch.Tensor,
    atol: float = 1e-5,
):
    """Verify that freeze hooks make gradients invariant to input changes (NNSight).

    When all nonlinearities are frozen, the gradient of any output with respect to
    the inputs should be the same regardless of what the actual input tokens are
    (except for the BOS token which must match).
    """
    torch.manual_seed(42)
    input1 = model.ensure_tokenized(inputs)

    special_ids = set(model.tokenizer.all_special_ids)  # type: ignore
    input2 = _fuzz_tokens(input1, model.tokenizer.vocab_size, special_ids)  # type: ignore

    n_layers: int = model.cfg.n_layers  # type: ignore

    _, freeze_hooks1 = model.setup_intervention_with_freeze(
        input1, constrained_layers=range(n_layers)
    )

    _, freeze_hooks2 = model.setup_intervention_with_freeze(
        input1, constrained_layers=range(n_layers)
    )

    def get_gradients_with_freeze(tokens: torch.Tensor, freeze_hooks):
        mlp_outs = []
        with model.trace() as tracer:  # type: ignore
            with tracer.invoke(tokens):
                pass

            barrier = tracer.barrier(2)

            for freeze_fn in freeze_hooks:
                with tracer.invoke():
                    freeze_fn(direct_effects_barrier=barrier)

            with tracer.invoke():
                embed_out = model.embed_location.output
                embed_out.requires_grad = True  # type: ignore
                save(embed_out)  # type: ignore
                for layer in range(n_layers):
                    barrier()
                    feature_output_loc = model.get_feature_output_loc(layer)
                    mlp_out = feature_output_loc.output  # type: ignore
                    if not mlp_out.requires_grad:
                        mlp_out.requires_grad = True
                    mlp_outs.append(save(mlp_out))
                logits = model.output.logits  # type: ignore
                max_logit = save(logits[0, -1].max())  # type: ignore

        mlp_out_grads = []
        with max_logit.backward():  # type: ignore
            for layer in reversed(range(n_layers)):
                mlp_out_grads.append(save(mlp_outs[layer].grad))  # type: ignore
            embed_grad = save(embed_out.grad)  # type: ignore

        return list(reversed(mlp_out_grads)), embed_grad

    grads1_mlp, grads1_embed = get_gradients_with_freeze(input1, freeze_hooks1)
    grads2_mlp, grads2_embed = get_gradients_with_freeze(input2, freeze_hooks2)
    assert torch.allclose(grads1_embed, grads2_embed, atol=atol), "Embed gradients differ"

    for layer in range(n_layers):
        assert torch.allclose(grads1_mlp[layer], grads2_mlp[layer], atol=atol), (
            f"MLP output gradients differ at layer {layer}"
        )


def run_gradient_invariance_test_tl(model_name: str, transcoder_set_name: str):
    model = ReplacementModel.from_pretrained(
        model_name, transcoder_set_name, backend="transformerlens"
    )
    assert isinstance(model, TransformerLensReplacementModel)

    with model.zero_softcap():
        verify_freeze_gradient_invariance_tl(model, "The quick brown fox jumps")


def run_gradient_invariance_test_nnsight(model_name: str, transcoder_set_name: str):
    model = ReplacementModel.from_pretrained(
        model_name,
        transcoder_set_name,
        backend="nnsight",
        lazy_decoder=True,
        lazy_encoder=True,
    )
    assert isinstance(model, NNSightReplacementModel)

    with model.zero_softcap():
        verify_freeze_gradient_invariance_nnsight(model, "The quick brown fox jumps")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("model_name,transcoder_set_name", MODEL_CONFIGS_TL)
def test_freeze_gradient_invariance_tl(model_name: str, transcoder_set_name: str):
    run_gradient_invariance_test_tl(model_name, transcoder_set_name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("model_name,transcoder_set_name", MODEL_CONFIGS_NNSIGHT)
def test_freeze_gradient_invariance_nnsight(model_name: str, transcoder_set_name: str):
    run_gradient_invariance_test_nnsight(model_name, transcoder_set_name)


if __name__ == "__main__":
    for model_name, transcoder_set_name in MODEL_CONFIGS_TL:
        print(f"Testing TL: {model_name} / {transcoder_set_name}")
        run_gradient_invariance_test_tl(model_name, transcoder_set_name)

    for model_name, transcoder_set_name in MODEL_CONFIGS_NNSIGHT:
        print(f"Testing NNSight: {model_name} / {transcoder_set_name}")
        run_gradient_invariance_test_nnsight(model_name, transcoder_set_name)
