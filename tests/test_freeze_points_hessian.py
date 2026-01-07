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


# Model configurations: (model_name, transcoder_set_name, backend)
# Add new models here to extend tests
MODEL_CONFIGS_TL = [
    ("google/gemma-2-2b", "gemma"),
    ("meta-llama/Llama-3.2-1B", "llama"),
]

MODEL_CONFIGS_NNSIGHT = [
    ("google/gemma-2-2b", "gemma"),
    ("meta-llama/Llama-3.2-1B", "llama"),
    (
        "google/gemma-3-1b-pt",
        "mwhanna/gemma-scope-2-1b-pt/transcoder_all/width_16k_l0_small_affine",
    ),
    (
        "google/gemma-3-4b-pt",
        "mwhanna/gemma-scope-2-4b-pt/transcoder_all/width_16k_l0_small_affine",
    ),
]


def verify_freeze_linearity_hessian_tl(
    model: TransformerLensReplacementModel,
    inputs: str | torch.Tensor,
    atol: float = 1e-5,
    n_samples: int = 20,
):
    """Verify linearity by checking that second derivatives are zero (TransformerLens).

    When all nonlinearities are frozen, the model becomes linear. We check:
    1. d²(loss)/d(embed)² = 0 (embeddings to logits path)
    2. d²(loss)/d(mlp_i)² = 0 (MLP outputs to logits path)
    """
    torch.manual_seed(42)
    input_tokens = model.ensure_tokenized(inputs)

    embed_output = [None]
    mlp_outputs = {}

    def capture_embed_hook(activations, hook):  # type: ignore
        activations = activations.clone()
        activations.requires_grad_(True)
        embed_output[0] = activations
        return activations

    def make_mlp_capture_hook(layer: int):
        def hook(activations, hook):  # type: ignore
            activations = activations.clone()
            activations.requires_grad_(True)
            mlp_outputs[layer] = activations
            return activations

        return hook

    hooks = [("hook_embed", capture_embed_hook)]
    hooks += [
        (f"blocks.{layer}.{model.feature_output_hook}", make_mlp_capture_hook(layer))
        for layer in range(model.cfg.n_layers)
    ]

    logits = model.run_with_hooks(input_tokens, fwd_hooks=hooks)  # type: ignore
    max_logit = logits[0, -1].max()

    # Test 1: Check that d²(loss)/d(embed)² = 0
    print("Testing second derivatives w.r.t. embeddings...")
    (grad_wrt_embed,) = torch.autograd.grad(
        max_logit,
        embed_output[0],  # type: ignore
        create_graph=True,
        retain_graph=True,
    )

    # Sample some elements of the gradient to check second derivatives
    flat_grad = grad_wrt_embed.flatten()
    n_check = min(n_samples, flat_grad.numel())
    max_second_deriv = 0.0

    for i in range(n_check):
        if flat_grad[i].requires_grad:
            (second_deriv,) = torch.autograd.grad(
                flat_grad[i],
                embed_output[0],  # type: ignore
                retain_graph=(i < n_check - 1),
                allow_unused=True,
            )
            if second_deriv is not None:
                max_second_deriv = max(max_second_deriv, second_deriv.abs().max().item())

    assert max_second_deriv < atol, (
        f"Second derivatives w.r.t. embeddings not zero: {max_second_deriv:.6e}"
    )

    for layer in range(min(3, model.cfg.n_layers)):
        (grad_wrt_mlp,) = torch.autograd.grad(
            max_logit,
            mlp_outputs[layer],
            create_graph=True,
            retain_graph=True,
        )

        # Sample some elements of the gradient to check second derivatives
        flat_grad = grad_wrt_mlp.flatten()
        n_check = min(n_samples, flat_grad.numel())
        max_second_deriv = 0.0

        for i in range(n_check):
            if flat_grad[i].requires_grad:
                (second_deriv,) = torch.autograd.grad(
                    flat_grad[i],
                    mlp_outputs[layer],
                    retain_graph=(i < n_check - 1) or (layer < min(2, model.cfg.n_layers - 1)),
                    allow_unused=True,
                )
                if second_deriv is not None:
                    max_second_deriv = max(max_second_deriv, second_deriv.abs().max().item())

        assert max_second_deriv < atol, (
            f"Second derivatives w.r.t. MLP {layer} output not zero: {max_second_deriv:.6e}"
        )


def verify_freeze_linearity_hessian_nnsight(
    model: NNSightReplacementModel,
    inputs: str | torch.Tensor,
    atol: float = 1e-5,
    n_samples: int = 20,
):
    """Verify linearity by checking that second derivatives are zero (NNSight).

    When all nonlinearities are frozen, the model becomes linear. We check:
    1. d²(loss)/d(embed)² = 0 (embeddings to logits path)
    2. d²(loss)/d(mlp_i)² = 0 (MLP outputs to logits path)
    """
    torch.manual_seed(42)
    input_tokens = model.ensure_tokenized(inputs)
    n_layers: int = model.cfg.n_layers  # type: ignore

    embed_out_saved = None
    mlp_outs_saved = []
    max_logit_saved = None

    with model.trace() as tracer:  # type: ignore
        with tracer.invoke(input_tokens):
            pass

        model.configure_gradient_flow(tracer)
        model.configure_skip_connection(tracer)

        with tracer.invoke():
            embed_out = model.embed_location.output
            embed_out.requires_grad = True  # type: ignore
            embed_out_saved = save(embed_out)  # type: ignore

            for layer in range(n_layers):
                feature_output_loc = model.get_feature_output_loc(layer)
                mlp_out = feature_output_loc.output  # type: ignore
                if not mlp_out.requires_grad:
                    mlp_out.requires_grad = True  # type: ignore
                mlp_outs_saved.append(save(mlp_out))

            logits = model.output.logits  # type: ignore
            max_logit = logits[0, -1].max()
            max_logit_saved = save(max_logit)  # type: ignore

    # Test 1: Check that d²(loss)/d(embed)² = 0
    (grad_wrt_embed,) = torch.autograd.grad(
        max_logit_saved,
        embed_out_saved,
        create_graph=True,
        retain_graph=True,
    )

    # Sample some elements to check second derivatives
    flat_grad = grad_wrt_embed.flatten()
    n_check = min(n_samples, flat_grad.numel())
    max_second_deriv = 0.0

    for i in range(n_check):
        if flat_grad[i].requires_grad:
            (second_deriv,) = torch.autograd.grad(
                flat_grad[i],
                embed_out_saved,
                retain_graph=(i < n_check - 1),
                allow_unused=True,
            )
            if second_deriv is not None:
                max_second_deriv = max(max_second_deriv, second_deriv.abs().max().item())

    assert max_second_deriv < atol, (
        f"Second derivatives w.r.t. embeddings not zero: {max_second_deriv:.6e}"
    )

    for layer in range(min(3, n_layers)):
        (grad_wrt_mlp,) = torch.autograd.grad(
            max_logit_saved,
            mlp_outs_saved[layer],
            create_graph=True,
            retain_graph=True,
        )

        # Sample some elements to check second derivatives
        flat_grad = grad_wrt_mlp.flatten()
        n_check = min(n_samples, flat_grad.numel())
        max_second_deriv = 0.0

        for i in range(n_check):
            if flat_grad[i].requires_grad:
                (second_deriv,) = torch.autograd.grad(
                    flat_grad[i],
                    mlp_outs_saved[layer],
                    retain_graph=(i < n_check - 1) or (layer < min(2, n_layers - 1)),
                    allow_unused=True,
                )
                if second_deriv is not None:
                    max_second_deriv = max(max_second_deriv, second_deriv.abs().max().item())

        assert max_second_deriv < atol, (
            f"Second derivatives w.r.t. MLP {layer} output not zero: {max_second_deriv:.6e}"
        )


def run_hessian_test_tl(model_name: str, transcoder_set_name: str):
    model = ReplacementModel.from_pretrained(
        model_name, transcoder_set_name, backend="transformerlens"
    )
    assert isinstance(model, TransformerLensReplacementModel)

    with model.zero_softcap():
        verify_freeze_linearity_hessian_tl(model, "The quick brown fox jumps")


def run_hessian_test_nnsight(model_name: str, transcoder_set_name: str):
    model = ReplacementModel.from_pretrained(
        model_name,
        transcoder_set_name,
        backend="nnsight",
        lazy_encoder=True,
    )
    assert isinstance(model, NNSightReplacementModel)

    with model.zero_softcap():
        verify_freeze_linearity_hessian_nnsight(model, "The quick brown fox jumps")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("model_name,transcoder_set_name", MODEL_CONFIGS_TL)
def test_freeze_linearity_hessian_tl(model_name: str, transcoder_set_name: str):
    run_hessian_test_tl(model_name, transcoder_set_name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("model_name,transcoder_set_name", MODEL_CONFIGS_NNSIGHT)
def test_freeze_linearity_hessian_nnsight(model_name: str, transcoder_set_name: str):
    run_hessian_test_nnsight(model_name, transcoder_set_name)


if __name__ == "__main__":
    for model_name, transcoder_set_name in MODEL_CONFIGS_TL:
        print(f"Testing TL: {model_name} / {transcoder_set_name}")
        run_hessian_test_tl(model_name, transcoder_set_name)

    for model_name, transcoder_set_name in MODEL_CONFIGS_NNSIGHT:
        print(f"Testing NNSight: {model_name} / {transcoder_set_name}")
        run_hessian_test_nnsight(model_name, transcoder_set_name)
