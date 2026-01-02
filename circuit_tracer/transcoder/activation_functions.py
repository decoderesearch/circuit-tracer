from typing import Any

import torch
from torch import nn


def rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x)


class jumprelu(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
        return (x * (x > threshold)).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold) * grad_output  # We don't apply STE to x input
        threshold_grad = torch.sum(
            -(threshold / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output,
            dim=0,
        )
        return x_grad, threshold_grad, None


class JumpReLU(torch.nn.Module):
    def __init__(self, threshold: float | torch.Tensor, bandwidth: float = 2) -> None:
        super().__init__()
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(threshold)

        self.threshold = nn.Parameter(threshold)
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return jumprelu.apply(x, self.threshold, self.bandwidth)  # type: ignore

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}, bandwidth={self.bandwidth}"


class TopK(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor):
        _, indices = torch.topk(x, k=self.k, dim=-1)
        gate = torch.zeros_like(x)
        gate.scatter_(dim=-1, index=indices, value=1)
        return x * gate.to(x.dtype)
