from __future__ import annotations

import torch
from torch import nn

from utils.tensor import gradients_to_vector
from .base import BaseGradientEstimator, GradientEstimate


class AutogradEstimator(BaseGradientEstimator):
    """Standard first-order stochastic gradient computed by PyTorch autograd."""

    name = "autograd"

    def estimate(
        self,
        model: nn.Module,
        batch,
        criterion: nn.Module,
        device: torch.device,
        *,
        client_id: int,
        round_id: int,
        batch_id: int,
    ) -> GradientEstimate:
        model.zero_grad(set_to_none=True)
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        gradient = gradients_to_vector(model)

        return GradientEstimate(
            gradient=gradient,
            loss=float(loss.detach().item()),
            num_queries=1,
            num_forward_queries=1,
            num_backward_calls=1,
        )
