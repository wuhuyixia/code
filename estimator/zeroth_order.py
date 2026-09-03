from __future__ import annotations

import torch
from torch import nn

from utils.tensor import parameters_to_vector, vector_to_parameters
from .base import BaseGradientEstimator, GradientEstimate


class TwoPointEstimator(BaseGradientEstimator):
    """Symmetric two-point zeroth-order gradient estimator.

    The smoothing radius is configuration-driven. Each estimate performs two
    forward objective evaluations and no backward pass.
    """

    name = "two_point"

    def __init__(self, smoothing: float, seed: int | None = None) -> None:
        if smoothing <= 0:
            raise ValueError("smoothing must be positive.")
        self.smoothing = float(smoothing)
        self.seed = seed
        self._generators: dict[tuple[int, str], torch.Generator] = {}

    def _generator(self, client_id: int, device: torch.device):
        if self.seed is None:
            return None
        key = (int(client_id), str(device))
        if key not in self._generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(self.seed) + 1_000_003 * int(client_id))
            self._generators[key] = generator
        return self._generators[key]

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
        inputs, targets = batch
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        x = parameters_to_vector(model).detach()
        direction = torch.randn(
            x.shape,
            dtype=x.dtype,
            device=x.device,
            generator=self._generator(client_id, x.device),
        )
        norm = direction.norm().clamp_min(torch.finfo(x.dtype).eps)
        direction = direction / norm

        plus = x + self.smoothing * direction
        minus = x - self.smoothing * direction

        try:
            with torch.no_grad():
                vector_to_parameters(plus, model)
                loss_plus = criterion(model(inputs), targets)
                vector_to_parameters(minus, model)
                loss_minus = criterion(model(inputs), targets)
        finally:
            with torch.no_grad():
                vector_to_parameters(x, model)

        gradient = ((loss_plus - loss_minus) / (2.0 * self.smoothing)) * direction
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Two-point gradient estimate became non-finite.")

        mean_loss = 0.5 * (loss_plus + loss_minus)
        return GradientEstimate(
            gradient=gradient.detach(),
            loss=float(mean_loss.detach().item()),
            num_queries=2,
            num_forward_queries=2,
            num_backward_calls=0,
            metadata={
                "smoothing": self.smoothing,
                "round_id": int(round_id),
                "batch_id": int(batch_id),
            },
        )

    def reset(self, client_id: int | None = None) -> None:
        if client_id is None:
            self._generators.clear()
            return
        client_id = int(client_id)
        for key in [key for key in self._generators if key[0] == client_id]:
            self._generators.pop(key, None)
