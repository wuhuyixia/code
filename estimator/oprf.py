from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from utils.tensor import parameters_to_vector, vector_to_parameters
from .base import BaseGradientEstimator, GradientEstimate


class OPRFEstimator(BaseGradientEstimator):
    """One-Point Residual Feedback zeroth-order gradient estimator.

    Core form
    ---------
    y_t = f(x_t + delta_t u_t)
    g_t = (y_t - y_{t-1}) / delta_t * u_t

    Only one new objective evaluation is performed per call. Residual history
    and random-number streams are stored independently for each client.

    This module intentionally contains no manuscript-specific numerical values.
    The smoothing parameter and initialization mode must be supplied explicitly
    by the experiment configuration.
    """

    name = "oprf"

    def __init__(
        self,
        *,
        delta: float,
        init_mode: str,
        seed: Optional[int] = None,
    ) -> None:
        if delta <= 0:
            raise ValueError("delta must be positive.")
        if init_mode not in {"one_point", "zero"}:
            raise ValueError("init_mode must be 'one_point' or 'zero'.")

        self.delta = float(delta)
        self.init_mode = str(init_mode)
        self.seed = seed
        self._prev_values: dict[int, torch.Tensor] = {}
        self._iterations: dict[int, int] = {}
        self._generators: dict[tuple[int, str], torch.Generator] = {}

    def _get_generator(self, client_id: int, device: torch.device):
        if self.seed is None:
            return None
        key = (int(client_id), str(device))
        if key not in self._generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(self.seed) + 1_000_003 * int(client_id))
            self._generators[key] = generator
        return self._generators[key]

    def _sample_direction(self, x: torch.Tensor, *, client_id: int) -> torch.Tensor:
        return torch.randn(
            x.shape,
            dtype=x.dtype,
            device=x.device,
            generator=self._get_generator(client_id, x.device),
        )

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
        client_id = int(client_id)
        delta_t = self.delta

        x_batch, y_batch = batch
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        x = parameters_to_vector(model).detach()
        if not x.is_floating_point():
            raise TypeError("Model parameters must use a floating-point dtype.")

        direction = self._sample_direction(x, client_id=client_id)
        query_point = x + delta_t * direction

        try:
            with torch.no_grad():
                vector_to_parameters(query_point, model)
                logits = model(x_batch)
                current_value = criterion(logits, y_batch)
        finally:
            with torch.no_grad():
                vector_to_parameters(x, model)

        current_value = torch.as_tensor(
            current_value,
            device=x.device,
            dtype=x.dtype,
        )
        if current_value.numel() != 1:
            raise ValueError("The objective function must return one scalar loss value.")
        current_value = current_value.reshape(())

        previous_value = self._prev_values.get(client_id)
        if previous_value is None:
            used_residual = False
            if self.init_mode == "one_point":
                residual = current_value
                gradient = (current_value / delta_t) * direction
            else:
                residual = torch.zeros((), device=x.device, dtype=x.dtype)
                gradient = torch.zeros_like(x)
        else:
            used_residual = True
            previous_value = previous_value.to(device=x.device, dtype=x.dtype)
            residual = current_value - previous_value
            gradient = (residual / delta_t) * direction

        self._prev_values[client_id] = current_value.detach().clone()
        iteration = self._iterations.get(client_id, 0)
        self._iterations[client_id] = iteration + 1

        if not torch.isfinite(gradient).all():
            raise FloatingPointError("OPRF gradient estimate became non-finite.")

        return GradientEstimate(
            gradient=gradient.detach(),
            loss=float(current_value.detach().item()),
            num_queries=1,
            num_forward_queries=1,
            num_backward_calls=0,
            metadata={
                "delta": delta_t,
                "residual": float(residual.detach().item()),
                "used_residual": used_residual,
                "iteration": iteration,
                "round_id": int(round_id),
                "batch_id": int(batch_id),
            },
        )

    def reset(self, client_id: int | None = None) -> None:
        if client_id is None:
            self._prev_values.clear()
            self._iterations.clear()
            self._generators.clear()
            return

        client_id = int(client_id)
        self._prev_values.pop(client_id, None)
        self._iterations.pop(client_id, None)
        for key in [key for key in self._generators if key[0] == client_id]:
            self._generators.pop(key, None)

    def state_dict(self) -> dict:
        return {
            "delta": self.delta,
            "init_mode": self.init_mode,
            "seed": self.seed,
            "prev_values": {
                client_id: value.detach().clone()
                for client_id, value in self._prev_values.items()
            },
            "iterations": dict(self._iterations),
        }

    def load_state_dict(self, state: dict) -> None:
        self.delta = float(state["delta"])
        self.init_mode = str(state["init_mode"])
        self.seed = state.get("seed")
        self._prev_values = {
            int(client_id): value.detach().clone()
            for client_id, value in state.get("prev_values", {}).items()
        }
        self._iterations = {
            int(client_id): int(iteration)
            for client_id, iteration in state.get("iterations", {}).items()
        }
        self._generators.clear()


OPRFGradientEstimator = OPRFEstimator
