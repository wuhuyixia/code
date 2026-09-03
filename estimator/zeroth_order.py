from __future__ import annotations

from collections.abc import Callable

import torch


def two_point_estimate(
    x: torch.Tensor,
    objective: Callable[[torch.Tensor], torch.Tensor],
    smoothing: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")
    direction = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    direction = direction / direction.norm().clamp_min(torch.finfo(x.dtype).eps)
    f_plus = objective(x + smoothing * direction)
    f_minus = objective(x - smoothing * direction)
    return ((f_plus - f_minus) / (2.0 * smoothing)) * direction
