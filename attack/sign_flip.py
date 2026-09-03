from __future__ import annotations

import torch


class SignFlipAttack:
    """Reverse and amplify a locally computed model update: delta -> -scale * delta."""

    def __init__(self, scale: float = 5.0):
        if scale < 0:
            raise ValueError("scale must be non-negative")
        self.scale = float(scale)

    def __call__(self, update: torch.Tensor, **_: object) -> torch.Tensor:
        return -self.scale * update
