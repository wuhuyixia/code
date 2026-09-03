from __future__ import annotations

import torch


class GaussianAttack:
    """Add zero-mean Gaussian noise to a model/update tensor."""

    def __init__(self, std: float = 1.0):
        if std < 0:
            raise ValueError("std must be non-negative")
        self.std = float(std)

    def __call__(self, update: torch.Tensor, **_: object) -> torch.Tensor:
        if self.std == 0:
            return update
        return update + torch.randn_like(update) * self.std
