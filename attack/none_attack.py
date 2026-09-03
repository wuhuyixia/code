from __future__ import annotations

import torch


class NoAttack:
    def __call__(self, update: torch.Tensor, **_: object) -> torch.Tensor:
        return update
