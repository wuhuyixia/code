from __future__ import annotations

import torch


def aggregate(updates: list[torch.Tensor], weights: list[float] | None = None) -> torch.Tensor:
    if not updates:
        raise ValueError("updates must be non-empty")
    stacked = torch.stack(updates)
    if weights is None:
        return stacked.mean(dim=0)
    w = torch.as_tensor(weights, dtype=stacked.dtype, device=stacked.device)
    w = w / w.sum()
    return (stacked * w.view(-1, *([1] * (stacked.ndim - 1)))).sum(dim=0)
