from __future__ import annotations

import torch


def aggregate(updates: list[torch.Tensor], trim_ratio: float = 0.2) -> torch.Tensor:
    if not updates:
        raise ValueError("updates must be non-empty")
    stacked = torch.stack(updates)
    n = stacked.shape[0]
    k = int(n * trim_ratio)
    if 2 * k >= n:
        raise ValueError("trim_ratio removes all samples")
    values, _ = torch.sort(stacked, dim=0)
    return values[k : n - k].mean(dim=0)
