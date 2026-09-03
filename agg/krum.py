from __future__ import annotations

import torch


def aggregate(updates: list[torch.Tensor], assumed_f: int) -> torch.Tensor:
    n = len(updates)
    if n < 2 * assumed_f + 3:
        raise ValueError("Krum requires n >= 2f + 3")
    stacked = torch.stack([u.reshape(-1) for u in updates])
    distances = torch.cdist(stacked, stacked, p=2).pow(2)
    neighbor_count = n - assumed_f - 2
    scores = []
    for i in range(n):
        d = torch.cat([distances[i, :i], distances[i, i + 1 :]])
        scores.append(torch.topk(d, k=neighbor_count, largest=False).values.sum())
    best = int(torch.argmin(torch.stack(scores)).item())
    return updates[best]
