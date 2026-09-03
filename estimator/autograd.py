from __future__ import annotations

import torch


def estimate(loss: torch.Tensor, parameters) -> torch.Tensor:
    grads = torch.autograd.grad(loss, tuple(parameters), retain_graph=False, create_graph=False)
    return torch.cat([g.detach().reshape(-1) for g in grads])
