from __future__ import annotations

import torch


def parameters_to_vector(model) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector([p.detach() for p in model.parameters()])


def vector_to_parameters(vector: torch.Tensor, model) -> None:
    torch.nn.utils.vector_to_parameters(vector, model.parameters())
