from __future__ import annotations

import torch


def parameters_to_vector(model) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector([p.detach() for p in model.parameters()])


def vector_to_parameters(vector: torch.Tensor, model) -> None:
    torch.nn.utils.vector_to_parameters(vector, model.parameters())


def gradients_to_vector(model) -> torch.Tensor:
    """Flatten parameter gradients into one vector in model parameter order."""
    gradients = []
    for parameter in model.parameters():
        if parameter.grad is None:
            gradients.append(torch.zeros_like(parameter).reshape(-1))
        else:
            gradients.append(parameter.grad.detach().reshape(-1))
    if not gradients:
        raise ValueError("Model contains no parameters.")
    return torch.cat(gradients)


@torch.no_grad()
def write_vector_to_gradients(vector: torch.Tensor, model) -> None:
    """Write a flat gradient vector into ``parameter.grad`` tensors."""
    offset = 0
    for parameter in model.parameters():
        numel = parameter.numel()
        if offset + numel > vector.numel():
            raise ValueError("Gradient vector is smaller than model parameter size.")
        chunk = vector[offset : offset + numel].view_as(parameter)
        if parameter.grad is None:
            parameter.grad = chunk.detach().clone()
        else:
            parameter.grad.copy_(chunk)
        offset += numel
    if offset != vector.numel():
        raise ValueError("Gradient vector size does not match model parameters.")
