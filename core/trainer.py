from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from estimator.base import BaseGradientEstimator
from utils.tensor import write_vector_to_gradients


@dataclass
class TrainStats:
    loss_sum: float = 0.0
    batches: int = 0
    queries: int = 0
    forward_queries: int = 0
    backward_calls: int = 0

    @property
    def mean_loss(self) -> float:
        return self.loss_sum / max(self.batches, 1)


def train_local(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    estimator: BaseGradientEstimator,
    device: torch.device,
    *,
    client_id: int,
    round_id: int,
    epochs: int,
    criterion: nn.Module | None = None,
) -> TrainStats:
    if epochs < 1:
        raise ValueError("epochs must be >= 1.")
    model.train()
    criterion = criterion or nn.CrossEntropyLoss()
    stats = TrainStats()

    batch_counter = 0
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            estimate = estimator.estimate(
                model,
                batch,
                criterion,
                device,
                client_id=int(client_id),
                round_id=int(round_id),
                batch_id=batch_counter,
            )
            write_vector_to_gradients(estimate.gradient, model)
            optimizer.step()

            stats.loss_sum += float(estimate.loss)
            stats.batches += 1
            stats.queries += int(estimate.num_queries)
            stats.forward_queries += int(estimate.num_forward_queries)
            stats.backward_calls += int(estimate.num_backward_calls)
            batch_counter += 1
    return stats
