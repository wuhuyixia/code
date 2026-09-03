from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class GradientEstimate:
    """Standardized result returned by all gradient estimators."""

    gradient: torch.Tensor
    loss: float
    num_queries: int
    num_forward_queries: int
    num_backward_calls: int
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseGradientEstimator(ABC):
    """Common interface for first- and zeroth-order gradient estimators."""

    name: str = "base"

    @abstractmethod
    def estimate(
        self,
        model: nn.Module,
        batch,
        criterion: nn.Module,
        device: torch.device,
        *,
        client_id: int,
        round_id: int,
        batch_id: int,
    ) -> GradientEstimate:
        raise NotImplementedError

    def reset(self, client_id: int | None = None) -> None:
        """Reset estimator state globally or for one client when stateful."""
        return None
