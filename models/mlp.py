from __future__ import annotations

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)
