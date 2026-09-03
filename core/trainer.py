from __future__ import annotations

import torch


def train_local(model, loader, optimizer, device, epochs: int = 1):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return model
