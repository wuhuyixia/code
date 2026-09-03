from __future__ import annotations

import torch


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += float(criterion(logits, y).item())
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(y.numel())
    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)}
