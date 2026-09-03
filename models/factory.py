from __future__ import annotations

from .cnn import SmallCNN
from .mlp import MLP


def build_model(cfg: dict, dataset_name: str):
    name = cfg.get("name", "cnn").lower()
    num_classes = int(cfg.get("num_classes", 10))
    in_channels = 3 if dataset_name.lower() == "gtsrb" else 1
    if name == "cnn":
        return SmallCNN(in_channels=in_channels, num_classes=num_classes)
    if name == "mlp":
        input_dim = 3 * 32 * 32 if dataset_name.lower() == "gtsrb" else 28 * 28
        return MLP(input_dim=input_dim, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")
