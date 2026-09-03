from __future__ import annotations

from .cnn import SmallCNN
from .mlp import MLP


def _require(cfg: dict, key: str):
    value = cfg.get(key)
    if value is None:
        raise ValueError(f"Missing required model configuration: {key}")
    return value


def build_model(cfg: dict, dataset_name: str | None = None):
    name = cfg.get("name")
    if name is None:
        raise ValueError("Missing required model configuration: name")
    name = str(name).lower()
    num_classes = int(_require(cfg, "num_classes"))

    if name == "cnn":
        in_channels = int(_require(cfg, "in_channels"))
        return SmallCNN(in_channels=in_channels, num_classes=num_classes)
    if name == "mlp":
        input_dim = int(_require(cfg, "input_dim"))
        return MLP(input_dim=input_dim, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")
