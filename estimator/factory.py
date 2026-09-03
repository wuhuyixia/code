from __future__ import annotations

from .autograd import AutogradEstimator
from .oprf import OPRFEstimator
from .zeroth_order import TwoPointEstimator


def _require(cfg: dict, key: str):
    value = cfg.get(key)
    if value is None:
        raise ValueError(f"Missing required estimator configuration: {key}")
    return value


def build_estimator(cfg: dict, *, seed: int | None = None):
    name = cfg.get("name")
    if name is None:
        raise ValueError("Missing required estimator configuration: name")
    name = str(name).lower()
    if name == "autograd":
        return AutogradEstimator()
    if name == "oprf":
        return OPRFEstimator(
            delta=float(_require(cfg, "delta")),
            init_mode=str(_require(cfg, "init_mode")),
            seed=seed,
        )
    if name in {"zeroth_order", "two_point", "zo"}:
        return TwoPointEstimator(
            smoothing=float(_require(cfg, "smoothing")),
            seed=seed,
        )
    raise ValueError(f"Unknown estimator: {name}")
