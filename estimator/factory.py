from __future__ import annotations

from .autograd import AutogradEstimator
from .oprf import OPRFEstimator
from . import zeroth_order


def build_estimator(cfg: dict, *, seed: int | None = None):
    name = cfg.get("name", "autograd").lower()
    if name == "autograd":
        return AutogradEstimator()
    if name == "oprf":
        return OPRFEstimator(
            delta=float(cfg.get("delta", 1e-3)),
            init_mode=str(cfg.get("init_mode", "one_point")),
            seed=seed,
        )
    if name in {"zeroth_order", "two_point", "zo"}:
        # Legacy function-style estimator; it will be migrated to the common
        # BaseGradientEstimator interface when its manuscript configuration is fixed.
        return zeroth_order
    raise ValueError(f"Unknown estimator: {name}")
