from __future__ import annotations

from . import autograd, oprf, zeroth_order


def build_estimator(cfg: dict):
    name = cfg.get("name", "autograd").lower()
    if name == "autograd":
        return autograd
    if name in {"zeroth_order", "two_point", "zo"}:
        return zeroth_order
    if name == "oprf":
        return oprf
    raise ValueError(f"Unknown estimator: {name}")
