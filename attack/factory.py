from __future__ import annotations

from .gaussian import GaussianAttack
from .none_attack import NoAttack
from .sign_flip import SignFlipAttack


def _require(cfg: dict, key: str):
    value = cfg.get(key)
    if value is None:
        raise ValueError(f"Missing required attack configuration: {key}")
    return value


def build_attack(cfg: dict):
    name = cfg.get("name")
    if name is None:
        raise ValueError("Missing required attack configuration: name")
    name = str(name).lower()
    if name == "none":
        return NoAttack()
    if name in {"sign", "sign_flip", "sign-flip", "sign_flipping"}:
        return SignFlipAttack(scale=float(_require(cfg, "sign_scale")))
    if name in {"gaussian", "gaussian_noise"}:
        return GaussianAttack(std=float(_require(cfg, "gaussian_std")))
    raise ValueError(f"Unknown attack: {name}")
