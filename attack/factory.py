from __future__ import annotations

from .gaussian import GaussianAttack
from .none_attack import NoAttack
from .sign_flip import SignFlipAttack


def build_attack(cfg: dict):
    name = cfg.get("name", "none").lower()
    if name == "none":
        return NoAttack()
    if name in {"sign", "sign_flip", "sign-flip", "sign_flipping"}:
        return SignFlipAttack(scale=float(cfg.get("sign_scale", 5.0)))
    if name in {"gaussian", "gaussian_noise"}:
        return GaussianAttack(std=float(cfg.get("gaussian_std", 1.0)))
    raise ValueError(f"Unknown attack: {name}")
