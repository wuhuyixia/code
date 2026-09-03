from __future__ import annotations

from . import algorithm1, dfedavg, krum, trimmed_mean


def _require(cfg: dict, key: str):
    value = cfg.get(key)
    if value is None:
        raise ValueError(f"Missing required aggregation configuration: {key}")
    return value


def build_aggregator(cfg: dict):
    name = cfg.get("name")
    if name is None:
        raise ValueError("Missing required aggregation configuration: name")
    name = str(name).lower()
    if name == "dfedavg":
        return lambda updates, **kwargs: dfedavg.aggregate(updates, kwargs.get("weights"))
    if name == "krum":
        f = int(_require(cfg, "assumed_f"))
        return lambda updates, **kwargs: krum.aggregate(updates, f)
    if name in {"trimmed_mean", "trmean"}:
        r = float(_require(cfg, "trim_ratio"))
        return lambda updates, **kwargs: trimmed_mean.aggregate(updates, r)
    if name in {"algorithm1", "algorithm_1"}:
        f = int(_require(cfg, "assumed_f"))
        return lambda updates, **kwargs: algorithm1.aggregate(updates, f=f, **kwargs)
    raise ValueError(f"Unknown aggregator: {name}")
