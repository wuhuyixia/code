from __future__ import annotations

from . import algorithm1, dfedavg, krum, trimmed_mean


def build_aggregator(cfg: dict):
    name = cfg.get("name", "dfedavg").lower()
    if name == "dfedavg":
        return lambda updates, **kwargs: dfedavg.aggregate(updates, kwargs.get("weights"))
    if name == "krum":
        f = int(cfg.get("assumed_f", 2))
        return lambda updates, **kwargs: krum.aggregate(updates, f)
    if name in {"trimmed_mean", "trmean"}:
        r = float(cfg.get("trim_ratio", 0.2))
        return lambda updates, **kwargs: trimmed_mean.aggregate(updates, r)
    if name in {"algorithm1", "algorithm_1"}:
        return algorithm1.aggregate
    raise ValueError(f"Unknown aggregator: {name}")
