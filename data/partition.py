from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def iid_partition(num_samples: int, num_clients: int, seed: int) -> list[np.ndarray]:
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    return [arr.astype(np.int64) for arr in np.array_split(indices, num_clients)]


def dirichlet_partition(labels, num_clients: int, alpha: float, seed: int) -> list[np.ndarray]:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for cls in np.unique(labels):
        cls_idx = np.flatnonzero(labels == cls)
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        cuts = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        splits = np.split(cls_idx, cuts)
        for cid, split in enumerate(splits):
            client_indices[cid].extend(split.tolist())
    return [np.asarray(sorted(v), dtype=np.int64) for v in client_indices]


def build_partition(labels, cfg: dict, seed: int) -> list[np.ndarray]:
    name = cfg.get("partition")
    num_clients = cfg.get("num_clients")
    if name is None or num_clients is None:
        raise ValueError("partition and num_clients must be configured")
    name = str(name).lower()
    num_clients = int(num_clients)
    if name == "iid":
        return iid_partition(len(labels), num_clients, seed)
    if name == "dirichlet":
        alpha = cfg.get("dirichlet_alpha")
        if alpha is None:
            raise ValueError("dirichlet_alpha is required for Dirichlet partitioning")
        return dirichlet_partition(labels, num_clients, float(alpha), seed)
    raise ValueError(f"Unknown partition method: {name}")


def partition_summary(partitions, labels) -> dict:
    labels = np.asarray(labels)
    classes = np.unique(labels)
    clients = {}
    for client_id, indices in enumerate(partitions):
        indices = np.asarray(indices, dtype=np.int64)
        counts = {str(int(cls)): int(np.sum(labels[indices] == cls)) for cls in classes}
        clients[str(client_id)] = {
            "num_samples": int(indices.size),
            "class_counts": counts,
            "indices": indices.tolist(),
        }
    return {"clients": clients}


def save_partition(path: str | Path, partitions, labels) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(partition_summary(partitions, labels), handle, indent=2)
