from __future__ import annotations

import numpy as np


def iid_partition(num_samples: int, num_clients: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    return [arr.astype(np.int64) for arr in np.array_split(indices, num_clients)]


def dirichlet_partition(labels, num_clients: int, alpha: float, seed: int) -> list[np.ndarray]:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
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
