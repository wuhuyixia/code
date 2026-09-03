from __future__ import annotations

import random


def complete_topology(num_clients: int, self_loop: bool = True) -> dict[int, list[int]]:
    return {
        i: [j for j in range(num_clients) if self_loop or j != i]
        for i in range(num_clients)
    }


def random_k_neighbor_topology(num_clients: int, k: int, seed: int) -> dict[int, list[int]]:
    if k >= num_clients:
        raise ValueError("k must be smaller than num_clients")
    rng = random.Random(seed)
    graph: dict[int, list[int]] = {}
    for i in range(num_clients):
        candidates = [j for j in range(num_clients) if j != i]
        graph[i] = sorted(rng.sample(candidates, k))
    return graph


def topology_for_round(cfg: dict, round_id: int, seed: int) -> dict[int, list[int]]:
    n = int(cfg["data"]["num_clients"])
    net = cfg["network"]
    if net.get("topology", "complete") == "complete":
        return complete_topology(n, bool(net.get("self_loop", True)))
    k = int(net.get("neighbors_per_node", 6))
    interval = int(net.get("switch_interval", 20))
    epoch = round_id // interval if net.get("dynamic", False) else 0
    return random_k_neighbor_topology(n, k, seed + epoch)
