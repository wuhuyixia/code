from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path


def complete_topology(num_clients: int, self_loop: bool = True) -> dict[int, list[int]]:
    return {
        i: [j for j in range(num_clients) if self_loop or j != i]
        for i in range(num_clients)
    }


def is_connected(topology: dict[int, list[int]]) -> bool:
    if not topology:
        return False
    start = next(iter(topology))
    visited = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        neighbors = set(topology.get(node, [])) - {node}
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited == set(topology)


def validate_topology(topology: dict[int, list[int]], *, require_symmetric: bool = True) -> None:
    nodes = set(topology)
    if not nodes:
        raise ValueError("topology must contain at least one node")
    for node, neighbors in topology.items():
        for neighbor in neighbors:
            if neighbor not in nodes:
                raise ValueError(f"topology references unknown node {neighbor}")
            if require_symmetric and node != neighbor and node not in topology.get(neighbor, []):
                raise ValueError(f"topology edge {node}->{neighbor} is not symmetric")
    if not is_connected(topology):
        raise ValueError("communication topology must be connected")


def random_connected_k_neighbor_topology(
    num_clients: int,
    k: int,
    seed: int,
    *,
    self_loop: bool = True,
) -> dict[int, list[int]]:
    """Build a deterministic connected undirected graph with at least k neighbors/node.

    The construction starts from a random spanning cycle to guarantee
    connectivity, then adds undirected edges until every node reaches the
    requested minimum degree. Exact regularity is not assumed by the framework.
    """
    if num_clients < 2:
        raise ValueError("num_clients must be >= 2")
    if k < 1 or k >= num_clients:
        raise ValueError("k must satisfy 1 <= k < num_clients")

    rng = random.Random(seed)
    adjacency = {i: set() for i in range(num_clients)}
    order = list(range(num_clients))
    rng.shuffle(order)
    for index, node in enumerate(order):
        nxt = order[(index + 1) % num_clients]
        adjacency[node].add(nxt)
        adjacency[nxt].add(node)

    candidates = [(i, j) for i in range(num_clients) for j in range(i + 1, num_clients)]
    rng.shuffle(candidates)
    while min(len(adjacency[i]) for i in adjacency) < k:
        added = False
        for i, j in candidates:
            if j in adjacency[i]:
                continue
            if len(adjacency[i]) < k or len(adjacency[j]) < k:
                adjacency[i].add(j)
                adjacency[j].add(i)
                added = True
                if min(len(adjacency[n]) for n in adjacency) >= k:
                    break
        if not added:
            break

    topology = {}
    for node in range(num_clients):
        neighbors = sorted(adjacency[node])
        if self_loop:
            neighbors = [node] + neighbors
        topology[node] = neighbors
    validate_topology(topology, require_symmetric=True)
    return topology


def topology_for_round(cfg: dict, round_id: int, seed: int) -> dict[int, list[int]]:
    num_clients = cfg["data"].get("num_clients")
    topology_name = cfg["network"].get("topology")
    if num_clients is None or topology_name is None:
        raise ValueError("data.num_clients and network.topology must be configured")
    num_clients = int(num_clients)
    net = cfg["network"]
    self_loop = bool(net.get("self_loop", True))

    if str(topology_name).lower() == "complete":
        topology = complete_topology(num_clients, self_loop)
        validate_topology(topology, require_symmetric=True)
        return topology

    if str(topology_name).lower() in {"k_neighbor", "random_k_neighbor"}:
        k = net.get("neighbors_per_node")
        if k is None:
            raise ValueError("network.neighbors_per_node must be configured")
        dynamic = bool(net.get("dynamic", False))
        if dynamic:
            interval = net.get("switch_interval")
            if interval is None or int(interval) < 1:
                raise ValueError("network.switch_interval must be >= 1 for dynamic topology")
            epoch = int(round_id) // int(interval)
        else:
            epoch = 0
        return random_connected_k_neighbor_topology(
            num_clients,
            int(k),
            int(seed) + epoch,
            self_loop=self_loop,
        )
    raise ValueError(f"Unknown topology type: {topology_name}")


def save_topology(path: str | Path, topology: dict[int, list[int]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(node): [int(v) for v in neighbors] for node, neighbors in topology.items()}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)
