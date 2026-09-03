from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from agg.factory import build_aggregator
from attack.factory import build_attack
from core.client import Client
from core.evaluator import evaluate
from core.metrics import append_metrics, communication_cost
from core.trainer import train_local
from data.datasets import extract_targets, load_dataset
from data.partition import build_partition, save_partition
from estimator.factory import build_estimator
from models.factory import build_model
from network.topology import save_topology, topology_for_round
from utils.tensor import parameters_to_vector, vector_to_parameters


@dataclass
class SimulationResult:
    rounds_completed: int
    elapsed_seconds: float


def _require(section: dict, key: str, prefix: str):
    value = section.get(key)
    if value is None:
        raise ValueError(f"Missing required configuration: {prefix}.{key}")
    return value


def _build_optimizer(model: nn.Module, cfg: dict):
    name = str(_require(cfg, "optimizer", "training")).lower()
    lr = float(_require(cfg, "lr", "training"))
    weight_decay = float(_require(cfg, "weight_decay", "training"))
    if name == "sgd":
        momentum = float(_require(cfg, "momentum", "training"))
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


class DecentralizedSimulator:
    """Synchronous configuration-driven decentralized FL simulator.

    Each communication round performs local training for every client, constructs
    the messages that are visible to neighbors (including configured Byzantine
    transformations), aggregates synchronously at every receiver, updates all
    client models, evaluates them, and writes reproducibility/resource metrics.
    """

    def __init__(self, config: dict):
        self.config = config

    def _run_dir(self) -> Path:
        run_dir = self.config["experiment"].get("run_dir")
        if run_dir is None:
            raise ValueError("experiment.run_dir must be resolved by main.py")
        path = Path(run_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _make_clients(self, train_dataset, partitions, device: torch.device):
        cfg = self.config
        training_cfg = cfg["training"]
        batch_size = int(_require(training_cfg, "batch_size", "training"))
        seed = int(_require(cfg["experiment"], "seed", "experiment"))

        template = build_model(cfg["model"], cfg["data"].get("dataset"))
        template = template.to(device)
        clients = []
        for client_id, indices in enumerate(partitions):
            model = copy.deepcopy(template)
            generator = torch.Generator().manual_seed(seed + 10_007 * client_id)
            loader = DataLoader(
                Subset(train_dataset, list(map(int, indices))),
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
                num_workers=int(training_cfg.get("num_workers", 0) or 0),
                pin_memory=device.type == "cuda",
            )
            clients.append(
                Client(
                    client_id=client_id,
                    model=model,
                    optimizer=_build_optimizer(model, training_cfg),
                    train_loader=loader,
                )
            )
        return clients

    def run(self) -> SimulationResult:
        cfg = self.config
        experiment_cfg = cfg["experiment"]
        training_cfg = cfg["training"]
        data_cfg = cfg["data"]
        seed = int(_require(experiment_cfg, "seed", "experiment"))
        rounds = int(_require(experiment_cfg, "rounds", "experiment"))
        if rounds < 1:
            raise ValueError("experiment.rounds must be >= 1")
        device = torch.device(_require(experiment_cfg, "resolved_device", "experiment"))
        run_dir = self._run_dir()

        train_dataset, val_dataset, test_dataset = load_dataset(data_cfg, seed)
        train_targets = extract_targets(train_dataset)
        partitions = build_partition(train_targets.cpu().numpy(), data_cfg, seed)
        if any(len(indices) == 0 for indices in partitions):
            raise ValueError("Client partition contains an empty client dataset.")
        save_partition(run_dir / "partition.json", partitions, train_targets.cpu().numpy())

        clients = self._make_clients(train_dataset, partitions, device)
        estimator = build_estimator(cfg["estimator"], seed=seed)
        aggregator = build_aggregator(cfg["aggregation"])
        attack = build_attack(cfg["attack"])

        eval_batch_size = int(_require(training_cfg, "eval_batch_size", "training"))
        val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
        local_epochs = int(_require(training_cfg, "local_epochs", "training"))

        attack_name = str(_require(cfg["attack"], "name", "attack")).lower()
        malicious_clients = {int(v) for v in cfg["attack"].get("malicious_clients", [])}
        attack_start = cfg["attack"].get("start_round")
        attack_target = cfg["attack"].get("target")
        if attack_name != "none":
            if attack_start is None:
                raise ValueError("attack.start_round must be configured when an attack is enabled")
            if attack_target not in {"update", "model"}:
                raise ValueError("attack.target must be either 'update' or 'model'")
            unknown = malicious_clients - set(range(len(clients)))
            if unknown:
                raise ValueError(f"Unknown malicious client ids: {sorted(unknown)}")

        start_time = time.perf_counter()
        previous_topology = None

        for round_id in range(rounds):
            round_start = time.perf_counter()
            topology = topology_for_round(cfg, round_id, seed)
            if topology != previous_topology:
                save_topology(run_dir / "topologies" / f"round_{round_id:04d}.json", topology)
                previous_topology = topology

            previous_vectors = {
                client.client_id: parameters_to_vector(client.model).detach().clone()
                for client in clients
            }
            train_stats = {}
            for client in clients:
                train_stats[client.client_id] = train_local(
                    client.model,
                    client.train_loader,
                    client.optimizer,
                    estimator,
                    device,
                    client_id=client.client_id,
                    round_id=round_id,
                    epochs=local_epochs,
                )

            local_vectors = {
                client.client_id: parameters_to_vector(client.model).detach().clone()
                for client in clients
            }
            messages = {}
            active_attack = attack_name != "none" and round_id >= int(attack_start)
            for client in clients:
                client_id = client.client_id
                vector = local_vectors[client_id]
                if active_attack and client_id in malicious_clients:
                    if attack_target == "update":
                        update = vector - previous_vectors[client_id]
                        attacked_update = attack(
                            update,
                            client_id=client_id,
                            round_id=round_id,
                            context={"previous_model": previous_vectors[client_id]},
                        )
                        vector = previous_vectors[client_id] + attacked_update
                    else:
                        vector = attack(
                            vector,
                            client_id=client_id,
                            round_id=round_id,
                            context={"previous_model": previous_vectors[client_id]},
                        )
                messages[client_id] = vector.detach().clone()

            aggregate_start = time.perf_counter()
            aggregated_vectors = {}
            for receiver in range(len(clients)):
                sender_ids = list(topology[receiver])
                if not sender_ids:
                    raise ValueError(f"Receiver {receiver} has no visible messages")
                received = [messages[sender] for sender in sender_ids]
                aggregated_vectors[receiver] = aggregator(
                    received,
                    context={
                        "receiver_id": receiver,
                        "sender_ids": sender_ids,
                    },
                ).detach()
            aggregation_seconds = time.perf_counter() - aggregate_start

            for client in clients:
                with torch.no_grad():
                    vector_to_parameters(aggregated_vectors[client.client_id], client.model)

            comm = communication_cost(topology, messages)
            val_accuracies, test_accuracies = [], []
            for client in clients:
                val_metrics = evaluate(client.model, val_loader, device)
                test_metrics = evaluate(client.model, test_loader, device)
                val_accuracies.append(float(val_metrics["accuracy"]))
                test_accuracies.append(float(test_metrics["accuracy"]))
                stats = train_stats[client.client_id]
                append_metrics(
                    run_dir / "client_metrics.csv",
                    {
                        "round": round_id,
                        "client_id": client.client_id,
                        "train_loss": stats.mean_loss,
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"],
                        "test_loss": test_metrics["loss"],
                        "test_accuracy": test_metrics["accuracy"],
                        "queries": stats.queries,
                        "forward_queries": stats.forward_queries,
                        "backward_calls": stats.backward_calls,
                    },
                )

            append_metrics(
                run_dir / "round_metrics.csv",
                {
                    "round": round_id,
                    "mean_val_accuracy": sum(val_accuracies) / len(val_accuracies),
                    "mean_test_accuracy": sum(test_accuracies) / len(test_accuracies),
                    "queries": sum(s.queries for s in train_stats.values()),
                    "forward_queries": sum(s.forward_queries for s in train_stats.values()),
                    "backward_calls": sum(s.backward_calls for s in train_stats.values()),
                    "message_count": comm["message_count"],
                    "transmitted_bytes": comm["transmitted_bytes"],
                    "aggregation_seconds": aggregation_seconds,
                    "round_seconds": time.perf_counter() - round_start,
                },
            )

        return SimulationResult(
            rounds_completed=rounds,
            elapsed_seconds=time.perf_counter() - start_time,
        )
