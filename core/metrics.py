from __future__ import annotations

import csv
import json
from pathlib import Path

import torch


def append_metrics(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
        if header != list(row.keys()):
            raise ValueError(
                f"Metric schema mismatch for {path}: existing={header}, new={list(row.keys())}"
            )
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def communication_cost(topology: dict[int, list[int]], messages: dict[int, torch.Tensor]) -> dict:
    """Count directed transmitted messages and payload bytes for one round."""
    message_count = 0
    transmitted_bytes = 0
    for receiver, senders in topology.items():
        for sender in senders:
            if sender == receiver:
                continue
            message_count += 1
            transmitted_bytes += tensor_nbytes(messages[sender])
    return {
        "message_count": int(message_count),
        "transmitted_bytes": int(transmitted_bytes),
    }


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
