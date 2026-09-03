from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from core.simulator import DecentralizedSimulator
from utils.config import apply_overrides, load_config, save_config
from utils.seed import resolve_device, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Decentralized federated learning experiments")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--set", action="append", default=[], help="Override config values, e.g. --set experiment.seed=43")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    seed = int(cfg["experiment"]["seed"])
    seed_everything(seed)
    device = resolve_device(str(cfg["experiment"].get("device", "auto")))
    cfg["experiment"]["resolved_device"] = str(device)

    run_name = f"{cfg['experiment'].get('name','run')}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cfg["experiment"].get("log_dir", "run_logs")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "resolved_config.yaml")

    simulator = DecentralizedSimulator(cfg)
    simulator.run()


if __name__ == "__main__":
    main()
