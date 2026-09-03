from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run one experiment configuration for multiple seeds.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    for seed in args.seeds:
        command = [
            sys.executable,
            "main.py",
            "--config",
            args.config,
            "--set",
            f"experiment.seed={seed}",
        ]
        for override in args.set:
            command.extend(["--set", override])
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
