"""Reinforcement learning entrypoint."""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL agent for tool use")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError("Implement PPO-style RL fine-tuning")


if __name__ == "__main__":
    main()
