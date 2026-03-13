"""Reinforcement learning entrypoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
from datasets import load_dataset

from rewards import RewardConfig, compute_reward

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def get_cfg(cfg: Dict[str, Any], *keys: str, default: Any) -> Any:
    cursor: Any = cfg
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_reward_config(cfg: Dict[str, Any]) -> RewardConfig:
    reward_cfg = get_cfg(cfg, "reward", default={})
    return RewardConfig(
        tool_selection=float(reward_cfg.get("tool_selection", 1.0)),
        schema_validity=float(reward_cfg.get("schema_validity", 1.0)),
        argument_accuracy=float(reward_cfg.get("argument_accuracy", 1.0)),
        no_call_penalty=float(reward_cfg.get("no_call_penalty", -0.2)),
    )


def parse_reference(target: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(target)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def stub_policy_prediction(reference: Dict[str, Any]) -> Dict[str, Any]:
    # Scaffold behavior: echo reference so reward path is verifiable.
    return {
        "name": reference.get("name"),
        "arguments": reference.get("arguments", {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL agent for tool use")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--max-train-samples", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true", help="Skip optimizer updates")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    train_path = resolve_path(get_cfg(cfg, "data", "processed_path", default="data/processed/train.jsonl"))
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    reward_config = build_reward_config(cfg)
    run_name = get_cfg(cfg, "run_name", default="rl")
    rl_steps = int(get_cfg(cfg, "train", "rl_steps", default=500))

    ds = load_dataset("json", data_files={"train": str(train_path)})["train"]
    if args.max_train_samples and args.max_train_samples < len(ds):
        ds = ds.select(range(args.max_train_samples))

    rewards: List[float] = []
    for row in ds:
        reference = parse_reference(row.get("target", "{}"))
        prediction = stub_policy_prediction(reference)
        reward = compute_reward(prediction, reference, reward_config)
        rewards.append(reward)

    if not rewards:
        raise RuntimeError("No usable RL training examples found")

    avg_reward = sum(rewards) / len(rewards)
    min_reward = min(rewards)
    max_reward = max(rewards)

    print(f"RL scaffold initialized for run: {run_name}")
    print(f"Examples processed: {len(rewards)}")
    print(f"Configured RL steps: {rl_steps}")
    print(f"Dry run: {args.dry_run}")
    print(f"Reward stats -> mean: {avg_reward:.4f}, min: {min_reward:.4f}, max: {max_reward:.4f}")

    if args.dry_run:
        print("Dry run complete. No policy updates were applied.")
        return

    print("TODO: integrate PPO rollout collection and policy/value optimization loop.")


if __name__ == "__main__":
    main()
