"""Reward shaping utilities for tool-use RL."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RewardConfig:
    """Configurable weights for composite reward."""

    tool_selection: float = 1.0
    schema_validity: float = 1.0
    argument_accuracy: float = 1.0
    no_call_penalty: float = -0.2


def compute_reward(prediction: Dict, reference: Dict, config: RewardConfig) -> float:
    """Return a scalar reward for a generated tool call."""

    if not prediction or not isinstance(prediction, dict):
        return config.no_call_penalty

    reward = 0.0

    pred_name = prediction.get("name") or prediction.get("tool_name")
    ref_name = reference.get("name") or reference.get("tool_name")
    if pred_name and ref_name and pred_name == ref_name:
        reward += config.tool_selection

    schema_score = _schema_validity_score(prediction)
    reward += config.schema_validity * schema_score

    pred_args = prediction.get("arguments", {})
    ref_args = reference.get("arguments", {})
    arg_score = _argument_match_score(pred_args, ref_args)
    reward += config.argument_accuracy * arg_score

    return reward


def _schema_validity_score(prediction: Dict[str, Any]) -> float:
    name = prediction.get("name") or prediction.get("tool_name")
    args = prediction.get("arguments")
    if not isinstance(name, str) or not name.strip():
        return 0.0
    if args is None:
        return 0.5
    if isinstance(args, dict):
        return 1.0
    return 0.0


def _argument_match_score(pred_args: Any, ref_args: Any) -> float:
    if not isinstance(ref_args, dict):
        return 1.0
    if not isinstance(pred_args, dict):
        return 0.0
    if not ref_args:
        return 1.0

    total = len(ref_args)
    matches = 0
    for key, ref_value in ref_args.items():
        if key in pred_args and _loosely_equal(pred_args[key], ref_value):
            matches += 1
    return matches / total


def _loosely_equal(lhs: Any, rhs: Any) -> bool:
    if lhs == rhs:
        return True
    return str(lhs).strip() == str(rhs).strip()
