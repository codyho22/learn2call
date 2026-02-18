"""Reward shaping utilities for tool-use RL."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class RewardConfig:
    """Configurable weights for composite reward."""

    tool_selection: float = 1.0
    schema_validity: float = 1.0
    argument_accuracy: float = 1.0
    no_call_penalty: float = -0.2


def compute_reward(prediction: Dict, reference: Dict, config: RewardConfig) -> float:
    """Return a scalar reward for a generated tool call."""

    raise NotImplementedError("Implement structured reward computation")
