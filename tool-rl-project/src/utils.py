"""Shared helpers (logging, configs, seeding)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    """Load an experiment config from YAML."""

    raise NotImplementedError("Implement config loading + validation")


def setup_logging(run_dir: Path) -> None:
    """Configure logging destinations for training loops."""

    raise NotImplementedError("Implement logging setup (tensorboard, wandb, etc.)")
