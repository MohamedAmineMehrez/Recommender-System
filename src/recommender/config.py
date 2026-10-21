"""Configuration loading helpers.

All paths in YAML configs are resolved relative to the repository root so that
experiments behave the same regardless of the current working directory.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(item) for item in value]
    if isinstance(value, str) and (
        value.startswith("data/")
        or value.startswith("artifacts/")
        or value.startswith("results/")
        or value.startswith("reports/")
        or value.startswith("configs/")
    ):
        return str(PROJECT_ROOT / value)
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config = _resolve_paths(config)
    config["_config_path"] = str(path)
    return config


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def load_experiment_config(path: str | Path, base_path: str | Path | None = None) -> dict[str, Any]:
    if base_path is None:
        base_path = PROJECT_ROOT / "configs" / "base.yaml"
    base = load_config(base_path)
    experiment = load_config(path)
    return merge_configs(base, experiment)
