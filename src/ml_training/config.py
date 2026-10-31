"""Load training configuration from YAML and merge DVC params."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Config must be a mapping: {config_path}")

    return data


def load_config(
    path: str | Path,
    params_path: str | Path = "params.yaml",
) -> dict[str, Any]:
    config = load_yaml(path)
    params_file = Path(params_path)
    if params_file.is_file():
        params = load_yaml(params_file)
        config = merge_params(config, params)
    return config


def merge_params(config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Overlay DVC params.yaml onto the static config."""
    merged = dict(config)

    prepare = params.get("prepare", {})
    if prepare:
        merged["split"] = {
            "test_size": prepare["test_size"],
            "random_state": prepare["random_state"],
            "stratify": prepare.get("stratify", True),
        }

    train = params.get("train", {})
    if train:
        merged["seed"] = train["seed"]
        merged["model"] = train["model"]

    evaluate = params.get("evaluate", {})
    metrics = dict(merged.get("metrics", {}))
    if evaluate:
        metrics["primary"] = evaluate["primary"]
        metrics["primary_direction"] = evaluate["primary_direction"]
    merged["metrics"] = metrics

    return merged