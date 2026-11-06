"""Frozen reference profile for population-stability checks.

Edges come from the training rows. A later job reads this file and does not
recompute the edges from live traffic.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml_training.config import load_config

PSI_EPSILON = 0.001
DEFAULT_BIN_COUNT = 10
REFERENCE_BUCKET = "ai-cloud-platform-899631475351-dvc"


def population_stability_index(
    expected_pct: list[float],
    actual_pct: list[float],
    eps: float = PSI_EPSILON,
) -> float:
    """PSI after adding eps to every share and renormalizing."""
    if len(expected_pct) != len(actual_pct):
        raise ValueError("expected_pct and actual_pct must have the same length")
    expected = [float(value) + eps for value in expected_pct]
    actual = [float(value) + eps for value in actual_pct]
    expected_sum = sum(expected)
    actual_sum = sum(actual)
    total = 0.0
    for exp, act in zip(expected, actual, strict=True):
        share_expected = exp / expected_sum
        share_actual = act / actual_sum
        total += (share_actual - share_expected) * math.log(share_actual / share_expected)
    return total


def collapsed_quantile_edges(values: list[float] | np.ndarray, bin_count: int = DEFAULT_BIN_COUNT) -> list[float]:
    """Quantile edges with duplicate cuts removed. A constant column keeps one bin."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("cannot build edges from an empty column")
    raw = np.quantile(array, np.linspace(0.0, 1.0, bin_count + 1))
    edges = [float(raw[0])]
    for value in raw[1:]:
        number = float(value)
        if number > edges[-1]:
            edges.append(number)
    if len(edges) == 1:
        edges.append(edges[0])
    return edges


def assign_bin(value: float, edges: list[float]) -> int:
    """Bin i is [edges[i], edges[i+1]). The last bin includes its right edge."""
    if len(edges) < 2:
        raise ValueError("edges need at least two cut points")
    last = len(edges) - 2
    number = float(value)
    if number >= edges[-1]:
        return last
    for index in range(last):
        if edges[index] <= number < edges[index + 1]:
            return index
    if edges[-2] <= number <= edges[-1]:
        return last
    return 0


def shares_for_edges(values: list[float] | np.ndarray, edges: list[float]) -> list[float]:
    counts = [0 for _ in range(len(edges) - 1)]
    for value in values:
        counts[assign_bin(float(value), edges)] += 1
    total = sum(counts)
    if total == 0:
        raise ValueError("cannot build shares from an empty column")
    return [count / total for count in counts]


def reference_object_key(model_version: str) -> str:
    return f"monitoring/reference/{model_version}/profile.json"


def reference_object_uri(model_version: str) -> str:
    return f"s3://{REFERENCE_BUCKET}/{reference_object_key(model_version)}"


def variable_profile(values: list[float] | np.ndarray, bin_count: int = DEFAULT_BIN_COUNT) -> dict[str, Any]:
    edges = collapsed_quantile_edges(values, bin_count)
    return {
        "edges": edges,
        "expected_pct": shares_for_edges(values, edges),
    }


def build_reference_profile(
    frame: pd.DataFrame,
    feature_names: list[str],
    probabilities: list[float] | np.ndarray,
    model_version: str,
    source: str,
    bin_count: int = DEFAULT_BIN_COUNT,
) -> dict[str, Any]:
    if len(probabilities) != len(frame):
        raise ValueError("probabilities must have one value per training row")
    return {
        "model_version": model_version,
        "source": source,
        "n_rows": len(frame),
        "bin_count": bin_count,
        "features": {
            name: variable_profile(frame[name].to_numpy(), bin_count) for name in feature_names
        },
        "probability_fraud": variable_profile(probabilities, bin_count),
    }


def write_reference_profile(profile: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return out


def build_profile_from_config(config: dict[str, Any]) -> dict[str, Any]:
    train_path = Path(config["data"]["train_path"])
    package_dir = Path(config["paths"]["package_dir"])
    frame = pd.read_csv(train_path)
    feature_names = list(config["features"])
    metadata = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(package_dir / "model.joblib")
    if list(metadata["feature_names"]) != feature_names:
        raise ValueError("config features and package metadata feature lists disagree")
    probabilities = model.predict_proba(frame[feature_names])[:, 1]
    profile = build_reference_profile(
        frame,
        feature_names,
        probabilities,
        model_version=str(metadata["model_version"]),
        source=str(train_path),
    )
    out = write_reference_profile(profile, "metrics/reference_profile.json")
    print(f"Wrote {out}")
    print(reference_object_uri(profile["model_version"]))
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    build_profile_from_config(load_config(args.config))


if __name__ == "__main__":
    main()
