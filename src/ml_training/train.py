"""Config-driven training with seed discipline."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ml_training.config import load_config


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_model(model_cfg: dict[str, Any], seed: int) -> LogisticRegression:
    model_type = model_cfg.get("type", "logistic_regression")
    params = dict(model_cfg.get("params", {}))

    if model_type != "logistic_regression":
        raise ValueError(f"Unsupported model.type for A4: {model_type}")

    return LogisticRegression(random_state=seed, **params)


def train_from_config(config: dict[str, Any]) -> dict[str, float]:
    seed = int(config["seed"])
    set_seeds(seed)

    raw_path = Path(config["data"]["raw_path"])
    features = list(config["features"])
    target = config["task"]["target"]
    split_cfg = config["split"]
    model_path = Path(config["paths"]["model_path"])

    df = pd.read_csv(raw_path)
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {raw_path}: {missing}")

    x = df[features]
    y = df[target]

    stratify = y if split_cfg.get("stratify", True) else None
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=float(split_cfg["test_size"]),
        random_state=int(split_cfg["random_state"]),
        stratify=stratify,
    )

    model = build_model(config["model"], seed=seed)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_val)
    y_prob = model.predict_proba(x_val)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_val, y_prob)),
        "f1": float(f1_score(y_val, y_pred)),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": features,
            "target": target,
            "seed": seed,
            "metrics": metrics,
        },
        model_path,
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to training config YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    metrics = train_from_config(config)

    print(f"Wrote model to {config['paths']['model_path']}")
    print(f"Validation metrics: {metrics}")


if __name__ == "__main__":
    main()