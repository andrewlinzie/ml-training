"""Train stage: fit model on prepared train split."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml_training.config import load_config


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_model(model_cfg: dict[str, Any], seed: int) -> LogisticRegression:
    model_type = model_cfg.get("type", "logistic_regression")
    params = dict(model_cfg.get("params", {}))

    if model_type != "logistic_regression":
        raise ValueError(f"Unsupported model.type: {model_type}")

    return LogisticRegression(random_state=seed, **params)


def train_from_config(config: dict[str, Any]) -> Path:
    seed = int(config["seed"])
    set_seeds(seed)

    train_path = Path(config["data"]["train_path"])
    features = list(config["features"])
    target = config["task"]["target"]
    model_path = Path(config["paths"]["model_path"])

    train_df = pd.read_csv(train_path)
    missing = [c for c in features + [target] if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing columns in {train_path}: {missing}")

    x_train = train_df[features]
    y_train = train_df[target]

    model = build_model(config["model"], seed=seed)
    model.fit(x_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": features,
            "target": target,
            "seed": seed,
        },
        model_path,
    )
    print(f"Wrote model to {model_path}")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train_from_config(load_config(args.config))


if __name__ == "__main__":
    main()