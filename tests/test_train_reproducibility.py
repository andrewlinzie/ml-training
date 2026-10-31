from pathlib import Path

import pandas as pd
import yaml

from ml_training.train import train_from_config


def _write_tiny_dataset(path: Path) -> None:
    rows = []
    for i in range(40):
        rows.append(
            {
                "amount": 10.0 + i,
                "hour_of_day": i % 24,
                "day_of_week": i % 7,
                "merchant_risk": (i % 10) / 10.0,
                "device_trust": ((9 - (i % 10)) / 10.0),
                "is_international": i % 2,
                "is_fraud": 1 if i % 3 == 0 else 0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_same_seed_same_metrics(tmp_path: Path) -> None:
    data_path = tmp_path / "tiny.csv"
    model_a = tmp_path / "model_a.joblib"
    model_b = tmp_path / "model_b.joblib"
    _write_tiny_dataset(data_path)

    base = {
        "task": {"target": "is_fraud"},
        "data": {"raw_path": str(data_path)},
        "features": [
            "amount",
            "hour_of_day",
            "day_of_week",
            "merchant_risk",
            "device_trust",
            "is_international",
        ],
        "split": {"test_size": 0.25, "random_state": 42, "stratify": True},
        "seed": 42,
        "model": {
            "type": "logistic_regression",
            "params": {"max_iter": 1000, "C": 1.0},
        },
    }

    config_a = {**base, "paths": {"model_path": str(model_a)}}
    config_b = {**base, "paths": {"model_path": str(model_b)}}

    metrics_a = train_from_config(config_a)
    metrics_b = train_from_config(config_b)

    assert metrics_a == metrics_b
    assert model_a.is_file()
    assert model_b.is_file()


def test_load_config_roundtrip(tmp_path: Path) -> None:
    from ml_training.config import load_config

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({"seed": 7, "task": {"target": "is_fraud"}}))
    loaded = load_config(cfg_path)
    assert loaded["seed"] == 7