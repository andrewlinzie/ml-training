from pathlib import Path

import pandas as pd

from ml_training.prepare import prepare_from_config
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


def _base_config(tmp_path: Path) -> dict:
    raw = tmp_path / "raw.csv"
    _write_tiny_dataset(raw)
    return {
        "task": {"name": "transaction_fraud", "target": "is_fraud"},
        "data": {
            "raw_path": str(raw),
            "train_path": str(tmp_path / "train.csv"),
            "val_path": str(tmp_path / "val.csv"),
        },
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
        "paths": {"model_path": str(tmp_path / "model.joblib")},
    }


def test_same_seed_same_model_file(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    prepare_from_config(config)

    path_a = tmp_path / "model_a.joblib"
    path_b = tmp_path / "model_b.joblib"

    config_a = {**config, "paths": {"model_path": str(path_a)}}
    config_b = {**config, "paths": {"model_path": str(path_b)}}

    train_from_config(config_a)
    train_from_config(config_b)

    assert path_a.is_file()
    assert path_b.is_file()
    assert path_a.read_bytes() == path_b.read_bytes()


def test_load_config_roundtrip(tmp_path: Path) -> None:
    from ml_training.config import load_config

    cfg_path = tmp_path / "cfg.yaml"
    params_path = tmp_path / "params.yaml"
    cfg_path.write_text(
        "task:\n  name: transaction_fraud\n  target: is_fraud\n"
        "metrics:\n  report_path: metrics/train_metrics.json\n"
    )
    params_path.write_text(
        "prepare:\n  test_size: 0.2\n  random_state: 42\n  stratify: true\n"
        "train:\n  seed: 7\n  model:\n    type: logistic_regression\n"
        "    params:\n      max_iter: 100\n      C: 1.0\n"
        "evaluate:\n  primary: roc_auc\n  primary_direction: maximize\n"
    )
    loaded = load_config(cfg_path, params_path=params_path)
    assert loaded["seed"] == 7
    assert loaded["split"]["test_size"] == 0.2
    assert loaded["metrics"]["primary"] == "roc_auc"