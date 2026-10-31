from pathlib import Path

import pandas as pd

from ml_training.evaluate import evaluate_from_config
from ml_training.package_model import load_package, package_from_config, predict_one
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


def test_package_load_and_predict(tmp_path: Path) -> None:
    raw = tmp_path / "tiny.csv"
    _write_tiny_dataset(raw)

    config = {
        "task": {
            "name": "transaction_fraud",
            "target": "is_fraud",
            "problem_type": "binary_classification",
        },
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
        "metrics": {
            "primary": "roc_auc",
            "primary_direction": "maximize",
            "report_path": str(tmp_path / "metrics.json"),
        },
        "paths": {
            "model_path": str(tmp_path / "model.joblib"),
            "package_dir": str(tmp_path / "packaged" / "fraud_classifier"),
        },
    }

    prepare_from_config(config)
    train_from_config(config)
    evaluate_from_config(config)
    metadata = package_from_config(config)

    assert metadata["model_version"] != "stub-v1"
    assert "fraud-logreg" in metadata["model_version"]

    package = load_package(config["paths"]["package_dir"])
    result = predict_one(
        package,
        {
            "amount": 100.0,
            "hour_of_day": 3,
            "day_of_week": 1,
            "merchant_risk": 0.9,
            "device_trust": 0.1,
            "is_international": 1,
        },
    )

    assert result["model_version"] == metadata["model_version"]
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability_fraud"] <= 1.0