import json

import pytest

from ml_training.log_run import (
    assert_package_matches_report,
    build_run_record,
    params_for_run,
    require_tracking_env,
)


def _config() -> dict:
    return {
        "seed": 42,
        "model": {
            "type": "logistic_regression",
            "params": {"C": 1.0, "max_iter": 1000},
        },
        "split": {"test_size": 0.2, "random_state": 42, "stratify": True},
        "metrics": {
            "primary": "roc_auc",
            "primary_direction": "maximize",
            "report_path": "metrics/train_metrics.json",
        },
        "task": {"name": "transaction_fraud"},
        "paths": {"package_dir": "models/packaged/fraud_classifier"},
    }


def _report() -> dict:
    return {
        "task": "transaction_fraud",
        "primary_metric": "roc_auc",
        "metrics": {"roc_auc": 0.8480595183310323},
    }


def _metadata() -> dict:
    return {
        "task": "transaction_fraud",
        "primary_metric": "roc_auc",
        "primary_metric_value": 0.8480595183310323,
    }


def test_require_tracking_env_rejects_missing_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        require_tracking_env()


def test_require_tracking_env_rejects_local_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "./mlruns")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "admin")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "secret")
    with pytest.raises(RuntimeError, match="http"):
        require_tracking_env()


def test_require_tracking_env_rejects_missing_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.example")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "admin")
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="PASSWORD"):
        require_tracking_env()


def test_params_for_run_records_training_choices() -> None:
    params = params_for_run(_config())
    assert params["seed"] == "42"
    assert params["C"] == "1.0"
    assert params["max_iter"] == "1000"
    assert params["stratify"] == "true"
    assert params["primary_metric"] == "roc_auc"
    assert "password" not in params


def test_package_must_match_report() -> None:
    assert_package_matches_report(_metadata(), _report())
    drifted = _metadata()
    drifted["primary_metric_value"] = 0.1
    with pytest.raises(ValueError, match="does not match"):
        assert_package_matches_report(drifted, _report())


def test_run_record_omits_credentials() -> None:
    record = build_run_record(
        tracking_uri="http://mlflow.example",
        experiment_name="transaction_fraud",
        experiment_id="1",
        run_id="abc",
        model_version="fraud-logreg-s42-5f3c0fdb-auc0p848060",
        primary_metric="roc_auc",
        primary_metric_value=0.8480595183310323,
    )
    blob = json.dumps(record)
    assert "password" not in blob
    assert record["artifact_subdir"] == "fraud_classifier"
    assert record["run_id"] == "abc"
