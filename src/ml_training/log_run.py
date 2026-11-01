"""Log one MLflow run for the current packaged model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ml_training.config import load_config
from ml_training.package_model import REQUIRED_PACKAGE_FILES

TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
TRACKING_USERNAME_ENV = "MLFLOW_TRACKING_USERNAME"
TRACKING_PASSWORD_ENV = "MLFLOW_TRACKING_PASSWORD"
ARTIFACT_SUBDIR = "fraud_classifier"


def require_tracking_env() -> str:
    uri = os.environ.get(TRACKING_URI_ENV, "").strip()
    if not uri.startswith(("http://", "https://")):
        raise RuntimeError(
            "MLFLOW_TRACKING_URI must be the http(s) tracking server. "
            "A missing value would log to a local mlruns directory."
        )
    username = os.environ.get(TRACKING_USERNAME_ENV, "").strip()
    password = os.environ.get(TRACKING_PASSWORD_ENV, "")
    if not username or not password:
        raise RuntimeError(
            "MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD must be set. "
            "Do not put them in the repo."
        )
    return uri


def params_for_run(config: dict[str, Any]) -> dict[str, str]:
    model = config["model"]
    model_params = model.get("params", {})
    split = config.get("split", {})
    metrics_cfg = config["metrics"]
    return {
        "seed": str(config["seed"]),
        "model_type": str(model["type"]),
        "C": str(model_params["C"]),
        "max_iter": str(model_params["max_iter"]),
        "test_size": str(split["test_size"]),
        "split_random_state": str(split["random_state"]),
        "stratify": str(bool(split.get("stratify", True))).lower(),
        "primary_metric": str(metrics_cfg["primary"]),
        "primary_direction": str(metrics_cfg["primary_direction"]),
    }


def assert_package_matches_report(metadata: dict[str, Any], report: dict[str, Any]) -> None:
    primary = str(report["primary_metric"])
    report_value = float(report["metrics"][primary])
    package_value = float(metadata["primary_metric_value"])
    if metadata.get("primary_metric") != primary:
        raise ValueError(
            f"Package primary metric is {metadata.get('primary_metric')}, report is {primary}."
        )
    if abs(package_value - report_value) > 1e-9:
        raise ValueError(
            "Package primary metric does not match metrics/train_metrics.json. "
            "Run python -m ml_training.package_model before logging."
        )
    if metadata.get("task") != report.get("task"):
        raise ValueError("Package task does not match the metrics report.")


def load_package_metadata(package_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_PACKAGE_FILES if not (package_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Package incomplete at {package_dir}: missing {missing}. "
            "Run python -m ml_training.package_model first."
        )
    return json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))


def run_record_path(config: dict[str, Any]) -> Path:
    return Path(config["metrics"]["report_path"]).parent / "mlflow_run.json"


def build_run_record(
    *,
    tracking_uri: str,
    experiment_name: str,
    experiment_id: str,
    run_id: str,
    model_version: str,
    primary_metric: str,
    primary_metric_value: float,
) -> dict[str, Any]:
    return {
        "artifact_subdir": ARTIFACT_SUBDIR,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "model_version": model_version,
        "primary_metric": primary_metric,
        "primary_metric_value": primary_metric_value,
        "run_id": run_id,
        "tracking_uri": tracking_uri,
    }


def write_run_record(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def log_from_config(config: dict[str, Any]) -> dict[str, Any]:
    tracking_uri = require_tracking_env()
    report_path = Path(config["metrics"]["report_path"])
    if not report_path.is_file():
        raise FileNotFoundError(f"Metrics report not found: {report_path}. Run evaluate first.")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    package_dir = Path(config["paths"]["package_dir"])
    metadata = load_package_metadata(package_dir)
    assert_package_matches_report(metadata, report)

    primary = str(report["primary_metric"])
    metric_values = {
        key: float(report["metrics"][key]) for key in ("precision", "recall", "f1", "roc_auc")
    }
    tags = {
        "model_version": str(metadata["model_version"]),
        "config_sha256": str(metadata["config_sha256"]),
        "task": str(metadata["task"]),
    }

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(str(config["task"]["name"]))
    with mlflow.start_run(run_name=str(metadata["model_version"])) as active:
        mlflow.log_params(params_for_run(config))
        mlflow.log_metrics(metric_values)
        mlflow.set_tags(tags)
        mlflow.log_artifacts(str(package_dir), artifact_path=ARTIFACT_SUBDIR)
        record = build_run_record(
            tracking_uri=tracking_uri,
            experiment_name=str(config["task"]["name"]),
            experiment_id=str(experiment.experiment_id),
            run_id=str(active.info.run_id),
            model_version=str(metadata["model_version"]),
            primary_metric=primary,
            primary_metric_value=float(report["metrics"][primary]),
        )

    write_run_record(record, run_record_path(config))
    print(f"experiment={record['experiment_name']}")
    print(f"run_id={record['run_id']}")
    print(f"model_version={record['model_version']}")
    print(f"primary={record['primary_metric']}={record['primary_metric_value']}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    log_from_config(load_config(args.config))


if __name__ == "__main__":
    main()
