"""Register the logged run and point the champion alias at it.

This is a promotion. It is not a DVC stage. `dvc repro` must not move the alias.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

from ml_training.config import load_config
from ml_training.log_run import require_tracking_env, run_record_path
from ml_training.package_model import load_package

CHAMPION_ALIAS = "champion"
MODEL_ARTIFACT_NAME = "model"


def champion_uri(registered_name: str) -> str:
    return f"models:/{registered_name}@{CHAMPION_ALIAS}"


def earns_champion(current_roc_auc: float, challenger_roc_auc: float) -> bool:
    """True when the challenger strictly beats the champion on roc_auc.

    This is the metric gate only. A person still has to run a promotion.
    An equal score does not earn the alias.
    """
    return float(challenger_roc_auc) > float(current_roc_auc)


def registry_record_path(config: dict[str, Any]) -> Path:
    return run_record_path(config).parent / "mlflow_registry.json"


def build_registry_record(
    *,
    registered_name: str,
    registry_version: str,
    run_id: str,
    model_version: str,
    primary_metric: str,
    primary_metric_value: float,
) -> dict[str, Any]:
    return {
        "alias": CHAMPION_ALIAS,
        "model_uri": champion_uri(registered_name),
        "model_version": model_version,
        "primary_metric": primary_metric,
        "primary_metric_value": primary_metric_value,
        "registered_name": registered_name,
        "registry_version": registry_version,
        "run_id": run_id,
    }


def write_registry_record(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def existing_alias_version(client: Any, registered_name: str) -> str | None:
    from mlflow.exceptions import MlflowException

    try:
        current = client.get_model_version_by_alias(registered_name, CHAMPION_ALIAS)
    except MlflowException as exc:
        code = getattr(exc, "error_code", "")
        message = str(exc).lower()
        model_missing = code == "RESOURCE_DOES_NOT_EXIST"
        alias_missing = (
            code == "INVALID_PARAMETER_VALUE" and "alias" in message and "not found" in message
        )
        if model_missing or alias_missing:
            return None
        raise
    return str(current.version)


def register_from_config(config: dict[str, Any]) -> dict[str, Any]:
    tracking_uri = require_tracking_env()
    run_path = run_record_path(config)
    if not run_path.is_file():
        raise FileNotFoundError(f"Run record not found: {run_path}. Run the log stage first.")

    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    package = load_package(config["paths"]["package_dir"])
    metadata = package["metadata"]
    if metadata["model_version"] != run_record["model_version"]:
        raise ValueError(
            "Package model_version does not match metrics/mlflow_run.json. "
            "Log this package before registering it."
        )

    registered_name = str(config["task"]["name"])
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    current = existing_alias_version(client, registered_name)
    if current is not None:
        raise RuntimeError(
            f"{champion_uri(registered_name)} already points at registry version {current}. "
            "This command does not move it. See CHAMPION.md."
        )

    model = package["model"]
    sample = pd.DataFrame([{name: 0.0 for name in package["feature_names"]}])
    signature = infer_signature(sample, model.predict(sample))
    with mlflow.start_run(run_id=str(run_record["run_id"])):
        info = mlflow.sklearn.log_model(
            model,
            name=MODEL_ARTIFACT_NAME,
            registered_model_name=registered_name,
            signature=signature,
            pip_requirements=[f"scikit-learn=={version('scikit-learn')}"],
        )

    registry_version = str(info.registered_model_version)
    client.set_model_version_tag(
        registered_name,
        registry_version,
        "model_version",
        str(metadata["model_version"]),
    )
    client.set_model_version_tag(
        registered_name,
        registry_version,
        "primary_metric",
        f"{metadata['primary_metric']}={metadata['primary_metric_value']}",
    )
    client.set_registered_model_alias(registered_name, CHAMPION_ALIAS, registry_version)

    record = build_registry_record(
        registered_name=registered_name,
        registry_version=registry_version,
        run_id=str(run_record["run_id"]),
        model_version=str(metadata["model_version"]),
        primary_metric=str(metadata["primary_metric"]),
        primary_metric_value=float(metadata["primary_metric_value"]),
    )
    write_registry_record(record, registry_record_path(config))
    print(f"model_uri={record['model_uri']}")
    print(f"registry_version={record['registry_version']}")
    print(f"run_id={record['run_id']}")
    print(f"model_version={record['model_version']}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    register_from_config(load_config(args.config))


if __name__ == "__main__":
    main()
