import json

import pytest

from ml_training.register_model import (
    build_registry_record,
    champion_uri,
    existing_alias_version,
)


def test_champion_uri() -> None:
    assert champion_uri("transaction_fraud") == "models:/transaction_fraud@champion"


def test_registry_record_keeps_both_versions() -> None:
    record = build_registry_record(
        registered_name="transaction_fraud",
        registry_version="1",
        run_id="abc",
        model_version="fraud-logreg-s42-5f3c0fdb-auc0p848060",
        primary_metric="roc_auc",
        primary_metric_value=0.8480595183310323,
    )
    blob = json.dumps(record)
    assert "password" not in blob
    assert record["model_uri"] == "models:/transaction_fraud@champion"
    assert record["registry_version"] == "1"
    assert record["model_version"].startswith("fraud-logreg-")


class _Missing:
    def get_model_version_by_alias(self, name: str, alias: str) -> None:
        from mlflow.exceptions import MlflowException
        from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

        raise MlflowException(f"Registered model {name} not found.", RESOURCE_DOES_NOT_EXIST)


class _Present:
    def get_model_version_by_alias(self, name: str, alias: str) -> object:
        return type("Version", (), {"version": "3"})()


def test_missing_alias_is_empty() -> None:
    assert existing_alias_version(_Missing(), "transaction_fraud") is None


def test_alias_not_found_on_existing_model_is_empty() -> None:
    class _AliasMissing:
        def get_model_version_by_alias(self, name: str, alias: str) -> None:
            from mlflow.exceptions import MlflowException
            from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

            raise MlflowException(
                f"Registered model alias {alias} not found.",
                INVALID_PARAMETER_VALUE,
            )

    assert existing_alias_version(_AliasMissing(), "transaction_fraud") is None


def test_present_alias_returns_version() -> None:
    assert existing_alias_version(_Present(), "transaction_fraud") == "3"


def test_other_registry_errors_propagate() -> None:
    class _Denied:
        def get_model_version_by_alias(self, name: str, alias: str) -> None:
            from mlflow.exceptions import MlflowException
            from mlflow.protos.databricks_pb2 import INTERNAL_ERROR

            raise MlflowException("denied", INTERNAL_ERROR)

    with pytest.raises(Exception, match="denied"):
        existing_alias_version(_Denied(), "transaction_fraud")
