import json
from pathlib import Path

from ml_training.evaluate import (
    REQUIRED_METRIC_KEYS,
    assert_report_schema,
    build_metrics_report,
    write_metrics_report,
)


def test_metrics_report_schema_and_roundtrip(tmp_path: Path) -> None:
    metrics = {
        "precision": 0.8,
        "recall": 0.7,
        "f1": 0.75,
        "roc_auc": 0.9,
    }
    report = build_metrics_report(
        metrics=metrics,
        primary="roc_auc",
        primary_direction="maximize",
        task_name="transaction_fraud",
        n_val=100,
    )
    assert_report_schema(report)

    path = tmp_path / "train_metrics.json"
    write_metrics_report(report, path)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert_report_schema(loaded)
    for key in REQUIRED_METRIC_KEYS:
        assert key in loaded["metrics"]
    assert loaded["primary_metric"] == "roc_auc"
    assert loaded["primary_direction"] == "maximize"