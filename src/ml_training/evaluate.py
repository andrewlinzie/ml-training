"""Evaluation helpers and evaluate stage CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from ml_training.config import load_config

REQUIRED_METRIC_KEYS = ("precision", "recall", "f1", "roc_auc")


def compute_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_prob: Any,
) -> dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def build_metrics_report(
    *,
    metrics: dict[str, float],
    primary: str,
    primary_direction: str,
    task_name: str,
    n_val: int,
) -> dict[str, Any]:
    missing = [k for k in REQUIRED_METRIC_KEYS if k not in metrics]
    if missing:
        raise KeyError(f"Metrics missing required keys: {missing}")

    if primary not in metrics:
        raise KeyError(f"Primary metric '{primary}' not present in metrics")

    if primary_direction not in {"maximize", "minimize"}:
        raise ValueError("primary_direction must be 'maximize' or 'minimize'")

    return {
        "task": task_name,
        "split": "validation",
        "n_samples": int(n_val),
        "primary_metric": primary,
        "primary_direction": primary_direction,
        "metrics": {k: float(metrics[k]) for k in REQUIRED_METRIC_KEYS},
    }


def write_metrics_report(report: dict[str, Any], path: str | Path) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path


def format_metrics_summary(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    primary = report["primary_metric"]
    lines = [
        f"Validation n={report['n_samples']}",
        (
            f"Primary metric: {primary}="
            f"{metrics[primary]:.6f} ({report['primary_direction']})"
        ),
        (
            "All metrics: "
            f"precision={metrics['precision']:.6f} "
            f"recall={metrics['recall']:.6f} "
            f"f1={metrics['f1']:.6f} "
            f"roc_auc={metrics['roc_auc']:.6f}"
        ),
    ]
    return "\n".join(lines)


def assert_report_schema(report: dict[str, Any]) -> None:
    for key in (
        "task",
        "split",
        "n_samples",
        "primary_metric",
        "primary_direction",
        "metrics",
    ):
        if key not in report:
            raise KeyError(f"Report missing key: {key}")

    for key in REQUIRED_METRIC_KEYS:
        if key not in report["metrics"]:
            raise KeyError(f"Report metrics missing key: {key}")
        if not isinstance(report["metrics"][key], (float, int, np.floating)):
            raise TypeError(f"Metric '{key}' must be numeric")


def evaluate_from_config(config: dict[str, Any]) -> dict[str, Any]:
    model_path = Path(config["paths"]["model_path"])
    val_path = Path(config["data"]["val_path"])
    metrics_cfg = config["metrics"]

    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = list(bundle["features"])
    target = bundle["target"]

    val_df = pd.read_csv(val_path)
    x_val = val_df[features]
    y_val = val_df[target]

    y_pred = model.predict(x_val)
    y_prob = model.predict_proba(x_val)[:, 1]
    metrics = compute_classification_metrics(y_val, y_pred, y_prob)

    report = build_metrics_report(
        metrics=metrics,
        primary=str(metrics_cfg["primary"]),
        primary_direction=str(metrics_cfg["primary_direction"]),
        task_name=str(config["task"]["name"]),
        n_val=len(y_val),
    )
    report_path = write_metrics_report(report, metrics_cfg["report_path"])
    print(f"Wrote metrics to {report_path}")
    print(format_metrics_summary(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    evaluate_from_config(load_config(args.config))


if __name__ == "__main__":
    main()