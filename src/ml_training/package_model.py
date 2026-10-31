"""Package a trained model into a versioned artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import yaml

from ml_training.config import load_config

REQUIRED_PACKAGE_FILES = (
    "model.joblib",
    "metadata.json",
    "feature_schema.json",
    "metrics.json",
    "config_snapshot.yaml",
)


def config_sha256(config: dict[str, Any]) -> str:
    payload = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_model_version(seed: int, config_hash: str, primary_metric: float) -> str:
    # Deterministic, human-readable version for Phase A (not semver release management).
    metric_tag = f"{primary_metric:.6f}".replace(".", "p")
    return f"fraud-logreg-s{seed}-{config_hash[:8]}-auc{metric_tag}"


def write_package(
    *,
    bundle: dict[str, Any],
    config: dict[str, Any],
    metrics_report: dict[str, Any],
    package_dir: str | Path,
) -> dict[str, Any]:
    out_dir = Path(package_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = list(bundle["features"])
    target = bundle["target"]
    seed = int(bundle["seed"])
    cfg_hash = config_sha256(config)
    primary = metrics_report["primary_metric"]
    primary_value = float(metrics_report["metrics"][primary])
    model_version = build_model_version(seed, cfg_hash, primary_value)

    # 1) model binary
    joblib.dump(bundle["model"], out_dir / "model.joblib")

    # 2) feature schema (order matters for inference)
    feature_schema = {
        "features": [
            {"name": name, "position": idx, "dtype": "number"}
            for idx, name in enumerate(features)
        ],
        "target": target,
    }
    (out_dir / "feature_schema.json").write_text(
        json.dumps(feature_schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # 3) metrics snapshot
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # 4) config snapshot
    (out_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True),
        encoding="utf-8",
    )

    # 5) metadata
    metadata = {
        "model_version": model_version,
        "model_type": config["model"]["type"],
        "task": config["task"]["name"],
        "framework": "scikit-learn",
        "seed": seed,
        "config_sha256": cfg_hash,
        "primary_metric": primary,
        "primary_metric_value": primary_value,
        "primary_direction": metrics_report["primary_direction"],
        "feature_names": features,
        "target": target,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return metadata


def load_package(package_dir: str | Path) -> dict[str, Any]:
    root = Path(package_dir)
    missing = [name for name in REQUIRED_PACKAGE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Package incomplete at {root}: missing {missing}")

    model = joblib.load(root / "model.joblib")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    feature_schema = json.loads((root / "feature_schema.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))

    feature_names = [f["name"] for f in feature_schema["features"]]
    if feature_names != metadata["feature_names"]:
        raise ValueError("feature_schema.json and metadata.json feature lists disagree")

    return {
        "model": model,
        "metadata": metadata,
        "feature_schema": feature_schema,
        "metrics": metrics,
        "feature_names": feature_names,
    }


def predict_one(package: dict[str, Any], features: dict[str, float]) -> dict[str, Any]:
    ordered = [float(features[name]) for name in package["feature_names"]]
    model = package["model"]
    pred = int(model.predict([ordered])[0])
    proba = float(model.predict_proba([ordered])[0][1])
    return {
        "model_version": package["metadata"]["model_version"],
        "prediction": pred,
        "probability_fraud": proba,
    }


def package_from_config(config: dict[str, Any]) -> dict[str, Any]:
    model_path = Path(config["paths"]["model_path"])
    metrics_path = Path(config["metrics"]["report_path"])
    package_dir = Path(config["paths"]["package_dir"])

    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}. Run train first.")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics report not found: {metrics_path}. Run train first.")

    bundle = joblib.load(model_path)
    metrics_report = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata = write_package(
        bundle=bundle,
        config=config,
        metrics_report=metrics_report,
        package_dir=package_dir,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    metadata = package_from_config(config)
    print(f"Wrote package to {config['paths']['package_dir']}")
    print(f"model_version={metadata['model_version']}")


if __name__ == "__main__":
    main()