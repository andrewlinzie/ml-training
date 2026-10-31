"""Prepare stage: split raw data into train/validation CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_training.config import load_config


def prepare_from_config(config: dict[str, Any]) -> tuple[Path, Path]:
    raw_path = Path(config["data"]["raw_path"])
    train_path = Path(config["data"]["train_path"])
    val_path = Path(config["data"]["val_path"])
    features = list(config["features"])
    target = config["task"]["target"]
    split_cfg = config["split"]

    df = pd.read_csv(raw_path)
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {raw_path}: {missing}")

    stratify = df[target] if split_cfg.get("stratify", True) else None
    train_df, val_df = train_test_split(
        df,
        test_size=float(split_cfg["test_size"]),
        random_state=int(split_cfg["random_state"]),
        stratify=stratify,
    )

    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Wrote train ({len(train_df)} rows) -> {train_path}")
    print(f"Wrote val   ({len(val_df)} rows) -> {val_path}")
    return train_path, val_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    prepare_from_config(load_config(args.config))


if __name__ == "__main__":
    main()