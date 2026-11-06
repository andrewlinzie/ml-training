import json
from pathlib import Path

import pandas as pd

from ml_training.reference_profile import (
    assign_bin,
    build_reference_profile,
    collapsed_quantile_edges,
    population_stability_index,
    reference_object_key,
)


def test_psi_matches_the_worked_shift() -> None:
    score = population_stability_index([0.5, 0.5], [0.1, 0.9])
    assert abs(score - 0.8736066829409495) < 1e-12


def test_psi_is_zero_when_shares_match() -> None:
    assert population_stability_index([0.5, 0.5], [0.5, 0.5]) == 0.0


def test_binary_column_keeps_a_usable_edge_list() -> None:
    values = [0, 1] * 20
    edges = collapsed_quantile_edges(values, bin_count=10)
    assert edges == sorted(edges)
    assert len(edges) >= 2
    assert len(edges) < 11
    profile = build_reference_profile(
        pd.DataFrame({"is_international": values, "amount": [float(i) for i in range(len(values))]}),
        ["is_international", "amount"],
        [0.2, 0.8] * 20,
        model_version="test-profile",
        source="fixture",
        bin_count=10,
    )
    international = profile["features"]["is_international"]
    assert international["edges"] == sorted(international["edges"])
    assert len(international["expected_pct"]) == len(international["edges"]) - 1
    assert abs(sum(international["expected_pct"]) - 1.0) < 1e-6
    assert abs(sum(profile["probability_fraud"]["expected_pct"]) - 1.0) < 1e-6
    assert assign_bin(1.0, [0.0, 0.5, 1.0]) == 1


def test_writer_roundtrip_keeps_the_contract(tmp_path: Path) -> None:
    from ml_training.reference_profile import write_reference_profile

    profile = build_reference_profile(
        pd.DataFrame({"amount": [1.0, 2.0, 3.0, 4.0]}),
        ["amount"],
        [0.1, 0.2, 0.3, 0.4],
        model_version="pkg",
        source="data/processed/train.csv",
    )
    path = write_reference_profile(profile, tmp_path / "reference_profile.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["model_version"] == "pkg"
    assert loaded["n_rows"] == 4
    assert reference_object_key("pkg") == "monitoring/reference/pkg/profile.json"
    assert abs(sum(loaded["features"]["amount"]["expected_pct"]) - 1.0) < 1e-6
