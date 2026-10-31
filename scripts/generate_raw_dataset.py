"""Generate a small synthetic transactions dataset for fraud classification."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

FEATURE_COLUMNS = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "merchant_risk",
    "device_trust",
    "is_international",
]

LABEL_COLUMN = "is_fraud"
ALL_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN]


def generate_row(rng: random.Random) -> dict[str, float | int]:
    amount = round(rng.uniform(1.0, 2500.0), 2)
    hour_of_day = rng.randint(0, 23)
    day_of_week = rng.randint(0, 6)
    merchant_risk = round(rng.uniform(0.0, 1.0), 3)
    device_trust = round(rng.uniform(0.0, 1.0), 3)
    is_international = rng.randint(0, 1)

    # Simple latent fraud probability (not a real fraud model — just labels for MLOps practice).
    fraud_logit = (
        0.002 * amount
        + 1.8 * merchant_risk
        - 1.5 * device_trust
        + 0.9 * is_international
        + (0.6 if hour_of_day in {0, 1, 2, 3} else 0.0)
        - 1.2
    )
    fraud_prob = 1.0 / (1.0 + pow(2.718281828, -fraud_logit))
    is_fraud = 1 if rng.random() < fraud_prob else 0

    return {
        "amount": amount,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "merchant_risk": merchant_risk,
        "device_trust": device_trust,
        "is_international": is_international,
        "is_fraud": is_fraud,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/transactions.csv"),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = [generate_row(rng) for _ in range(args.n_rows)]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    fraud_count = sum(int(r["is_fraud"]) for r in rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Fraud rate: {fraud_count / len(rows):.3f} ({fraud_count}/{len(rows)})")


if __name__ == "__main__":
    main()