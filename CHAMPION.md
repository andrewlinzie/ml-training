# Champion promotion

Inference may pin one URI:

`models:/transaction_fraud@champion`

That alias points at one registry version of the registered model `transaction_fraud`. Moving the alias changes the version underneath the URI. The URI string stays the same.

The current pointer is recorded in `metrics/mlflow_registry.json`. Registry version 1 is package `fraud-logreg-s42-5f3c0fdb-auc0p848060` with validation `roc_auc` `0.8480595183310323`.

## Rule

1. The comparison metric is `roc_auc`. Higher is better. Precision, recall, and f1 are recorded on the run and are not the promotion metric.
2. A challenger earns the alias only when its validation `roc_auc` is strictly greater than the champion's. An equal score does not move the alias. `earns_champion` in `src/ml_training/register_model.py` is that comparison.
3. A person runs the promotion after reading both scores. `dvc repro` does not move the alias. `python -m ml_training.register_model` creates the first alias and refuses once the alias exists.
4. There is no minimum improvement in this phase. Any strict increase qualifies. The person is the gate that stops a trivial increase from being promoted by accident.
5. The package id (`model_version`, for example `fraud-logreg-s42-5f3c0fdb-auc0p848060`) and the registry version (`1`, `2`, …) are different strings. The alias moves the registry version.

## What this rule does not do yet

No command moves an existing alias. The refusal is intentional until a promotion command exists that calls `earns_champion` and is run by a person. A training job must not gain that ability.
