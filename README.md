# ml-training

## Purpose

`ml-training` is the training project for the AI Cloud Platform MLOps loop.

It is responsible for:

- training and evaluating models
- writing metrics
- packaging versioned model artifacts
- versioning data and model outputs with DVC (Phase A+)
- local and CI training jobs

It is not responsible for:

- deploying `ai-inference-service`
- ArgoCD / GitOps desired state
- the public `api-service`
- Terraform infrastructure

## Owns / Does not own

| Component | Owns | Does not own |
|---|---|---|
| `ml-training` | train/eval code, DVC pointers, metrics, packaged artifacts, local/CI train jobs | ArgoCD apps, public API, EKS deploy of inference |
| `ai-inference-service` | load artifact + predict + `/health` `/metrics` | training data, hyperparameter search, DVC remotes |
| `terraform-platform` | S3 bucket + IAM for DVC (and later MLflow store) | training code |
| `gitops-infra` | later: MLflow, Workflows, CronJobs (Phases B/E) | how the model is trained |

## Artifact boundary

Training produces a versioned artifact; serving only consumes a pinned version (Git/DVC/MLflow pointer)—never an untracked mutable `latest`.

## Learning task

**Question:** Is this transaction fraudulent?

| Field | Value |
|---|---|
| Task | Transaction fraud detection |
| Problem type | Binary classification |
| Target | `is_fraud` (0 = legit, 1 = fraud) |
| Features | `amount`, `hour_of_day`, `day_of_week`, `merchant_risk`, `device_trust`, `is_international` |
| Data source | Synthetic generator (`scripts/generate_raw_dataset.py`) |
| License | Generated in-repo; no third-party dataset license |
| Split policy | 80/20 train/validation, stratified by `is_fraud`, `random_state=42` |

Raw data path: `data/raw/transactions.csv` (gitignored; DVC-tracked later).

## Champion metric

| Field | Value |
|---|---|
| Primary metric | `roc_auc` |
| Direction | maximize |
| Why | Threshold-free ranking quality; better default than accuracy on imbalanced labels |
| Secondary | `precision`, `recall`, `f1` |

Report path: `metrics/train_metrics.json`

## Model artifact contract

Package directory: `models/packaged/fraud_classifier/`

| File | Purpose |
|---|---|
| `model.joblib` | Fitted sklearn model |
| `metadata.json` | `model_version`, seed, config hash, primary metric |
| `feature_schema.json` | Feature names + order for inference |
| `metrics.json` | Validation metrics snapshot |
| `config_snapshot.yaml` | Exact config used for the run |

`model_version` is never `stub-v1`.

## Local setup

```bash
cd ml-training
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

## DVC remote

Default remote: `s3://ai-cloud-platform-899631475351-dvc/dvc` (us-east-2).

```bash
dvc push data/raw/transactions.csv.dvc
dvc pull data/raw/transactions.csv.dvc