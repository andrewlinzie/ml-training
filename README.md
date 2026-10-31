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

## Local setup

```bash
cd ml-training
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"