# GH Issue Resolution Predictor

## Problem

Predicting whether a GitHub issue will be resolved — and how quickly — is a practical proxy for understanding open-source project health and contributor responsiveness. This project builds a model ladder on top of structured event data from GH Archive to benchmark increasingly expressive approaches on the same task.

## Data Sources

| Source | Description |
|--------|-------------|
| [GH Archive](https://www.gharchive.org/) | Hourly JSON dumps of public GitHub events (issues, PRs, comments) |
| BigQuery (`githubarchive.*`) | SQL-accessible mirror of GH Archive used for bulk feature extraction |

Raw exports land in `data/` (gitignored). Processed features are also kept local.

## Model Ladder

| # | Model | Notes |
|---|-------|-------|
| 1 | Logistic Regression | Baseline — bag-of-words + metadata features |
| 2 | Gradient Boosted Trees | Structured features only (XGBoost / LightGBM) |
| 3 | Fine-tuned BERT | Issue title + body embeddings |
| 4 | LLM-based classifier | Few-shot or fine-tuned on structured prompt |

## Results

> **WIP** — table will be populated as experiments complete.

| Model | Accuracy | F1 | AUC-ROC | Notes |
|-------|----------|----|---------|-------|
| Logistic Regression | — | — | — | |
| GBT | — | — | — | |
| Fine-tuned BERT | — | — | — | |
| LLM classifier | — | — | — | |

## Repo Layout

```
data/        raw + processed data (gitignored)
notebooks/   exploratory analysis
src/         pipeline code
results/     model artifacts, eval outputs
```
