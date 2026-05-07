# GH Issue Resolution Predictor

Predict GitHub issue resolution speed at filing time using issue text and repository activity
signals. The task is a four-class classification problem:

| Class | Definition |
|-------|------------|
| Fast | resolved in <=7 days |
| Medium | resolved in <=30 days |
| Slow | resolved in <=90 days |
| Stale | >90 days or closed as `not_planned` |

The primary metric is macro-F1.

## Data

The project uses GH Archive data exported through BigQuery and Google Cloud Storage.

| Resource | Location |
|----------|----------|
| Labeled issues | `gs://gh_issue_ml-data/issues/issues_labeled_2025/*.parquet` |
| Issues with repo signals | `gs://gh_issue_ml-data/issues/issues_with_signals/*.parquet` |
| Qwen summaries | `gs://gh_issue_ml-data/llm_features/qwen_summaries/*.parquet` |

Temporal split:

```text
Train: issue_created_at < 2025-08-01
Test:  2025-08-01 <= issue_created_at < 2025-11-01
```

Issues after 2025-11-01 are discarded to avoid truncated labels.

## Model Ladder

| # | Model | Features | Macro-F1 |
|---|-------|----------|----------|
| 1 | Logistic Regression text-only | TF-IDF title + body | 0.298 |
| 2 | Logistic Regression text+signals | TF-IDF + repo signals | 0.3005 |
| 3 | DeBERTa text-only | title + body | 0.356 |
| 4 | DeBERTa text+signals | title + body + repo signals | 0.361 |
| 5 | Qwen-enriched DeBERTa | structured Qwen summary + title/body + repo signals | 0.3565 on Qwen-covered subset |

For the fair Model 5 comparison, Model 4 was also evaluated on the exact same Qwen-covered
temporal test subset:

```text
Model 4 same-subset macro-F1: 0.3606
Model 5 same-subset macro-F1: 0.3565
Delta: -0.0041
```

Conclusion: structured zero-shot Qwen summaries did not improve macro-F1 over the raw-text
DeBERTa+signals baseline on the same test rows.

## Key Scripts

| Script | Purpose |
|--------|---------|
| `src/train_lr_baseline.py` | Model 1: streaming logistic regression text baseline |
| `src/train_lr_signals.py` | Model 2: logistic regression with repo signals |
| `src/train_deberta_text.py` | Model 3: fine-tuned DeBERTa text-only |
| `src/train_deberta_signals.py` | Model 4: DeBERTa + repo signals |
| `src/extract_qwen_summaries.py` | Qwen3.5 structured summary extraction |
| `src/train_deberta_hybrid.py` | Model 5: Qwen summary + DeBERTa + repo signals |
| `src/eval_deberta_signals_qwen_subset.py` | Model 4 evaluation on Model 5's Qwen-covered subset |
| `src/analyze_deberta_errors.py` | Saved-checkpoint error analysis for Models 4-5 |

## Results Files

| File | Contents |
|------|----------|
| `results/lr_text_only_eval.txt` | Model 1 evaluation |
| `results/lr_signals_eval.txt` | Model 2 evaluation |
| `results/deberta_text_eval.txt` | Model 3 evaluation |
| `results/deberta_signals_eval.txt` | Model 4 full-test evaluation |
| `results/deberta_hybrid_eval.txt` | Model 5 Qwen-covered subset evaluation |
| `results/deberta_signals_qwen_subset_eval.txt` | Model 4 same-subset baseline for Model 5 |

## Current Status

Model training is complete. Remaining work is error analysis and report writing:

- sample and characterize misclassified examples across Models 3-5
- compare Model 4 vs Model 5 errors on the same Qwen-covered subset
- add representative Qwen summaries as interpretability examples
- write Related Work, Results, Error Analysis, Limitations, and Discussion sections

## Error Analysis

Run the saved-checkpoint analyzer on the machine that has GCS credentials and the model
checkpoints:

```bash
python src/analyze_deberta_errors.py \
  --model-kind signals \
  --checkpoint /scratch/zh2312/gh-issue-resolution/results/deberta_signals/pytorch_model.bin \
  --output-dir results/error_analysis/model4
```

For the Qwen-enriched model:

```bash
python src/analyze_deberta_errors.py \
  --model-kind hybrid \
  --checkpoint /scratch/zh2312/gh-issue-resolution/results/deberta_hybrid/pytorch_model.bin \
  --qwen-subset \
  --output-dir results/error_analysis/model5
```

The analyzer writes normalized confusion matrices, subgroup metrics, ranked confusion pairs,
row-level probabilities, representative high-confidence mistakes, and Fast/not-Fast feature
contrasts. For human-interpretable Fast-boundary analysis, inspect:

| File | Question |
|------|----------|
| `fast_numeric_contrasts.csv` | Do false Fast predictions differ in text length, confidence, repo activity signals, or probabilities? |
| `fast_categorical_contrasts.csv` | Which author roles, text-length bins, repo-activity bins, and Qwen fields appear in each Fast-boundary bucket? |
| `fast_feature_lift.csv` | Which categorical values are overrepresented in `true_fast_pred_not_fast` or `true_not_fast_pred_fast` errors? |
| `representative_errors.csv` | What do concrete high-confidence Fast/not-Fast mistakes look like? |
