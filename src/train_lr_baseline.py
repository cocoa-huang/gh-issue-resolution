"""
train_lr_baseline.py — Model 1: LR text-only baseline.

Features: TF-IDF on (title + body) + author_association (one-hot)
Metric:   macro-F1

Usage:
    python3 src/train_lr_baseline.py           # full run
    python3 src/train_lr_baseline.py --sample  # smoke test: 3 shards only

Outputs:
    results/lr_text_only.joblib
    results/lr_text_only_eval.txt
"""

import os
import sys
import gcsfs
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# ── Config ────────────────────────────────────────────────────────────────────

GCS_ISSUES  = "gs://gh_issue_ml-data/issues/issues_labeled_2025/"
RESULTS_DIR = "results"
LABEL_ORDER = ["Fast", "Medium", "Slow", "Stale"]

# ── Load ──────────────────────────────────────────────────────────────────────

def load_gcs_parquet(gcs_prefix: str, sample: bool = False) -> pd.DataFrame:
    fs = gcsfs.GCSFileSystem()
    paths = sorted(fs.glob(gcs_prefix.rstrip("/") + "/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquets found at {gcs_prefix}")
    if sample:
        paths = paths[:3]
        print(f"  [sample mode] Loading {len(paths)} shards...")
    else:
        print(f"  Loading {len(paths)} shards...")
    return pd.concat([pd.read_parquet(fs.open(p)) for p in paths], ignore_index=True)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sample = "--sample" in sys.argv
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load issues
    print("Loading issues...")
    df = load_gcs_parquet(GCS_ISSUES, sample=sample)
    df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")
    df["author_association"] = df["author_association"].fillna("NONE")
    print(f"  {len(df):,} issues loaded")
    print("  Label distribution:")
    print(df["label"].value_counts(normalize=True).mul(100).round(1).to_string())

    # 2. Train / test split
    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"\nTrain: {len(train):,}  Test: {len(test):,}")

    # 3. Pipeline
    model = Pipeline([
        ("features", ColumnTransformer([
            ("tfidf",  TfidfVectorizer(max_features=100_000, sublinear_tf=True, min_df=5), "text"),
            ("author", OneHotEncoder(handle_unknown="ignore", sparse_output=True), ["author_association"]),
        ])),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="saga",
            n_jobs=-1,
            random_state=42,
        )),
    ])

    # 4. Train
    print("\nTraining...")
    model.fit(train[["text", "author_association"]], train["label"])

    # 5. Evaluate
    preds    = model.predict(test[["text", "author_association"]])
    macro_f1 = f1_score(test["label"], preds, average="macro", labels=LABEL_ORDER)
    report   = classification_report(test["label"], preds, labels=LABEL_ORDER, digits=3)
    cm       = confusion_matrix(test["label"], preds, labels=LABEL_ORDER)

    output = (
        f"Model 1 — LR text-only\n"
        f"{'='*50}\n"
        f"Macro-F1: {macro_f1:.4f}\n\n"
        f"{report}\n"
        f"Confusion matrix (rows=true, cols=pred)\n"
        f"Order: {LABEL_ORDER}\n{cm}\n"
    )
    print(output)

    suffix = "_sample" if sample else ""

    eval_path = f"{RESULTS_DIR}/lr_text_only{suffix}_eval.txt"
    with open(eval_path, "w") as f:
        f.write(output)
    print(f"Eval written to {eval_path}")

    # 6. Save model (skip saving for sample runs — model is not meaningful)
    if not sample:
        model_path = f"{RESULTS_DIR}/lr_text_only.joblib"
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    else:
        print("Sample run — model not saved.")


if __name__ == "__main__":
    main()
