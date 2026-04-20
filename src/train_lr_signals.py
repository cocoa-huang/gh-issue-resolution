"""
train_lr_signals.py — Model 2: LR text + repo signals.

Features:
  - Hashed TF-IDF on (title + body)
  - author_association one-hot (5 categories)
  - 5 repo-level signals (log1p-scaled): pr_merged_30d, avg_merge_hours_30d,
    push_count_30d, release_count_90d, star_count_30d

All signals are pre-aggregated in BigQuery with a temporal window strictly
before issue_created_at — no future leakage.

Temporal split:
    Train: issue_created_at < 2025-08-01
    Test:  2025-08-01 <= issue_created_at < 2025-11-01
    Discard: >= 2025-11-01 (labels truncated)

Usage:
    python3 src/train_lr_signals.py           # full run
    python3 src/train_lr_signals.py --sample  # smoke test: 3 shards only

Outputs:
    results/lr_signals.joblib
    results/lr_signals_eval.txt
"""

import os
import sys
from collections import Counter

import gcsfs
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# ── Config ────────────────────────────────────────────────────────────────────

GCS_ISSUES  = "gs://gh_issue_ml-data/issues/issues_with_signals/"
RESULTS_DIR = "results"
LABEL_ORDER = ["Fast", "Medium", "Slow", "Stale"]
AUTHOR_CATS = ["COLLABORATOR", "CONTRIBUTOR", "MEMBER", "NONE", "OWNER"]
SIGNAL_COLS = [
    "pr_merged_30d",
    "avg_merge_hours_30d",
    "push_count_30d",
    "release_count_90d",
    "star_count_30d",
]

TRAIN_CUTOFF = pd.Timestamp("2025-08-01", tz="UTC")
TEST_CUTOFF  = pd.Timestamp("2025-11-01", tz="UTC")

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_shards(gcs_prefix: str, sample: bool = False):
    fs = gcsfs.GCSFileSystem()
    paths = sorted(fs.glob(gcs_prefix.rstrip("/") + "/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquets found at {gcs_prefix}")
    if sample:
        paths = paths[:3]
        print(f"  [sample mode] Using {len(paths)} shards")
    else:
        print(f"  Using {len(paths)} shards")
    return fs, paths


def load_shard(fs, path: str) -> pd.DataFrame:
    df = pd.read_parquet(fs.open(path))
    df["created_at"] = pd.to_datetime(df["issue_created_at"], utc=True)
    df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")
    df["author_association"] = df["author_association"].fillna("NONE")
    for col in SIGNAL_COLS:
        df[col] = df[col].fillna(0.0)
    return df


def encode_author(series: pd.Series) -> np.ndarray:
    col = series.fillna("NONE").str.upper()
    result = np.zeros((len(col), len(AUTHOR_CATS)), dtype=np.float32)
    for i, cat in enumerate(AUTHOR_CATS):
        result[:, i] = (col == cat).astype(np.float32)
    return result


def encode_signals(df: pd.DataFrame) -> np.ndarray:
    return np.log1p(df[SIGNAL_COLS].values).astype(np.float32)


def featurize(vectorizer: HashingVectorizer, df: pd.DataFrame):
    text_X   = vectorizer.transform(df["text"])
    author_X = csr_matrix(encode_author(df["author_association"]))
    signal_X = csr_matrix(encode_signals(df))
    return hstack([text_X, author_X, signal_X])

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sample = "--sample" in sys.argv
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fs, paths = get_shards(GCS_ISSUES, sample=sample)
    print(f"  Temporal split: train < {TRAIN_CUTOFF.date()}  |  test {TRAIN_CUTOFF.date()} – {TEST_CUTOFF.date()}  |  discard >= {TEST_CUTOFF.date()}")

    # ── 1. Count train labels for class weights ───────────────────────────────
    print("\nCounting label distribution (train split)...")
    label_counts: Counter = Counter()
    for p in paths:
        df = load_shard(fs, p)
        train_df = df[df["created_at"] < TRAIN_CUTOFF]
        label_counts.update(train_df["label"].tolist())

    total = sum(label_counts.values())
    print(f"  Train issues: {total:,}")
    print("  Label distribution (train):")
    for lbl in LABEL_ORDER:
        print(f"    {lbl}: {label_counts[lbl]:,} ({100*label_counts[lbl]/total:.1f}%)")

    y_for_weights = np.array([lbl for lbl, cnt in label_counts.items() for _ in range(cnt)])
    class_weights = compute_class_weight("balanced", classes=np.array(LABEL_ORDER), y=y_for_weights)
    sample_weight_map = dict(zip(LABEL_ORDER, class_weights))
    print(f"  Class weights: { {k: round(v, 3) for k, v in sample_weight_map.items()} }")

    # ── 2. Build vectorizer and classifier ───────────────────────────────────
    vectorizer = HashingVectorizer(
        n_features=2**18,
        ngram_range=(1, 2),
        alternate_sign=False,
        norm="l2",
    )

    clf = SGDClassifier(
        loss="log_loss",
        max_iter=1,
        tol=None,
        random_state=42,
    )

    # ── 3. Train (streaming) ─────────────────────────────────────────────────
    print("\nTraining (streaming)...")
    n_train_shards = 0
    for p in paths:
        df = load_shard(fs, p)
        train_df = df[df["created_at"] < TRAIN_CUTOFF]
        if train_df.empty:
            continue
        X  = featurize(vectorizer, train_df)
        y  = train_df["label"].values
        sw = np.array([sample_weight_map[lbl] for lbl in y])
        clf.partial_fit(X, y, classes=LABEL_ORDER, sample_weight=sw)
        n_train_shards += 1
        if n_train_shards % 10 == 0:
            print(f"  Trained {n_train_shards} shards...")

    print(f"  Done. {n_train_shards} shards contributed training rows.")

    # ── 4. Evaluate on test split ─────────────────────────────────────────────
    print(f"\nEvaluating on test split ({TRAIN_CUTOFF.date()} – {TEST_CUTOFF.date()})...")
    all_preds = []
    all_true  = []
    n_test = 0
    for p in paths:
        df = load_shard(fs, p)
        test_df = df[(df["created_at"] >= TRAIN_CUTOFF) & (df["created_at"] < TEST_CUTOFF)]
        if test_df.empty:
            continue
        X = featurize(vectorizer, test_df)
        all_preds.extend(clf.predict(X).tolist())
        all_true.extend(test_df["label"].tolist())
        n_test += len(test_df)

    print(f"  Test issues: {n_test:,}")

    macro_f1 = f1_score(all_true, all_preds, average="macro", labels=LABEL_ORDER)
    report   = classification_report(all_true, all_preds, labels=LABEL_ORDER, digits=3)
    cm       = confusion_matrix(all_true, all_preds, labels=LABEL_ORDER)

    output = (
        f"Model 2 — LR text+signals (SGD/HashingVectorizer + repo signals log1p)\n"
        f"Temporal split: train < 2025-08-01  |  test 2025-08-01 – 2025-10-31\n"
        f"Signal features: {SIGNAL_COLS}\n"
        f"{'='*60}\n"
        f"Train issues: {total:,}  |  Test issues: {n_test:,}\n"
        f"Macro-F1: {macro_f1:.4f}\n\n"
        f"{report}\n"
        f"Confusion matrix (rows=true, cols=pred)\n"
        f"Order: {LABEL_ORDER}\n{cm}\n"
    )
    print(output)

    suffix = "_sample" if sample else ""
    eval_path = f"{RESULTS_DIR}/lr_signals{suffix}_eval.txt"
    with open(eval_path, "w") as f:
        f.write(output)
    print(f"Eval written to {eval_path}")

    if not sample:
        model_path = f"{RESULTS_DIR}/lr_signals.joblib"
        joblib.dump({"vectorizer": vectorizer, "clf": clf}, model_path)
        print(f"Model saved to {model_path}")
    else:
        print("Sample run — model not saved.")


if __name__ == "__main__":
    main()
