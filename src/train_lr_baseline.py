"""
train_lr_baseline.py — Model 1: LR text-only baseline.

Features: Hashed TF-IDF on (title + body) + author_association (one-hot)
Metric:   macro-F1

Uses SGDClassifier with partial_fit to stream shards — never loads full
dataset into memory.

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
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# ── Config ────────────────────────────────────────────────────────────────────

GCS_ISSUES  = "gs://gh_issue_ml-data/issues/issues_labeled_2025/"
RESULTS_DIR = "results"
LABEL_ORDER = ["Fast", "Medium", "Slow", "Stale"]

AUTHOR_CATS = ["COLLABORATOR", "CONTRIBUTOR", "MEMBER", "NONE", "OWNER"]

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


def encode_author(series: pd.Series) -> np.ndarray:
    """One-hot encode author_association into a dense array."""
    col = series.fillna("NONE").str.upper()
    result = np.zeros((len(col), len(AUTHOR_CATS)), dtype=np.float32)
    for i, cat in enumerate(AUTHOR_CATS):
        result[:, i] = (col == cat).astype(np.float32)
    return result


def shard_iter(fs, paths):
    for p in paths:
        df = pd.read_parquet(fs.open(p))
        df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")
        df["author_association"] = df["author_association"].fillna("NONE")
        yield df

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sample = "--sample" in sys.argv
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fs, paths = get_shards(GCS_ISSUES, sample=sample)

    # Split shards: last 20% → test, rest → train
    n_test = max(1, int(len(paths) * 0.2))
    train_paths = paths[:-n_test]
    test_paths  = paths[-n_test:]
    print(f"  Train shards: {len(train_paths)}  Test shards: {len(test_paths)}")

    # ── 1. Compute class weights from training shards ─────────────────────────
    print("\nCounting label distribution (train shards)...")
    from collections import Counter
    label_counts: Counter = Counter()
    for df in shard_iter(fs, train_paths):
        label_counts.update(df["label"].tolist())

    print("  Label distribution (train):")
    total = sum(label_counts.values())
    for lbl in LABEL_ORDER:
        print(f"    {lbl}: {label_counts[lbl]:,} ({100*label_counts[lbl]/total:.1f}%)")

    labels_arr = np.array([lbl for lbl, cnt in label_counts.items() for _ in range(cnt)])
    class_weights = compute_class_weight("balanced", classes=np.array(LABEL_ORDER), y=labels_arr)
    sample_weight_map = dict(zip(LABEL_ORDER, class_weights))
    print(f"  Class weights: { {k: round(v,3) for k,v in sample_weight_map.items()} }")

    # ── 2. Build vectorizer and classifier ───────────────────────────────────
    vectorizer = HashingVectorizer(
        n_features=2**18,   # 262144 buckets — good balance of speed vs. collision
        ngram_range=(1, 2),
        alternate_sign=False,
        norm="l2",
    )

    clf = SGDClassifier(
        loss="modified_huber",   # produces calibrated probabilities, robust to outliers
        max_iter=1,
        tol=None,
        random_state=42,
    )

    # ── 3. Train (streaming, one shard at a time) ─────────────────────────────
    print("\nTraining (streaming)...")
    for i, df in enumerate(shard_iter(fs, train_paths)):
        text_X   = vectorizer.transform(df["text"])
        author_X = encode_author(df["author_association"])
        from scipy.sparse import csr_matrix
        X = hstack([text_X, csr_matrix(author_X)])
        y = df["label"].values
        sw = np.array([sample_weight_map[lbl] for lbl in y])
        clf.partial_fit(X, y, classes=LABEL_ORDER, sample_weight=sw)
        if (i + 1) % 10 == 0 or (i + 1) == len(train_paths):
            print(f"  Trained shard {i+1}/{len(train_paths)}")

    # ── 4. Evaluate on test shards ────────────────────────────────────────────
    print("\nEvaluating on test shards...")
    all_preds = []
    all_true  = []
    for df in shard_iter(fs, test_paths):
        text_X   = vectorizer.transform(df["text"])
        author_X = encode_author(df["author_association"])
        from scipy.sparse import csr_matrix
        X = hstack([text_X, csr_matrix(author_X)])
        all_preds.extend(clf.predict(X).tolist())
        all_true.extend(df["label"].tolist())

    macro_f1 = f1_score(all_true, all_preds, average="macro", labels=LABEL_ORDER)
    report   = classification_report(all_true, all_preds, labels=LABEL_ORDER, digits=3)
    cm       = confusion_matrix(all_true, all_preds, labels=LABEL_ORDER)

    output = (
        f"Model 1 — LR text-only (SGD/HashingVectorizer)\n"
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

    if not sample:
        model_path = f"{RESULTS_DIR}/lr_text_only.joblib"
        joblib.dump({"vectorizer": vectorizer, "clf": clf}, model_path)
        print(f"Model saved to {model_path}")
    else:
        print("Sample run — model not saved.")


if __name__ == "__main__":
    main()
