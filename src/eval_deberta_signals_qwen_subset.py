"""
eval_deberta_signals_qwen_subset.py - Evaluate Model 4 on the Model 5 Qwen-covered subset.

This does not retrain Model 4. It loads the saved DeBERTa+signals checkpoint and
evaluates it on the exact Aug-Oct 2025 test rows that have cached Qwen summaries.

Data source: gs://gh_issue_ml-data/issues/issues_with_signals/*.parquet
Subset keys: gs://gh_issue_ml-data/llm_features/qwen_summaries/*.parquet
Output: results/deberta_signals_qwen_subset_eval.txt
"""

import argparse
import hashlib
import os
import re

import gcsfs
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import classification_report, confusion_matrix, f1_score

GCS_SIGNALS = "gs://gh_issue_ml-data/issues/issues_with_signals/"
GCS_QWEN = "gs://gh_issue_ml-data/llm_features/qwen_summaries/"
MODEL_NAME = "microsoft/deberta-v3-base"
RESULTS_DIR = "results"
DEFAULT_CHECKPOINT = os.path.join(RESULTS_DIR, "deberta_signals", "pytorch_model.bin")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "deberta_signals_qwen_subset_eval.txt")

LABEL_ORDER = ["Fast", "Medium", "Slow", "Stale"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_ORDER)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_ORDER)}

SIGNAL_COLS = [
    "pr_merged_30d",
    "avg_merge_hours_30d",
    "push_count_30d",
    "release_count_90d",
    "star_count_30d",
]
AUTHOR_CATS = ["COLLABORATOR", "CONTRIBUTOR", "MEMBER", "NONE", "OWNER"]
NUM_SIGNALS = len(SIGNAL_COLS) + len(AUTHOR_CATS)

TRAIN_CUTOFF = pd.Timestamp("2025-08-01", tz="UTC")
TEST_CUTOFF = pd.Timestamp("2025-11-01", tz="UTC")
MAX_LEN = 512
INF_BATCH = 64

LOAD_COLS = [
    "repo_name",
    "title",
    "body",
    "issue_created_at",
    "label",
    "author_association",
] + SIGNAL_COLS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", default=GCS_SIGNALS)
    parser.add_argument("--qwen", default=GCS_QWEN)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--sample", action="store_true", help="Use first 3 Qwen/source shards.")
    return parser.parse_args()


def issue_key(repo_name, issue_created_at, title, body) -> str:
    parts = [
        "" if pd.isna(repo_name) else str(repo_name),
        "" if pd.isna(issue_created_at) else str(issue_created_at),
        "" if pd.isna(title) else str(title),
        "" if pd.isna(body) else str(body),
    ]
    payload = "\x1f".join(parts).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def add_issue_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["issue_key"] = [
        issue_key(repo, created, title, body)
        for repo, created, title, body in zip(
            df["repo_name"], df["issue_created_at"], df["title"], df["body"]
        )
    ]
    return df


def get_parquet_shards(fs, base_path: str, sample: bool = False):
    paths = sorted(fs.glob(base_path.rstrip("/") + "/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquets found at {base_path}")
    if sample:
        paths = paths[:3]
    return paths


def qwen_source_indices(qwen_paths) -> set[int]:
    indices = set()
    for path in qwen_paths:
        match = re.search(r"part-(\d+)\.parquet$", os.path.basename(path))
        if match:
            indices.add(int(match.group(1)))
    return indices


def restrict_signal_paths_to_qwen(paths, qwen_paths):
    indices = qwen_source_indices(qwen_paths)
    if not indices:
        return paths
    if max(indices) >= len(paths):
        print("Qwen shard indices do not match signal shard list; scanning all signal shards.")
        return paths
    selected = [paths[i] for i in sorted(indices)]
    print(f"Restricting signal scan to {len(selected)}/{len(paths)} source shards.")
    return selected


def load_qwen_keys(fs, qwen_paths) -> set[str]:
    frames = [pd.read_parquet(fs.open(p), columns=["issue_key", "qwen_summary"]) for p in qwen_paths]
    qwen = pd.concat(frames, ignore_index=True).drop_duplicates("issue_key")
    qwen = qwen[qwen["qwen_summary"].fillna("").astype(str).str.len() > 0]
    keys = set(qwen["issue_key"].astype(str))
    print(f"Loaded {len(keys):,} non-empty Qwen-covered issue keys.")
    return keys


def load_shard(fs, path: str) -> pd.DataFrame:
    df = pd.read_parquet(fs.open(path), columns=LOAD_COLS)
    df["created_at"] = pd.to_datetime(df["issue_created_at"], utc=True)
    df["author_association"] = df["author_association"].fillna("NONE").str.upper()
    df["text"] = df["title"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)
    for col in SIGNAL_COLS:
        df[col] = df[col].fillna(0.0)
    return add_issue_keys(df)


def encode_signals(df: pd.DataFrame) -> np.ndarray:
    numeric = np.log1p(df[SIGNAL_COLS].values.astype(np.float32))
    author = df["author_association"].values
    auth_oh = np.zeros((len(df), len(AUTHOR_CATS)), dtype=np.float32)
    for i, cat in enumerate(AUTHOR_CATS):
        auth_oh[:, i] = (author == cat).astype(np.float32)
    return np.concatenate([numeric, auth_oh], axis=1)


class DeBERTaWithSignals(nn.Module):
    def __init__(self, backbone, pooler, pooler_dim: int, num_signals: int, num_labels: int):
        super().__init__()
        self.deberta = backbone
        self.pooler = pooler
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(pooler_dim + num_signals, num_labels)
        self.num_signals = num_signals

    def forward(self, input_ids=None, attention_mask=None, signals=None, labels=None, **kwargs):
        hidden = self.deberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self.dropout(self.pooler(hidden))
        if signals is None:
            signals = torch.zeros(
                pooled.size(0), self.num_signals, device=pooled.device, dtype=pooled.dtype
            )
        combined = torch.cat([pooled, signals.to(pooled.device, pooled.dtype)], dim=1)
        return SequenceClassifierOutput(logits=self.classifier(combined))


def build_model(checkpoint: str):
    print(f"Loading pretrained model shell: {MODEL_NAME}")
    pretrained = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_ORDER),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    pooler_dim = getattr(pretrained.pooler, "output_dim", None) or pretrained.config.pooler_hidden_size
    model = DeBERTaWithSignals(
        backbone=pretrained.deberta,
        pooler=pretrained.pooler,
        pooler_dim=pooler_dim,
        num_signals=NUM_SIGNALS,
        num_labels=len(LABEL_ORDER),
    )
    del pretrained

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Missing Model 4 checkpoint: {checkpoint}")
    print(f"Loading Model 4 checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Missing keys: {len(missing)}  |  Unexpected keys: {len(unexpected)}")
    return model


def stream_subset_eval(model, tokenizer, fs, paths, qwen_keys: set[str], device):
    all_preds, all_true = [], []
    model.eval()
    n_processed = 0

    for p in paths:
        df = load_shard(fs, p)
        test = df[
            (df["created_at"] >= TRAIN_CUTOFF)
            & (df["created_at"] < TEST_CUTOFF)
            & (df["issue_key"].isin(qwen_keys))
        ].reset_index(drop=True)
        if test.empty:
            continue

        print(f"Evaluating {len(test):,} Qwen-covered test rows from {os.path.basename(p)}")
        sig_arr = encode_signals(test)
        for start in range(0, len(test), INF_BATCH):
            end = min(start + INF_BATCH, len(test))
            batch = test.iloc[start:end]
            enc = tokenizer(
                batch["text"].tolist(),
                max_length=MAX_LEN,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            sig_t = torch.tensor(sig_arr[start:end], dtype=torch.float32).to(device)
            labels = [LABEL2ID[l] for l in batch["label"]]
            with torch.no_grad():
                out = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    signals=sig_t,
                )
            all_preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
            all_true.extend(labels)
            n_processed += len(batch)
            if n_processed % 5_000 == 0:
                print(f"  Inferred {n_processed:,} subset test rows...", flush=True)

    return all_preds, all_true


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    fs = gcsfs.GCSFileSystem()
    signal_paths = get_parquet_shards(fs, args.signals, sample=args.sample)
    qwen_paths = get_parquet_shards(fs, args.qwen, sample=args.sample)
    signal_paths = restrict_signal_paths_to_qwen(signal_paths, qwen_paths)
    qwen_keys = load_qwen_keys(fs, qwen_paths)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = build_model(args.checkpoint).to(device)

    all_preds, all_true = stream_subset_eval(model, tokenizer, fs, signal_paths, qwen_keys, device)
    print(f"Total Qwen-covered test rows evaluated: {len(all_true):,}")
    if not all_true:
        raise ValueError("No Qwen-covered test rows found for Model 4 evaluation.")

    pred_labels = [ID2LABEL[p] for p in all_preds]
    true_labels = [ID2LABEL[l] for l in all_true]
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", labels=LABEL_ORDER)
    report = classification_report(true_labels, pred_labels, labels=LABEL_ORDER, digits=3)
    cm = confusion_matrix(true_labels, pred_labels, labels=LABEL_ORDER)

    output = (
        f"Model 4 - DeBERTa raw issue text + repo signals on Qwen-covered subset ({MODEL_NAME})\n"
        f"Checkpoint: {args.checkpoint}\n"
        f"Temporal split: test 2025-08-01 - 2025-10-31\n"
        f"Test: {len(all_true):,} (same Qwen-covered joined subset used by Model 5)\n"
        f"Input: [ISSUE: title body] equivalent raw title + body, no Qwen summary\n"
        f"Signal features ({NUM_SIGNALS}): {SIGNAL_COLS} + author_association one-hot\n"
        f"{'='*60}\n"
        f"Macro-F1: {macro_f1:.4f}\n\n"
        f"{report}\n"
        f"Confusion matrix (rows=true, cols=pred)\n"
        f"Order: {LABEL_ORDER}\n{cm}\n"
    )
    print(output)
    with open(args.output, "w") as f:
        f.write(output)
    print(f"Eval written to {args.output}")


if __name__ == "__main__":
    main()
