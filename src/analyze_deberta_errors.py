"""
Analyze DeBERTa issue-resolution errors from a saved checkpoint.

This script runs inference on the temporal test split and writes paper-ready
error-analysis artifacts:
  - prediction table with probabilities/margins
  - confusion counts and row-normalized confusion rates
  - per-class precision/recall/F1
  - subgroup metrics by author, text length, repo activity, and Qwen fields
  - feature contrasts for the Fast vs not-Fast decision boundary
  - top confusion pairs
  - representative high-confidence and low-margin mistakes
  - Markdown summary for the final report

Examples:
  python src/analyze_deberta_errors.py \
    --model-kind signals \
    --checkpoint /scratch/zh2312/gh-issue-resolution/results/deberta_signals/pytorch_model.bin \
    --output-dir results/error_analysis/model4

  python src/analyze_deberta_errors.py \
    --model-kind hybrid \
    --checkpoint /scratch/zh2312/gh-issue-resolution/results/deberta_hybrid/pytorch_model.bin \
    --qwen-subset \
    --output-dir results/error_analysis/model5
"""

print("analyze_deberta_errors.py: starting imports", flush=True)

import argparse
import hashlib
import os
import re
from collections import Counter, defaultdict

print("analyze_deberta_errors.py: importing gcsfs", flush=True)
import gcsfs
print("analyze_deberta_errors.py: importing numpy", flush=True)
import numpy as np
print("analyze_deberta_errors.py: importing pandas", flush=True)
import pandas as pd
print("analyze_deberta_errors.py: importing torch", flush=True)
import torch
print("analyze_deberta_errors.py: importing sklearn metrics", flush=True)
from sklearn.metrics import classification_report, confusion_matrix, f1_score
print("analyze_deberta_errors.py: importing torch.nn", flush=True)
from torch import nn
print("analyze_deberta_errors.py: importing transformers", flush=True)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

print("analyze_deberta_errors.py: imports complete", flush=True)


GCS_SIGNALS = "gs://gh_issue_ml-data/issues/issues_with_signals/"
GCS_QWEN = "gs://gh_issue_ml-data/llm_features/qwen_summaries/"
MODEL_NAME = "microsoft/deberta-v3-base"

LABEL_ORDER = ["Fast", "Medium", "Slow", "Stale"]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_ORDER)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_ORDER)}

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

BASE_LOAD_COLS = [
    "repo_name",
    "title",
    "body",
    "issue_created_at",
    "label",
    "author_association",
] + SIGNAL_COLS
OPTIONAL_LOAD_COLS = ["resolution_days", "state_reason"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-kind", choices=["signals", "hybrid"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--signals", default=GCS_SIGNALS)
    parser.add_argument("--qwen", default=GCS_QWEN)
    parser.add_argument("--qwen-subset", action="store_true")
    parser.add_argument("--sample", action="store_true", help="Use first 3 signal/Qwen shards.")
    parser.add_argument("--output-dir", default="results/error_analysis")
    parser.add_argument("--max-examples-per-pair", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap for quick debugging.")
    parser.add_argument("--batch-size", type=int, default=INF_BATCH)
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
        raise FileNotFoundError(f"No parquet shards found at {base_path}")
    return paths[:3] if sample else paths


def qwen_source_indices(qwen_paths) -> set[int]:
    indices = set()
    for path in qwen_paths:
        match = re.search(r"part-(\d+)\.parquet$", os.path.basename(path))
        if match:
            indices.add(int(match.group(1)))
    return indices


def restrict_signal_paths_to_qwen(paths, qwen_paths):
    indices = qwen_source_indices(qwen_paths)
    if not indices or max(indices) >= len(paths):
        return paths
    selected = [paths[i] for i in sorted(indices)]
    print(f"Restricting signal scan to {len(selected)}/{len(paths)} Qwen-covered source shards.")
    return selected


def load_qwen(fs, qwen_paths, include_summary: bool) -> pd.DataFrame:
    cols = ["issue_key", "qwen_summary"] if include_summary else ["issue_key"]
    frames = [pd.read_parquet(fs.open(path), columns=cols) for path in qwen_paths]
    qwen = pd.concat(frames, ignore_index=True).drop_duplicates("issue_key")
    if "qwen_summary" in qwen.columns:
        qwen = qwen[qwen["qwen_summary"].fillna("").astype(str).str.len() > 0]
    return qwen


def load_signal_shard(fs, path: str) -> pd.DataFrame:
    try:
        df = pd.read_parquet(fs.open(path), columns=BASE_LOAD_COLS + OPTIONAL_LOAD_COLS)
    except Exception:
        df = pd.read_parquet(fs.open(path), columns=BASE_LOAD_COLS)
    df["created_at"] = pd.to_datetime(df["issue_created_at"], utc=True)
    df["author_association"] = df["author_association"].fillna("NONE").str.upper()
    for col in SIGNAL_COLS:
        df[col] = df[col].fillna(0.0)
    return add_issue_keys(df)


def build_hybrid_text(df: pd.DataFrame) -> pd.Series:
    summaries = df["qwen_summary"].fillna("").astype(str)
    issue_text = df["title"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)
    return "[SUMMARY: " + summaries + "] [ISSUE: " + issue_text + "]"


def attach_qwen_if_needed(df: pd.DataFrame, qwen: pd.DataFrame | None, model_kind: str):
    if qwen is None:
        df["text"] = df["title"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)
        return df
    df = df.merge(qwen, on="issue_key", how="inner")
    if model_kind == "hybrid":
        df["text"] = build_hybrid_text(df)
    else:
        df["text"] = df["title"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)
    return df


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
        combined = combined.to(self.classifier.weight.dtype)
        return SequenceClassifierOutput(logits=self.classifier(combined))


def build_model(checkpoint: str):
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
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint. Missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
    return model


def parse_qwen_field(summary: str, field: str) -> str:
    if not isinstance(summary, str) or not summary.strip():
        return "missing"
    match = re.search(rf"{re.escape(field)}\s*:\s*([^\n\r]+)", summary, flags=re.IGNORECASE)
    if not match:
        return "missing"
    value = match.group(1).strip().split("|")[0].strip().lower()
    return re.sub(r"[^a-z0-9_ -]", "", value)[:80] or "missing"


def text_len_bin(token_count: int) -> str:
    if token_count <= 30:
        return "000-030"
    if token_count <= 100:
        return "031-100"
    if token_count <= 300:
        return "101-300"
    return "301+"


def activity_bin(row: pd.Series) -> str:
    active = (
        row["pr_merged_30d"] > 0
        or row["push_count_30d"] > 0
        or row["release_count_90d"] > 0
        or row["star_count_30d"] > 0
    )
    return "active_repo_window" if active else "quiet_repo_window"


def confidence_bin(confidence: float) -> str:
    if confidence < 0.40:
        return "0.00-0.40"
    if confidence < 0.55:
        return "0.40-0.55"
    if confidence < 0.70:
        return "0.55-0.70"
    if confidence < 0.85:
        return "0.70-0.85"
    return "0.85-1.00"


def update_group(groups, group_name: str, group_value: str, true_label: str, pred_label: str):
    key = (group_name, str(group_value))
    groups[key]["y_true"].append(true_label)
    groups[key]["y_pred"].append(pred_label)


def truncate_text(text: str, limit: int = 500) -> str:
    text = " ".join(str(text).split())
    return text[: limit - 3] + "..." if len(text) > limit else text


def stream_predictions(args, model, tokenizer, fs, signal_paths, qwen, device):
    records = []
    groups = defaultdict(lambda: {"y_true": [], "y_pred": []})
    pair_counts = Counter()
    example_pool = defaultdict(list)
    n_processed = 0

    model.eval()
    for path in signal_paths:
        df = load_signal_shard(fs, path)
        df = attach_qwen_if_needed(df, qwen, args.model_kind)
        test = df[(df["created_at"] >= TRAIN_CUTOFF) & (df["created_at"] < TEST_CUTOFF)]
        if test.empty:
            continue
        test = test.reset_index(drop=True)
        sig_arr = encode_signals(test)

        for start in range(0, len(test), args.batch_size):
            end = min(start + args.batch_size, len(test))
            batch = test.iloc[start:end].copy()
            enc = tokenizer(
                batch["text"].tolist(),
                max_length=MAX_LEN,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            sig_t = torch.tensor(sig_arr[start:end], dtype=torch.float32).to(device)

            with torch.no_grad():
                logits = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    signals=sig_t,
                ).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            pred_ids = probs.argmax(axis=1)
            sorted_probs = np.sort(probs, axis=1)
            margins = sorted_probs[:, -1] - sorted_probs[:, -2]
            confidences = sorted_probs[:, -1]

            for i, (_, row) in enumerate(batch.iterrows()):
                true_label = row["label"]
                pred_label = ID2LABEL[int(pred_ids[i])]
                correct = true_label == pred_label
                raw_text = f"{row.get('title', '')} {row.get('body', '')}"
                word_count = len(str(raw_text).split())

                record = {
                    "issue_key": row["issue_key"],
                    "repo_name": row["repo_name"],
                    "issue_created_at": row["issue_created_at"],
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "correct": correct,
                    "confidence": float(confidences[i]),
                    "margin": float(margins[i]),
                    "p_fast": float(probs[i, LABEL2ID["Fast"]]),
                    "p_medium": float(probs[i, LABEL2ID["Medium"]]),
                    "p_slow": float(probs[i, LABEL2ID["Slow"]]),
                    "p_stale": float(probs[i, LABEL2ID["Stale"]]),
                    "author_association": row["author_association"],
                    "text_words": word_count,
                    "text_len_bin": text_len_bin(word_count),
                    "repo_activity_bin": activity_bin(row),
                    "title": truncate_text(row.get("title", ""), 240),
                    "body_excerpt": truncate_text(row.get("body", ""), 500),
                }
                for col in OPTIONAL_LOAD_COLS + SIGNAL_COLS:
                    if col in row:
                        record[col] = row[col]
                if "qwen_summary" in row:
                    record["qwen_summary"] = truncate_text(row["qwen_summary"], 500)
                    record["qwen_issue_type"] = parse_qwen_field(row["qwen_summary"], "Issue type")
                    record["qwen_spec_quality"] = parse_qwen_field(
                        row["qwen_summary"], "Specification quality"
                    )
                    record["qwen_complexity"] = parse_qwen_field(row["qwen_summary"], "Complexity")
                    record["qwen_urgency"] = parse_qwen_field(row["qwen_summary"], "Urgency signals")

                records.append(record)
                pair_counts[(true_label, pred_label)] += 1
                update_group(groups, "author_association", row["author_association"], true_label, pred_label)
                update_group(groups, "text_len_bin", record["text_len_bin"], true_label, pred_label)
                update_group(groups, "repo_activity_bin", record["repo_activity_bin"], true_label, pred_label)
                update_group(groups, "confidence_bin", confidence_bin(record["confidence"]), true_label, pred_label)
                if "qwen_issue_type" in record:
                    update_group(groups, "qwen_issue_type", record["qwen_issue_type"], true_label, pred_label)
                    update_group(groups, "qwen_spec_quality", record["qwen_spec_quality"], true_label, pred_label)
                    update_group(groups, "qwen_complexity", record["qwen_complexity"], true_label, pred_label)
                    update_group(groups, "qwen_urgency", record["qwen_urgency"], true_label, pred_label)

                if not correct:
                    pair = (true_label, pred_label)
                    example_pool[pair].append(record)

            n_processed += len(batch)
            if n_processed % 50_000 == 0:
                print(f"Processed {n_processed:,} test rows...", flush=True)
            if args.max_rows and n_processed >= args.max_rows:
                return pd.DataFrame(records), groups, pair_counts, example_pool

    return pd.DataFrame(records), groups, pair_counts, example_pool


def per_class_metrics_df(y_true, y_pred) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label in LABEL_ORDER:
        rows.append(
            {
                "label": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            }
        )
    return pd.DataFrame(rows)


def subgroup_metrics_df(groups) -> pd.DataFrame:
    rows = []
    for (group_name, group_value), values in groups.items():
        y_true = values["y_true"]
        y_pred = values["y_pred"]
        if not y_true:
            continue
        rows.append(
            {
                "group": group_name,
                "value": group_value,
                "n": len(y_true),
                "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
                "macro_f1": f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    labels=LABEL_ORDER,
                    zero_division=0,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["group", "macro_f1", "n"], ascending=[True, True, False])


def representative_examples(example_pool, max_per_pair: int) -> pd.DataFrame:
    rows = []
    for pair, examples in example_pool.items():
        ranked = sorted(examples, key=lambda r: (-r["confidence"], r["margin"]))
        for record in ranked[:max_per_pair]:
            rows.append(record)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["true_label", "pred_label", "confidence"], ascending=[True, True, False])


def add_fast_boundary_bucket(pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_df = pred_df.copy()
    true_fast = pred_df["true_label"] == "Fast"
    pred_fast = pred_df["pred_label"] == "Fast"
    pred_df["fast_boundary_bucket"] = np.select(
        [
            true_fast & pred_fast,
            true_fast & ~pred_fast,
            ~true_fast & pred_fast,
            ~true_fast & ~pred_fast,
        ],
        [
            "true_fast_pred_fast",
            "true_fast_pred_not_fast",
            "true_not_fast_pred_fast",
            "true_not_fast_pred_not_fast",
        ],
        default="unknown",
    )
    pred_df["true_is_fast"] = true_fast
    pred_df["pred_is_fast"] = pred_fast
    return pred_df


def fast_numeric_contrasts(pred_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "text_words",
        "confidence",
        "margin",
        "p_fast",
        "p_medium",
        "p_slow",
        "p_stale",
    ]
    for col in SIGNAL_COLS + ["resolution_days"]:
        if col in pred_df.columns:
            numeric_cols.append(col)

    rows = []
    for bucket, group in pred_df.groupby("fast_boundary_bucket"):
        for col in numeric_cols:
            series = pd.to_numeric(group[col], errors="coerce").dropna()
            if series.empty:
                continue
            rows.append(
                {
                    "bucket": bucket,
                    "feature": col,
                    "n": int(series.shape[0]),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "p25": float(series.quantile(0.25)),
                    "p75": float(series.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature", "bucket"])


def fast_categorical_contrasts(pred_df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = ["author_association", "text_len_bin", "repo_activity_bin"]
    for col in ["qwen_issue_type", "qwen_spec_quality", "qwen_complexity", "qwen_urgency"]:
        if col in pred_df.columns:
            categorical_cols.append(col)

    rows = []
    for bucket, group in pred_df.groupby("fast_boundary_bucket"):
        bucket_n = len(group)
        for col in categorical_cols:
            counts = group[col].fillna("missing").astype(str).value_counts()
            for value, count in counts.items():
                rows.append(
                    {
                        "bucket": bucket,
                        "feature": col,
                        "value": value,
                        "count": int(count),
                        "share_within_bucket": float(count / max(bucket_n, 1)),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["feature", "value", "share_within_bucket"], ascending=[True, True, False]
    )


def fast_feature_lift(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Human-readable association table for the Fast/not-Fast boundary.

    Positive lift means a categorical value appears more often in the target
    error bucket than in the full analyzed population.
    """
    categorical_cols = ["author_association", "text_len_bin", "repo_activity_bin"]
    for col in ["qwen_issue_type", "qwen_spec_quality", "qwen_complexity", "qwen_urgency"]:
        if col in pred_df.columns:
            categorical_cols.append(col)

    target_buckets = [
        "true_fast_pred_not_fast",
        "true_not_fast_pred_fast",
    ]
    rows = []
    for col in categorical_cols:
        base = pred_df[col].fillna("missing").astype(str).value_counts(normalize=True)
        for bucket in target_buckets:
            group = pred_df[pred_df["fast_boundary_bucket"] == bucket]
            if group.empty:
                continue
            shares = group[col].fillna("missing").astype(str).value_counts(normalize=True)
            for value, share in shares.items():
                base_share = float(base.get(value, 0.0))
                rows.append(
                    {
                        "bucket": bucket,
                        "feature": col,
                        "value": value,
                        "bucket_share": float(share),
                        "overall_share": base_share,
                        "share_lift": float(share - base_share),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("share_lift", ascending=False)


def write_summary(args, pred_df, cm, row_norm, per_class, subgroup, pair_df, out_path):
    y_true = pred_df["true_label"].tolist()
    y_pred = pred_df["pred_label"].tolist()
    macro = f1_score(y_true, y_pred, average="macro", labels=LABEL_ORDER, zero_division=0)
    accuracy = float((pred_df["true_label"] == pred_df["pred_label"]).mean())
    worst_classes = per_class.sort_values("f1").head(2)
    worst_groups = subgroup[subgroup["n"] >= 50].head(8) if not subgroup.empty else pd.DataFrame()
    off_diag = pair_df[pair_df["true_label"] != pair_df["pred_label"]].head(8)
    bucket_counts = pred_df["fast_boundary_bucket"].value_counts()

    lines = [
        f"# Error analysis: {args.model_kind}",
        "",
        f"Rows analyzed: {len(pred_df):,}",
        f"Macro-F1: {macro:.4f}",
        f"Accuracy: {accuracy:.4f}",
        "",
        "## Per-class failure modes",
        "",
    ]
    for _, row in worst_classes.iterrows():
        lines.append(
            f"- {row['label']}: F1={row['f1']:.3f}, precision={row['precision']:.3f}, "
            f"recall={row['recall']:.3f}, support={int(row['support']):,}"
        )
    lines.extend(["", "## Largest confusions", ""])
    for _, row in off_diag.iterrows():
        lines.append(
            f"- true {row['true_label']} -> predicted {row['pred_label']}: "
            f"{int(row['count']):,} examples ({100 * row['share_of_true']:.1f}% of true {row['true_label']})"
        )
    lines.extend(["", "## Weakest subgroups with at least 50 rows", ""])
    for _, row in worst_groups.iterrows():
        lines.append(
            f"- {row['group']}={row['value']}: n={int(row['n']):,}, "
            f"macro-F1={row['macro_f1']:.3f}, accuracy={row['accuracy']:.3f}"
        )
    lines.extend(["", "## Fast vs not-Fast boundary", ""])
    for bucket in [
        "true_fast_pred_fast",
        "true_fast_pred_not_fast",
        "true_not_fast_pred_fast",
        "true_not_fast_pred_not_fast",
    ]:
        lines.append(f"- {bucket}: {int(bucket_counts.get(bucket, 0)):,} rows")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `predictions.parquet`: row-level predictions, probabilities, margins, and metadata",
            "- `confusion_counts.csv`: raw confusion matrix",
            "- `confusion_row_normalized.csv`: P(predicted class | true class)",
            "- `per_class_metrics.csv`: precision/recall/F1/support by class",
            "- `subgroup_metrics.csv`: macro-F1/accuracy by interpretable slice",
            "- `fast_numeric_contrasts.csv`: numeric feature summaries by Fast-boundary bucket",
            "- `fast_categorical_contrasts.csv`: categorical feature shares by Fast-boundary bucket",
            "- `fast_feature_lift.csv`: values overrepresented in Fast-boundary error buckets",
            "- `top_confusions.csv`: ranked true/predicted label pairs",
            "- `representative_errors.csv`: high-confidence mistakes for manual inspection",
        ]
    )
    with open(out_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    fs = gcsfs.GCSFileSystem()
    signal_paths = get_parquet_shards(fs, args.signals, sample=args.sample)
    qwen = None
    if args.model_kind == "hybrid" or args.qwen_subset:
        qwen_paths = get_parquet_shards(fs, args.qwen, sample=args.sample)
        signal_paths = restrict_signal_paths_to_qwen(signal_paths, qwen_paths)
        qwen = load_qwen(fs, qwen_paths, include_summary=args.model_kind == "hybrid")
        print(f"Loaded {len(qwen):,} Qwen-covered rows.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = build_model(args.checkpoint).to(device)

    pred_df, groups, pair_counts, example_pool = stream_predictions(
        args, model, tokenizer, fs, signal_paths, qwen, device
    )
    if pred_df.empty:
        raise ValueError("No predictions were produced. Check split dates, shard paths, and Qwen subset.")

    pred_df = add_fast_boundary_bucket(pred_df)
    y_true = pred_df["true_label"].tolist()
    y_pred = pred_df["pred_label"].tolist()
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    row_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_df = pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER)
    row_norm_df = pd.DataFrame(row_norm, index=LABEL_ORDER, columns=LABEL_ORDER)
    per_class = per_class_metrics_df(y_true, y_pred)
    subgroup = subgroup_metrics_df(groups)

    pair_rows = []
    true_counts = Counter(y_true)
    for (true_label, pred_label), count in pair_counts.items():
        pair_rows.append(
            {
                "true_label": true_label,
                "pred_label": pred_label,
                "count": count,
                "share_of_true": count / max(true_counts[true_label], 1),
            }
        )
    pair_df = pd.DataFrame(pair_rows).sort_values("count", ascending=False)
    examples = representative_examples(example_pool, args.max_examples_per_pair)
    fast_numeric = fast_numeric_contrasts(pred_df)
    fast_categorical = fast_categorical_contrasts(pred_df)
    fast_lift = fast_feature_lift(pred_df)

    pred_df.to_parquet(os.path.join(args.output_dir, "predictions.parquet"), index=False)
    cm_df.to_csv(os.path.join(args.output_dir, "confusion_counts.csv"))
    row_norm_df.to_csv(os.path.join(args.output_dir, "confusion_row_normalized.csv"))
    per_class.to_csv(os.path.join(args.output_dir, "per_class_metrics.csv"), index=False)
    subgroup.to_csv(os.path.join(args.output_dir, "subgroup_metrics.csv"), index=False)
    fast_numeric.to_csv(os.path.join(args.output_dir, "fast_numeric_contrasts.csv"), index=False)
    fast_categorical.to_csv(
        os.path.join(args.output_dir, "fast_categorical_contrasts.csv"), index=False
    )
    fast_lift.to_csv(os.path.join(args.output_dir, "fast_feature_lift.csv"), index=False)
    pair_df.to_csv(os.path.join(args.output_dir, "top_confusions.csv"), index=False)
    examples.to_csv(os.path.join(args.output_dir, "representative_errors.csv"), index=False)
    write_summary(
        args,
        pred_df,
        cm_df,
        row_norm_df,
        per_class,
        subgroup,
        pair_df,
        os.path.join(args.output_dir, "summary.md"),
    )

    macro = f1_score(y_true, y_pred, average="macro", labels=LABEL_ORDER, zero_division=0)
    print(f"Analyzed {len(pred_df):,} rows. Macro-F1={macro:.4f}")
    print(f"Wrote artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
