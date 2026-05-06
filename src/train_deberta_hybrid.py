"""
train_deberta_hybrid.py - Model 5: Qwen summary + raw issue text + repo signals.

Warm-starts from the saved Model 4 DeBERTa+signals checkpoint, then fine-tunes
on inputs shaped as:
  [SUMMARY: {qwen_summary}] [ISSUE: {title} {body}]

Data source: gs://gh_issue_ml-data/issues/issues_with_signals/*.parquet
Summary cache: gs://gh_issue_ml-data/llm_features/qwen_summaries/*.parquet
"""

import argparse
import faulthandler
import hashlib
import os
import re
import signal
import sys
from collections import Counter

import gcsfs
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

GCS_SIGNALS = "gs://gh_issue_ml-data/issues/issues_with_signals/"
GCS_QWEN = "gs://gh_issue_ml-data/llm_features/qwen_summaries/"
MODEL_NAME = "microsoft/deberta-v3-base"
RESULTS_DIR = "results"
CKPT_DIR = os.path.join(RESULTS_DIR, "deberta_hybrid")
DEFAULT_WARM_START = os.environ.get(
    "MODEL4_CHECKPOINT",
    os.path.join(RESULTS_DIR, "deberta_signals", "pytorch_model.bin"),
)

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

TRAIN_SAMPLE = 500_000
MAX_LEN = 512
TRAIN_BATCH = 16
EVAL_BATCH = 32
GRAD_ACCUM = 4
EPOCHS = 3
LR = 2e-5
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
EVAL_STEPS = 2_000
SAVE_STEPS = 2_000
FP16 = torch.cuda.is_available()
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
    parser.add_argument("--sample", action="store_true", help="Smoke test: 3 shards, 5K train.")
    parser.add_argument("--signals", default=GCS_SIGNALS)
    parser.add_argument("--qwen", default=GCS_QWEN)
    parser.add_argument("--warm-start", default=DEFAULT_WARM_START)
    parser.add_argument(
        "--allow-missing-warm-start",
        action="store_true",
        help="Only for local smoke tests; full Model 5 should use the Model 4 checkpoint.",
    )
    return parser.parse_args()


def dump_stack_on_signal(signum, frame):
    print(f"\nReceived signal {signum}; dumping Python stack before exit.", file=sys.stderr, flush=True)
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    raise SystemExit(128 + signum)


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


def build_hybrid_text(df: pd.DataFrame) -> pd.Series:
    summaries = df["qwen_summary"].fillna("").astype(str)
    issue_text = df["title"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)
    return "[SUMMARY: " + summaries + "] [ISSUE: " + issue_text + "]"


def get_shards(fs, base_path: str, sample: bool = False):
    paths = sorted(fs.glob(base_path.rstrip("/") + "/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquets found at {base_path}")
    if sample:
        paths = paths[:3]
        print(f"  [sample mode] {len(paths)} shards")
    else:
        print(f"  {len(paths)} shards")
    return paths


def get_qwen_shards(fs, base_path: str, sample: bool = False):
    paths = sorted(fs.glob(base_path.rstrip("/") + "/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Qwen summary parquets found at {base_path}")
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
        print(
            "  Qwen shard indices do not match signal shard list; scanning all signal shards."
        )
        return paths
    selected = [paths[i] for i in sorted(indices)]
    print(
        f"  Restricting signal scan to {len(selected)}/{len(paths)} source shards "
        f"with Qwen summaries"
    )
    return selected


def load_qwen_summaries(fs, qwen_paths) -> pd.DataFrame:
    frames = [
        pd.read_parquet(fs.open(p), columns=["issue_key", "qwen_summary"])
        for p in qwen_paths
    ]
    qwen = pd.concat(frames, ignore_index=True).drop_duplicates("issue_key")
    qwen = qwen[qwen["qwen_summary"].fillna("").astype(str).str.len() > 0]
    print(f"  Loaded {len(qwen):,} non-empty Qwen summaries")
    return qwen


def load_shard(fs, path: str) -> pd.DataFrame:
    df = pd.read_parquet(fs.open(path), columns=LOAD_COLS)
    df["created_at"] = pd.to_datetime(df["issue_created_at"], utc=True)
    df["author_association"] = df["author_association"].fillna("NONE").str.upper()
    for col in SIGNAL_COLS:
        df[col] = df[col].fillna(0.0)
    return add_issue_keys(df)


def attach_summaries(df: pd.DataFrame, qwen: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    merged = df.merge(qwen, on="issue_key", how="inner")
    missing = before - len(merged)
    if missing:
        print(f"  Dropped {missing:,}/{before:,} rows without Qwen summaries")
    merged["text"] = build_hybrid_text(merged)
    return merged


def encode_signals(df: pd.DataFrame) -> np.ndarray:
    numeric = np.log1p(df[SIGNAL_COLS].values.astype(np.float32))
    author = df["author_association"].values
    auth_oh = np.zeros((len(df), len(AUTHOR_CATS)), dtype=np.float32)
    for i, cat in enumerate(AUTHOR_CATS):
        auth_oh[:, i] = (author == cat).astype(np.float32)
    return np.concatenate([numeric, auth_oh], axis=1)


def collect_train_sample(fs, paths, qwen: pd.DataFrame, sample_mode: bool) -> pd.DataFrame:
    target = 5_000 if sample_mode else TRAIN_SAMPLE
    rng = np.random.default_rng(42)

    print("Pass 1: counting train labels with Qwen summaries...")
    label_counts = Counter()
    for p in paths:
        df = attach_summaries(load_shard(fs, p), qwen)
        label_counts.update(df[df["created_at"] < TRAIN_CUTOFF]["label"].tolist())

    total = sum(label_counts.values())
    if total == 0:
        raise ValueError("No training rows remain after joining Qwen summaries.")
    print(f"  Total joinable training rows: {total:,}")
    rates = {
        lbl: min(1.0, (target * cnt / total) / max(cnt, 1))
        for lbl, cnt in label_counts.items()
    }
    print(f"  Sampling rates: { {k: round(v, 4) for k, v in rates.items()} }")

    print("Pass 2: sampling training rows...")
    frames = []
    for p in paths:
        df = attach_summaries(load_shard(fs, p), qwen)
        train = df[df["created_at"] < TRAIN_CUTOFF]
        if train.empty:
            continue
        parts = []
        for lbl in LABEL_ORDER:
            rows = train[train["label"] == lbl]
            if rows.empty:
                continue
            rate = rates.get(lbl, 0.0)
            parts.append(rows if rate >= 1.0 else rows[rng.random(len(rows)) < rate])
        if parts:
            frames.append(pd.concat(parts))

    if not frames:
        raise ValueError("No sampled training rows remain after joining Qwen summaries.")
    result = pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Sampled {len(result):,} training rows")
    for lbl in LABEL_ORDER:
        n = (result["label"] == lbl).sum()
        print(f"    {lbl}: {n:,}  ({100*n/len(result):.1f}%)")
    return result


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


class IssueSignalDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], signals: np.ndarray, tokenizer):
        print(f"  Tokenizing {len(texts):,} texts...")
        self.encodings = tokenizer(
            texts,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.signals = torch.tensor(signals, dtype=torch.float32)
        print("  Done.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "signals": self.signals[idx],
            "labels": self.labels[idx],
        }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=self._class_weights.to(outputs.logits.device))(
            outputs.logits, labels
        )
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(
            labels, preds, average="macro", labels=list(range(len(LABEL_ORDER)))
        )
    }


class SlurmProgressCallback(TrainerCallback):
    def __init__(self, total_steps: int, log_every: int = 200):
        self.total = total_steps
        self.log_every = log_every

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or state.global_step % self.log_every != 0:
            return
        pct = 100 * state.global_step / max(self.total, 1)
        loss = logs.get("loss", "?")
        epoch = round(logs.get("epoch", 0), 1)
        print(
            f"[Step {state.global_step}/{self.total} | "
            f"Epoch {epoch}/{args.num_train_epochs} | "
            f"{pct:.1f}% | loss={loss}]",
            flush=True,
        )


def build_model(warm_start: str, allow_missing_warm_start: bool):
    print(f"\nLoading pretrained model shell: {MODEL_NAME}")
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

    if os.path.exists(warm_start):
        print(f"Warm-starting from Model 4 checkpoint: {warm_start}")
        state = torch.load(warm_start, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Missing keys: {len(missing)}  |  Unexpected keys: {len(unexpected)}")
    elif allow_missing_warm_start:
        print(f"WARNING: missing warm-start checkpoint {warm_start}; continuing for smoke test only.")
    else:
        raise FileNotFoundError(
            f"Model 5 requires a saved Model 4 checkpoint. Pass --warm-start or set "
            f"MODEL4_CHECKPOINT. Missing: {warm_start}"
        )

    print(
        f"  Pooler output dim: {pooler_dim}  |  Signal dim: {NUM_SIGNALS}  |  "
        f"Classifier in: {pooler_dim + NUM_SIGNALS}"
    )
    return model


def stream_test_eval(model, tokenizer, fs, paths, qwen: pd.DataFrame, device):
    all_preds, all_true = [], []
    model.eval()
    n_processed = 0

    for p in paths:
        df = attach_summaries(load_shard(fs, p), qwen)
        test = df[
            (df["created_at"] >= TRAIN_CUTOFF) & (df["created_at"] < TEST_CUTOFF)
        ].reset_index(drop=True)
        if test.empty:
            continue
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
            if n_processed % 50_000 == 0:
                print(f"  Inferred {n_processed:,} test rows...", flush=True)
    return all_preds, all_true


def main():
    signal.signal(signal.SIGTERM, dump_stack_on_signal)
    faulthandler.enable(file=sys.stderr, all_threads=True)

    args = parse_args()
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  fp16: {FP16}")

    fs = gcsfs.GCSFileSystem()
    paths = get_shards(fs, args.signals, sample=args.sample)

    print("\nLoading Qwen summaries...")
    qwen_paths = get_qwen_shards(fs, args.qwen, sample=args.sample)
    paths = restrict_signal_paths_to_qwen(paths, qwen_paths)
    qwen = load_qwen_summaries(fs, qwen_paths)
    sample_summaries = qwen["qwen_summary"].head(20).to_frame()
    sample_summaries.to_csv(os.path.join(RESULTS_DIR, "qwen_summary_examples.csv"), index=False)

    train_df = collect_train_sample(fs, paths, qwen, sample_mode=args.sample)

    n_val = min(max(500, len(train_df) // 10), max(1, len(train_df) - 1))
    n_tr = len(train_df) - n_val
    tr_df = train_df.iloc[:n_tr].reset_index(drop=True)
    val_df = train_df.iloc[n_tr:].reset_index(drop=True)
    print(f"\nTrain: {n_tr:,}  |  Val: {n_val:,}")

    cw = compute_class_weight("balanced", classes=np.array(LABEL_ORDER), y=tr_df["label"].values)
    class_weights = torch.tensor(cw, dtype=torch.float32)
    print(f"Class weights: { {l: round(w, 3) for l, w in zip(LABEL_ORDER, cw)} }")

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\nEncoding signals...")
    tr_signals = encode_signals(tr_df)
    val_signals = encode_signals(val_df)

    print("Building train dataset...")
    train_dataset = IssueSignalDataset(
        tr_df["text"].tolist(), [LABEL2ID[l] for l in tr_df["label"]], tr_signals, tokenizer
    )
    print("Building val dataset...")
    val_dataset = IssueSignalDataset(
        val_df["text"].tolist(), [LABEL2ID[l] for l in val_df["label"]], val_signals, tokenizer
    )

    model = build_model(args.warm_start, args.allow_missing_warm_start)

    steps_per_epoch = max(1, n_tr // (TRAIN_BATCH * GRAD_ACCUM))
    total_steps = steps_per_epoch * EPOCHS
    print(f"\nSteps/epoch ~= {steps_per_epoch:,}  |  Total steps ~= {total_steps:,}")

    training_args = TrainingArguments(
        output_dir=CKPT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=FP16,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=200,
        report_to="none",
        save_total_limit=2,
        dataloader_num_workers=4,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[SlurmProgressCallback(total_steps)],
    )

    print("\nStarting hybrid fine-tuning...")
    trainer.train()
    print("Training complete. Saving best checkpoint...")
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "pytorch_model.bin"))
    tokenizer.save_pretrained(CKPT_DIR)

    print(f"\nEvaluating on joined full test split (streaming, batch={INF_BATCH})...")
    model.to(device)
    all_preds, all_true = stream_test_eval(model, tokenizer, fs, paths, qwen, device)
    print(f"  Total test rows evaluated: {len(all_true):,}")

    pred_labels = [ID2LABEL[p] for p in all_preds]
    true_labels = [ID2LABEL[l] for l in all_true]
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", labels=LABEL_ORDER)
    report = classification_report(true_labels, pred_labels, labels=LABEL_ORDER, digits=3)
    cm = confusion_matrix(true_labels, pred_labels, labels=LABEL_ORDER)

    output = (
        f"Model 5 - Qwen summary + raw issue text + repo signals ({MODEL_NAME})\n"
        f"Warm start: saved Model 4 DeBERTa+signals checkpoint ({args.warm_start})\n"
        f"Comparison baseline: Model 4 raw issue text + repo signals macro-F1 = 0.361\n"
        f"Temporal split: train < 2025-08-01  |  test 2025-08-01 - 2025-10-31\n"
        f"Train: {n_tr:,} (stratified sample)  |  Val: {n_val:,}  |  Test: {len(all_true):,} (joined full)\n"
        f"Input: [SUMMARY: qwen_summary] [ISSUE: title body]\n"
        f"Signal features ({NUM_SIGNALS}): {SIGNAL_COLS} + author_association one-hot\n"
        f"Epochs: {EPOCHS}  |  Effective batch: {TRAIN_BATCH * GRAD_ACCUM}  |  LR: {LR}  |  fp16: {FP16}\n"
        f"{'='*60}\n"
        f"Macro-F1: {macro_f1:.4f}\n\n"
        f"{report}\n"
        f"Confusion matrix (rows=true, cols=pred)\n"
        f"Order: {LABEL_ORDER}\n{cm}\n"
    )
    print(output)

    eval_path = os.path.join(RESULTS_DIR, "deberta_hybrid_eval.txt")
    with open(eval_path, "w") as f:
        f.write(output)
    print(f"Eval written to {eval_path}")


if __name__ == "__main__":
    main()
