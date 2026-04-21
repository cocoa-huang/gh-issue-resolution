"""
train_deberta_text.py — Model 3: Fine-tuned DeBERTa text-only.

Features: Raw title + body (truncated to 512 tokens)
Metric:   macro-F1

Data collection is a two-pass process: pass 1 counts per-label totals to
derive stratified sampling probabilities; pass 2 streams each shard and
samples probabilistically so the full dataset is never loaded into memory.
Training uses 500K examples (stratified), which is ~30% of the temporal
training split and more than sufficient for fine-tuning a pre-trained encoder.

Test evaluation streams the full Aug–Oct 2025 split (~1.35M rows) through
the model in batches so both LR and DeBERTa results are comparable on
identical data.

Temporal split:
    Train: issue_created_at < 2025-08-01  →  500K stratified sample
    Val:   10% of the sampled train set   →  used for checkpoint selection
    Test:  2025-08-01 <= issue_created_at < 2025-11-01  →  full streaming eval
    Discard: >= 2025-11-01 (truncated labels)

Usage:
    python3 src/train_deberta_text.py           # full run
    python3 src/train_deberta_text.py --sample  # smoke test: 3 shards, 5K train

Outputs:
    results/deberta_text/          # best HuggingFace checkpoint
    results/deberta_text_eval.txt  # macro-F1 + per-class report + confusion matrix
"""

import os
import sys
from collections import Counter

import gcsfs
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ── Config ────────────────────────────────────────────────────────────────────

GCS_ISSUES   = "gs://gh_issue_ml-data/issues/issues_labeled_2025/"
MODEL_NAME   = "microsoft/deberta-v3-base"
RESULTS_DIR  = "results"
CKPT_DIR     = os.path.join(RESULTS_DIR, "deberta_text")

LABEL_ORDER  = ["Fast", "Medium", "Slow", "Stale"]
LABEL2ID     = {l: i for i, l in enumerate(LABEL_ORDER)}
ID2LABEL     = {i: l for i, l in enumerate(LABEL_ORDER)}

TRAIN_CUTOFF = pd.Timestamp("2025-08-01", tz="UTC")
TEST_CUTOFF  = pd.Timestamp("2025-11-01", tz="UTC")

TRAIN_SAMPLE   = 500_000   # stratified sample from training split
MAX_LEN        = 512
TRAIN_BATCH    = 16
EVAL_BATCH     = 32
GRAD_ACCUM     = 4         # effective batch size = 64
EPOCHS         = 3
LR             = 2e-5
WARMUP_RATIO   = 0.06
WEIGHT_DECAY   = 0.01
EVAL_STEPS     = 2_000
SAVE_STEPS     = 2_000
FP16           = torch.cuda.is_available()
INF_BATCH      = 64        # inference batch for streaming test eval

# ── Data loading ──────────────────────────────────────────────────────────────

def get_shards(sample: bool = False):
    fs = gcsfs.GCSFileSystem()
    paths = sorted(fs.glob(GCS_ISSUES.rstrip("/") + "/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquets found at {GCS_ISSUES}")
    if sample:
        paths = paths[:3]
        print(f"  [sample mode] {len(paths)} shards")
    else:
        print(f"  {len(paths)} shards")
    return fs, paths


def load_shard_text(fs, path: str) -> pd.DataFrame:
    df = pd.read_parquet(
        fs.open(path),
        columns=["title", "body", "issue_created_at", "label"],
    )
    df["created_at"] = pd.to_datetime(df["issue_created_at"], utc=True)
    df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")
    return df[["text", "created_at", "label"]]


def collect_train_sample(fs, paths, sample_mode: bool) -> pd.DataFrame:
    """
    Two-pass stratified sampling that never loads the full training set.

    Pass 1: count per-label totals to derive per-label sampling probabilities.
    Pass 2: stream each shard, sample each class at its computed rate.
    """
    target = 5_000 if sample_mode else TRAIN_SAMPLE
    rng = np.random.default_rng(42)

    # Pass 1 — count
    print("Pass 1: counting train labels...")
    label_counts: Counter = Counter()
    for p in paths:
        df = load_shard_text(fs, p)
        label_counts.update(df[df["created_at"] < TRAIN_CUTOFF]["label"].tolist())

    total = sum(label_counts.values())
    print(f"  Total training rows: {total:,}")
    rates = {lbl: min(1.0, (target * cnt / total) / max(cnt, 1))
             for lbl, cnt in label_counts.items()}
    print(f"  Sampling rates: { {k: round(v, 4) for k, v in rates.items()} }")

    # Pass 2 — sample
    print("Pass 2: sampling training rows...")
    frames = []
    for p in paths:
        df = load_shard_text(fs, p)
        train = df[df["created_at"] < TRAIN_CUTOFF]
        if train.empty:
            continue
        parts = []
        for lbl in LABEL_ORDER:
            rows = train[train["label"] == lbl]
            if rows.empty:
                continue
            rate = rates.get(lbl, 0.0)
            if rate >= 1.0:
                parts.append(rows[["text", "label"]])
            else:
                mask = rng.random(len(rows)) < rate
                parts.append(rows[mask][["text", "label"]])
        if parts:
            frames.append(pd.concat(parts))

    result = (
        pd.concat(frames)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    print(f"  Sampled {len(result):,} training rows")
    for lbl in LABEL_ORDER:
        n = (result["label"] == lbl).sum()
        print(f"    {lbl}: {n:,}  ({100*n/len(result):.1f}%)")
    return result


def stream_test_eval(model, tokenizer, fs, paths, device) -> tuple[list, list]:
    """
    Stream the full test split through the model in fixed-size batches.
    Tokenizes on the fly to avoid allocating a 1.35M-row tensor upfront.
    """
    all_preds, all_true = [], []
    model.eval()
    buf_texts, buf_labels = [], []
    n_processed = 0

    def flush(texts, labels):
        enc = tokenizer(
            texts,
            max_length=MAX_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_true.extend(labels)

    for p in paths:
        df = load_shard_text(fs, p)
        test = df[
            (df["created_at"] >= TRAIN_CUTOFF) & (df["created_at"] < TEST_CUTOFF)
        ]
        if test.empty:
            continue
        for _, row in test.iterrows():
            buf_texts.append(row["text"])
            buf_labels.append(LABEL2ID[row["label"]])
            if len(buf_texts) == INF_BATCH:
                flush(buf_texts, buf_labels)
                buf_texts, buf_labels = [], []
                n_processed += INF_BATCH
                if n_processed % 50_000 == 0:
                    print(f"  Inferred {n_processed:,} test rows...")

    if buf_texts:
        flush(buf_texts, buf_labels)

    return all_preds, all_true

# ── Dataset ───────────────────────────────────────────────────────────────────

class IssueDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer):
        print(f"  Tokenizing {len(texts):,} texts...")
        self.encodings = tokenizer(
            texts,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
        print("  Done.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }

# ── Weighted-loss Trainer ─────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(
            weight=self._class_weights.to(outputs.logits.device)
        )(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(
            labels, preds,
            average="macro",
            labels=list(range(len(LABEL_ORDER))),
        )
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sample = "--sample" in sys.argv
    os.makedirs(CKPT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  fp16: {FP16}")

    fs, paths = get_shards(sample=sample)

    # ── 1. Collect data ───────────────────────────────────────────────────────
    train_df = collect_train_sample(fs, paths, sample_mode=sample)

    # Val split: last 10% of the shuffled sample (random_state=42 above)
    n_val = max(500, len(train_df) // 10)
    n_tr  = len(train_df) - n_val
    tr_df  = train_df.iloc[:n_tr].reset_index(drop=True)
    val_df = train_df.iloc[n_tr:].reset_index(drop=True)
    print(f"\nTrain: {n_tr:,}  |  Val: {n_val:,}")

    # ── 2. Class weights from training portion ─────────────────────────────────
    cw = compute_class_weight(
        "balanced",
        classes=np.array(LABEL_ORDER),
        y=tr_df["label"].values,
    )
    class_weights = torch.tensor(cw, dtype=torch.float32)
    print(f"Class weights: { {l: round(w, 3) for l, w in zip(LABEL_ORDER, cw)} }")

    # ── 3. Tokenizer + datasets ───────────────────────────────────────────────
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\nBuilding train dataset...")
    train_dataset = IssueDataset(
        tr_df["text"].tolist(),
        [LABEL2ID[l] for l in tr_df["label"]],
        tokenizer,
    )
    print("Building val dataset...")
    val_dataset = IssueDataset(
        val_df["text"].tolist(),
        [LABEL2ID[l] for l in val_df["label"]],
        tokenizer,
    )

    # ── 4. Model ──────────────────────────────────────────────────────────────
    print(f"\nLoading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_ORDER),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # ── 5. Training ───────────────────────────────────────────────────────────
    steps_per_epoch = max(1, n_tr // (TRAIN_BATCH * GRAD_ACCUM))
    print(f"\nSteps/epoch ≈ {steps_per_epoch:,}  |  Total steps ≈ {steps_per_epoch * EPOCHS:,}")

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
        tokenizer=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()
    print("Training complete. Saving best checkpoint...")
    trainer.save_model(CKPT_DIR)
    tokenizer.save_pretrained(CKPT_DIR)

    # ── 6. Full test evaluation (streaming) ───────────────────────────────────
    print(f"\nEvaluating on full test split (streaming, batch={INF_BATCH})...")
    model.to(device)
    all_preds, all_true = stream_test_eval(model, tokenizer, fs, paths, device)
    print(f"  Total test rows evaluated: {len(all_true):,}")

    pred_labels = [ID2LABEL[p] for p in all_preds]
    true_labels = [ID2LABEL[l] for l in all_true]

    macro_f1 = f1_score(true_labels, pred_labels, average="macro", labels=LABEL_ORDER)
    report   = classification_report(true_labels, pred_labels, labels=LABEL_ORDER, digits=3)
    cm       = confusion_matrix(true_labels, pred_labels, labels=LABEL_ORDER)

    output = (
        f"Model 3 — Fine-tuned DeBERTa text-only ({MODEL_NAME})\n"
        f"Temporal split: train < 2025-08-01  |  test 2025-08-01 – 2025-10-31\n"
        f"Train: {n_tr:,} (stratified sample)  |  Val: {n_val:,}  |  Test: {len(all_true):,} (full)\n"
        f"Epochs: {EPOCHS}  |  Effective batch: {TRAIN_BATCH * GRAD_ACCUM}  |  LR: {LR}  |  fp16: {FP16}\n"
        f"{'='*60}\n"
        f"Macro-F1: {macro_f1:.4f}\n\n"
        f"{report}\n"
        f"Confusion matrix (rows=true, cols=pred)\n"
        f"Order: {LABEL_ORDER}\n{cm}\n"
    )
    print(output)

    eval_path = os.path.join(RESULTS_DIR, "deberta_text_eval.txt")
    with open(eval_path, "w") as f:
        f.write(output)
    print(f"Eval written to {eval_path}")


if __name__ == "__main__":
    main()
