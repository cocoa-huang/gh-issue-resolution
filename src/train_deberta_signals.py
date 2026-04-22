"""
train_deberta_signals.py — Model 4: Fine-tuned DeBERTa text + repo signals.

Features:
  - Raw title + body (DeBERTa-v3-base, CLS via pretrained ContextPooler)
  - 5 log1p-scaled repo signals: pr_merged_30d, avg_merge_hours_30d,
    push_count_30d, release_count_90d, star_count_30d
  - author_association one-hot (5 categories: COLLABORATOR, CONTRIBUTOR,
    MEMBER, NONE, OWNER)
  Combined numeric dim: 10

Architecture:
  DeBERTa backbone → ContextPooler (768) → dropout
  concat with 10-dim signal vector → Linear(778, 4)

  Backbone and pooler are initialised from microsoft/deberta-v3-base.
  Only the classifier head is randomly initialised.

Data source: gs://gh_issue_ml-data/issues/issues_with_signals/*.parquet

Temporal split:
    Train: issue_created_at < 2025-08-01  →  500K stratified sample
    Val:   10% of the sampled train set   →  checkpoint selection
    Test:  2025-08-01 <= issue_created_at < 2025-11-01  →  full streaming eval
    Discard: >= 2025-11-01 (truncated labels)

Usage:
    python3 src/train_deberta_signals.py           # full run
    python3 src/train_deberta_signals.py --sample  # smoke test: 3 shards, 5K train

Outputs:
    results/deberta_signals/          # best checkpoint (state dict + tokenizer)
    results/deberta_signals_eval.txt  # macro-F1, per-class report, confusion matrix
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
    TrainerCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ── Config ─────────────────────────────────────────────────────────────────────

GCS_SIGNALS  = "gs://gh_issue_ml-data/issues/issues_with_signals/"
MODEL_NAME   = "microsoft/deberta-v3-base"
RESULTS_DIR  = "results"
CKPT_DIR     = os.path.join(RESULTS_DIR, "deberta_signals")

LABEL_ORDER  = ["Fast", "Medium", "Slow", "Stale"]
LABEL2ID     = {l: i for i, l in enumerate(LABEL_ORDER)}
ID2LABEL     = {i: l for i, l in enumerate(LABEL_ORDER)}

SIGNAL_COLS  = [
    "pr_merged_30d",
    "avg_merge_hours_30d",
    "push_count_30d",
    "release_count_90d",
    "star_count_30d",
]
AUTHOR_CATS  = ["COLLABORATOR", "CONTRIBUTOR", "MEMBER", "NONE", "OWNER"]
NUM_SIGNALS  = len(SIGNAL_COLS) + len(AUTHOR_CATS)  # 10

TRAIN_CUTOFF = pd.Timestamp("2025-08-01", tz="UTC")
TEST_CUTOFF  = pd.Timestamp("2025-11-01", tz="UTC")

TRAIN_SAMPLE  = 500_000
MAX_LEN       = 512
TRAIN_BATCH   = 16
EVAL_BATCH    = 32
GRAD_ACCUM    = 4           # effective batch = 64
EPOCHS        = 3
LR            = 2e-5
WARMUP_RATIO  = 0.06
WEIGHT_DECAY  = 0.01
EVAL_STEPS    = 2_000
SAVE_STEPS    = 2_000
FP16          = torch.cuda.is_available()
INF_BATCH     = 64

LOAD_COLS = ["title", "body", "issue_created_at", "label", "author_association"] + SIGNAL_COLS

# ── Data loading ───────────────────────────────────────────────────────────────

def get_shards(sample: bool = False):
    fs = gcsfs.GCSFileSystem()
    paths = sorted(fs.glob(GCS_SIGNALS.rstrip("/") + "/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquets found at {GCS_SIGNALS}")
    if sample:
        paths = paths[:3]
        print(f"  [sample mode] {len(paths)} shards")
    else:
        print(f"  {len(paths)} shards")
    return fs, paths


def load_shard(fs, path: str) -> pd.DataFrame:
    df = pd.read_parquet(fs.open(path), columns=LOAD_COLS)
    df["created_at"] = pd.to_datetime(df["issue_created_at"], utc=True)
    df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")
    df["author_association"] = df["author_association"].fillna("NONE").str.upper()
    for col in SIGNAL_COLS:
        df[col] = df[col].fillna(0.0)
    return df


def encode_signals(df: pd.DataFrame) -> np.ndarray:
    numeric  = np.log1p(df[SIGNAL_COLS].values.astype(np.float32))
    author   = df["author_association"].values
    auth_oh  = np.zeros((len(df), len(AUTHOR_CATS)), dtype=np.float32)
    for i, cat in enumerate(AUTHOR_CATS):
        auth_oh[:, i] = (author == cat).astype(np.float32)
    return np.concatenate([numeric, auth_oh], axis=1)  # (N, 10)

# ── Two-pass stratified sampling ───────────────────────────────────────────────

def collect_train_sample(fs, paths, sample_mode: bool) -> pd.DataFrame:
    target = 5_000 if sample_mode else TRAIN_SAMPLE
    rng = np.random.default_rng(42)

    print("Pass 1: counting train labels...")
    label_counts: Counter = Counter()
    for p in paths:
        df = load_shard(fs, p)
        label_counts.update(df[df["created_at"] < TRAIN_CUTOFF]["label"].tolist())

    total = sum(label_counts.values())
    print(f"  Total training rows: {total:,}")
    rates = {lbl: min(1.0, (target * cnt / total) / max(cnt, 1))
             for lbl, cnt in label_counts.items()}
    print(f"  Sampling rates: { {k: round(v, 4) for k, v in rates.items()} }")

    print("Pass 2: sampling training rows...")
    frames = []
    for p in paths:
        df = load_shard(fs, p)
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
                parts.append(rows)
            else:
                mask = rng.random(len(rows)) < rate
                parts.append(rows[mask])
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

# ── Model ──────────────────────────────────────────────────────────────────────

class DeBERTaWithSignals(nn.Module):
    """
    DeBERTa-v3-base backbone with signal concatenation.

    Takes the pretrained backbone and pooler from
    AutoModelForSequenceClassification, discards its classifier, and adds a
    new linear head that operates on [pooler_out | signals].
    """

    def __init__(self, backbone, pooler, pooler_dim: int, num_signals: int, num_labels: int):
        super().__init__()
        self.deberta    = backbone
        self.pooler     = pooler
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Linear(pooler_dim + num_signals, num_labels)
        self.num_signals = num_signals

    def forward(self, input_ids=None, attention_mask=None, signals=None,
                labels=None, **kwargs):
        hidden = self.deberta(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        pooled  = self.pooler(hidden)
        pooled  = self.dropout(pooled)
        if signals is not None:
            combined = torch.cat([pooled, signals.to(pooled.device, pooled.dtype)], dim=1)
        else:
            zeros    = torch.zeros(pooled.size(0), self.num_signals,
                                   device=pooled.device, dtype=pooled.dtype)
            combined = torch.cat([pooled, zeros], dim=1)
        logits = self.classifier(combined)
        return SequenceClassifierOutput(logits=logits)

# ── Dataset ────────────────────────────────────────────────────────────────────

class IssueSignalDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int],
                 signals: np.ndarray, tokenizer):
        print(f"  Tokenizing {len(texts):,} texts...")
        self.encodings = tokenizer(
            texts,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels  = torch.tensor(labels,  dtype=torch.long)
        self.signals = torch.tensor(signals, dtype=torch.float32)
        print("  Done.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "signals":        self.signals[idx],
            "labels":         self.labels[idx],
        }

# ── Weighted-loss Trainer ──────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        loss    = nn.CrossEntropyLoss(
            weight=self._class_weights.to(outputs.logits.device)
        )(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# ── Metrics ────────────────────────────────────────────────────────────────────

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

# ── SLURM-compatible progress callback ────────────────────────────────────────

class SlurmProgressCallback(TrainerCallback):
    def __init__(self, total_steps: int, log_every: int = 200):
        self.total    = total_steps
        self.log_every = log_every

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or state.global_step % self.log_every != 0:
            return
        pct   = 100 * state.global_step / max(self.total, 1)
        loss  = logs.get("loss", "?")
        epoch = round(logs.get("epoch", 0), 1)
        print(
            f"[Step {state.global_step}/{self.total} | "
            f"Epoch {epoch}/{args.num_train_epochs} | "
            f"{pct:.1f}% | loss={loss}]",
            flush=True,
        )

# ── Streaming test eval ────────────────────────────────────────────────────────

def stream_test_eval(model, tokenizer, fs, paths, device):
    all_preds, all_true = [], []
    model.eval()
    n_processed = 0

    for p in paths:
        df   = load_shard(fs, p)
        test = df[
            (df["created_at"] >= TRAIN_CUTOFF) & (df["created_at"] < TEST_CUTOFF)
        ].reset_index(drop=True)
        if test.empty:
            continue

        sig_arr = encode_signals(test)

        for start in range(0, len(test), INF_BATCH):
            end     = min(start + INF_BATCH, len(test))
            batch   = test.iloc[start:end]
            texts   = batch["text"].tolist()
            signals = sig_arr[start:end]
            labels  = [LABEL2ID[l] for l in batch["label"]]

            enc = tokenizer(
                texts,
                max_length=MAX_LEN,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            sig_t = torch.tensor(signals, dtype=torch.float32).to(device)

            with torch.no_grad():
                out = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    signals=sig_t,
                )
            all_preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
            all_true.extend(labels)
            n_processed += len(texts)
            if n_processed % 50_000 == 0:
                print(f"  Inferred {n_processed:,} test rows...", flush=True)

    return all_preds, all_true

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    sample = "--sample" in sys.argv
    os.makedirs(CKPT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  fp16: {FP16}")

    fs, paths = get_shards(sample=sample)

    # ── 1. Collect data ───────────────────────────────────────────────────────
    train_df = collect_train_sample(fs, paths, sample_mode=sample)

    n_val = max(500, len(train_df) // 10)
    n_tr  = len(train_df) - n_val
    tr_df  = train_df.iloc[:n_tr].reset_index(drop=True)
    val_df = train_df.iloc[n_tr:].reset_index(drop=True)
    print(f"\nTrain: {n_tr:,}  |  Val: {n_val:,}")

    # ── 2. Class weights ──────────────────────────────────────────────────────
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

    print("\nEncoding signals...")
    tr_signals  = encode_signals(tr_df)
    val_signals = encode_signals(val_df)

    print("Building train dataset...")
    train_dataset = IssueSignalDataset(
        tr_df["text"].tolist(),
        [LABEL2ID[l] for l in tr_df["label"]],
        tr_signals,
        tokenizer,
    )
    print("Building val dataset...")
    val_dataset = IssueSignalDataset(
        val_df["text"].tolist(),
        [LABEL2ID[l] for l in val_df["label"]],
        val_signals,
        tokenizer,
    )

    # ── 4. Model: load pretrained backbone + pooler, new classifier head ──────
    print(f"\nLoading pretrained model: {MODEL_NAME}")
    pretrained  = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_ORDER),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    pooler_dim = getattr(pretrained.pooler, "output_dim", None) or pretrained.config.pooler_hidden_size
    model = DeBERTaWithSignals(
        backbone    = pretrained.deberta,
        pooler      = pretrained.pooler,
        pooler_dim  = pooler_dim,
        num_signals = NUM_SIGNALS,
        num_labels  = len(LABEL_ORDER),
    )
    del pretrained
    print(f"  Pooler output dim: {pooler_dim}  |  Signal dim: {NUM_SIGNALS}  |  Classifier in: {pooler_dim + NUM_SIGNALS}")

    # ── 5. Training ───────────────────────────────────────────────────────────
    steps_per_epoch = max(1, n_tr // (TRAIN_BATCH * GRAD_ACCUM))
    total_steps     = steps_per_epoch * EPOCHS
    print(f"\nSteps/epoch ≈ {steps_per_epoch:,}  |  Total steps ≈ {total_steps:,}")

    training_args = TrainingArguments(
        output_dir                  = CKPT_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = TRAIN_BATCH,
        per_device_eval_batch_size  = EVAL_BATCH,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LR,
        warmup_ratio                = WARMUP_RATIO,
        weight_decay                = WEIGHT_DECAY,
        fp16                        = FP16,
        eval_strategy               = "steps",
        eval_steps                  = EVAL_STEPS,
        save_strategy               = "steps",
        save_steps                  = SAVE_STEPS,
        load_best_model_at_end      = True,
        metric_for_best_model       = "macro_f1",
        greater_is_better           = True,
        logging_steps               = 200,
        report_to                   = "none",
        save_total_limit            = 2,
        dataloader_num_workers      = 4,
    )

    trainer = WeightedTrainer(
        class_weights   = class_weights,
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        compute_metrics = compute_metrics,
        callbacks       = [SlurmProgressCallback(total_steps)],
    )

    print("\nStarting training...")
    trainer.train()
    print("Training complete. Saving best checkpoint...")
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "pytorch_model.bin"))
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
        f"Model 4 — Fine-tuned DeBERTa text + repo signals ({MODEL_NAME})\n"
        f"Temporal split: train < 2025-08-01  |  test 2025-08-01 – 2025-10-31\n"
        f"Train: {n_tr:,} (stratified sample)  |  Val: {n_val:,}  |  Test: {len(all_true):,} (full)\n"
        f"Epochs: {EPOCHS}  |  Effective batch: {TRAIN_BATCH * GRAD_ACCUM}  |  LR: {LR}  |  fp16: {FP16}\n"
        f"Signal features ({NUM_SIGNALS}): {SIGNAL_COLS} + author_association one-hot\n"
        f"{'='*60}\n"
        f"Macro-F1: {macro_f1:.4f}\n\n"
        f"{report}\n"
        f"Confusion matrix (rows=true, cols=pred)\n"
        f"Order: {LABEL_ORDER}\n{cm}\n"
    )
    print(output)

    eval_path = os.path.join(RESULTS_DIR, "deberta_signals_eval.txt")
    with open(eval_path, "w") as f:
        f.write(output)
    print(f"Eval written to {eval_path}")


if __name__ == "__main__":
    main()
