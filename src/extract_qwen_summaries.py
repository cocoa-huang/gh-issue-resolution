"""
extract_qwen_summaries.py - Generate Model 5 Qwen semantic summaries.

Reads issues_with_signals parquet shards from GCS, sends only title/body to
Qwen, and writes cached summaries keyed by a deterministic issue hash.

Output columns:
  issue_key, repo_name, issue_created_at, qwen_summary

Usage:
  python3 src/extract_qwen_summaries.py
  python3 src/extract_qwen_summaries.py --sample --max-issues 64
  python3 src/extract_qwen_summaries.py --max-train-issues 40000 --max-test-issues 10000 --overwrite --clear-output
"""

import argparse
import hashlib
from typing import Optional

import gcsfs
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GCS_SIGNALS = "gs://gh_issue_ml-data/issues/issues_with_signals/"
GCS_OUTPUT = "gs://gh_issue_ml-data/llm_features/qwen_summaries/"
MODEL_NAME = "Qwen/Qwen3.5-2B"

TRAIN_CUTOFF = pd.Timestamp("2025-08-01", tz="UTC")
TEST_CUTOFF = pd.Timestamp("2025-11-01", tz="UTC")

LOAD_COLS = ["repo_name", "title", "body", "issue_created_at"]

PROMPT = """Analyze this GitHub issue using only the title and body. Extract filing-time
semantic attributes that could affect maintainer triage speed. Do not predict a
resolution-time class, do not use labels such as Fast/Medium/Slow/Stale, and do
not assume any information beyond the issue text.

Return exactly these five short fields:
Issue type: bug | feature | regression | docs | question | maintenance | other
Specification quality: clear | partial | vague
Complexity: low | medium | high
Urgency signals: none | moderate | strong
Resolution-speed rationale: one concise analytical sentence

Issue title: {title}
Issue body (excerpt): {body}

Structured summary:"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=GCS_SIGNALS)
    parser.add_argument("--output", default=GCS_OUTPUT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--body-chars", type=int, default=4000)
    parser.add_argument("--sample", action="store_true", help="Use first 3 shards only.")
    parser.add_argument("--max-issues", type=int, default=None)
    parser.add_argument("--max-train-issues", type=int, default=None)
    parser.add_argument("--max-test-issues", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete existing output parquet parts before writing a new sampled cache.",
    )
    parser.add_argument(
        "--include-test-only",
        action="store_true",
        help="Generate only Aug-Oct 2025 summaries; default includes train and test rows before 2025-11-01.",
    )
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


def load_shard(fs, path: str, include_test_only: bool) -> pd.DataFrame:
    df = pd.read_parquet(fs.open(path), columns=LOAD_COLS)
    created_at = pd.to_datetime(df["issue_created_at"], utc=True)
    if include_test_only:
        df = df[(created_at >= TRAIN_CUTOFF) & (created_at < TEST_CUTOFF)]
    else:
        df = df[created_at < TEST_CUTOFF]
    df = df.reset_index(drop=True)
    return add_issue_keys(df)


def temporal_counts(fs, paths) -> tuple[list[int], list[int]]:
    train_counts, test_counts = [], []
    print("Counting eligible rows per shard for temporal sampling...")
    for i, path in enumerate(paths):
        df = pd.read_parquet(fs.open(path), columns=["issue_created_at"])
        created_at = pd.to_datetime(df["issue_created_at"], utc=True)
        train_counts.append(int((created_at < TRAIN_CUTOFF).sum()))
        test_counts.append(int(((created_at >= TRAIN_CUTOFF) & (created_at < TEST_CUTOFF)).sum()))
        if (i + 1) % 25 == 0 or i + 1 == len(paths):
            print(f"  Counted {i + 1:,}/{len(paths):,} shards", flush=True)
    print(f"  Total train-eligible rows: {sum(train_counts):,}")
    print(f"  Total test-eligible rows: {sum(test_counts):,}")
    return train_counts, test_counts


def allocate_quotas(counts: list[int], target: Optional[int]) -> list[int]:
    if target is None:
        return counts
    total = sum(counts)
    target = min(target, total)
    if target <= 0 or total <= 0:
        return [0 for _ in counts]

    raw = np.array(counts, dtype=np.float64) * (target / total)
    quotas = np.floor(raw).astype(int)
    quotas = np.minimum(quotas, np.array(counts, dtype=int))
    remaining = target - int(quotas.sum())
    if remaining > 0:
        order = np.argsort(-(raw - quotas))
        for idx in order:
            if remaining <= 0:
                break
            if quotas[idx] < counts[idx]:
                quotas[idx] += 1
                remaining -= 1
    return quotas.tolist()


def load_temporal_sampled_shard(
    fs,
    path: str,
    train_quota: int,
    test_quota: int,
    random_state: int,
) -> pd.DataFrame:
    if train_quota <= 0 and test_quota <= 0:
        return pd.DataFrame(columns=LOAD_COLS + ["issue_key"])

    df = pd.read_parquet(fs.open(path), columns=LOAD_COLS)
    created_at = pd.to_datetime(df["issue_created_at"], utc=True)
    train = df[created_at < TRAIN_CUTOFF]
    test = df[(created_at >= TRAIN_CUTOFF) & (created_at < TEST_CUTOFF)]

    parts = []
    if train_quota > 0 and not train.empty:
        parts.append(train.sample(n=min(train_quota, len(train)), random_state=random_state))
    if test_quota > 0 and not test.empty:
        parts.append(test.sample(n=min(test_quota, len(test)), random_state=random_state + 1))
    if not parts:
        return pd.DataFrame(columns=LOAD_COLS + ["issue_key"])
    sampled = pd.concat(parts).sample(frac=1, random_state=random_state + 2).reset_index(drop=True)
    return add_issue_keys(sampled)


def format_prompt(row, body_chars: int) -> str:
    title = "" if pd.isna(row["title"]) else str(row["title"])
    body = "" if pd.isna(row["body"]) else str(row["body"])
    return PROMPT.format(title=title, body=body[:body_chars])


def generate_batch(model, tokenizer, prompts, max_new_tokens: int):
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[:, enc["input_ids"].shape[1]:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def output_path(base_output: str, shard_index: int) -> str:
    return base_output.rstrip("/") + f"/part-{shard_index:05d}.parquet"


def write_parquet(fs, df: pd.DataFrame, path: str):
    with fs.open(path, "wb") as f:
        df.to_parquet(f, index=False)


def clear_output_parts(fs, base_output: str):
    existing = sorted(fs.glob(base_output.rstrip("/") + "/*.parquet"))
    if not existing:
        return
    print(f"Deleting {len(existing):,} existing output parquet part(s) from {base_output}")
    for path in existing:
        fs.rm(path)


def main():
    args = parse_args()
    fs = gcsfs.GCSFileSystem()
    paths = sorted(fs.glob(args.input.rstrip("/") + "/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquets found at {args.input}")
    if args.sample:
        paths = paths[:3]
        print(f"[sample mode] {len(paths)} input shards")
    else:
        print(f"{len(paths)} input shards")

    temporal_sample = args.max_train_issues is not None or args.max_test_issues is not None
    if temporal_sample and args.include_test_only:
        raise ValueError("--include-test-only cannot be combined with temporal train/test quotas.")
    if temporal_sample and args.max_issues is not None:
        raise ValueError("--max-issues cannot be combined with --max-train-issues/--max-test-issues.")
    if args.clear_output:
        clear_output_parts(fs, args.output)

    train_quotas = test_quotas = None
    if temporal_sample:
        train_counts, test_counts = temporal_counts(fs, paths)
        train_quotas = allocate_quotas(train_counts, args.max_train_issues)
        test_quotas = allocate_quotas(test_counts, args.max_test_issues)
        print(f"  Target train summaries: {sum(train_quotas):,}")
        print(f"  Target test summaries: {sum(test_quotas):,}")

    print(f"Loading Qwen model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    processed = 0
    processed_train = 0
    processed_test = 0
    for i, path in enumerate(paths):
        out_path = output_path(args.output, i)
        if fs.exists(out_path) and not args.overwrite:
            print(f"Skipping existing {out_path}")
            continue

        if temporal_sample:
            df = load_temporal_sampled_shard(
                fs,
                path,
                train_quota=train_quotas[i],
                test_quota=test_quotas[i],
                random_state=args.seed + i * 10,
            )
        else:
            df = load_shard(fs, path, include_test_only=args.include_test_only)
        if not temporal_sample and args.max_issues is not None:
            remaining = args.max_issues - processed
            if remaining <= 0:
                break
            df = df.iloc[:remaining].reset_index(drop=True)
        if df.empty:
            continue

        if temporal_sample:
            created_at = pd.to_datetime(df["issue_created_at"], utc=True)
            shard_train = int((created_at < TRAIN_CUTOFF).sum())
            shard_test = int(((created_at >= TRAIN_CUTOFF) & (created_at < TEST_CUTOFF)).sum())
            processed_train += shard_train
            processed_test += shard_test
            split_msg = f" ({shard_train:,} train / {shard_test:,} test)"
        else:
            split_msg = ""
        print(f"Shard {i}: generating {len(df):,} summaries{split_msg} from title/body only")
        summaries = []
        prompts = [format_prompt(row, args.body_chars) for _, row in df.iterrows()]
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start:start + args.batch_size]
            summaries.extend(generate_batch(model, tokenizer, batch, args.max_new_tokens))
            done = start + len(batch)
            if done % 500 == 0 or done == len(prompts):
                print(f"  {done:,}/{len(prompts):,}", flush=True)

        out_df = df[["issue_key", "repo_name", "issue_created_at"]].copy()
        out_df["qwen_summary"] = [s.strip() for s in summaries]
        write_parquet(fs, out_df, out_path)
        processed += len(out_df)
        print(f"Wrote {len(out_df):,} rows to {out_path}")

    print(f"Total summaries generated this run: {processed:,}")
    if temporal_sample:
        print(f"  Train summaries generated: {processed_train:,}")
        print(f"  Test summaries generated: {processed_test:,}")


if __name__ == "__main__":
    main()
