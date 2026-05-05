"""
extract_qwen_summaries.py - Generate Model 5 Qwen semantic summaries.

Reads issues_with_signals parquet shards from GCS, sends only title/body to
Qwen, and writes cached summaries keyed by a deterministic issue hash.

Output columns:
  issue_key, repo_name, issue_created_at, qwen_summary

Usage:
  python3 src/extract_qwen_summaries.py
  python3 src/extract_qwen_summaries.py --sample --max-issues 64
"""

import argparse
import hashlib

import gcsfs
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GCS_SIGNALS = "gs://gh_issue_ml-data/issues/issues_with_signals/"
GCS_OUTPUT = "gs://gh_issue_ml-data/llm_features/qwen_summaries/"
MODEL_NAME = "Qwen/Qwen3.5-2B"

TRAIN_CUTOFF = pd.Timestamp("2025-08-01", tz="UTC")
TEST_CUTOFF = pd.Timestamp("2025-11-01", tz="UTC")

LOAD_COLS = ["repo_name", "title", "body", "issue_created_at"]

PROMPT = """You are analyzing a GitHub issue to predict how quickly it will be resolved.
Summarize the key properties of this issue in 2-3 sentences, focusing ONLY on
factors relevant to resolution speed: issue type, specification quality,
complexity, urgency, and anything that signals how a maintainer would prioritize it.
Do not restate the issue. Be analytical, not descriptive.

Issue title: {title}
Issue body (excerpt): {body}

Summary:"""


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
    parser.add_argument("--overwrite", action="store_true")
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
    for i, path in enumerate(paths):
        out_path = output_path(args.output, i)
        if fs.exists(out_path) and not args.overwrite:
            print(f"Skipping existing {out_path}")
            continue

        df = load_shard(fs, path, include_test_only=args.include_test_only)
        if args.max_issues is not None:
            remaining = args.max_issues - processed
            if remaining <= 0:
                break
            df = df.iloc[:remaining].reset_index(drop=True)
        if df.empty:
            continue

        print(f"Shard {i}: generating {len(df):,} summaries from title/body only")
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


if __name__ == "__main__":
    main()
