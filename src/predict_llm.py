import os
import json
import time
import argparse
import pandas as pd
import numpy as np
import gcsfs
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from openai import OpenAI

GCS_SIGNALS = "gs://gh_issue_ml-data/issues/issues_with_signals/"
LABEL_ORDER = ["Fast", "Medium", "Slow", "Stale"]

TRAIN_CUTOFF = pd.Timestamp("2025-08-01", tz="UTC")
TEST_CUTOFF = pd.Timestamp("2025-11-01", tz="UTC")

LOAD_COLS = [
    "title", "body", "issue_created_at", "label",
    "author_association",
    "pr_merged_30d",
    "avg_merge_hours_30d",
    "push_count_30d",
    "release_count_90d",
    "star_count_30d",
]

SYSTEM_PROMPT = """
You are predicting GitHub issue resolution time bins.

Return exactly one JSON object:
{
  "label": "Fast" | "Medium" | "Slow" | "Stale",
  "confidence": number,
  "rationale": string
}

Label definitions:
Fast: issue is likely resolved within 7 days.
Medium: issue is likely resolved within 8–30 days.
Slow: issue is likely resolved after 30 days.
Stale: issue is unlikely to be resolved or remains unresolved.
"""

def load_test_sample(n, seed=42):
    fs = gcsfs.GCSFileSystem()
    paths = sorted(fs.glob(GCS_SIGNALS.rstrip("/") + "/*.parquet"))

    frames = []
    for p in paths:
        df = pd.read_parquet(fs.open(p), columns=LOAD_COLS)
        df["created_at"] = pd.to_datetime(df["issue_created_at"], utc=True)
        test = df[
            (df["created_at"] >= TRAIN_CUTOFF) &
            (df["created_at"] < TEST_CUTOFF)
        ]
        if not test.empty:
            frames.append(test)

        if sum(len(x) for x in frames) >= n * 3:
            break

    out = pd.concat(frames).sample(n=min(n, sum(len(x) for x in frames)), random_state=seed)
    return out.reset_index(drop=True)

def build_prompt(row):
    title = str(row["title"] or "")
    body = str(row["body"] or "")
    body = body[:4000]

    return f"""
Predict the GitHub issue resolution time bin.

Issue title:
{title}

Issue body:
{body}

Repository/activity signals before issue creation:
- author_association: {row.get("author_association")}
- pr_merged_30d: {row.get("pr_merged_30d")}
- avg_merge_hours_30d: {row.get("avg_merge_hours_30d")}
- push_count_30d: {row.get("push_count_30d")}
- release_count_90d: {row.get("release_count_90d")}
- star_count_30d: {row.get("star_count_30d")}

Choose one label: Fast, Medium, Slow, Stale.
"""

def call_llm(client, model, prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "issue_resolution_prediction",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "enum": LABEL_ORDER,
                                },
                                "confidence": {
                                    "type": "number",
                                },
                                "rationale": {
                                    "type": "string",
                                },
                            },
                            "required": ["label", "confidence", "rationale"],
                            "additionalProperties": False,
                        },
                    }
                },
            )
            return json.loads(resp.output_text)

        except Exception as e:
            wait = 2 ** attempt
            print(f"API error: {e}; retrying in {wait}s", flush=True)
            time.sleep(wait)

    return {
        "label": "Stale",
        "confidence": 0.0,
        "rationale": "API failed after retries.",
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--model", type=str, default="gpt-5.5-mini")
    parser.add_argument("--output", type=str, default="results/llm_zero_shot_predictions.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    provider = os.environ.get("LLM_PROVIDER", "openai")

    if provider == "deepseek":
        client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com"
        )
        model = "deepseek-chat"

    elif provider == "openai":
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        )
        model = "gpt-5.5-mini"

    else:
        raise ValueError("Unknown LLM_PROVIDER")    
    
    df = load_test_sample(args.n)

    done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                obj = json.loads(line)
                done.add(obj["row_id"])

    with open(args.output, "a", buffering=1) as f:
        for i, row in df.iterrows():
            if i in done:
                continue

            prompt = build_prompt(row)
            pred = call_llm(client, args.model, prompt)

            record = {
                "row_id": int(i),
                "true_label": row["label"],
                "pred_label": pred["label"],
                "confidence": pred.get("confidence"),
                "rationale": pred.get("rationale"),
                "title": row.get("title"),
            }

            f.write(json.dumps(record) + "\n")
            print(record, flush=True)

    preds, true = [], []
    with open(args.output) as f:
        for line in f:
            obj = json.loads(line)
            true.append(obj["true_label"])
            preds.append(obj["pred_label"])

    macro_f1 = f1_score(true, preds, average="macro", labels=LABEL_ORDER)
    report = classification_report(true, preds, labels=LABEL_ORDER, digits=3)
    cm = confusion_matrix(true, preds, labels=LABEL_ORDER)

    eval_text = (
        f"LLM zero-shot baseline\n"
        f"Model: {args.model}\n"
        f"N: {len(true)}\n"
        f"Macro-F1: {macro_f1:.4f}\n\n"
        f"{report}\n"
        f"Confusion matrix rows=true cols=pred\n"
        f"Order: {LABEL_ORDER}\n{cm}\n"
    )

    eval_path = args.output.replace("_predictions.jsonl", "_eval.txt")
    with open(eval_path, "w") as f:
        f.write(eval_text)

    print(eval_text)

if __name__ == "__main__":
    main()