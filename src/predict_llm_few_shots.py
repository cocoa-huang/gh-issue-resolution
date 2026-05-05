import os
import json
import time
import argparse
import pandas as pd
import numpy as np
#import gcsfs
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from openai import OpenAI


#GCS_SIGNALS = "gs://gh_issue_ml-data/issues/issues_with_signals/"
LOCAL_PARQUET  = "E:/gh-issue-resolution/data/issues_issues_with_signals_000000000006.parquet"
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

Your rationale should very briefly explain the key factors influencing your prediction, such as relevant signals or issue content.

User:
Here are labeled examples.

Example 1:
title: 🛠️ Task: Conversation List (Sidebar)
body: ### 🔗 Parent Issue\n\n#77\n\n### 🧩 Module\n\nMessages\n\n### 📝 Description\n\nImplement the left-side conversation list where users can see their recent chats.\n\n### ✅ Acceptance Criteria\n\n- [x] Create a scrollable list of recent conversations.\n- [x] Add a search bar for filtering messages.\n- [x] Display user avatars, names, and last message preview.\n- [ ] Connect to a mock API or backend endpoint.\n\n### 🔥 Priority\n\nHigh\n\n### 💬 Additional Notes\n\n_No response
Signals:
- author_association: NONE
- pr_merged_30d: 0
- avg_merge_hours_30d: 0
- push_count_30d: 0
- release_count_90d: 0
- star_count_30d: 0

Correct label: Slow

Example 2:
title: 🛑 Manhuagui is down	
body: In [`a8d7dd2`](https://github.com/http403/uptime_monitor/commit/a8d7dd2b975ee0cc5693fc7173c1b137a2756178\n), Manhuagui (https://www.manhuagui.com) was **down**:\n- HTTP code: 0\n- Response time: 0 ms\n	
Signals:
- author_association: OWNER
- pr_merged_30d: 40
- avg_merge_hours_30d: 2.636364
- push_count_30d: 3284
- release_count_90d: 0
- star_count_30d: 0

Correct label: Fast

Example 3:
title: [Registry] 👤 Admin 📖 Add an authorized bot	
body: - [x] Backend development\n- [x] Writing documentation\n- [x] Tests implementation\n- [x] Internal security audit and optimization\n\nFrontend : no user interface is initially planned for this feature\nBackend (payable) : Registry adds an address of an authorized bot to its data
Signals:
- author_association: NONE
- pr_merged_30d: 0
- avg_merge_hours_30d: NAN
- push_count_30d: 0
- release_count_90d: 0
- star_count_30d: 0

Correct label: Medium

Example 4:
title: Fix Bug #12345678
body: This issue tracks fixing a critical bug in the application.
- author_association: OWNER
- pr_merged_30d: 0
- avg_merge_hours_30d: NAN
- push_count_30d: 6
- release_count_90d: 0
- star_count_30d: 0

Correct label: Stale

"""

def load_mock_data(n=5):
    data = []
    for i in range(n):
        data.append({
            "title": f"Bug {i}",
            "body": "The system crashes when clicking button.",
            "label": np.random.choice(LABEL_ORDER),
            "author_association": "NONE",
            "pr_merged_30d": 2,
            "avg_merge_hours_30d": 10,
            "push_count_30d": 5,
            "release_count_90d": 1,
            "star_count_30d": 3,
        })
    return pd.DataFrame(data)

'''
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
'''
def load_test_sample(n, seed=42):
    df = pd.read_parquet(LOCAL_PARQUET, columns=LOAD_COLS)

    df["created_at"] = pd.to_datetime(df["issue_created_at"], utc=True)

    test = df[
        (df["created_at"] >= TRAIN_CUTOFF) &
        (df["created_at"] < TEST_CUTOFF)
    ].copy()

    if test.empty:
        raise ValueError(
            "No test rows found in this parquet shard. "
            "This shard may not contain Aug–Oct 2025 issues."
        )

    out = test.sample(
        n=min(n, len(test)),
        random_state=seed
    ).reset_index(drop=True)

    out["sample_id"] = np.arange(len(out))

    print(f"Loaded {len(test):,} eligible test rows from local parquet")
    print(f"Sampled {len(out):,} rows for LLM evaluation")

    return out

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
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )

            text = resp.choices[0].message.content
            pred = json.loads(text)

            label = pred.get("label", "Stale")
            if label not in LABEL_ORDER:
                label = "Stale"
            pred["label"] = label
            return pred

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
        model = "deepseek-v4-flash"

    elif provider == "openai":
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        )
        model = "gpt-5.4-mini"

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
            pred = call_llm(client, model, prompt)

            time.sleep(0.5)  

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
    report = classification_report(
        true,
        preds,
        labels=LABEL_ORDER,
        digits=3,
        zero_division=0,
    )
    cm = confusion_matrix(true, preds, labels=LABEL_ORDER)

    eval_text = (
        f"LLM zero-shot — text + repo signals ({model})\n"
        f"Provider: {provider}\n"
        f"Temporal split: train < 2025-08-01  |  test 2025-08-01 – 2025-10-31\n"
        f"Evaluation sample: {len(true):,} sampled test issues from local parquet\n"
        f"Signal features (10): {LOAD_COLS}\n"
        f"{'='*60}\n"
        f"Macro-F1: {macro_f1:.4f}\n\n"
        f"{report}\n"
        f"Confusion matrix (rows=true, cols=pred)\n"
        f"Order: {LABEL_ORDER}\n{cm}\n"
    )

    eval_path = args.output.replace("_predictions.jsonl", "_eval.txt")
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write(eval_text)

    print(eval_text)

if __name__ == "__main__":
    main()