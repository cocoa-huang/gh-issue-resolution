import pyarrow.parquet as pq
import gcsfs

fs = gcsfs.GCSFileSystem()

files = [
    ("issues", "gh_issue_ml-data/issues/issues_labeled_2025/000000000000.parquet"),
    ("pr_daily", "gh_issue_ml-data/repo_signals/pr_daily/000000000000.parquet"),
    ("push_daily", "gh_issue_ml-data/repo_signals/push_daily/000000000000.parquet"),
    ("release_daily", "gh_issue_ml-data/repo_signals/release_daily/000000000000.parquet"),
    ("watch_daily", "gh_issue_ml-data/repo_signals/watch_daily/000000000000.parquet"),
]

for name, path in files:
    t = pq.read_table(fs.open(path))
    print(f"=== {name} ({t.num_rows} rows) ===")
    print(t.schema)
    print()
