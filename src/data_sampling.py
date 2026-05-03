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