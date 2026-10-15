"""Shared utilities for dataset processing."""
import json
import os
import pandas as pd


SCHEMA_COLUMNS = ["uid", "image_path", "question", "answer_counts"]


def counts_to_ranked(answer_counts: dict, min_count: int = 1) -> list:
    """Sort labels by count descending, filter to count >= min_count."""
    return [
        (label, count)
        for label, count in sorted(answer_counts.items(), key=lambda x: -x[1])
        if count >= min_count
    ]


def majority_vote_gt(answer_counts_json: str) -> str:
    """Return the top-1 label (highest count) from answer_counts."""
    counts = json.loads(answer_counts_json)
    return max(counts, key=counts.get)


def validate_df(df: pd.DataFrame, dataset_name: str) -> None:
    missing = set(SCHEMA_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"[{dataset_name}] Missing columns: {missing}")
    assert df["uid"].is_unique, f"[{dataset_name}] uid is not unique"
    assert df["uid"].notna().all(), f"[{dataset_name}] uid has NaN"
    assert df["image_path"].notna().all(), f"[{dataset_name}] image_path has NaN"
    assert df["answer_counts"].notna().all(), f"[{dataset_name}] answer_counts has NaN"
    for _, row in df.head(5).iterrows():
        counts = json.loads(row["answer_counts"])
        assert isinstance(counts, dict), f"answer_counts is not a dict: {row['answer_counts']}"


def sort_answer_counts(answer_counts_json: str) -> str:
    """Re-encode answer_counts dict sorted by count descending, then alphabetically."""
    counts = json.loads(answer_counts_json)
    sorted_counts = dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))
    return json.dumps(sorted_counts)


def _get_top_counts(answer_counts_json: str):
    """Return (top1, top2, top3) counts from answer_counts."""
    pos = sorted([v for v in json.loads(answer_counts_json).values() if v > 0], reverse=True)
    return (
        pos[0] if len(pos) >= 1 else 0,
        pos[1] if len(pos) >= 2 else 0,
        pos[2] if len(pos) >= 3 else 0,
    )


def _is_excluded(answer_counts_json: str) -> bool:
    t1, t2, t3 = _get_top_counts(answer_counts_json)
    if t2 == 0:
        return True                     # k=1: single label
    if t1 >= 9 and t2 == 1 and t3 == 0:
        return True                     # near-unanimous with 1 outlier dissenter
    return False


def write_parquet(df: pd.DataFrame, dataset_name: str, base_dir: str = "data/parquet",
                  suffix: str = "") -> str:
    """Sort answer_counts, apply exclusion filters, write parquet.

    Args:
        suffix: appended to filename, e.g. suffix="_raw" → {dataset}_raw.parquet
    """
    df = df.copy()
    df["answer_counts"] = df["answer_counts"].apply(sort_answer_counts)
    before = len(df)
    excluded = df["answer_counts"].apply(_is_excluded)
    df = df[~excluded].reset_index(drop=True)
    print(f"  Filtered: {before} → {len(df)} ({before - len(df)} removed)")
    out_dir = os.path.join(base_dir, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_name}{suffix}.parquet")
    df[SCHEMA_COLUMNS].to_parquet(out_path, index=False)
    print_stats(df, dataset_name)
    return out_path


def print_stats(df: pd.DataFrame, dataset_name: str) -> None:
    k_values = df["answer_counts"].apply(
        lambda s: len([v for v in json.loads(s).values() if v > 0])
    )
    print(f"\n[{dataset_name}]")
    print(f"  Samples: {len(df)}")
    print(f"  k distribution: {k_values.value_counts().sort_index().to_dict()}")
    print(f"  Example:")
    row = df.iloc[0]
    print(f"    uid:           {row['uid']}")
    print(f"    image_path:    {row['image_path']}")
    print(f"    question:      {row['question'][:80]}")
    counts = json.loads(row["answer_counts"])
    ranked = counts_to_ranked(counts)
    print(f"    answer_counts: {counts}")
    print(f"    ranked:        {', '.join(f'{l}({c})' for l, c in ranked)}")
