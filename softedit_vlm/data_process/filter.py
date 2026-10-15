"""Filter dataset parquets: drop singletons, non-alpha labels, long labels, yes/no-only.

Filters applied in order:
  1. Drop singleton labels (count=1) — noise from single annotators.
  2. Remove samples with any label containing non-alphabetic characters.
  3. Remove samples with any label longer than 2 words.
  4. Remove samples whose only labels are {yes, no} — binary polls, not ranking.
After each filter, discard samples that become k<2 (fewer than 2 distinct labels).

Usage:
    python -m data_process.filter
"""
import json
import os
import re
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PARQUET_DIR = "data/parquet"
DATASETS = ["vqav2"]


def _parse_counts(ac_json: str) -> dict:
    return json.loads(ac_json)


def _dump_counts(counts: dict) -> str:
    return json.dumps(dict(sorted(counts.items(), key=lambda x: (-x[1], x[0]))))


def filter_singletons(counts: dict) -> dict:
    """Remove labels with count=1."""
    return {l: c for l, c in counts.items() if c > 1}


def is_alpha_only(label: str) -> bool:
    """True if label contains only alphabetic chars and spaces."""
    return bool(re.fullmatch(r'[a-zA-Z ]+', label.strip()))


def filter_non_alpha(counts: dict) -> dict:
    """Remove labels containing non-alphabetic characters. Returns empty dict if any label fails."""
    if not all(is_alpha_only(l) for l in counts.keys()):
        return {}
    return counts


def filter_long_labels(counts: dict, max_words: int = 2) -> dict:
    """Remove labels with more than max_words words. Returns empty dict if any label fails."""
    if not all(len(l.strip().split()) <= max_words for l in counts.keys()):
        return {}
    return counts


def is_yes_no_only(counts: dict) -> bool:
    """True if the only labels with count>0 are subsets of {yes, no}."""
    pos = {l for l, c in counts.items() if c > 0}
    return pos <= {"yes", "no"}


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all filters sequentially, return filtered dataframe."""
    n0 = len(df)

    # Filter 1: drop singleton labels
    df["answer_counts"] = df["answer_counts"].apply(
        lambda s: _dump_counts(filter_singletons(_parse_counts(s)))
    )
    df["_k"] = df["answer_counts"].apply(lambda s: len([v for v in _parse_counts(s).values() if v > 0]))
    df = df[df["_k"] >= 2].copy()
    n1 = len(df)

    # Filter 2: remove samples with any non-alpha label
    df["answer_counts"] = df["answer_counts"].apply(
        lambda s: _dump_counts(filter_non_alpha(_parse_counts(s)))
    )
    df["_k"] = df["answer_counts"].apply(lambda s: len([v for v in _parse_counts(s).values() if v > 0]))
    df = df[df["_k"] >= 2].copy()
    n2 = len(df)

    # Filter 3: remove samples with any label > 2 words
    df["answer_counts"] = df["answer_counts"].apply(
        lambda s: _dump_counts(filter_long_labels(_parse_counts(s)))
    )
    df["_k"] = df["answer_counts"].apply(lambda s: len([v for v in _parse_counts(s).values() if v > 0]))
    df = df[df["_k"] >= 2].copy()
    n3 = len(df)

    # Filter 4: remove yes/no-only samples
    df = df[~df["answer_counts"].apply(lambda s: is_yes_no_only(_parse_counts(s)))].copy()
    n4 = len(df)

    df.drop(columns=["_k"], inplace=True)
    print(f"    {n0} → singleton:{n1} → alpha-only:{n2} → ≤2words:{n3} → no-yesno:{n4}")
    return df


def main():
    print(f"{'Dataset':<12} {'Original':>10} {'Final':>8} {'% kept':>8}")
    print("-" * 42)

    for name in DATASETS:
        pq_path = os.path.join(PARQUET_DIR, name, f"{name}.parquet")
        if not os.path.exists(pq_path):
            print(f"[{name}] SKIP — {pq_path} not found")
            continue

        df = pd.read_parquet(pq_path)
        n_orig = len(df)
        print(f"[{name}]")
        df = apply_filters(df)
        n_final = len(df)

        df.to_parquet(pq_path, index=False)
        print(f"    Written → {pq_path} ({n_orig} → {n_final}, {100*n_final/n_orig:.1f}% kept)")

    print("\nDone.")


if __name__ == "__main__":
    main()
