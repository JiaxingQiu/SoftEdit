"""Load SoftQA samples from unified parquet (dataset-agnostic)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]

PARQUET_PATHS = {
    "vqav2": REPO_ROOT / "data" / "parquet" / "vqav2" / "vqav2.parquet",
}


def load_parquet(dataset: str) -> pd.DataFrame:
    if dataset not in PARQUET_PATHS:
        raise KeyError(f"Unknown dataset {dataset!r}. Choose from {sorted(PARQUET_PATHS)}")
    path = PARQUET_PATHS[dataset]
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def parse_answer_counts(raw: Any) -> dict[str, int]:
    if isinstance(raw, dict):
        return {str(k): int(v) for k, v in raw.items()}
    if isinstance(raw, str):
        return {str(k): int(v) for k, v in json.loads(raw).items()}
    raise TypeError(f"Unsupported answer_counts type: {type(raw)}")


def ranked_labels(answer_counts: dict[str, int]) -> list[str]:
    return [lab for lab, _ in sorted(answer_counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def resolve_image_path(image_path: str, repo_root: Path | None = None) -> Path:
    root = repo_root or REPO_ROOT
    path = Path(image_path)
    if not path.is_absolute():
        path = root / path
    return path


def list_datasets() -> list[str]:
    return sorted(PARQUET_PATHS)


def _row_to_sample(dataset: str, row: Any) -> dict[str, Any]:
    counts = parse_answer_counts(row["answer_counts"])
    image_path = resolve_image_path(row["image_path"])
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    return {
        "dataset": dataset,
        "uid": row["uid"],
        "question": row["question"],
        "answer_counts": counts,
        "ranked_labels": ranked_labels(counts),
        "image_path": image_path,
        "image": Image.open(image_path).convert("RGB"),
    }


def load_sample(
    dataset: str,
    uid: str | None = None,
    *,
    index: int | None = None,
    min_labels: int = 2,
) -> dict[str, Any]:
    """Load one sample (dataset-agnostic).

    Selection (first match wins):
      1. ``uid`` if provided
      2. ``index`` among rows with >= min_labels answers (0-based)
      3. otherwise index=0 among those multi-label rows
    """
    df = load_parquet(dataset)
    if uid is not None:
        rows = df[df["uid"] == uid]
        if rows.empty:
            raise KeyError(f"uid {uid!r} not found in {dataset}")
        return _row_to_sample(dataset, rows.iloc[0])

    multi = []
    for _, r in df.iterrows():
        counts = parse_answer_counts(r["answer_counts"])
        if len(counts) >= min_labels:
            multi.append(r)
    if not multi:
        raise ValueError(f"No sample with >= {min_labels} labels in {dataset}")

    i = 0 if index is None else index
    if i < 0 or i >= len(multi):
        raise IndexError(
            f"index {i} out of range for {dataset} "
            f"(0..{len(multi) - 1} multi-label samples with min_labels={min_labels})"
        )
    return _row_to_sample(dataset, multi[i])
