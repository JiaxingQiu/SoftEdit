"""Sample loading + output paths for generality set generation.

Reuses the soft_generality parquet loaders (same source parquets) and adds a
``soft_gen_available``-filtered iterator plus canonical output locations.
"""

from __future__ import annotations

import random as _random
from pathlib import Path
from typing import Any, Iterator, List

from PIL import Image

from prep_generality.soft_generality.shared.sample_io import (
    REPO_ROOT,
    PARQUET_PATHS,
    load_parquet,
    parse_answer_counts,
    ranked_labels,
    resolve_image_path,
)

# ---- output layout (parallel to data/soft_generality/) ----
DATA_ROOT = REPO_ROOT / "data" / "generality"
TGEN_ROOT = DATA_ROOT / "tgen"          # tgen/{dataset}.parquet  (uid, variants)
IGEN_ROOT = DATA_ROOT / "igen"          # igen/{dataset}/{uid}/aug{i}.png


def tgen_parquet_path(dataset: str) -> Path:
    return TGEN_ROOT / f"{dataset}.parquet"


def igen_dir(dataset: str, uid: str) -> Path:
    return IGEN_ROOT / dataset / str(uid)


def iter_soft_gen_samples(dataset: str) -> Iterator[dict[str, Any]]:
    """Yield soft_gen_available samples for a dataset.

    Each sample: dataset, uid, question, answer_counts, ranked_labels, image_path.
    The image itself is opened lazily by the caller (only when i-gen is needed).
    """
    df = load_parquet(dataset)
    if "soft_gen_available" in df.columns:
        df = df[df["soft_gen_available"].astype(bool)]
    for _, row in df.iterrows():
        counts = parse_answer_counts(row["answer_counts"])
        yield {
            "dataset": dataset,
            "uid": str(row["uid"]),
            "question": str(row["question"]),
            "answer_counts": counts,
            "ranked_labels": ranked_labels(counts),
            "image_path": resolve_image_path(row["image_path"]),
        }


def load_bg_pool(
    dataset: str = "vqav2",
    n: int = 64,
    seed: int = 0,
    size: int = 256,
) -> List[Image.Image]:
    """Load n random real images to use as "fake new scene" backgrounds (i-gen).

    Sampled deterministically (fixed seed) from ``dataset``'s image pool and
    downscaled to ``size`` for speed.
    """
    df = load_parquet(dataset)
    paths = list(df["image_path"])
    rng = _random.Random(seed)
    rng.shuffle(paths)
    pool: List[Image.Image] = []
    for p in paths:
        if len(pool) >= n:
            break
        try:
            im = Image.open(resolve_image_path(p)).convert("RGB")
            im.thumbnail((size, size), Image.LANCZOS)
            pool.append(im)
        except Exception:  # noqa: BLE001 - skip unreadable images
            continue
    return pool


__all__ = [
    "REPO_ROOT",
    "PARQUET_PATHS",
    "DATA_ROOT",
    "TGEN_ROOT",
    "IGEN_ROOT",
    "tgen_parquet_path",
    "igen_dir",
    "iter_soft_gen_samples",
    "load_bg_pool",
]
