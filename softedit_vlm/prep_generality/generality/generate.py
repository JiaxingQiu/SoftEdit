"""Unified single-pass generality builder (t-gen + i-gen).

For each soft_gen_available sample it produces BOTH sets in one pass:
  - t-gen: k_text rephrased questions (gpt-4o-mini)  -> tgen/{dataset}.parquet
  - i-gen: k_image label-safe augmented images       -> igen/{dataset}/{uid}/aug{i}.png

Idempotent: t-gen skips uids already in the parquet; i-gen skips images already
on disk. Safe to resubmit after a timeout. The t-gen parquet is checkpointed
every N samples so partial progress is never lost.

Usage:
    python -m prep_generality.generality.generate --dataset vqav2
    python -m prep_generality.generality.generate --dataset vqav2 --limit 20   # smoke test
    python -m prep_generality.generality.generate --dataset vqav2 --overwrite
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from prep_generality.generality.augment import make_augmentations, DEFAULT_AUG_STRENGTH
from prep_generality.generality.shared.sample_io import (
    iter_soft_gen_samples,
    load_bg_pool,
    tgen_parquet_path,
    igen_dir,
    IGEN_ROOT,
)
from prep_generality.generality.tgen import make_client, rephrase_question, DEFAULT_DEPLOYMENT

DATASETS = ("vqav2",)
BG_DATASET = "vqav2"   # source of "fake new scene" backgrounds for i-gen
BG_POOL_SIZE = 64


def _load_existing_tgen(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {str(u): list(v) for u, v in zip(df["uid"], df["variants"])}


def _save_tgen(rows: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {"uid": list(rows.keys()), "variants": [list(v) for v in rows.values()]}
    )
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def build(
    dataset: str,
    *,
    k_text: int = 2,
    k_image: int = 2,
    aug_strength: int = DEFAULT_AUG_STRENGTH,
    deployment: str = DEFAULT_DEPLOYMENT,
    overwrite: bool = False,
    limit: int = 0,
    checkpoint_every: int = 50,
) -> None:
    tgen_path = tgen_parquet_path(dataset)
    tgen_rows = {} if overwrite else _load_existing_tgen(tgen_path)

    samples = list(iter_soft_gen_samples(dataset))
    if limit:
        samples = samples[:limit]

    print("=" * 60, flush=True)
    print(f"Generality build | dataset={dataset} | n={len(samples)}", flush=True)
    print(f"t-gen → {tgen_path}  (existing: {len(tgen_rows)})", flush=True)
    print(f"i-gen → {IGEN_ROOT / dataset}/<uid>/aug*.png", flush=True)
    print(f"k_text={k_text} k_image={k_image} aug_strength={aug_strength} "
          f"deployment={deployment} overwrite={overwrite}", flush=True)
    print("=" * 60, flush=True)

    client = make_client()
    bg_pool = load_bg_pool(dataset=BG_DATASET, n=BG_POOL_SIZE)
    print(f"scene bg pool: {len(bg_pool)} images from {BG_DATASET}", flush=True)
    t0 = time.time()
    n_tgen_new = n_igen_new = 0

    for i, s in enumerate(samples, 1):
        uid = s["uid"]

        # ---- t-gen ----
        if overwrite or uid not in tgen_rows:
            variants = rephrase_question(client, s["question"], k=k_text, deployment=deployment)
            if variants:
                tgen_rows[uid] = variants
                n_tgen_new += 1

        # ---- i-gen ----
        out_dir = igen_dir(dataset, uid)
        targets = [out_dir / f"aug{j}.png" for j in range(k_image)]
        if overwrite or not all(p.exists() for p in targets):
            img = Image.open(s["image_path"]).convert("RGB")
            augs = make_augmentations(img, uid, k=k_image, bg_pool=bg_pool, aug_strength=aug_strength)
            out_dir.mkdir(parents=True, exist_ok=True)
            for (op, aug), p in zip(augs, targets):
                if overwrite or not p.exists():
                    aug.save(p)
                    n_igen_new += 1

        if i % checkpoint_every == 0:
            _save_tgen(tgen_rows, tgen_path)
            rate = i / max(time.time() - t0, 1e-6)
            print(f"[{dataset}] {i}/{len(samples)}  tgen+{n_tgen_new} igen+{n_igen_new}  "
                  f"{rate:.1f} samp/s", flush=True)

    _save_tgen(tgen_rows, tgen_path)
    print("=" * 60, flush=True)
    print(f"[done] {dataset}: tgen uids={len(tgen_rows)} (+{n_tgen_new}), "
          f"igen images +{n_igen_new}", flush=True)
    print(f"  t-gen parquet: {tgen_path}", flush=True)
    print(f"  i-gen images:  {IGEN_ROOT / dataset}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build t-gen + i-gen generality sets")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--k_text", type=int, default=2, help="rephrasings per sample")
    parser.add_argument("--k_image", type=int, default=2, help="augmented images per sample")
    parser.add_argument("--aug_strength", type=int, default=DEFAULT_AUG_STRENGTH,
                        choices=[1, 2, 3, 4, 5],
                        help="i-gen strength bin (1=mild .. 4=today .. 5=strong)")
    parser.add_argument("--deployment", type=str, default=DEFAULT_DEPLOYMENT,
                        help="Azure gpt-4o deployment name")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate even if outputs exist")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process first N samples (0=all; for smoke tests)")
    parser.add_argument("--checkpoint_every", type=int, default=50)
    args = parser.parse_args()

    build(
        args.dataset,
        k_text=args.k_text,
        k_image=args.k_image,
        aug_strength=args.aug_strength,
        deployment=args.deployment,
        overwrite=args.overwrite,
        limit=args.limit,
        checkpoint_every=args.checkpoint_every,
    )
