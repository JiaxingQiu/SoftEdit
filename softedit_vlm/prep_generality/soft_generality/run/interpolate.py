"""Step 3: Interpolate valid samples at t ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}.

Reads the GPT-4o validation parquet to determine which UIDs passed.
For each valid UID, encodes original + t=1.0 edit in VAE latent space,
interpolates, decodes, and saves as JPEG.

Idempotent — skips existing files.

Usage:
    python -m prep_generality.soft_generality.run.interpolate --dataset vqav2 --start_idx 0 --end_idx 100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image

from ..shared.sample_io import load_parquet, parse_answer_counts, ranked_labels, resolve_image_path, REPO_ROOT
from ..backends.flux_klein.config import DEFAULT_MODEL_ID, DEFAULT_SIZE, DEFAULT_SEED


OUTPUT_ROOT = REPO_ROOT / "data" / "soft_generality" / "images"
VALIDATION_ROOT = REPO_ROOT / "data" / "soft_generality" / "validation"
INTERP_STRENGTHS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
JPEG_QUALITY = 92


def _t1_path(dataset: str, uid: str, label: str) -> Path:
    safe_label = label.replace("/", "_").replace(" ", "_")
    return OUTPUT_ROOT / dataset / uid / f"{safe_label}_t1.0.png"


def _interp_path(dataset: str, uid: str, label: str, t: float) -> Path:
    safe_label = label.replace("/", "_").replace(" ", "_")
    return OUTPUT_ROOT / dataset / uid / f"{safe_label}_t{t}.jpg"


def _load_valid_uids(dataset: str) -> set[str]:
    """Load passed UIDs from per-UID validation parquet (uid, gpt4o_valid)."""
    import pandas as pd
    parquet_path = VALIDATION_ROOT / dataset / f"{dataset}_gpt4o.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Validation parquet not found: {parquet_path}. Run validate_gpt4o first."
        )
    df = pd.read_parquet(parquet_path)
    return set(df.loc[df["gpt4o_valid"], "uid"].tolist())


def log(msg: str):
    print(msg, flush=True)


def run(dataset: str, start_idx: int, end_idx: int):
    valid_uids = _load_valid_uids(dataset)
    log(f"[interpolate] dataset={dataset}  valid_uids={len(valid_uids)}")

    df = load_parquet(dataset)
    df = df.iloc[start_idx:end_idx]
    n_total = len(df)
    log(f"[interpolate] processing rows {start_idx}..{end_idx} ({n_total} rows)")

    from ..backends.flux_klein.edit import FluxKleinEditor
    log("[interpolate] loading FluxKleinEditor (offload=False)...")
    editor = FluxKleinEditor(offload=False)
    log("[interpolate] model loaded.")

    t0 = time.time()
    n_generated = 0
    n_skipped = 0
    n_invalid = 0

    for row_i, (_, row) in enumerate(df.iterrows()):
        uid = row["uid"]
        if uid not in valid_uids:
            n_invalid += 1
            continue

        counts = parse_answer_counts(row["answer_counts"])
        labels = ranked_labels(counts)
        image_path = resolve_image_path(row["image_path"])

        if not image_path.exists():
            continue

        source_img = Image.open(image_path).convert("RGB")
        original_resized = source_img.resize(
            (DEFAULT_SIZE, DEFAULT_SIZE), Image.Resampling.LANCZOS
        )

        for label in labels:
            t1_file = _t1_path(dataset, uid, label)
            if not t1_file.exists():
                continue

            all_exist = all(
                _interp_path(dataset, uid, label, t).exists() for t in INTERP_STRENGTHS
            )
            if all_exist:
                n_skipped += len(INTERP_STRENGTHS)
                continue

            edited_img = Image.open(t1_file).convert("RGB")

            for t in INTERP_STRENGTHS:
                out_file = _interp_path(dataset, uid, label, t)
                if out_file.exists():
                    n_skipped += 1
                    continue

                interp_img = editor.interpolate(
                    original_resized, edited_img, t, DEFAULT_SIZE, DEFAULT_SIZE
                )

                out_file.parent.mkdir(parents=True, exist_ok=True)
                interp_img.save(out_file, "JPEG", quality=JPEG_QUALITY)
                n_generated += 1

        if n_generated > 0 and n_generated % 30 == 0:
            elapsed = time.time() - t0
            rate = n_generated / elapsed
            log(f"  [{row_i+1}/{n_total}] uid={uid} "
                f"generated={n_generated} ({rate:.1f} img/s)")

    elapsed = time.time() - t0
    log(f"\n[interpolate] done. generated={n_generated}  skipped={n_skipped}  "
        f"invalid_uids={n_invalid}  time={elapsed:.0f}s ({elapsed/3600:.1f}h)")


def main():
    parser = argparse.ArgumentParser(description="Interpolate valid samples (Step 3)")
    parser.add_argument("--dataset", required=True, choices=["vqav2"])
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    args = parser.parse_args()

    end_idx = args.end_idx
    if end_idx is None:
        end_idx = len(load_parquet(args.dataset))

    run(args.dataset, args.start_idx, end_idx)


if __name__ == "__main__":
    main()
