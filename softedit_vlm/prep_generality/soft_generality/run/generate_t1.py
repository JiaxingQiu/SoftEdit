"""Step 1: Generate t=1.0 (full) edits for all labels of each sample.

Uses FLUX.2-klein-9B. Saves one JPEG per (uid, label).
Idempotent — skips existing files.

Usage:
    python -m prep_generality.soft_generality.run.generate_t1 --dataset vqav2 --start_idx 0 --end_idx 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from ..shared.sample_io import load_parquet, parse_answer_counts, ranked_labels, resolve_image_path, REPO_ROOT
from ..shared.prompts import build_prompt
from ..backends.flux_klein.config import DEFAULT_MODEL_ID, DEFAULT_SEED, DEFAULT_SIZE, DEFAULT_STEPS


OUTPUT_ROOT = REPO_ROOT / "data" / "soft_generality" / "images"
JPEG_QUALITY = 92


def output_path(dataset: str, uid: str, label: str) -> Path:
    """t=1.0 saved as PNG (lossless) so interpolation matches the one-sample experiment."""
    safe_label = label.replace("/", "_").replace(" ", "_")
    return OUTPUT_ROOT / dataset / uid / f"{safe_label}_t1.0.png"


def log(msg: str):
    print(msg, flush=True)


def run(dataset: str, start_idx: int, end_idx: int):
    df = load_parquet(dataset)
    df = df.iloc[start_idx:end_idx]
    n_total = len(df)
    log(f"[generate_t1] dataset={dataset}  rows={n_total} (idx {start_idx}..{end_idx})")

    from ..backends.flux_klein.edit import FluxKleinEditor
    log("[generate_t1] loading FluxKleinEditor (offload=False)...")
    editor = FluxKleinEditor(offload=False)
    log("[generate_t1] model loaded.")

    t0 = time.time()
    n_generated = 0
    n_skipped = 0

    for row_i, (_, row) in enumerate(df.iterrows()):
        uid = row["uid"]
        question = row["question"]
        counts = parse_answer_counts(row["answer_counts"])
        labels = ranked_labels(counts)
        image_path = resolve_image_path(row["image_path"])

        if not image_path.exists():
            log(f"  [{row_i+1}/{n_total}] uid={uid}  SKIP (image not found)")
            continue

        all_exist = all(output_path(dataset, uid, lab).exists() for lab in labels)
        if all_exist:
            n_skipped += len(labels)
            continue

        source_img = Image.open(image_path).convert("RGB").resize(
            (DEFAULT_SIZE, DEFAULT_SIZE), Image.Resampling.LANCZOS
        )
        log(f"  [{row_i+1}/{n_total}] uid={uid}  labels={len(labels)}")

        for lab_i, label in enumerate(labels):
            out_file = output_path(dataset, uid, label)
            if out_file.exists():
                n_skipped += 1
                continue

            prompt = build_prompt(question, label, dataset=dataset)
            result = editor.edit(
                source_img,
                prompt,
                num_inference_steps=DEFAULT_STEPS,
                height=DEFAULT_SIZE,
                width=DEFAULT_SIZE,
                seed=DEFAULT_SEED,
            )

            out_file.parent.mkdir(parents=True, exist_ok=True)
            result.image.save(out_file, "PNG")
            n_generated += 1

            elapsed = time.time() - t0
            rate = n_generated / elapsed if elapsed > 0 else 0
            log(f"    label {lab_i+1}/{len(labels)} \"{label}\"  "
                f"total={n_generated} ({rate:.2f} img/s, {elapsed/60:.1f}min)")

    elapsed = time.time() - t0
    log(f"\n[generate_t1] done. generated={n_generated}  skipped={n_skipped}  "
        f"time={elapsed:.0f}s ({elapsed/3600:.1f}h)")


def main():
    parser = argparse.ArgumentParser(description="Generate t=1.0 edits (Step 1)")
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
