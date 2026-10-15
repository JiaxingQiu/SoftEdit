"""Step 4: Domain-refine soft-generality images to native resolution.

Soft-gen outputs are 512×512 RGB (FLUX edit scale). For FERPlus the native
domain is 48×48 grayscale. This script reuses ``blend_generality.refine.refine_image``
but with its own PRESETS: soft-gen images are FLUX *edits* of the upscaled
original, so the face framing already matches native FER — no crop needed
(unlike blend, where T2I generates from scratch with wider head+neck framing
that requires crop_frac=0.75).

Reads  data/soft_generality/images/{dataset}/          (512² originals)
Writes data/soft_generality/images/{dataset}_refined/   (native domain)

Idempotent — skips existing files. After verifying output, swap manually:
    cd data/soft_generality/images
    mv {dataset} {dataset}_raw && mv {dataset}_refined {dataset}

Usage (env: soft_gen, CPU-only):
    python -m prep_generality.soft_generality.run.refine --dataset vqav2 [--limit 20]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from PIL import Image

from prep_generality.blend_generality.refine import refine_image

# Soft-gen presets differ from blend: no crop needed because FLUX edits
# preserve the original FER face framing (blend's T2I generates from scratch
# with wider head+neck framing, hence its crop_frac=0.75).
PRESETS = {
    "ferplus": dict(size=48, grayscale=True, hist_eq=False, noise=0.0,
                    crop_frac=1.0, crop_y=0.5),
    "cifar10h": dict(size=32, grayscale=False, hist_eq=False, noise=0.0,
                     crop_frac=1.0, crop_y=0.5),
}

SOFT_GEN_ROOT = PROJECT_ROOT / "data" / "soft_generality" / "images"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _uid_seed(uid: str, filename: str) -> int:
    """Stable per-(uid, filename) seed so output is reproducible."""
    return (abs(hash((uid, filename))) % 1_000_000_000)


def build(dataset: str, *, suffix: str = "refined", overwrite: bool = False,
          limit: int = 0, **cfg) -> None:
    if dataset not in PRESETS:
        raise KeyError(f"No refine preset for {dataset!r}. Have: {sorted(PRESETS)}")
    params = {**PRESETS[dataset], **{k: v for k, v in cfg.items() if v is not None}}

    src_root = SOFT_GEN_ROOT / dataset
    dst_root = SOFT_GEN_ROOT / f"{dataset}_{suffix}"
    if not src_root.exists():
        raise FileNotFoundError(f"soft-gen images not found: {src_root}")

    uid_dirs = sorted(p for p in src_root.iterdir() if p.is_dir())
    if limit:
        uid_dirs = uid_dirs[:limit]

    print("=" * 60, flush=True)
    print(f"refine soft-gen | dataset={dataset} | uids={len(uid_dirs)}", flush=True)
    print(f"params: {params}", flush=True)
    print(f"{src_root}  ->  {dst_root}", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    n_new = n_skip = 0
    for i, uid_dir in enumerate(uid_dirs, 1):
        uid = uid_dir.name
        for src in sorted(uid_dir.iterdir()):
            if not src.is_file() or src.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            dst = dst_root / uid / src.name
            if dst.exists() and not overwrite:
                n_skip += 1
                continue
            out = refine_image(
                Image.open(src),
                seed=_uid_seed(uid, src.name),
                **params,
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix.lower() in {".jpg", ".jpeg"}:
                out.save(dst, "JPEG", quality=92)
            else:
                out.save(dst)
            n_new += 1
        if i % 500 == 0:
            rate = i / max(time.time() - t0, 1e-6)
            print(f"[{dataset}] {i}/{len(uid_dirs)}  +{n_new} skip{n_skip}  "
                  f"{rate:.1f} uid/s", flush=True)

    elapsed = time.time() - t0
    print("=" * 60, flush=True)
    print(f"[done] {dataset}: refined +{n_new} (skipped {n_skip}) -> {dst_root}  "
          f"({elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refine soft-gen images into a dataset's native domain (Step 4)"
    )
    parser.add_argument("--dataset", required=True, choices=sorted(PRESETS))
    parser.add_argument("--suffix", default="refined",
                        help="output folder = {dataset}_{suffix}")
    parser.add_argument("--size", type=int, default=None,
                        help="override native size")
    parser.add_argument("--crop_frac", type=float, default=None,
                        help="central square crop as fraction of shorter side")
    parser.add_argument("--crop_y", type=float, default=None,
                        help="vertical crop center (0=top..1=bottom)")
    parser.add_argument("--noise", type=float, default=None,
                        help="override gaussian noise frac (0-1)")
    parser.add_argument("--hist_eq", dest="hist_eq", action="store_true",
                        default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0,
                        help="process at most N uids (0=all)")
    args = parser.parse_args()

    build(
        args.dataset,
        suffix=args.suffix,
        overwrite=args.overwrite,
        limit=args.limit,
        size=args.size,
        noise=args.noise,
        crop_frac=args.crop_frac,
        crop_y=args.crop_y,
        hist_eq=args.hist_eq,
    )
