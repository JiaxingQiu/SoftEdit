"""Refine blend images into a dataset's native domain (center-crop -> [grayscale]
-> downscale). Clean 512² FLUX renders are "too easy"; matching the native low-res
form (FERPlus 48² gray, CIFAR-10H 32² RGB) makes blend i-gen reflect the original
sample's difficulty. Label-safe: it's the same form annotators labeled.

Reads  data/blend_generality/{dataset}/         (clean blends)
Writes data/blend_generality/{dataset}_refined/  (idempotent; clean folder untouched)

To make refined canonical (eval/editing read .../{dataset} unchanged), manually
swap after refining, then re-run affected jobs. Cancel any job reading the folder
first, or the mv corrupts it:
    cd data/blend_generality && mv {dataset} {dataset}_clean && mv {dataset}_refined {dataset}

Usage (env: soft_gen, CPU-only):
    python -m prep_generality.blend_generality.refine --dataset vqav2 [--limit 20]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
from PIL import Image, ImageOps

from prep_generality.blend_generality.paths import BLEND_ROOT

# Native-domain presets per dataset (measured from the source parquets).
# size      : native resolution (square)
# grayscale : collapse to single channel (replicated back to RGB on save)
# hist_eq   : histogram equalization (off by default)
# noise     : std of additive gaussian noise as a fraction of 255 (0 = none)
# crop_frac : central square crop as a fraction of the shorter side (1.0 = full).
#             FLUX blends frame the whole head/neck; FER is a tight forehead-to-chin
#             crop, so <1 zooms into the face.
# crop_y    : vertical center of the crop (0=top..1=bottom); <0.5 biases upward
#             onto the face rather than the neck.
PRESETS = {
    "ferplus": dict(size=48, grayscale=True, hist_eq=False, noise=0.0,
                    crop_frac=0.75, crop_y=0.45),
    "cifar10h": dict(size=32, grayscale=False, hist_eq=False, noise=0.0,
                     crop_frac=1.0, crop_y=0.5),
}


def refine_image(
    img: Image.Image,
    *,
    size: int,
    grayscale: bool = True,
    hist_eq: bool = False,
    noise: float = 0.0,
    crop_frac: float = 1.0,
    crop_y: float = 0.5,
    seed: int = 0,
) -> Image.Image:
    """Domain-match a single blend image. Deterministic given ``seed``.

    central square crop (zoomed by ``crop_frac``, vertically centered at ``crop_y``)
    -> [grayscale] -> [hist_eq] -> downscale to size -> [gaussian noise] -> RGB
    """
    im = img.convert("RGB")

    # central square crop, zoomed to `crop_frac` of the shorter side and centered
    # horizontally / at vertical position `crop_y` (tightens onto the face).
    w, h = im.size
    s = max(1, int(round(min(w, h) * crop_frac)))
    left = (w - s) // 2
    top = int(round(crop_y * h - s / 2))
    top = max(0, min(top, h - s))
    im = im.crop((left, top, left + s, top + s))

    if grayscale:
        im = im.convert("L")
    if hist_eq:
        im = ImageOps.equalize(im)

    # downscale to the native resolution (the key domain feature)
    im = im.resize((size, size), Image.BICUBIC)

    if noise > 0:
        arr = np.asarray(im).astype(np.float32)
        rng = np.random.default_rng(seed)
        arr = arr + rng.normal(0.0, noise * 255.0, arr.shape)
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode=im.mode)

    return im.convert("RGB")


def _uid_seed(uid: str, round_idx: int) -> int:
    """Stable per-(uid, round) seed so noise is reproducible."""
    return (abs(hash(uid)) % 1_000_000) * 10 + round_idx


def build(dataset: str, *, suffix: str = "refined", overwrite: bool = False,
          limit: int = 0, **cfg) -> None:
    if dataset not in PRESETS:
        raise KeyError(f"No refine preset for {dataset!r}. Have: {sorted(PRESETS)}")
    params = {**PRESETS[dataset], **{k: v for k, v in cfg.items() if v is not None}}

    src_root = BLEND_ROOT / dataset
    dst_root = BLEND_ROOT / f"{dataset}_{suffix}"
    if not src_root.exists():
        raise FileNotFoundError(f"clean blends not found: {src_root}")

    uid_dirs = sorted(p for p in src_root.iterdir() if p.is_dir())
    if limit:
        uid_dirs = uid_dirs[:limit]

    print("=" * 60, flush=True)
    print(f"refine blends | dataset={dataset} | uids={len(uid_dirs)}", flush=True)
    print(f"params: {params}", flush=True)
    print(f"{src_root}  ->  {dst_root}", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    n_new = n_skip = 0
    for i, uid_dir in enumerate(uid_dirs, 1):
        uid = uid_dir.name
        for src in sorted(uid_dir.glob("blend*.png")):
            dst = dst_root / uid / src.name
            if dst.exists() and not overwrite:
                n_skip += 1
                continue
            round_idx = int("".join(c for c in src.stem if c.isdigit()) or 0)
            out = refine_image(Image.open(src), seed=_uid_seed(uid, round_idx), **params)
            dst.parent.mkdir(parents=True, exist_ok=True)
            out.save(dst)
            n_new += 1
        if i % 500 == 0:
            rate = i / max(time.time() - t0, 1e-6)
            print(f"[{dataset}] {i}/{len(uid_dirs)}  +{n_new} skip{n_skip}  {rate:.1f} uid/s",
                  flush=True)

    print("=" * 60, flush=True)
    print(f"[done] {dataset}: refined +{n_new} (skipped {n_skip}) -> {dst_root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine blend images into a dataset's native domain")
    parser.add_argument("--dataset", required=True, choices=sorted(PRESETS))
    parser.add_argument("--suffix", default="refined", help="output folder = {dataset}_{suffix}")
    parser.add_argument("--size", type=int, default=None, help="override native size")
    parser.add_argument("--noise", type=float, default=None, help="override gaussian noise frac (0-1)")
    parser.add_argument("--crop_frac", type=float, default=None,
                        help="central square crop as fraction of shorter side (1.0=full)")
    parser.add_argument("--crop_y", type=float, default=None,
                        help="vertical crop center (0=top..1=bottom; <0.5 biases up onto face)")
    parser.add_argument("--hist_eq", dest="hist_eq", action="store_true", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="process at most N uids (0=all)")
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
