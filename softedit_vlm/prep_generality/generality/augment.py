"""Label-safe image augmentations for image-generality (i-gen).

Each sample gets ``k`` label-preserving variants. Every variant runs the same
base "jitter":

  * "jitter" — jointly apply the FULL combo of photometric/geometric effects
    (gaussian noise, brightness, contrast, aspect stretch, rotation) with random
    magnitudes. All effects always fire; magnitudes are randomized per
    (uid, index) so variants differ. Stretch/rotation empty areas are filled
    with a neutral bg image (not black).

A "scene" wrap is then optionally added at the very END of jitter: place the
finished jittered image (mild padding) into a "fake new scene" formed by a
RANDOM REAL background image (sampled from vqav2). Probes background-context /
scale robustness. With the default k=2: one jitter image + one jitter+scene.

Global strength (``aug_strength``) is a discrete bin in 1..5 that scales EVERY
axis (jitter magnitudes + scene padding) together:

    scale = aug_strength / 4      # bin 4 == "today"; bin 5 is stronger

    bin 1 (mild)  ... bin 3 (default) ... bin 4 (today) ... bin 5 (strong)

Label safety: the original content is never cropped, flipped, or recolored; the
scene variant only adds a modest border (subject stays dominant), so the answer
is preserved. Deterministic per (uid, index).

Deliberately EXCLUDED (can flip vqav2 answers): flip, crop, hue/color shift,
grayscale.
"""

from __future__ import annotations

import hashlib
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance

# strength bins: 1..5, converted to a continuous scale (bin 4 == today's level)
DEFAULT_AUG_STRENGTH = 3
_ANCHOR = 4.0  # bin whose scale == 1.0 ("today")

# base magnitude ranges at scale 1.0 (== bin 4)
_NOISE = (6.0, 14.0)    # gaussian noise sigma
_BC = 0.2               # brightness / contrast max deviation from 1.0
_ROT = 24.0             # rotation degrees at scale 1.0 (bin 4); max possible = ±30 at bin 5
_STRETCH = 0.25         # max per-axis squeeze fraction at scale 1.0 (aspect distortion)
_PAD = 0.3              # scene padding fraction: area = 1 - _PAD * scale
_AREA_FLOOR = 0.2       # never shrink the subject below this area fraction


def _seed(uid: str, tag) -> int:
    h = hashlib.sha256(f"{uid}:{tag}".encode()).hexdigest()
    return int(h[:8], 16)


def _scale(aug_strength: float) -> float:
    return float(aug_strength) / _ANCHOR


def _photometric(img: Image.Image, rng: np.random.Generator, scale: float) -> Image.Image:
    """Gaussian noise + brightness + contrast, magnitudes scaled by ``scale``."""
    img = img.convert("RGB")
    sigma = float(rng.uniform(*_NOISE)) * scale
    arr = np.asarray(img).astype(np.float32) + rng.normal(0.0, sigma, np.asarray(img).shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img = ImageEnhance.Brightness(img).enhance(1.0 + float(rng.uniform(-_BC, _BC)) * scale)
    img = ImageEnhance.Contrast(img).enhance(1.0 + float(rng.uniform(-_BC, _BC)) * scale)
    return img


def _stretch(
    img: Image.Image,
    rng: np.random.Generator,
    scale: float,
    fill_bg: Image.Image | None = None,
) -> Image.Image:
    """Anisotropic aspect distortion, keeping the frame size and all content.

    Each axis is independently squeezed by up to ``_STRETCH * scale`` and the
    result is centered on a same-size canvas; empty bands are filled with
    ``fill_bg`` (neutral bg) instead of black. Squeeze-only (factors <= 1) so no
    content is ever cropped.
    """
    w, h = img.size
    fx = 1.0 - float(rng.uniform(0.0, _STRETCH)) * scale
    fy = 1.0 - float(rng.uniform(0.0, _STRETCH)) * scale
    nw, nh = max(1, int(round(w * fx))), max(1, int(round(h * fy)))
    resized = img.resize((nw, nh), Image.LANCZOS)
    if fill_bg is not None:
        canvas = fill_bg.convert("RGB").resize((w, h), Image.LANCZOS)
    else:
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(resized, ((w - nw) // 2, (h - nh) // 2))
    return canvas


def _jitter(
    img: Image.Image,
    rng: np.random.Generator,
    scale: float,
    fill_bg: Image.Image | None = None,
) -> Image.Image:
    """Full base effect: photometric combo, then aspect stretch, then rotation.

    Stretch and rotation run last; their exposed areas are filled with ``fill_bg``
    (a neutral background image) instead of black. Falls back to black if no bg
    is given.
    """
    img = _photometric(img, rng, scale)
    img = _stretch(img, rng, scale, fill_bg=fill_bg)
    angle = float(rng.uniform(-_ROT, _ROT)) * scale
    rotated = img.rotate(angle, resample=Image.BILINEAR, expand=False)
    if fill_bg is None:
        return rotated
    mask = Image.new("L", img.size, 255).rotate(angle, resample=Image.BILINEAR, expand=False)
    canvas = fill_bg.convert("RGB").resize(img.size, Image.LANCZOS)
    canvas.paste(rotated, (0, 0), mask)
    return canvas


def _scene_pad(
    img: Image.Image,
    rng: np.random.Generator,
    bg_pool: Sequence[Image.Image],
    scale: float,
) -> Image.Image:
    """Wrap a finished image in a "fake new scene": mild padding onto a real bg."""
    orig_w, orig_h = img.size
    area_pct = max(_AREA_FLOOR, min(1.0, 1.0 - _PAD * scale))
    pad_ratio = (1.0 / area_pct) ** 0.5 - 1.0
    new_w = int(orig_w * (1 + pad_ratio))
    new_h = int(orig_h * (1 + pad_ratio))

    bg = bg_pool[int(rng.integers(len(bg_pool)))].convert("RGB")
    canvas = bg.resize((new_w, new_h), Image.LANCZOS)

    max_x, max_y = new_w - orig_w, new_h - orig_h
    x = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
    y = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
    canvas.paste(img, (x, y))
    return canvas.resize((orig_w, orig_h), Image.LANCZOS)


def make_augmentations(
    img: Image.Image,
    uid: str,
    k: int = 2,
    bg_pool: Sequence[Image.Image] | None = None,
    aug_strength: int = DEFAULT_AUG_STRENGTH,
) -> List[Tuple[str, Image.Image]]:
    """Return k (tag, augmented_image) pairs, deterministic per uid.

    Every variant runs the same base jitter (photometric + rotation, rotation
    corners filled with a neutral bg). A scene wrap (mild padding onto a random
    real bg) is then optionally added at the very end. With the default k=2 that
    yields one jitter image and one jitter+scene image. ``aug_strength`` (bin
    1..5) scales every axis together.
    """
    img = img.convert("RGB")
    scale = _scale(aug_strength)
    out: List[Tuple[str, Image.Image]] = []
    for i in range(k):
        rng = np.random.default_rng(_seed(uid, i))
        fill_bg = bg_pool[int(rng.integers(len(bg_pool)))] if bg_pool else None
        aug = _jitter(img, rng, scale, fill_bg=fill_bg)

        use_scene = bool(bg_pool) and (i % 2 == 1)
        if use_scene:
            aug = _scene_pad(aug, rng, bg_pool, scale)

        tag = ("jitter+scene" if use_scene else "jitter") + f"(s{aug_strength})"
        out.append((tag, aug))
    return out


__all__ = ["make_augmentations", "DEFAULT_AUG_STRENGTH"]
