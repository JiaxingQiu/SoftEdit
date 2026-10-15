"""Blend-generality input loader (ladder layout).

Loads distributional-blend ladder images from:
    data/blend_generality_tune/{dataset}/{uid}/tune/{label}/t{strength}.png

Each label subfolder contains images at t=0.0 (distributional blend),
0.3, 0.5, 0.7, and 1.0 (pure anchor).

Gold at strength t:
    t=0.0  → edit's full answer_counts (ambiguity-preserving blend)
    t=1.0  → one_hot(label) (pure single-class anchor)
    0<t<1  → (1-t)*p_edit + t*one_hot(label)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
BLEND_ROOT = REPO_ROOT / "data" / "blend_generality_tune"

BLEND_CAP = 3
DEFAULT_STRENGTHS = (0.0, 0.5, 1.0)
ALL_STRENGTHS = (0.0, 0.3, 0.5, 0.7, 1.0)


def _safe_label(label: str) -> str:
    return str(label).replace("/", "_").replace(" ", "_")


def blend_gold_counts(answer_counts, cap: int = BLEND_CAP) -> Dict[str, int]:
    """Gold answer_counts for blend images: the top-``cap`` labels by frequency.

    Keeps original counts so ranked metrics preserve frequency-tier structure.
    Accepts a dict or a JSON string.
    """
    if isinstance(answer_counts, str):
        answer_counts = json.loads(answer_counts)
    ranked = sorted(answer_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {lab: cnt for lab, cnt in ranked[:cap]}


def blend_gen_label_from_path(image_path: str) -> str:
    """Extract the target label from a ladder image path.

    Path pattern: .../tune/{label}/t{strength}.png
    Returns the label directory name.
    """
    p = Path(image_path)
    return p.parent.name


def blend_gen_strength_from_path(image_path: str) -> float | None:
    """Extract the strength from a ladder image filename like 't0.5.png'."""
    stem = Path(image_path).stem
    if not stem.startswith("t"):
        return None
    try:
        return float(stem[1:])
    except ValueError:
        return None


def get_blend_gen_input(
    dataset_name: str,
    edit_ds,
    strengths: Sequence[float] = DEFAULT_STRENGTHS,
    *,
    k: int | None = None,
    dedupe_t0: bool = True,
) -> Dict[str, List[str]]:
    """Load blend_gen ladder image paths (flat, all strengths merged).

    Args:
        k: ignored (kept for backward compatibility with old callers).
        dedupe_t0: if True (eval default), keep only one t=0.0 image per uid
            because gold is shared. Set False to keep every label's t=0.0
            (gallery / per-label ladders).

    Returns:
        {uid: [path1, path2, ...]} filtered to edit-set UIDs.
    """
    ds_dir = BLEND_ROOT / dataset_name
    if not ds_dir.exists():
        print(f"[blend_gen] WARNING: {ds_dir} not found — returning empty")
        return {}

    strength_set = {round(s, 2) for s in strengths}
    edit_uids = {str(ex["uid"]) for ex in edit_ds.data}
    related_images: Dict[str, List[str]] = {}

    for uid in edit_uids:
        tune_dir = ds_dir / uid / "tune"
        if not tune_dir.exists():
            continue
        matched = []
        t0_taken = False
        for label_dir in sorted(tune_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            for p in sorted(label_dir.glob("t*.png")):
                t = blend_gen_strength_from_path(str(p))
                if t is None or round(t, 2) not in strength_set:
                    continue
                if dedupe_t0 and round(t, 2) == 0.0 and t0_taken:
                    continue
                matched.append(str(p))
                if round(t, 2) == 0.0:
                    t0_taken = True
        if matched:
            related_images[uid] = matched

    return related_images


def get_blend_gen_input_by_strength(
    dataset_name: str,
    edit_ds,
    strengths: Sequence[float] | None = DEFAULT_STRENGTHS,
) -> Dict[float, Dict[str, List[str]]]:
    """Blend-gen ladder paths grouped by strength, for per-strength eval.

    Returns ``{strength: {uid: [path, ...]}}`` so each ambiguity level can
    be scored separately (blend_generality@0.0, @0.5, @1.0).
    """
    ds_dir = BLEND_ROOT / dataset_name
    if not ds_dir.exists():
        print(f"[blend_gen] WARNING: {ds_dir} not found — returning empty")
        return {}

    want = {round(float(s), 2) for s in strengths} if strengths is not None else None
    edit_uids = {str(ex["uid"]) for ex in edit_ds.data}
    by_strength: Dict[float, Dict[str, List[str]]] = {}

    for uid in edit_uids:
        tune_dir = ds_dir / uid / "tune"
        if not tune_dir.exists():
            continue
        t0_taken = False
        for label_dir in sorted(tune_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            for p in sorted(label_dir.glob("t*.png")):
                t = blend_gen_strength_from_path(str(p))
                if t is None:
                    continue
                t = round(t, 2)
                if want is not None and t not in want:
                    continue
                if t == 0.0 and t0_taken:
                    continue
                by_strength.setdefault(t, {}).setdefault(uid, []).append(str(p))
                if t == 0.0:
                    t0_taken = True

    return by_strength
