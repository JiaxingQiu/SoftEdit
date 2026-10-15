"""Soft-generality input loader (single-label eval).

Loads FLUX-generated single-label images from:
    data/soft_generality/images/{dataset}/{uid}/{label}_t{strength}.{png,jpg}

For single-label eval, gold = [target_label] (the label in the filename).
The caller chooses which strength levels to include via the ``strengths``
parameter.

Returns: {uid: [path1, path2, ...]} for matching images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
SOFT_GEN_ROOT = REPO_ROOT / "data" / "soft_generality" / "images"

DEFAULT_STRENGTHS = (0.1, 0.3, 0.5, 0.7, 1.0)


def _strength_from_path(p: Path) -> float | None:
    """Parse the strength value from a filename like 'cat_t0.7.jpg'."""
    stem = p.stem  # e.g. "cat_t0.7"
    parts = stem.rsplit("_t", 1)
    if len(parts) != 2:
        return None
    try:
        return float(parts[1])
    except ValueError:
        return None


def get_soft_gen_input(
    dataset_name: str,
    edit_ds,
    strengths: Sequence[float] = DEFAULT_STRENGTHS,
) -> Dict[str, List[str]]:
    """Load soft_generality image paths for edit-set samples.

    Args:
        dataset_name: vqav2
        edit_ds: VQADataset with .data containing edit-set examples
        strengths: which interpolation strengths to include (default 0.1, 0.3, 0.5, 0.7, 1.0)

    Returns:
        {uid: [path1, path2, ...]} filtered to edit-set UIDs and matching strengths.
    """
    ds_dir = SOFT_GEN_ROOT / dataset_name
    if not ds_dir.exists():
        print(f"[soft_gen] WARNING: {ds_dir} not found — returning empty")
        return {}

    strength_set = {round(s, 2) for s in strengths}
    edit_uids = {str(ex["uid"]) for ex in edit_ds.data}
    related_images: Dict[str, List[str]] = {}

    for uid in edit_uids:
        uid_dir = ds_dir / uid
        if not uid_dir.exists():
            continue
        matched = []
        for p in sorted(uid_dir.iterdir()):
            if not p.is_file():
                continue
            t = _strength_from_path(p)
            if t is not None and round(t, 2) in strength_set:
                matched.append(str(p))
        if matched:
            related_images[uid] = matched

    return related_images


def get_soft_gen_input_by_strength(
    dataset_name: str,
    edit_ds,
    strengths: Sequence[float] | None = DEFAULT_STRENGTHS,
) -> Dict[float, Dict[str, List[str]]]:
    """Soft-gen image paths grouped by edit strength, for per-strength eval.

    Unlike ``get_soft_gen_input`` (which collapses all strengths into one set),
    this returns ``{strength: {uid: [path, ...]}}`` so each ambiguity level can
    be scored separately (soft_generality@0.1, @0.3, ...). Defaults to
    ``DEFAULT_STRENGTHS`` (0.1, 0.3, 0.5, 0.7, 1.0); pass ``strengths=None`` to take
    every strength present on disk instead.
    """
    ds_dir = SOFT_GEN_ROOT / dataset_name
    if not ds_dir.exists():
        print(f"[soft_gen] WARNING: {ds_dir} not found — returning empty")
        return {}

    want = {round(float(s), 2) for s in strengths} if strengths is not None else None
    edit_uids = {str(ex["uid"]) for ex in edit_ds.data}
    by_strength: Dict[float, Dict[str, List[str]]] = {}

    for uid in edit_uids:
        uid_dir = ds_dir / uid
        if not uid_dir.exists():
            continue
        for p in sorted(uid_dir.iterdir()):
            if not p.is_file():
                continue
            t = _strength_from_path(p)
            if t is None:
                continue
            t = round(t, 2)
            if want is not None and t not in want:
                continue
            by_strength.setdefault(t, {}).setdefault(uid, []).append(str(p))

    return by_strength


def soft_gen_gold_from_path(image_path: str) -> List[str]:
    """Extract the single gold label from a soft_gen image filename.

    Filename pattern: {label}_t{strength}.{png,jpg}
    Gold = [label] regardless of strength (single-label eval).
    """
    stem = Path(image_path).stem  # e.g. "cat_t0.7"
    label = stem.rsplit("_t", 1)[0]
    return [label]
