"""Image-generality input loader.

Loads label-safe augmented images from:
    data/generality/igen/{dataset}/{uid}/aug{0,1}.png

Returns the same interface as t-gen: {uid: [path1, path2, ...]}.
Gold labels are the source sample's ``answer_counts`` (augmentations are
label-preserving).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
IGEN_ROOT = REPO_ROOT / "data" / "generality" / "igen"


def get_i_gen_input(
    dataset_name: str,
    edit_ds,
    k: int = 2,
) -> Dict[str, List[str]]:
    """Load i-gen (augmented) image paths for edit-set samples.

    Args:
        dataset_name: vqav2
        edit_ds: VQADataset with .data containing edit-set examples
        k: max augmented images per sample (default 2 = aug0 + aug1)

    Returns:
        {uid: [aug0_path, aug1_path, ...]} filtered to edit-set UIDs
    """
    ds_dir = IGEN_ROOT / dataset_name
    if not ds_dir.exists():
        print(f"[i_gen] WARNING: {ds_dir} not found — returning empty")
        return {}

    edit_uids = {str(ex["uid"]) for ex in edit_ds.data}
    related_images: Dict[str, List[str]] = {}

    for uid in edit_uids:
        uid_dir = ds_dir / uid
        if not uid_dir.exists():
            continue
        augs = sorted(uid_dir.glob("aug*.png"))[:k]
        if augs:
            related_images[uid] = [str(p) for p in augs]

    return related_images
