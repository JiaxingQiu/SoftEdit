"""Text-generality input loader.

Loads rephrased question variants from:
    data/generality/tgen/{dataset}.parquet  (columns: uid, variants)

Returns the same interface as the legacy HF loader: {uid: [q1, q2, ...]}.
Gold labels are derived on the fly from the source parquet's answer_counts
(rephrasing is meaning-preserving → labels invariant).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
TGEN_ROOT = REPO_ROOT / "data" / "generality" / "tgen"


def get_t_gen_input(
    dataset_name: str,
    edit_ds,
    k: int = 2,
) -> Dict[str, List[str]]:
    """Load rephrased question variants for edit-set samples.

    Args:
        dataset_name: vqav2
        edit_ds: VQADataset with .data containing edit-set examples
        k: max variants per sample (default 2 to match generation)

    Returns:
        {uid: [rephrased_q1, rephrased_q2, ...]} filtered to edit-set UIDs
    """
    parquet_path = TGEN_ROOT / f"{dataset_name}.parquet"
    if not parquet_path.exists():
        print(f"[t_gen] WARNING: {parquet_path} not found — returning empty")
        return {}

    edit_uids = {str(ex["uid"]) for ex in edit_ds.data}
    df = pd.read_parquet(parquet_path)
    df = df[df["uid"].astype(str).isin(edit_uids)]

    related_texts: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        uid = str(row["uid"])
        variants = list(row["variants"])[:k]
        if variants:
            related_texts[uid] = variants

    return related_texts
