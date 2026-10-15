"""Text-generality (t-gen) input loader.

Loads rephrased question variants from `data/generality/tgen/{dataset}.parquet`
(columns: uid, variants) — local-first, else HuggingFace. Returns
``{uid: [q1, q2, ...]}`` filtered to the edit-set UIDs.

Gold labels are derived on the fly from the source parquet's `answer_counts`
(rephrasing is meaning-preserving → labels invariant), so `text_generality`
reuses the same reliability pass.
"""
from __future__ import annotations

from typing import Dict, List

from rellm.dataset.utils.data import load_generality_parquet


def get_t_gen_input(dataset_name: str, edit_ds, k: int = 4) -> Dict[str, List[str]]:
    """Rephrased question variants for edit-set samples.

    Args:
        dataset_name: e.g. "protoqa"
        edit_ds: TQADataset whose ``.data`` holds the edit-set examples
        k: max variants per sample (default 4, matches generation)
    Returns:
        {uid: [rephrased_q1, ...]} filtered to edit-set UIDs (empty if none).
    """
    try:
        df = load_generality_parquet("tgen", dataset_name)
    except FileNotFoundError as e:
        print(f"[t_gen] WARNING: {e} — returning empty")
        return {}

    edit_uids = {str(ex["uid"]) for ex in edit_ds.data}
    df = df[df["uid"].astype(str).isin(edit_uids)]

    related_texts: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        variants = list(row["variants"])[:k]
        if variants:
            related_texts[str(row["uid"])] = variants
    return related_texts
