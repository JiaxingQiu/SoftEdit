"""Soft-generality (soft-gen) input loader — text, one-hot gold.

Loads per-target premises from `data/generality/softgen/{dataset}.parquet`
(columns: uid, target_label, context) — local-first, else HuggingFace. For each
edit-set sample it composes ``context + question`` and pairs it with the target
label, whose gold is **one-hot** ``{target_label: 1}``.

Returns ``{uid: [(target_label, composed_question), ...]}`` filtered to the
edit-set UIDs. `soft_generality` (editeval) turns each into a one-hot eval row;
the alignment on these is ``sloc``.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from rellm.dataset.utils.data import load_generality_parquet


def _compose(context: str, question: str) -> str:
    """Prepend a premise to the question (matches softgen.compose_question)."""
    context = (context or "").strip()
    return f"{context} {question}".strip() if context else str(question).strip()


def get_soft_gen_input(dataset_name: str, edit_ds) -> Dict[str, List[Tuple[str, str]]]:
    """Per-target composed prompts for edit-set samples.

    Returns {uid: [(target_label, "context question"), ...]} for edit-set UIDs
    (empty if the parquet is missing).
    """
    try:
        df = load_generality_parquet("softgen", dataset_name)
    except FileNotFoundError as e:
        print(f"[soft_gen] WARNING: {e} — returning empty")
        return {}

    q_by_uid = {str(ex["uid"]): ex["question"] for ex in edit_ds.data}
    df = df[df["uid"].astype(str).isin(q_by_uid.keys())]

    out: Dict[str, List[Tuple[str, str]]] = {}
    for _, row in df.iterrows():
        uid = str(row["uid"])
        target = str(row["target_label"])
        context = str(row["context"]) if row["context"] is not None else ""
        if not target or not context:
            continue
        out.setdefault(uid, []).append((target, _compose(context, q_by_uid[uid])))
    return out
