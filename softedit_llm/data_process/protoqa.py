"""Build the ProtoQA ranked-uncertainty parquet for rankedit_llm.

ProtoQA (community-datasets/proto_qa) is commonsense "Family Feud" QA where each
question has clustered human answers with vote counts — a natural human-uncertainty
distribution. We keep the **top-3 clusters by count** per question as the gold
distribution (cluster name = canonical label, count = votes).

Output: data/parquet/protoqa/protoqa.parquet with columns
    uid           : str   (ProtoQA id, e.g. "train_q0")
    question      : str   (normalized question, first letter capitalized)
    answer_counts : str   (JSON {label: count} over the kept top-k clusters)

Usage:
    python -m data_process.protoqa                       # train split, top-3
    python -m data_process.protoqa --split train --topk 3 --min_labels 2
    python -m data_process.protoqa --limit 50            # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

HF_REPO = "community-datasets/proto_qa"
HF_CONFIG = "proto_qa"


def _norm_label(s: str) -> str:
    return str(s).strip().lower()


def _clean_question(ex: dict) -> str:
    """Prefer the normalized (lowercased) question; capitalize the first letter."""
    q = (ex.get("normalized-question") or ex.get("question") or "").strip()
    return (q[:1].upper() + q[1:]) if q else q


def _top_clusters(ex: dict, topk: int) -> dict:
    """Return {label: count} for the top-k answer clusters by count.

    Cluster label = the cluster's canonical answer string (`answers[i][0]`;
    scraped train has one string per cluster). Duplicate normalized labels merge
    by summing counts.
    """
    ac = ex.get("answer-clusters") or {}
    counts = ac.get("count") or []
    answers = ac.get("answers") or []
    pairs = []
    for i in range(min(len(counts), len(answers))):
        cluster_answers = answers[i] or []
        if not cluster_answers:
            continue
        label = _norm_label(cluster_answers[0])
        try:
            c = int(counts[i])
        except (TypeError, ValueError):
            continue
        if label and c > 0:
            pairs.append((c, label))
    pairs.sort(key=lambda x: -x[0])
    out: dict = {}
    for c, label in pairs[:topk]:
        out[label] = out.get(label, 0) + c
    return out


def _parse_splits(split: str) -> list:
    """Accept 'train', 'validation', 'all', or 'train+validation' / 'train,validation'."""
    s = str(split).strip().lower()
    if s in ("all", "full", "train+validation", "train+valid"):
        return ["train", "validation"]
    return [p.strip() for p in s.replace(",", "+").split("+") if p.strip()]


def build(split: str = "all", topk: int = 3, min_labels: int = 1, limit: int = 0) -> Path:
    from datasets import load_dataset, concatenate_datasets

    out_path = PROJECT_ROOT / "data" / "parquet" / "protoqa" / "protoqa.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    splits = _parse_splits(split)
    print(f"Loading {HF_REPO} [{HF_CONFIG}] splits={splits} ...", flush=True)
    parts = [load_dataset(HF_REPO, HF_CONFIG, split=sp) for sp in splits]
    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)

    rows = []
    seen_uids = set()
    n_seen = 0
    n_kept = 0
    for ex in ds:
        n_seen += 1
        if limit and n_seen > limit:
            break
        counts = _top_clusters(ex, topk)
        if len(counts) < min_labels:
            continue
        q = _clean_question(ex)
        if not q:
            continue
        uid = str(ex.get("id"))
        if uid in seen_uids:
            continue
        seen_uids.add(uid)
        rows.append({
            "uid": uid,
            "question": q,
            "answer_counts": json.dumps(counts),
        })
        n_kept += 1

    df = pd.DataFrame(rows, columns=["uid", "question", "answer_counts"])
    tmp = out_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    print(f"[done] kept {n_kept}/{n_seen} → {out_path}", flush=True)
    if n_kept:
        ex0 = df.iloc[0]
        print(f"  example: uid={ex0['uid']} q={ex0['question']!r} counts={ex0['answer_counts']}",
              flush=True)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build ProtoQA ranked-uncertainty parquet")
    p.add_argument("--split", type=str, default="all",
                   help="HF split(s): 'all' (train+validation, default) | 'train' | 'validation' "
                        "| 'train+validation'")
    p.add_argument("--topk", type=int, default=3, help="keep top-k clusters by count")
    p.add_argument("--min_labels", type=int, default=1,
                   help="drop questions with fewer than this many kept clusters")
    p.add_argument("--limit", type=int, default=0, help="only process first N (smoke test)")
    args = p.parse_args()
    build(split=args.split, topk=args.topk, min_labels=args.min_labels, limit=args.limit)
