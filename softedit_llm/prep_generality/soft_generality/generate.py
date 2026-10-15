"""Build the soft-gen set: one premise per (question, target) over all top-3
targets → softgen/protoqa.parquet.

Output columns: uid, target_label, context.  A soft-gen eval sample is
`compose_question(context, question)` with **one-hot gold** on target_label
(built at load time in Stage 9); the alignment metric on these is `sloc`.

custom_id = "<uid>#<target_index>" where target_index indexes the top-3 labels
in count-descending order (json.loads(answer_counts).keys()). At collect time the
target string is recovered from the source parquet, so no sidecar is needed.

Usage (sync prototype):
    python -m prep_generality.soft_generality.generate --limit 10
Usage (batch, full pool):
    python -m prep_generality.soft_generality.generate --backend batch --step gen
    python -m prep_generality.soft_generality.generate --backend batch --step submit
    python -m prep_generality.soft_generality.generate --backend batch --step status
    python -m prep_generality.soft_generality.generate --backend batch --step collect
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from prep_generality.soft_generality import softgen
from prep_generality.azure_batch import BatchRunner

DATASET = "protoqa"
SRC_PARQUET = f"data/parquet/{DATASET}/{DATASET}.parquet"


def softgen_parquet_path() -> Path:
    return Path(f"data/generality/softgen/{DATASET}.parquet")


def _iter_targets(limit: int = 0):
    """Yield (uid, question, target_index, target, others) over top-3 labels."""
    df = pd.read_parquet(SRC_PARQUET)
    if limit:
        df = df.head(limit)
    for _, r in df.iterrows():
        labels = list(json.loads(r["answer_counts"]).keys())
        for i, target in enumerate(labels):
            others = [l for j, l in enumerate(labels) if j != i]
            yield str(r["uid"]), r["question"], i, target, others


def _labels_by_uid() -> dict:
    df = pd.read_parquet(SRC_PARQUET)
    return {str(u): list(json.loads(c).keys()) for u, c in zip(df["uid"], df["answer_counts"])}


def _question_by_uid() -> dict:
    df = pd.read_parquet(SRC_PARQUET)
    return {str(u): q for u, q in zip(df["uid"], df["question"])}


def _save(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["uid", "target_label", "context"])
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Sync backend (prototype)
# ---------------------------------------------------------------------------

def build_sync(limit: int = 0, checkpoint_every: int = 50) -> Path:
    out_path = softgen_parquet_path()
    client = softgen.make_client()
    rows = []
    items = list(_iter_targets(limit=limit))
    print(f"[softgen sync] contexts to generate={len(items)} → {out_path}", flush=True)
    t0 = time.time()
    for n, (uid, q, _, target, others) in enumerate(items, 1):
        ctx = softgen.generate_context(client, q, target, others)
        rows.append({"uid": uid, "target_label": target, "context": ctx})
        if n % checkpoint_every == 0:
            _save(rows, out_path)
            print(f"  {n}/{len(items)}  {n/max(time.time()-t0,1e-6):.1f} ctx/s", flush=True)
    _save(rows, out_path)
    print(f"[softgen sync] done: {len(rows)} contexts → {out_path}", flush=True)
    return out_path


# ---------------------------------------------------------------------------
# Batch backend
# ---------------------------------------------------------------------------

def _runner(n_batches: int) -> BatchRunner:
    return BatchRunner("softgen", DATASET, n_batches=n_batches)


def batch_gen(n_batches: int = 50, limit: int = 0) -> int:
    records = [{"custom_id": f"{uid}#{i}",
                "body": softgen.build_request_body(q, target, others)}
               for uid, q, i, target, others in _iter_targets(limit=limit)]
    return _runner(n_batches).gen_requests(records)


def batch_submit(n_batches: int = 50):
    return _runner(n_batches).submit()


def batch_status(n_batches: int = 50):
    return _runner(n_batches).status()


def batch_collect(n_batches: int = 50) -> Path:
    results = _runner(n_batches).collect()      # {"<uid>#<i>": content}
    labels_by_uid = _labels_by_uid()
    rows = []
    for custom_id, content in results.items():
        uid, _, idx = custom_id.rpartition("#")
        labels = labels_by_uid.get(uid, [])
        try:
            target = labels[int(idx)]
        except (ValueError, IndexError):
            continue
        rows.append({"uid": uid, "target_label": target,
                     "context": softgen.parse_context(content)})
    out_path = softgen_parquet_path()
    _save(rows, out_path)
    print(f"[softgen batch] wrote {len(rows)} contexts → {out_path}", flush=True)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build soft-gen premises for ProtoQA (all top-3)")
    p.add_argument("--backend", choices=["sync", "batch"], default="sync")
    p.add_argument("--step", choices=["gen", "submit", "status", "collect"],
                   help="batch backend step")
    p.add_argument("--n_batches", type=int, default=50, help="batch backend: #JSONL shards")
    p.add_argument("--limit", type=int, default=0, help="only first N questions (smoke test)")
    p.add_argument("--checkpoint_every", type=int, default=50)
    args = p.parse_args()

    if args.backend == "sync":
        build_sync(limit=args.limit, checkpoint_every=args.checkpoint_every)
    else:
        if args.step == "gen":
            batch_gen(n_batches=args.n_batches, limit=args.limit)
        elif args.step == "submit":
            batch_submit(n_batches=args.n_batches)
        elif args.step == "status":
            batch_status(n_batches=args.n_batches)
        elif args.step == "collect":
            batch_collect(n_batches=args.n_batches)
        else:
            p.error("--backend batch requires --step {gen,submit,status,collect}")
