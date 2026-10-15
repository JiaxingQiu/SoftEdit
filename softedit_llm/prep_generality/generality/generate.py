"""Build the t-gen set: k rephrasings per ProtoQA question → tgen/protoqa.parquet.

Output columns: uid, variants (list[str]).  A t-gen eval sample reuses the base
question's `answer_counts` as gold (same-gold rephrasings → `sgen` / accuracy `gen`).

Sync backend (default, faithful to the VLM t-gen): idempotent + checkpointed, so
it is safe to interrupt/resume. Optional Azure batch backend for the full pool.

Usage (sync):
    python -m prep_generality.generality.generate --limit 10          # prototype
    python -m prep_generality.generality.generate                     # full pool
Usage (batch):
    python -m prep_generality.generality.generate --backend batch --step gen
    python -m prep_generality.generality.generate --backend batch --step submit
    python -m prep_generality.generality.generate --backend batch --step status
    python -m prep_generality.generality.generate --backend batch --step collect
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

from prep_generality.generality import tgen
from prep_generality.azure_batch import BatchRunner

DATASET = "protoqa"
SRC_PARQUET = f"data/parquet/{DATASET}/{DATASET}.parquet"


def tgen_parquet_path() -> Path:
    return Path(f"data/generality/tgen/{DATASET}.parquet")


def _load_src() -> pd.DataFrame:
    return pd.read_parquet(SRC_PARQUET)[["uid", "question"]]


def _load_existing(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {str(u): list(v) for u, v in zip(df["uid"], df["variants"])}


def _save(rows: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"uid": list(rows.keys()),
                       "variants": [list(v) for v in rows.values()]})
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Sync backend
# ---------------------------------------------------------------------------

def build_sync(k: int = 4, limit: int = 0, overwrite: bool = False,
               checkpoint_every: int = 50) -> Path:
    out_path = tgen_parquet_path()
    rows = {} if overwrite else _load_existing(out_path)
    src = _load_src()
    if limit:
        src = src.head(limit)

    print(f"[tgen sync] questions={len(src)}  existing={len(rows)}  k={k}  → {out_path}",
          flush=True)
    client = tgen.make_client()
    t0 = time.time()
    n_new = 0
    for i, (_, r) in enumerate(src.iterrows(), 1):
        uid = str(r["uid"])
        if not overwrite and uid in rows:
            continue
        variants = tgen.rephrase_question(client, r["question"], k=k)
        if variants:
            rows[uid] = variants
            n_new += 1
        if i % checkpoint_every == 0:
            _save(rows, out_path)
            rate = i / max(time.time() - t0, 1e-6)
            print(f"  {i}/{len(src)}  +{n_new} new  {rate:.1f} q/s", flush=True)
    _save(rows, out_path)
    print(f"[tgen sync] done: {len(rows)} uids (+{n_new}) → {out_path}", flush=True)
    return out_path


# ---------------------------------------------------------------------------
# Batch backend
# ---------------------------------------------------------------------------

def _runner(n_batches: int) -> BatchRunner:
    return BatchRunner("tgen", DATASET, n_batches=n_batches)


def batch_gen(k: int = 4, n_batches: int = 30, limit: int = 0) -> int:
    src = _load_src()
    if limit:
        src = src.head(limit)
    records = [{"custom_id": str(r["uid"]),
                "body": tgen.build_request_body(r["question"], k)}
               for _, r in src.iterrows()]
    return _runner(n_batches).gen_requests(records)


def batch_submit(n_batches: int = 30):
    return _runner(n_batches).submit()


def batch_status(n_batches: int = 30):
    return _runner(n_batches).status()


def batch_collect(k: int = 4, n_batches: int = 30) -> Path:
    results = _runner(n_batches).collect()          # {uid: content}
    q_by_uid = {str(u): q for u, q in zip(*[_load_src()[c] for c in ("uid", "question")])}
    rows = {}
    for uid, content in results.items():
        variants = tgen.parse_variants(content, q_by_uid.get(uid, ""), k)
        if variants:
            rows[uid] = variants
    out_path = tgen_parquet_path()
    _save(rows, out_path)
    print(f"[tgen batch] wrote {len(rows)} uids → {out_path}", flush=True)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build t-gen rephrasings for ProtoQA")
    p.add_argument("--backend", choices=["sync", "batch"], default="sync")
    p.add_argument("--step", choices=["gen", "submit", "status", "collect"],
                   help="batch backend step")
    p.add_argument("--k", type=int, default=4, help="rephrasings per question")
    p.add_argument("--n_batches", type=int, default=30, help="batch backend: #JSONL shards")
    p.add_argument("--limit", type=int, default=0, help="only first N questions (smoke test)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--checkpoint_every", type=int, default=50)
    args = p.parse_args()

    if args.backend == "sync":
        build_sync(k=args.k, limit=args.limit, overwrite=args.overwrite,
                   checkpoint_every=args.checkpoint_every)
    else:
        if args.step == "gen":
            batch_gen(k=args.k, n_batches=args.n_batches, limit=args.limit)
        elif args.step == "submit":
            batch_submit(n_batches=args.n_batches)
        elif args.step == "status":
            batch_status(n_batches=args.n_batches)
        elif args.step == "collect":
            batch_collect(k=args.k, n_batches=args.n_batches)
        else:
            p.error("--backend batch requires --step {gen,submit,status,collect}")
