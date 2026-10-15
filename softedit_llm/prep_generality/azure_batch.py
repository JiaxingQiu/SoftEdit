"""Generic Azure OpenAI Batch runner (faithful to data_process/gen_support.py).

Reusable 4-step batch lifecycle for any chat-completions job:
    1. gen_requests(records) — write + split request JSONL files
    2. submit()             — upload & create Azure batch jobs
    3. status()             — poll job progress
    4. collect()            — download results → {custom_id: content}

`records` is a list of ``{"custom_id": str, "body": {...chat body...}}``. The
runner stores everything under ``data/generality/<task>/{requests,meta,outputs}``
so it is idempotent and safe to resubmit after a timeout.

Batch endpoint/deployment mirror the working config in gen_support.py.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

from openai import AzureOpenAI

from data_process.tokens import azure_key

# Batch endpoint is the Azure resource root, without a /openai/... path.
BATCH_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "").split("/openai/")[0].rstrip("/")
BATCH_API_VERSION = "2025-01-01-preview"
BATCH_DEPLOYMENT = "gpt-4o-batch"


class BatchRunner:
    def __init__(self, task: str, dataset: str, n_batches: int = 50,
                 *, deployment: str = BATCH_DEPLOYMENT,
                 endpoint: str = BATCH_ENDPOINT,
                 api_version: str = BATCH_API_VERSION,
                 api_key: Optional[str] = None):
        self.task = task
        self.dataset = dataset
        self.n_batches = n_batches
        self.deployment = deployment

        # Dataset-scoped scratch (mirrors gen_support's data/support/{dataset}/),
        # so adding a new dataset never collides with an existing one.
        self.base_dir = Path(f"data/generality/{task}/{dataset}")
        self.request_dir = self.base_dir / "requests"
        self.meta_dir = self.base_dir / "meta"
        self.output_dir = self.base_dir / "outputs"
        for d in (self.request_dir, self.meta_dir, self.output_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.client = AzureOpenAI(
            api_key=api_key or azure_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )

    # ---- Step 1: generate + split request JSONL files ----
    def gen_requests(self, records: List[dict]) -> int:
        existing = list(self.request_dir.glob("batch_*.jsonl"))
        if existing:
            print(f"Request files already exist ({len(existing)} batches), skipping. "
                  f"Delete {self.request_dir}/ to regenerate.")
            return -1
        reqs = []
        for r in records:
            reqs.append({
                "custom_id": str(r["custom_id"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": r["body"],
            })
        all_path = self.request_dir / "all.jsonl"
        with open(all_path, "w") as f:
            for r in reqs:
                f.write(json.dumps(r) + "\n")
        print(f"Generated {len(reqs)} requests → {all_path}")
        self._split_file(all_path, self.n_batches)
        return len(reqs)

    def _split_file(self, input_file: Path, n_batches: int):
        with open(input_file) as f:
            lines = f.readlines()
        per_batch = max(1, math.ceil(len(lines) / n_batches))
        actual = 0
        for i in range(n_batches):
            chunk = lines[i * per_batch:(i + 1) * per_batch]
            if not chunk:
                break
            with open(self.request_dir / f"batch_{i}.jsonl", "w") as f:
                f.writelines(chunk)
            actual += 1
        os.remove(input_file)
        print(f"Split into {actual} batch files (~{per_batch}/batch)")

    # ---- Step 2: submit ----
    def _submit_single(self, b: int) -> bool:
        batch_file = self.request_dir / f"batch_{b}.jsonl"
        meta_file = self.meta_dir / f"meta_{b}.json"
        if not batch_file.exists():
            return False
        if meta_file.exists():
            print(f"  Batch {b}: already submitted, skipping")
            return False
        with open(batch_file, "rb") as f:
            upload = self.client.files.create(file=f, purpose="batch")
        job = self.client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"generality_{self.task}_{self.dataset}", "batch": str(b)},
        )
        with open(meta_file, "w") as f:
            json.dump({"job_id": job.id, "batch": b, "created_at": str(job.created_at)}, f)
        print(f"  Batch {b}: submitted (job_id={job.id})")
        return True

    def submit(self, batch_ids: Optional[list] = None) -> int:
        ids = batch_ids if batch_ids is not None else range(self.n_batches)
        n = sum(1 for b in ids if self._submit_single(b))
        print(f"Submitted {n} batches")
        return n

    def cancel_all(self) -> int:
        n = 0
        for b in range(self.n_batches):
            meta_file = self.meta_dir / f"meta_{b}.json"
            if not meta_file.exists():
                continue
            meta = json.loads(meta_file.read_text())
            try:
                self.client.batches.cancel(meta["job_id"])
                print(f"  Batch {b}: cancelled ({meta['job_id']})")
            except Exception as e:  # noqa: BLE001
                print(f"  Batch {b}: cancel failed ({e})")
            meta_file.unlink()
            n += 1
        print(f"Cancelled {n} batches")
        return n

    def resubmit(self, batch_ids: list):
        for b in batch_ids:
            meta_file = self.meta_dir / f"meta_{b}.json"
            if meta_file.exists():
                try:
                    self.client.batches.cancel(json.loads(meta_file.read_text())["job_id"])
                except Exception:  # noqa: BLE001
                    pass
                meta_file.unlink()
            out = self.output_dir / f"outputs_{b}.jsonl"
            if out.exists():
                out.unlink()
            self._submit_single(b)

    # ---- Step 3: status ----
    def status(self):
        counts = {"completed": 0, "in_progress": 0, "validating": 0,
                  "finalizing": 0, "failed": 0, "other": 0}
        failed = []
        for b in range(self.n_batches):
            meta_file = self.meta_dir / f"meta_{b}.json"
            if not meta_file.exists():
                continue
            job = self.client.batches.retrieve(json.loads(meta_file.read_text())["job_id"])
            s = job.status
            counts[s if s in counts else "other"] += 1
            if s == "failed":
                failed.append(b)
            if s != "completed":
                print(f"  Batch {b}: {s}")
        print(f"\nSummary: {counts}")
        if failed:
            print(f"Failed batch IDs: {failed}")
        return counts, failed

    # ---- Step 4: collect ----
    def _collect_batch(self, b: int) -> List[dict]:
        meta_file = self.meta_dir / f"meta_{b}.json"
        out_file = self.output_dir / f"outputs_{b}.jsonl"
        if out_file.exists():
            with open(out_file) as f:
                return [json.loads(line) for line in f]
        if not meta_file.exists():
            return []
        job = self.client.batches.retrieve(json.loads(meta_file.read_text())["job_id"])
        if job.status != "completed":
            print(f"  Batch {b}: not completed ({job.status})")
            return []
        resp = self.client.files.content(job.output_file_id)
        recs = []
        for line in resp.iter_lines():
            if not line:
                continue
            payload = json.loads(line)
            content = (payload.get("response", {}).get("body", {})
                       .get("choices", [{}])[0].get("message", {}).get("content", ""))
            recs.append({"custom_id": payload.get("custom_id"), "content": content})
        with open(out_file, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"  Batch {b}: collected {len(recs)} results")
        return recs

    def collect(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for b in range(self.n_batches):
            if not (self.meta_dir / f"meta_{b}.json").exists():
                continue
            for rec in self._collect_batch(b):
                out[rec["custom_id"]] = rec["content"]
        print(f"\nCollected {len(out)} total results")
        return out
