"""Step 2: Validate t=1.0 edits with Azure GPT-4o batch API.

For each (uid, label) pair, ask GPT-4o:
  "Based on this image, is the answer to '{question}' '{label}'? YES or NO."

Samples where any label gets NO → marked failed, excluded from interpolation.

Driven interactively via notebooks/validate_gpt4o.ipynb.

Usage (CLI, optional):
    python -m prep_generality.soft_generality.run.validate_gpt4o --dataset vqav2 --action gen_requests
"""

from __future__ import annotations

import base64
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AzureOpenAI

from ..shared.sample_io import load_parquet, parse_answer_counts, ranked_labels, REPO_ROOT


AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "").split("/openai/")[0].rstrip("/")
AZURE_API_VERSION = "2025-01-01-preview"
AZURE_DEPLOYMENT = "gpt-4o-batch"

OUTPUT_ROOT = REPO_ROOT / "data" / "soft_generality"
IMAGE_ROOT = OUTPUT_ROOT / "images"
VALIDATION_ROOT = OUTPUT_ROOT / "validation"

MAX_PER_BATCH = 250  # keep batch files under 200MB (each request ~700KB with base64 PNG)

# Labels to exclude from pass/fail aggregation (GPT-4o unreliable on these)
EXCLUDE_FROM_VALIDATION = {
    "ferplus": {"neutral"},
}

SYSTEM_PROMPT = (
    "You are a visual validator. Answer two questions about the image. "
    "Reply ONLY in this exact format:\n"
    "Q1:YES or Q1:NO\n"
    "Q2:YES or Q2:NO\n"
    "Do not explain."
)


def _user_prompt(question: str, label: str, dataset: str) -> str:
    if dataset == "cifar10h":
        q1 = f'Q1: Does this image show a {label}?'
    elif dataset == "ferplus":
        q1 = f'Q1: Does this face express {label}?'
    else:
        q1 = f'Q1: Based on this image, is the answer to "{question}" "{label}"?'
    q2 = 'Q2: Is there any stamped or watermark-style text on this image?'
    return f'{q1}\n{q2}'


def _image_path(dataset: str, uid: str, label: str) -> Path:
    safe_label = label.replace("/", "_").replace(" ", "_")
    return IMAGE_ROOT / dataset / uid / f"{safe_label}_t1.0.png"


def _encode_image_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class SoftGenValidator:
    def __init__(
        self,
        dataset: str,
        azure_key: str,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        max_per_batch: int = MAX_PER_BATCH,
    ):
        self.dataset = dataset
        self.max_per_batch = max_per_batch
        self.azure_deployment = azure_deployment or AZURE_DEPLOYMENT

        self.base_dir = VALIDATION_ROOT / dataset
        self.request_dir = self.base_dir / "requests"
        self.meta_dir = self.base_dir / "meta"
        self.output_dir = self.base_dir / "outputs"
        for d in [self.request_dir, self.meta_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.client = AzureOpenAI(
            api_key=azure_key or os.environ.get("AZURE_OPENAI_API_KEY", ""),
            api_version=api_version or AZURE_API_VERSION,
            azure_endpoint=azure_endpoint or AZURE_ENDPOINT,
        )

    @property
    def n_batches(self) -> int:
        """Discover number of batch files on disk."""
        return len(list(self.request_dir.glob("batch_*.jsonl")))

    def count_t1_images(self) -> int:
        image_dir = IMAGE_ROOT / self.dataset
        if not image_dir.exists():
            return 0
        return sum(1 for _ in image_dir.rglob("*_t1.0.png"))

    # ---- Step 1: generate request JSONL files ----
    def gen_requests(self) -> int:
        """Generate batch request files, streaming directly to splits (memory-safe)."""
        existing = list(self.request_dir.glob("batch_*.jsonl"))
        if existing:
            print(f"Request files already exist ({len(existing)} batches), skipping. "
                  f"Delete {self.request_dir}/ to regenerate.")
            return -1

        df = load_parquet(self.dataset)

        # First pass: count how many requests we'll generate
        pairs = []
        for _, row in df.iterrows():
            uid = row["uid"]
            question = row["question"]
            counts = parse_answer_counts(row["answer_counts"])
            labels = ranked_labels(counts)
            for label in labels:
                img_path = _image_path(self.dataset, uid, label)
                if img_path.exists():
                    pairs.append((uid, question, label, img_path))

        if not pairs:
            print("No t=1.0 images found. Run generate_t1 first.")
            return 0

        n_total = len(pairs)
        per_batch = self.max_per_batch
        n_batches = math.ceil(n_total / per_batch)
        self._n_batches = n_batches
        print(f"Generating {n_total} requests into {n_batches} batch files "
              f"({per_batch} per batch)...")

        # Stream one request at a time into batch files
        batch_idx = 0
        in_batch = 0
        f_out = open(self.request_dir / f"batch_{batch_idx}.jsonl", "w")

        for i, (uid, question, label, img_path) in enumerate(pairs):
            b64 = _encode_image_base64(img_path)
            request = {
                "custom_id": f"{uid}___{label}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.azure_deployment,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64}",
                                    "detail": "low",
                                },
                            },
                            {
                                "type": "text",
                                "text": _user_prompt(question, label, self.dataset),
                            },
                        ]},
                    ],
                    "max_tokens": 20,
                    "temperature": 0.0,
                }
            }
            f_out.write(json.dumps(request) + "\n")
            del b64, request
            in_batch += 1

            if in_batch >= per_batch and batch_idx < n_batches - 1:
                f_out.close()
                batch_idx += 1
                in_batch = 0
                f_out = open(self.request_dir / f"batch_{batch_idx}.jsonl", "w")

            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{n_total} requests written...")

        f_out.close()
        print(f"Generated {n_total} requests → {batch_idx + 1} batch files")
        return n_total

    # ---- Step 2: submit batches ----
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
            metadata={"description": f"soft_gen_validate_{self.dataset}", "batch": str(b)},
        )
        meta = {"job_id": job.id, "batch": b, "created_at": str(job.created_at)}
        with open(meta_file, "w") as f:
            json.dump(meta, f)
        print(f"  Batch {b}: submitted (job_id={job.id})")
        return True

    def submit(self, batch_ids: list[int] | None = None) -> int:
        """Submit batch jobs. If batch_ids given, submit only those; otherwise all."""
        ids = batch_ids if batch_ids is not None else range(self.n_batches)
        submitted = 0
        for b in ids:
            if self._submit_single(b):
                submitted += 1
        print(f"Submitted {submitted} batches total")
        return submitted

    def resubmit(self, batch_ids: list[int]):
        """Cancel and resubmit specific failed batches."""
        for b in batch_ids:
            meta_file = self.meta_dir / f"meta_{b}.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                try:
                    self.client.batches.cancel(meta["job_id"])
                except Exception:
                    pass
                meta_file.unlink()
            output_file = self.output_dir / f"outputs_{b}.jsonl"
            if output_file.exists():
                output_file.unlink()
            self._submit_single(b)

    # ---- Step 3: check status ----
    def status(self) -> tuple[dict[str, int], list[int]]:
        counts = {"completed": 0, "in_progress": 0, "failed": 0, "other": 0}
        failed_ids = []
        for b in range(self.n_batches):
            meta_file = self.meta_dir / f"meta_{b}.json"
            if not meta_file.exists():
                continue
            with open(meta_file) as f:
                meta = json.load(f)
            job = self.client.batches.retrieve(meta["job_id"])
            s = job.status
            bucket = s if s in counts else "other"
            counts[bucket] += 1
            if s == "failed":
                failed_ids.append(b)
            if s not in ("completed",):
                print(f"  Batch {b}: {s}")
        print(f"\nSummary: {counts}")
        if failed_ids:
            print(f"Failed batch IDs: {failed_ids}")
        return counts, failed_ids

    # ---- Step 4: collect results ----
    def collect_batch(self, b: int) -> list[dict[str, Any]]:
        """Collect results from a single batch."""
        output_file = self.output_dir / f"outputs_{b}.jsonl"
        if output_file.exists():
            with open(output_file) as f:
                return [json.loads(line) for line in f]

        meta_file = self.meta_dir / f"meta_{b}.json"
        if not meta_file.exists():
            return []
        with open(meta_file) as f:
            meta = json.load(f)
        job = self.client.batches.retrieve(meta["job_id"])
        if job.status != "completed":
            print(f"  Batch {b}: not completed yet ({job.status})")
            return []

        file_resp = self.client.files.content(job.output_file_id)
        records = []
        for line in file_resp.iter_lines():
            if not line:
                continue
            obj = json.loads(line)
            custom_id = obj["custom_id"]
            uid, label = custom_id.split("___", 1)
            content = obj["response"]["body"]["choices"][0]["message"]["content"].strip().upper()
            valid = "Q1:YES" in content or "Q1: YES" in content
            has_text = "Q2:YES" in content or "Q2: YES" in content
            records.append({
                "uid": uid,
                "label": label,
                "valid": valid,
                "has_text_overlay": has_text,
                "response": content,
            })

        with open(output_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"  Batch {b}: collected {len(records)} results")
        return records

    def collect(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Collect all batch results → (df_raw, df_uid).

        Returns:
            df_raw: one row per (uid, label) with gpt4o_valid, has_text_overlay, gpt4o_response
            df_uid: one row per uid with aggregated gpt4o_valid (False if any label failed)
        """
        all_records = []
        for b in range(self.n_batches):
            records = self.collect_batch(b)
            all_records.extend(records)

        if not all_records:
            print("No results collected yet.")
            return pd.DataFrame(), pd.DataFrame()

        df_raw = pd.DataFrame(all_records)
        df_raw.columns = ["uid", "label", "gpt4o_valid", "has_text_overlay", "gpt4o_response"]

        # Aggregate: failed UID if:
        #   - ANY label has text overlay (Q2=YES), OR
        #   - ANY non-excluded label has bad edit (Q1=NO)
        # (e.g. "neutral" in ferplus excluded from Q1 check only)
        exclude = EXCLUDE_FROM_VALIDATION.get(self.dataset, set())

        def _uid_passes(g):
            if g["has_text_overlay"].any():
                return False
            non_excluded = g[~g["label"].isin(exclude)]
            if non_excluded.empty:
                return True
            return non_excluded["gpt4o_valid"].all()

        df_uid = df_raw.groupby("uid").apply(_uid_passes).rename("gpt4o_valid").reset_index()

        n_pass = df_uid["gpt4o_valid"].sum()
        n_total = len(df_uid)
        n_text = int(df_raw["has_text_overlay"].sum())
        print(f"Collected {len(df_raw)} image results → {n_total} UIDs")
        print(f"  Pass: {n_pass}/{n_total} ({100*n_pass/n_total:.1f}%)")
        print(f"  Text overlays: {n_text} images")

        raw_path = self.base_dir / f"{self.dataset}_gpt4o_raw.parquet"
        uid_path = self.base_dir / f"{self.dataset}_gpt4o.parquet"
        df_raw.to_parquet(raw_path, index=False)
        df_uid.to_parquet(uid_path, index=False)
        print(f"Saved raw → {raw_path}")
        print(f"Saved per-UID → {uid_path}")
        return df_raw, df_uid

    # ---- Step 5: summary ----
    def summary(self) -> dict[str, Any]:
        """Summarize validation: identify failed UIDs (any label invalid or text overlay → fail)."""
        raw_path = self.base_dir / f"{self.dataset}_gpt4o_raw.parquet"
        if raw_path.exists():
            df = pd.read_parquet(raw_path)
        else:
            df, _ = self.collect()
        if df.empty:
            return {"n_pass": 0, "n_fail": 0, "fail_rate": 0.0, "failed_uids": []}

        # A UID fails if:
        #   - ANY label has text overlay (Q2=YES), OR
        #   - ANY non-excluded label has bad edit (Q1=NO)
        exclude = EXCLUDE_FROM_VALIDATION.get(self.dataset, set())

        def _uid_passes(g):
            if g["has_text_overlay"].any():
                return False
            non_excluded = g[~g["label"].isin(exclude)]
            if non_excluded.empty:
                return True
            return non_excluded["gpt4o_valid"].all()

        uid_pass = df.groupby("uid").apply(_uid_passes)

        failed_uids = uid_pass[~uid_pass].index.tolist()
        passed_uids = uid_pass[uid_pass].index.tolist()

        n_pass = len(passed_uids)
        n_fail = len(failed_uids)
        n_text = int(df["has_text_overlay"].sum())
        fail_rate = n_fail / (n_pass + n_fail) if (n_pass + n_fail) > 0 else 0.0

        print(f"Pass: {n_pass} UIDs")
        print(f"Fail: {n_fail} UIDs (fail_rate={fail_rate:.1%})")
        print(f"Text overlay detected: {n_text} images")

        return {
            "n_pass": n_pass,
            "n_fail": n_fail,
            "fail_rate": fail_rate,
            "passed_uids": passed_uids,
            "failed_uids": failed_uids,
        }

    # ---- Step 6: remove failed UID folders ----
    def remove_failed(self, dry_run: bool = True) -> int:
        """Remove image folders for failed UIDs."""
        info = self.summary()
        failed_uids = info["failed_uids"]
        if not failed_uids:
            print("No failed UIDs to remove.")
            return 0

        removed = 0
        for uid in failed_uids:
            uid_dir = IMAGE_ROOT / self.dataset / uid
            if uid_dir.exists():
                if dry_run:
                    print(f"  [dry_run] would remove: {uid_dir}")
                else:
                    shutil.rmtree(uid_dir)
                removed += 1

        action = "would remove" if dry_run else "removed"
        print(f"\n{action} {removed} UID folders (dry_run={dry_run})")
        return removed

    def valid_uids(self) -> list[str]:
        """Return list of UIDs that passed validation (for use by interpolate.py)."""
        parquet_path = self.base_dir / f"{self.dataset}_gpt4o.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Validation parquet not found: {parquet_path}. Run collect() first."
            )
        df = pd.read_parquet(parquet_path)
        uid_valid = df.groupby("uid")["gpt4o_valid"].all()
        return uid_valid[uid_valid].index.tolist()
