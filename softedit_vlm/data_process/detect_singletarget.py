"""
Single-target question detection via OpenAI Batch API (VQAv2 only).

Classifies each question as single-target (one correct answer, annotator
disagreement = ambiguity) or multi-target (multiple answers can be
simultaneously correct). Keeps only single-target questions.

Pipeline (mirrors detect_hypernymy.py):
    1. gen_requests  — build JSONL request files from filtered parquet
    2. submit        — upload & create OpenAI batch jobs
    3. status        — check batch job progress
    4. collect       — download results, parse YES/NO classifications
    5. summary       — inspect before filtering
    6. filter        — remove multi-target samples from parquet

Driven interactively via data_process/detect_singletarget.ipynb.
"""
import json
import math
import os
from pathlib import Path

import pandas as pd
from openai import AzureOpenAI


SYSTEM_PROMPT = (
    "You are given a question and the set of answers provided by multiple annotators.\n\n"
    "Decide: for this question, can multiple of these answers be correct AT THE SAME TIME?\n\n"
    "YES — the question is open-ended and allows multiple simultaneous correct answers "
    "(for example, for question 'What is on the table?', book, cup, plate can all be present).\n"
    "NO — the question targets a single thing and only one answer should be correct "
    "(for example, for question 'What color is the car?', red or blue can be correct, but not both).\n\n"
    "Reply with YES or NO only."
)

AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "").split("/openai/")[0].rstrip("/")
AZURE_API_VERSION = "2025-01-01-preview"
AZURE_DEPLOYMENT = "gpt-4o-batch"

N_BATCHES = 50
PARQUET_PATH = "data/parquet/vqav2/vqav2.parquet"


def user_prompt(question: str, labels: list) -> str:
    label_strs = [f'"{l}"' for l in labels]
    return (
        f'Question: "{question}"\n'
        f"Answers: [{', '.join(label_strs)}]"
    )


class SingleTargetDetector:
    def __init__(self, azure_key: str = None,
                 azure_endpoint: str = None, azure_deployment: str = None,
                 api_version: str = None):
        self.azure_deployment = azure_deployment or AZURE_DEPLOYMENT

        self.base_dir = Path("data/singletarget/vqav2")
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

    def load_df(self):
        return pd.read_parquet(PARQUET_PATH)

    # ---- Demo: test prompt on a single sample ----
    def demo(self, question: str, labels: list) -> str:
        """Run single-target detection on one sample (non-batch, for testing)."""
        response = self.client.chat.completions.create(
            model=self.azure_deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(question, labels)},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip().upper()
        print(f"Q: {question}")
        print(f"Labels: {labels}")
        print(f"Multi-target? {content}")
        return content

    # ---- Step 1: generate request JSONL files ----
    def gen_requests(self):
        existing = list(self.request_dir.glob("batch_*.jsonl"))
        if existing:
            print(f"Request files already exist ({len(existing)} batches), skipping. "
                  f"Delete {self.request_dir}/ to regenerate.")
            return -1
        df = self.load_df()
        requests = []
        for _, row in df.iterrows():
            counts = json.loads(row["answer_counts"]) if isinstance(row["answer_counts"], str) else row["answer_counts"]
            labels = [k for k, v in counts.items() if v > 0]
            if len(labels) < 2:
                continue
            requests.append({
                "custom_id": row["uid"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.azure_deployment,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt(row["question"], labels)},
                    ],
                    "max_tokens": 10,
                    "temperature": 0.0,
                }
            })

        all_path = self.request_dir / "all.jsonl"
        with open(all_path, "w") as f:
            for r in requests:
                f.write(json.dumps(r) + "\n")
        print(f"Generated {len(requests)} requests → {all_path}")

        self._split_file(all_path, N_BATCHES)
        return len(requests)

    def _split_file(self, input_file: Path, n_batches: int):
        with open(input_file) as f:
            lines = f.readlines()
        per_batch = max(1, math.ceil(len(lines) / n_batches))
        actual_batches = 0
        for i in range(n_batches):
            chunk = lines[i * per_batch : (i + 1) * per_batch]
            if not chunk:
                break
            split_path = self.request_dir / f"batch_{i}.jsonl"
            with open(split_path, "w") as f:
                f.writelines(chunk)
            actual_batches += 1
        os.remove(input_file)
        print(f"Split into {actual_batches} batch files")

    # ---- Step 2: submit batches ----
    def _submit_single(self, b: int):
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
            metadata={
                "description": "singletarget_vqav2",
                "batch": str(b),
            }
        )
        meta = {"job_id": job.id, "batch": b, "created_at": str(job.created_at)}
        with open(meta_file, "w") as f:
            json.dump(meta, f)
        print(f"  Batch {b}: submitted (job_id={job.id})")
        return True

    def submit(self, batch_ids: list = None):
        """Submit batch jobs. If batch_ids given, submit only those; otherwise all."""
        ids = batch_ids if batch_ids is not None else range(N_BATCHES)
        submitted = 0
        for b in ids:
            if self._submit_single(b):
                submitted += 1
        print(f"Submitted {submitted} batches total")
        return submitted

    def cancel_all(self):
        """Cancel all submitted batch jobs and remove meta files."""
        cancelled = 0
        for b in range(N_BATCHES):
            meta_file = self.meta_dir / f"meta_{b}.json"
            if not meta_file.exists():
                continue
            with open(meta_file) as f:
                meta = json.load(f)
            try:
                self.client.batches.cancel(meta["job_id"])
                print(f"  Batch {b}: cancelled ({meta['job_id']})")
            except Exception as e:
                print(f"  Batch {b}: cancel failed ({e})")
            meta_file.unlink()
            cancelled += 1
        print(f"Cancelled {cancelled} batches")
        return cancelled

    def resubmit(self, batch_ids: list):
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
    def status(self):
        counts = {"completed": 0, "in_progress": 0, "failed": 0, "other": 0}
        failed_ids = []
        for b in range(N_BATCHES):
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
    def _collect_batch(self, b: int) -> list:
        """Collect results from a single batch. Returns list of {uid, is_multi}."""
        meta_file = self.meta_dir / f"meta_{b}.json"
        output_file = self.output_dir / f"outputs_{b}.jsonl"
        if output_file.exists():
            with open(output_file) as f:
                return [json.loads(line) for line in f]

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
            payload = json.loads(line)
            uid = payload.get("custom_id")
            content = (
                payload.get("response", {})
                .get("body", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            ).strip().upper()
            is_multi = content.startswith("YES")
            records.append({"uid": uid, "is_multi": is_multi, "raw": content})

        with open(output_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"  Batch {b}: collected {len(records)} results")
        return records

    def collect_batch(self, b: int) -> list:
        """Collect and preview results from a single batch."""
        results = self._collect_batch(b)
        print(f"Got {len(results)} results from batch {b}")
        return results

    def collect(self):
        """Collect results from all batches → all_classifications.json."""
        all_cls = {}
        for b in range(N_BATCHES):
            if not (self.meta_dir / f"meta_{b}.json").exists():
                continue
            for rec in self._collect_batch(b):
                all_cls[rec["uid"]] = rec["is_multi"]

        merged_path = self.output_dir / "all_classifications.json"
        with open(merged_path, "w") as f:
            json.dump(all_cls, f)

        n_multi = sum(1 for v in all_cls.values() if v)
        n_single = sum(1 for v in all_cls.values() if not v)
        print(f"\nCollected {len(all_cls)} classifications → {merged_path}")
        print(f"  Multi-target (YES): {n_multi} ({100*n_multi/len(all_cls):.1f}%)")
        print(f"  Single-target (NO): {n_single} ({100*n_single/len(all_cls):.1f}%)")
        return all_cls

    # ---- Step 5: summary ----
    def summary(self):
        """Load classifications and print statistics without modifying the parquet."""
        cls_path = self.output_dir / "all_classifications.json"
        if not cls_path.exists():
            raise FileNotFoundError(f"Run 'collect' first: {cls_path}")
        with open(cls_path) as f:
            all_cls = json.load(f)

        df = self.load_df()
        multi_uids = {uid for uid, is_multi in all_cls.items() if is_multi}
        mask = df["uid"].isin(multi_uids)

        print(f"Total samples:              {len(df):>8,}")
        print(f"Multi-target (drop):        {mask.sum():>8,} ({100*mask.sum()/len(df):.1f}%)")
        print(f"Single-target (keep):       {(~mask).sum():>8,} ({100*(~mask).sum()/len(df):.1f}%)")

        print(f"\nExample multi-target questions (will be dropped):")
        shown = 0
        for uid in multi_uids:
            row = df[df["uid"] == uid]
            if row.empty:
                continue
            row = row.iloc[0]
            counts = json.loads(row["answer_counts"]) if isinstance(row["answer_counts"], str) else row["answer_counts"]
            pos = {l: c for l, c in counts.items() if c > 0}
            print(f"  Q: {row['question']}")
            print(f"    answers: {pos}")
            shown += 1
            if shown >= 15:
                break

        return all_cls, multi_uids

    # ---- Step 6: filter parquet ----
    def filter_parquet(self, dry_run: bool = True):
        """Remove multi-target samples from the parquet.

        Args:
            dry_run: If True, print stats without writing.
        """
        cls_path = self.output_dir / "all_classifications.json"
        if not cls_path.exists():
            raise FileNotFoundError(f"Run 'collect' first: {cls_path}")
        with open(cls_path) as f:
            all_cls = json.load(f)

        df = self.load_df()
        n_before = len(df)

        multi_uids = {uid for uid, is_multi in all_cls.items() if is_multi}
        df = df[~df["uid"].isin(multi_uids)].copy()
        n_after = len(df)

        print(f"Filter: {n_before} → {n_after} "
              f"({n_before - n_after} multi-target removed, {100*n_after/n_before:.1f}% kept)")

        if not dry_run:
            df.to_parquet(PARQUET_PATH, index=False)
            print(f"Written → {PARQUET_PATH}")
        else:
            print("[DRY RUN] Skipping write.")

        return n_before, n_after
