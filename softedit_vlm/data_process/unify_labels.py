"""
Label unification via OpenAI Batch API.

For open-vocabulary VQAv2, annotator answers may contain
synonymous labels ("no one" / "nobody", "suitcase" / "luggage"). This module
uses GPT-4o to identify and merge them.

Pipeline (same pattern as reasonedit/t_gen.py):
    1. gen_requests  — build JSONL request files from parquet
    2. submit        — upload & create OpenAI batch jobs
    3. status        — check batch job progress
    4. collect       — download results, parse mappings
    5. merge         — apply mappings to parquet, re-aggregate answer_counts

Driven interactively via data_process/unify_labels.ipynb.
"""
import json
import math
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI, AzureOpenAI

from data_process.common import write_parquet


SYSTEM_PROMPT = (
    "Given a question and its free text answer labels from multiple annotators, merge synonymous labels into "
    "mutually exclusive canonical labels. "
    "Return a JSON mapping each label to its canonical form.\n\n"
    "Example:\n"
    '  Question: "Who threw up in this toilet?"\n'
    '  Labels: ["no one", "nobody", "man", "guy"]\n'
    '  Output: {"no one": "no one", "nobody": "no one", "man": "man", "guy": "man"}\n\n'
    "Only merge true synonyms/variants. Keep related-but-different labels separate "
    '(e.g., "dog" and "animal" should NOT merge). '
    "Return ONLY the JSON object."
)

AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "").split("/openai/")[0].rstrip("/")
AZURE_API_VERSION = "2025-01-01-preview"
AZURE_DEPLOYMENT = "gpt-4o-batch"

DATASETS = {
    "vqav2": {"n_batches": 200},
}


def user_prompt(question: str, labels: list) -> str:
    label_strs = [f'"{l}"' for l in labels]
    return (
        f"Question: \"{question}\"\n"
        f"Labels: [{', '.join(label_strs)}]"
    )


class LabelUnifier:
    def __init__(self, dataset_name: str, azure_key: str = None,
                 azure_endpoint: str = None, azure_deployment: str = None,
                 api_version: str = None):
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASETS.keys())}")
        self.dataset_name = dataset_name
        self.cfg = DATASETS[dataset_name]
        self.azure_deployment = azure_deployment or AZURE_DEPLOYMENT

        self.base_dir = Path(f"data/unify/{dataset_name}")
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

    def _raw_parquet_path(self):
        return f"data/parquet/{self.dataset_name}/{self.dataset_name}_raw.parquet"

    def load_df(self):
        return pd.read_parquet(self._raw_parquet_path())

    # ---- Demo: test prompt on a single sample ----
    def demo(self, question: str, labels: list) -> dict:
        """Run label unification on a single sample (non-batch, for testing)."""
        response = self.client.chat.completions.create(
            model=self.azure_deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(question, labels)},
            ],
            max_tokens=200,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        try:
            return json.loads(content.strip().strip("`").removeprefix("json").strip())
        except json.JSONDecodeError:
            print(f"Failed to parse: {content}")
            return {}

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
            counts = json.loads(row["answer_counts"])
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
                    "max_tokens": 200,
                    "temperature": 0.0,
                }
            })

        all_path = self.request_dir / "all.jsonl"
        with open(all_path, "w") as f:
            for r in requests:
                f.write(json.dumps(r) + "\n")
        print(f"Generated {len(requests)} requests → {all_path}")

        self._split_file(all_path, self.cfg["n_batches"])
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
                "description": f"label_unify_{self.dataset_name}",
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
        ids = batch_ids if batch_ids is not None else range(self.cfg["n_batches"])
        submitted = 0
        for b in ids:
            if self._submit_single(b):
                submitted += 1
        print(f"Submitted {submitted} batches total")
        return submitted

    def cancel_all(self):
        """Cancel all submitted batch jobs and remove meta files."""
        cancelled = 0
        for b in range(self.cfg["n_batches"]):
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
        for b in range(self.cfg["n_batches"]):
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
        """Collect results from a single batch. Returns list of {uid, mapping}."""
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
            )
            try:
                mapping = json.loads(content.strip().strip("`").removeprefix("json").strip())
            except json.JSONDecodeError:
                mapping = {}
            records.append({"uid": uid, "mapping": mapping})

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
        """Collect results from all batches."""
        all_mappings = {}
        for b in range(self.cfg["n_batches"]):
            if not (self.meta_dir / f"meta_{b}.json").exists():
                continue
            for rec in self._collect_batch(b):
                all_mappings[rec["uid"]] = rec["mapping"]

        merged_path = self.output_dir / "all_mappings.json"
        with open(merged_path, "w") as f:
            json.dump(all_mappings, f)
        print(f"\nCollected {len(all_mappings)} total mappings → {merged_path}")
        return all_mappings

    # ---- Step 5: apply mappings to parquet ----
    def _apply_mappings(self, df, all_mappings):
        """Apply label mappings to df. Returns (df, changed_count, examples)."""
        changed = 0
        examples = []
        for idx, row in df.iterrows():
            uid = row["uid"]
            mapping = all_mappings.get(uid)
            if not mapping:
                continue

            counts = json.loads(row["answer_counts"])
            new_counts = {}
            did_merge = False
            for label, count in counts.items():
                canonical = mapping.get(label, label).strip().lower()
                if canonical != label:
                    did_merge = True
                new_counts[canonical] = new_counts.get(canonical, 0) + count
            if did_merge:
                sorted_counts = dict(sorted(new_counts.items(), key=lambda x: (-x[1], x[0])))
                if len(examples) < 10:
                    examples.append({
                        "uid": uid,
                        "before": counts,
                        "after": sorted_counts,
                        "mapping": {k: v for k, v in mapping.items() if v != k},
                    })
                df.at[idx, "answer_counts"] = json.dumps(sorted_counts)
                changed += 1
        return df, changed, examples

    def merge(self, dry_run: bool = False):
        """Apply label mappings + re-apply exclusion → {dataset}.parquet."""
        mappings_path = self.output_dir / "all_mappings.json"
        if not mappings_path.exists():
            raise FileNotFoundError(f"Run 'collect' first: {mappings_path}")
        with open(mappings_path) as f:
            all_mappings = json.load(f)

        df = self.load_df()
        df, changed, examples = self._apply_mappings(df, all_mappings)
        print(f"Unified labels for {changed}/{len(df)} samples")

        if not dry_run:
            write_parquet(df, self.dataset_name)
        else:
            print("[DRY RUN] Skipping write.")

        return changed, examples
