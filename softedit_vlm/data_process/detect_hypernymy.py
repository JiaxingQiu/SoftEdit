"""
Hypernymy detection via OpenAI Batch API (VQAv2 only).

For each sample, ask GPT-4o whether any answer label is a hyponym (more specific)
of another answer label (its hypernym). Returns all hyponym→hypernym pairs found.

Pipeline (mirrors unify_labels.py):
    1. gen_requests  — build JSONL request files from filtered parquet
    2. submit        — upload & create OpenAI batch jobs
    3. status        — check batch job progress
    4. collect       — download results, parse hyponym→hypernym mappings
    5. (optional) filter — use mappings to remove hypernymy samples from parquet

Driven interactively via data_process/detect_hypernymy.ipynb.
"""
import json
import math
import os
from pathlib import Path

import pandas as pd
from openai import AzureOpenAI


SYSTEM_PROMPT = (
    "Given a question and answer labels from multiple annotators, identify ALL pairs "
    "where one label is a hyponym (more specific version) of another label (its hypernym).\n\n"
    "A hyponym adds detail to the same concept as its hypernym:\n"
    '  "polar bear" is a hyponym of "bear" (specific species)\n'
    '  "kitchen table" is a hyponym of "table" (specific type)\n'
    '  "dark blue" is a hyponym of "blue" (specific shade)\n'
    '  "dog" is a hyponym of "animal" (specific category)\n\n'
    "NOT hypernym-hyponym:\n"
    '  "red" / "blue" (different values)\n'
    '  "cat" / "dog" (siblings, neither contains the other)\n'
    '  "left" / "right" (opposites)\n\n'
    "Return a JSON object mapping each hyponym to its hypernym. "
    "Include ALL such pairs. If none exist, return {}."
)

AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "").split("/openai/")[0].rstrip("/")
AZURE_API_VERSION = "2025-01-01-preview"
AZURE_DEPLOYMENT = "gpt-4o-batch"

N_BATCHES = 100
PARQUET_PATH = "data/parquet/vqav2/vqav2.parquet"


def user_prompt(question: str, labels: list) -> str:
    label_strs = [f'"{l}"' for l in labels]
    return (
        f'Question: "{question}"\n'
        f"Labels: [{', '.join(label_strs)}]"
    )


class HypernymyDetector:
    def __init__(self, azure_key: str = None,
                 azure_endpoint: str = None, azure_deployment: str = None,
                 api_version: str = None):
        self.azure_deployment = azure_deployment or AZURE_DEPLOYMENT

        self.base_dir = Path("data/hypernymy/vqav2")
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
    def demo(self, question: str, labels: list) -> dict:
        """Run hypernymy detection on a single sample (non-batch, for testing)."""
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
                "description": "hypernymy_vqav2",
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
        """Collect results from all batches → all_mappings.json."""
        all_mappings = {}
        for b in range(N_BATCHES):
            if not (self.meta_dir / f"meta_{b}.json").exists():
                continue
            for rec in self._collect_batch(b):
                all_mappings[rec["uid"]] = rec["mapping"]

        merged_path = self.output_dir / "all_mappings.json"
        with open(merged_path, "w") as f:
            json.dump(all_mappings, f)

        n_with_pairs = sum(1 for m in all_mappings.values() if m)
        print(f"\nCollected {len(all_mappings)} total mappings → {merged_path}")
        print(f"  Samples with hypernymy pairs: {n_with_pairs} "
              f"({100*n_with_pairs/len(all_mappings):.1f}%)")
        return all_mappings

    # ---- Step 5 (optional): filter parquet using mappings ----
    def summary(self):
        """Load mappings and print statistics without modifying the parquet."""
        mappings_path = self.output_dir / "all_mappings.json"
        if not mappings_path.exists():
            raise FileNotFoundError(f"Run 'collect' first: {mappings_path}")
        with open(mappings_path) as f:
            all_mappings = json.load(f)

        df = self.load_df()
        has_hypernymy = {uid for uid, m in all_mappings.items() if m}
        mask = df["uid"].isin(has_hypernymy)

        print(f"Total samples:            {len(df):>8,}")
        print(f"With hypernymy pairs:     {mask.sum():>8,} ({100*mask.sum()/len(df):.1f}%)")
        print(f"Without (would remain):   {(~mask).sum():>8,} ({100*(~mask).sum()/len(df):.1f}%)")

        print(f"\nExample hypernymy pairs:")
        shown = 0
        for uid, mapping in all_mappings.items():
            if not mapping:
                continue
            row = df[df["uid"] == uid]
            if row.empty:
                continue
            row = row.iloc[0]
            counts = json.loads(row["answer_counts"])
            pos = {l: c for l, c in counts.items() if c > 0}
            print(f"  Q: {row['question']}")
            print(f"    answers: {pos}")
            print(f"    hypernymy: {mapping}")
            shown += 1
            if shown >= 15:
                break

        return all_mappings, has_hypernymy

    @staticmethod
    def _merge_counts(counts: dict, mapping: dict) -> dict:
        """Fold hyponym counts into hypernyms. Only applies pairs present in counts."""
        pos = {l for l, c in counts.items() if c > 0}
        valid = {c: p for c, p in mapping.items() if c in pos and p in pos}

        def resolve(label):
            seen = set()
            while label in valid and label not in seen:
                seen.add(label)
                label = valid[label]
            return label

        new_counts = {}
        for label, count in counts.items():
            if count <= 0:
                continue
            target = resolve(label)
            new_counts[target] = new_counts.get(target, 0) + count
        return dict(sorted(new_counts.items(), key=lambda x: (-x[1], x[0])))

    UNKNOWN_SYNONYMS = {
        "unknown",
        "unanswerable",
        "unsuitable",
        "unsure",
        "unclear",
        "not sure",
        "dont know",
        "don't know",
        "do not know",
        "idk",
        "cannot tell",
        "can't tell",
        "cant tell",
        "n/a",
        "na",
    }

    def _filter_unknown(self, df: pd.DataFrame, all_mappings: dict = None) -> pd.DataFrame:
        """Drop samples where any label token/phrase matches an unknown synonym.

        Matches whole tokens (or multi-word phrases) bounded by spaces / ends,
        not substrings inside a word. e.g. 'na' matches 'na' / 'foo na' but not 'banana'.
        """
        n_before = len(df)
        patterns = tuple(sorted(self.UNKNOWN_SYNONYMS, key=len, reverse=True))

        def has_unknown(ac_json: str) -> bool:
            for label, c in json.loads(ac_json).items():
                if c <= 0:
                    continue
                padded = f" {label.lower().strip()} "
                if any(f" {p} " in padded for p in patterns):
                    return True
            return False

        mask = df["answer_counts"].apply(has_unknown)
        df = df[~mask].copy()
        print(f"mode=unknown: {n_before} → {len(df)} "
              f"({mask.sum()} removed, {100*len(df)/n_before:.1f}% kept)")
        print(f"  whole-token patterns: {sorted(self.UNKNOWN_SYNONYMS)}")
        return df

    def _filter_drop(self, df: pd.DataFrame, all_mappings: dict) -> pd.DataFrame:
        """Drop every sample that has any hypernymy pair."""
        has_hypernymy = {uid for uid, m in all_mappings.items() if m}
        n_before = len(df)
        df = df[~df["uid"].isin(has_hypernymy)].copy()
        print(f"mode=drop: {n_before} → {len(df)} "
              f"({n_before - len(df)} removed, {100*len(df)/n_before:.1f}% kept)")
        return df

    NEGATION_TOKENS = ("no", "none", "not", "nothing")

    def _filter_others(self, df: pd.DataFrame, all_mappings: dict = None) -> pd.DataFrame:
        """Drop samples with negation / yes / multi-word / single-letter labels, or 'or' questions.

        Removes a sample if:
          1. any positive-count label contains whole-token "no" / "none" / "not" / "nothing"
          2. any positive-count label is exactly "yes"
          3. any positive-count label has more than 1 word
          4. the question contains whole-token "or"
          5. any positive-count label is a single letter
        """
        n_before = len(df)
        n_neg = n_yes = n_multi = n_or = n_letter = 0

        def should_drop(row) -> bool:
            nonlocal n_neg, n_yes, n_multi, n_or, n_letter
            q = str(row.get("question", "")).lower().strip()
            q_padded = f" {q} "
            if " or " in q_padded:
                n_or += 1
                return True

            for label, c in json.loads(row["answer_counts"]).items():
                if c <= 0:
                    continue
                text = label.lower().strip()
                padded = f" {text} "
                if any(f" {tok} " in padded for tok in self.NEGATION_TOKENS):
                    n_neg += 1
                    return True
                if text == "yes":
                    n_yes += 1
                    return True
                if len(text.split()) > 1:
                    n_multi += 1
                    return True
                if len(text) == 1 and text.isalpha():
                    n_letter += 1
                    return True
            return False

        mask = df.apply(should_drop, axis=1)
        df = df[~mask].copy()
        print(f"mode=others: {n_before} → {len(df)} "
              f"({mask.sum()} removed, {100*len(df)/n_before:.1f}% kept)")
        print(f"  by reason (first match): or-question={n_or}, negation={n_neg}, "
              f"yes={n_yes}, multi-word={n_multi}, single-letter={n_letter}")
        return df

    def _filter_merge(self, df: pd.DataFrame, all_mappings: dict) -> pd.DataFrame:
        """Merge hyponym→hypernym counts; drop only samples that become k=1."""
        n_before = len(df)
        n_merged = 0
        n_became_k1 = 0
        keep_rows = []
        examples = []

        for _, row in df.iterrows():
            mapping = all_mappings.get(row["uid"]) or {}
            if not mapping:
                keep_rows.append(row)
                continue

            counts = json.loads(row["answer_counts"])
            before = {l: c for l, c in counts.items() if c > 0}
            new_counts = self._merge_counts(counts, mapping)
            if len(new_counts) < 2:
                n_became_k1 += 1
                continue

            n_merged += 1
            row = row.copy()
            row["answer_counts"] = json.dumps(new_counts)
            keep_rows.append(row)
            if len(examples) < 10 and new_counts != before:
                examples.append({
                    "uid": row["uid"],
                    "question": row["question"],
                    "before": before,
                    "after": new_counts,
                    "mapping": mapping,
                })

        df = pd.DataFrame(keep_rows).reset_index(drop=True)
        print(f"mode=merge: {n_before} → {len(df)} "
              f"({n_became_k1} became k=1 and dropped, "
              f"{n_merged} merged & kept, "
              f"{100*len(df)/n_before:.1f}% kept)")
        for ex in examples:
            print(f"\n  uid: {ex['uid']}")
            print(f"  Q: {ex['question']}")
            print(f"  mapping: {ex['mapping']}")
            print(f"  before:  {ex['before']}")
            print(f"  after:   {ex['after']}")
        return df

    def filter_parquet(self, mode: str = "merge", dry_run: bool = True):
        """Apply hypernymy / unknown / others filters to the parquet.

        Args:
            mode: Filter strategy —
                "merge"   — fold hyponym counts into hypernyms; drop only k=1
                "drop"    — remove all samples that have any hypernymy pair
                "unknown" — remove samples containing 'unknown' or a synonym
                "others"  — remove samples with negation / yes / multi-word /
                            single-letter labels, or questions containing whole-token "or"
            dry_run: If True, print stats without writing.
        """
        if mode not in ("merge", "drop", "unknown", "others"):
            raise ValueError(
                f"Unknown mode={mode!r}. Choose from 'merge', 'drop', 'unknown', 'others'."
            )

        df = self.load_df()
        n_before = len(df)

        if mode == "unknown":
            df = self._filter_unknown(df)
        elif mode == "others":
            df = self._filter_others(df)
        else:
            mappings_path = self.output_dir / "all_mappings.json"
            if not mappings_path.exists():
                raise FileNotFoundError(f"Run 'collect' first: {mappings_path}")
            with open(mappings_path) as f:
                all_mappings = json.load(f)

            if mode == "drop":
                df = self._filter_drop(df, all_mappings)
            else:
                df = self._filter_merge(df, all_mappings)

        n_after = len(df)
        if not dry_run:
            df.to_parquet(PARQUET_PATH, index=False)
            print(f"Written → {PARQUET_PATH}")
        else:
            print("[DRY RUN] Skipping write.")

        return n_before, n_after
