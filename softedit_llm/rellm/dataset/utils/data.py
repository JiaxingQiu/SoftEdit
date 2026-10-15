import os
import logging

import pandas as pd
from huggingface_hub import snapshot_download

LOG = logging.getLogger(__name__)

# Text ranked-uncertainty datasets (LLM analog of RANKED_DATASETS in rankedit).
RANKED_DATASETS = {"protoqa"}

# LLM data is published under an `llm/` prefix inside the shared RankedVQA repo
# (mirrors the VLM flow: build locally → hf upload → snapshot_download fallback).
HF_REPO_ID = "JJoy333/RankedVQA"
HF_PARQUET_PREFIX = "llm/parquet"
HF_GENERALITY_PREFIX = "llm/generality"


def load_ranked_parquet(dataset_name: str) -> pd.DataFrame:
    """Load a ranked-uncertainty parquet: local `data/parquet/` first, else HF.

    Built by `data_process/{dataset_name}.py` (e.g. `python -m data_process.protoqa`)
    and published via `hf upload JJoy333/RankedVQA data/parquet llm/parquet ...`.
    """
    local_path = os.path.join("data", "parquet", dataset_name, f"{dataset_name}.parquet")
    if os.path.exists(local_path):
        return pd.read_parquet(local_path)
    local_root = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=[f"{HF_PARQUET_PREFIX}/{dataset_name}/*.parquet"],
    )
    hf_path = os.path.join(local_root, HF_PARQUET_PREFIX, dataset_name, f"{dataset_name}.parquet")
    if os.path.exists(hf_path):
        return pd.read_parquet(hf_path)
    raise FileNotFoundError(
        f"Cannot find parquet for '{dataset_name}' at {local_path} or "
        f"HF {HF_REPO_ID}:{HF_PARQUET_PREFIX}/{dataset_name}/. Build it first, "
        f"e.g. `python -m data_process.{dataset_name}`."
    )


def load_generality_parquet(kind: str, dataset_name: str) -> pd.DataFrame:
    """Load a generality parquet (kind in {'tgen','softgen'}): local first, else HF.

    Built by `prep_generality/{generality,soft_generality}/generate.py` and published
    via `hf upload JJoy333/RankedVQA data/generality llm/generality ...`.
    """
    if kind not in ("tgen", "softgen"):
        raise ValueError(f"Unknown generality kind: {kind!r} (expected 'tgen' or 'softgen')")
    local_path = os.path.join("data", "generality", kind, f"{dataset_name}.parquet")
    if os.path.exists(local_path):
        return pd.read_parquet(local_path)
    local_root = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=[f"{HF_GENERALITY_PREFIX}/{kind}/{dataset_name}.parquet"],
    )
    hf_path = os.path.join(local_root, HF_GENERALITY_PREFIX, kind, f"{dataset_name}.parquet")
    if os.path.exists(hf_path):
        return pd.read_parquet(hf_path)
    raise FileNotFoundError(
        f"Cannot find {kind} parquet for '{dataset_name}' at {local_path} or "
        f"HF {HF_REPO_ID}:{HF_GENERALITY_PREFIX}/{kind}/. Build it first, e.g. "
        f"`python -m prep_generality.{'generality' if kind == 'tgen' else 'soft_generality'}.generate`."
    )
