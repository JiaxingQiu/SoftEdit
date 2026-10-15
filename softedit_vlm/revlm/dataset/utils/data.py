import os
import logging
from typing import List, Dict

import pandas as pd
from huggingface_hub import snapshot_download

LOG = logging.getLogger(__name__)

RANKED_DATASETS = {"vqav2"}


def load_ranked_parquet(dataset_name: str) -> pd.DataFrame:
    """Load a RankedVQA parquet from local data/parquet/ or HuggingFace."""
    local_path = os.path.join("data", "parquet", dataset_name, f"{dataset_name}.parquet")
    if os.path.exists(local_path):
        return pd.read_parquet(local_path)
    local_root = snapshot_download(
        repo_id="JJoy333/RankedVQA",
        repo_type="dataset",
        allow_patterns=[f"{dataset_name}/*.parquet"],
    )
    hf_path = os.path.join(local_root, dataset_name, f"{dataset_name}.parquet")
    if os.path.exists(hf_path):
        return pd.read_parquet(hf_path)
    raise FileNotFoundError(
        f"Cannot find parquet for '{dataset_name}' at {local_path} or HF JJoy333/RankedVQA"
    )
