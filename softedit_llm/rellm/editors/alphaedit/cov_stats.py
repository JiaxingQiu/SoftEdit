"""Covariance statistics for AlphaEdit's null space (wikipedia mom2).

Faithful port of ``rome/layer_stats.py`` + ``AlphaEdit_main.get_cov`` from the
official repo. Computes the uncentered second moment ``E[k kᵀ]`` of the *input*
to each edited ``mlp.down_proj`` (the layer's fan-in keys) over a text corpus
(wikipedia by default), caching the result under ``data/stats/`` so edit jobs only
load it.

Only the imports (relative) and the stats dir (``data/stats``) differ from
upstream; the streaming accumulation (``SecondMoment`` via ``tally``) is verbatim.
"""
import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .nethook import Trace, set_requires_grad
from .runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally
from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}

# Default cache root: <repo>/data/stats (run entrypoints chdir to repo root).
STATS_DIR = "data/stats"

COV_CACHE = {}


def _hf_wikipedia(ds_name):
    """Load the text corpus for stats. Wikipedia config names have drifted across
    `datasets` versions, so try a few; fall back to wikitext-103 if unavailable."""
    if ds_name == "wikitext":
        return load_dataset("wikitext", "wikitext-103-raw-v1")
    if ds_name == "wikipedia":
        for cfg in ("20220301.en", "20200501.en"):
            try:
                return load_dataset("wikipedia", cfg, trust_remote_code=True)
            except Exception as e:  # noqa: BLE001
                print(f"[alphaedit.cov] wikipedia/{cfg} unavailable ({e}); trying next", flush=True)
        print("[alphaedit.cov] falling back to wikitext-103-raw-v1", flush=True)
        return load_dataset("wikitext", "wikitext-103-raw-v1")
    return load_dataset(ds_name)


def _model_maxlen(model):
    for attr in ("n_positions", "max_sequence_length", "max_position_embeddings", "seq_length"):
        if hasattr(model.config, attr):
            maxlen = getattr(model.config, attr)
            break
    else:
        raise NotImplementedError("Could not determine model max sequence length")
    mt = getattr(model.config, "model_type", "")
    if "mistral" in mt:
        maxlen = getattr(model.config, "sliding_window", None) or 4096
    if "qwen2" in mt:
        maxlen = 4096
    return maxlen


def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
):
    """Load or compute cached stats for the *input* of ``layer_name``."""

    def get_ds():
        raw_ds = _hf_wikipedia(ds_name)
        maxlen = _model_maxlen(model)
        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    batch_size = 1
    npos = _model_maxlen(model)
    if batch_tokens is None:
        batch_tokens = npos * 3
    if precision is None:
        precision = "float32"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        model_name = model.config._name_or_path.rsplit("/", 1)[-1]

    stats_dir = Path(stats_dir)
    file_extension = (
        f"{model_name}/{ds_name}_stats/"
        f"{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    )
    filename = stats_dir / file_extension

    ds = get_ds() if (not filename.exists() or force_recompute) else None
    if progress is None:
        progress = lambda x: x  # noqa: E731

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        # num_workers=0: this env launches DataLoader workers with `spawn`, which
        # can't pickle the upstream local `length_collation.<locals>.collate_fn`.
        # Collation is cheap here, so run it in the main process.
        num_workers=0,
    )
    batch_count = -(-(sample_size or (len(ds) if ds is not None else 0)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, "cuda")
                with Trace(
                    model, layer_name, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)
    return stat


def get_cov(
    model,
    tokenizer,
    layer_name,
    mom2_dataset,
    mom2_n_samples,
    mom2_dtype,
    stats_dir=STATS_DIR,
    inv=False,
    force_recompute=False,
):
    """Uncentered second moment ``E[k kᵀ]`` of ``layer_name``'s input, cached."""
    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"[alphaedit.cov] retrieving covariance for {model_name} @ {layer_name}", flush=True)
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tokenizer,
            layer_name,
            stats_dir,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=(mom2_n_samples if not force_recompute else mom2_n_samples // 10),
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    cov = COV_CACHE[key]
    return torch.inverse(cov.cuda()) if inv else cov
