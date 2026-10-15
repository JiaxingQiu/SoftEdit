"""Ablation 5 — editing miscalibrated but correctly-answered samples (LLM).

The main pipeline edits **errors** (rank_acc == 0) and evaluates locality on
the **correct** complement.  This ablation **swaps** the two: it treats a
subsample of baseline-correct samples as edits (they answered correctly but
their distribution is likely miscalibrated) and uses the error set as the
unrelated locality pool.

Goal: show that SoftEdit can improve calibration (Align / SGen / SLoc) on
already-correct samples without degrading accuracy, whereas a baseline (no
edit) cannot.  Locality metric is skipped (``loc`` is meaningless when the
"edit" samples are already correct).

Results land in ``results/ee/<editor>_abl_miscalib/<model>/<dataset>/``.

Usage::

    python -m rellm.run.ablate_miscalib --editor softedit  --model_name qwen  --dataset_name protoqa
    python -m rellm.run.ablate_miscalib --editor baseline  --model_name llama --dataset_name protoqa
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from rellm.config_utils import configure_args
from .edit_utils import find_errors, edit_n_eval_seq, edit_n_eval_all

_SUBSAMPLE = 1000
_SKIP = ("loc",)


def _maybe_resolve_lambda_loc(config, model, edit_ds):
    cfg = getattr(config, "editor", config)
    raw_loc = getattr(cfg, "lambda_loc", "auto")
    raw_gen = getattr(cfg, "lambda_gen", "auto")
    loc_auto = isinstance(raw_loc, str) and raw_loc.strip().lower() == "auto"
    gen_auto = isinstance(raw_gen, str) and raw_gen.strip().lower() == "auto"

    if loc_auto or gen_auto:
        from rellm.editors.softedit.auto_lambda import resolve_lambda_loc
        resolved_loc = resolve_lambda_loc(config, model, edit_ds.data,
                                          force_pilot=gen_auto)
        if loc_auto:
            config.editor.lambda_loc = float(resolved_loc)

    if gen_auto:
        from rellm.editors.softedit.auto_lambda import resolve_lambda_gen
        config.editor.lambda_gen = float(resolve_lambda_gen(
            config, float(config.editor.lambda_loc)))


def run_miscalib(config, sequential=True, eval_every=200):
    """Like ``run_edit`` but swaps edit↔locality and subsamples correct set."""
    out_path = os.path.join(config.edit_dir, config.fname)
    if sequential:
        from .resume import is_complete, clear_resume
        if getattr(config, "overwrite", False):
            clear_resume(config.edit_dir)
        elif is_complete(config.edit_dir):
            print(f"Edit run already COMPLETE at {config.edit_dir}. Skipping.", flush=True)
            print("-" * 50, flush=True)
            return
    else:
        if os.path.exists(out_path) and not config.overwrite:
            print(f"Result already exists at {out_path}. Skipping.", flush=True)
            print("-" * 50, flush=True)
            return

    # Step 1: load baseline preds and split into errors / correct
    model, edit_ds, loc_ds = find_errors(config)

    # Step 2: SWAP — correct samples become "edits", errors become locality
    edit_ds, loc_ds = loc_ds, edit_ds
    print(f"[ablate_miscalib] Swapped: editing {len(edit_ds.data)} baseline-correct samples "
          f"(locality pool = {len(loc_ds.data)} errors)", flush=True)

    # Step 3: subsample the (now large) edit set
    subsample = getattr(config, "subsample_edits", 0) or _SUBSAMPLE
    if subsample and len(edit_ds.data) > subsample:
        print(f"[ablate_miscalib] Subsampling edit set from {len(edit_ds.data)} to {subsample}", flush=True)
        edit_ds.data = random.sample(edit_ds.data, subsample)
    edit_ds.set_dataloader()

    _maybe_resolve_lambda_loc(config, model, edit_ds)

    if sequential:
        edit_n_eval_seq(config, model, edit_ds, out_path, eval_every=eval_every,
                        max_batches=getattr(config, "max_batches", None),
                        skip=_SKIP, loc_ds=loc_ds)
    else:
        edit_n_eval_all(config, model, edit_ds, out_path, skip=_SKIP, loc_ds=loc_ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation 5: edit miscalibrated but correctly-answered samples (LLM)")
    parser.add_argument("--editor", type=str, required=True,
                        choices=["softedit", "baseline"],
                        help="Editor to evaluate")
    parser.add_argument("--config", type=str, default="rellm/config/config.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "test", "all"])
    parser.add_argument("--sequential", action="store_true", default=True)
    parser.add_argument("--no-sequential", dest="sequential", action="store_false")
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--subsample", type=int, default=0)
    parser.add_argument("--subsample_edits", type=int, default=_SUBSAMPLE,
                        help=f"Subsample correct-answer edit set (default {_SUBSAMPLE})")
    parser.add_argument("--pred_path", type=str, default=None)
    parser.add_argument("--pred_postedit_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.suffix = "_abl_miscalib"
    args.task = "postedit_qa"
    config = configure_args(args, config_path=args.config)

    config.subsample = args.subsample
    config.pred_path = args.pred_path or os.path.join(
        config.pred_dir, f"softqa_{config.experiment.split}.json"
    )
    config.overwrite = args.overwrite
    config.debug = getattr(args, "debug", False)
    config.subsample_edits = args.subsample_edits
    config.max_batches = args.max_batches
    config.wandb = bool(args.wandb) and bool(args.sequential)
    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens
    run_miscalib(config, sequential=args.sequential, eval_every=args.eval_every)
