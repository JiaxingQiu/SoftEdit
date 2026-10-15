import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import torch
import time
import numpy as np

# Add project root to path so we can run as a module or script
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from revlm import *
from revlm.config_utils import configure_args
from .edit_utils import *


def _maybe_resolve_lambda_loc(config, model, edit_ds):
    """Pre-step: resolve SoftEdit's ``lambda_loc="auto"`` and ``lambda_gen="auto"``.

    λ_loc is resolved first (100-edit pilot). Then λ_gen is resolved from the
    same pilot statistics so that ``l_align ≈ 100 × λ_gen·(l_gen + λ_loc·l_loc)``
    at initialization.  No-op when the respective config values are not "auto".
    """
    cfg = getattr(config, "editor", config)
    raw_loc = getattr(cfg, "lambda_loc", "auto")
    raw_gen = getattr(cfg, "lambda_gen", "auto")
    loc_auto = isinstance(raw_loc, str) and raw_loc.strip().lower() == "auto"
    gen_auto = isinstance(raw_gen, str) and raw_gen.strip().lower() == "auto"

    # ---- λ_loc (the pilot also stashes stats that λ_gen reuses). Run the
    # pilot when either lambda is "auto"; force_pilot lets it fire even when
    # λ_loc itself is a fixed value. ----
    if loc_auto or gen_auto:
        from revlm.editors.softedit.auto_lambda import resolve_lambda_loc
        resolved_loc = resolve_lambda_loc(config, model, edit_ds.data,
                                          force_pilot=gen_auto)
        if loc_auto:
            config.editor.lambda_loc = float(resolved_loc)

    # ---- λ_gen (depends on pilot stats + resolved λ_loc) ----
    if gen_auto:
        from revlm.editors.softedit.auto_lambda import resolve_lambda_gen
        config.editor.lambda_gen = float(resolve_lambda_gen(
            config, float(config.editor.lambda_loc)))


def run_edit(config, sequential=False, eval_every=200, skip=()):
    """Universal edit runner: find errors, edit with chosen editor, report reliability.

    skip: metric sets to skip for quick notebook sessions, default none.
    Aliases accepted, e.g. skip=["soft_gen", "i_gen", "t_gen", "loc", "rel"];
    skipped sets are None in the saved output.
    """
    
    # Enable wandb if sequential mode is used
    config.wandb = sequential

    out_path = os.path.join(config.edit_dir, config.fname)
    if sequential:
        # Resume is the default for sequential editing. --overwrite forces a fresh
        # start (wipes any checkpoint); otherwise a COMPLETE marker means the run
        # already finished and a resubmit is a no-op.
        from .resume import is_complete, clear_resume
        if getattr(config, "overwrite", False):
            clear_resume(config.edit_dir)
        elif is_complete(config.edit_dir):
            print(f"Edit run already COMPLETE at {config.edit_dir}. Skipping.", flush=True)
            print("-"*50, flush=True)
            return
    else:
        # Batch path: a finished result JSON is the completion marker (resubmit is
        # a no-op). Incomplete runs resume inside edit_n_eval_all: editor weights
        # from `_resume/batch_ckpt.pt` (or a finished retrain ckpt) and eval sets
        # from existing pred_postedit dumps. Sequential `_resume/ckpt.pt` is unused.
        if os.path.exists(out_path) and not config.overwrite:
            print(f"Edit evaluation result already exists at {out_path}. Skipping edit evaluation.", flush=True)
            print("-"*50, flush=True)
            return

    # Step 1: Find errors
    model, edit_ds, loc_ds = find_errors(config)

    # Subsample edit set if requested
    subsample_edits = getattr(config, "subsample_edits", 0)
    if subsample_edits and len(edit_ds.data) > subsample_edits:
        print(f"Subsampling edit set from {len(edit_ds.data)} to {subsample_edits} examples", flush=True)
        edit_ds.data = random.sample(edit_ds.data, subsample_edits)
        edit_ds.set_dataloader()

    # Step 1.5: resolve SoftEdit's λ_loc="auto" over the full edit set (a
    # standalone pre-step, before any editing). Rewrites config.editor.lambda_loc
    # in place so the editor built downstream sees a concrete float.
    _maybe_resolve_lambda_loc(config, model, edit_ds)

    # Step 2-3: Edit and Evaluate on all errors
    if sequential:
        out_dict = edit_n_eval_seq(config, model, edit_ds, out_path, eval_every=eval_every,
                                   max_batches=getattr(config, "max_batches", None), skip=skip,
                                   loc_ds=loc_ds)
    else:
        out_dict = edit_n_eval_all(config, model, edit_ds, out_path, skip=skip, loc_ds=loc_ds)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Editing Evaluation")

    # Config
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml", help="Path to YAML config file (CLI overrides YAML)")
    parser.add_argument(
        "--editor",
        type=str,
        required=True,
        choices=[
            "ft", "ft_soft", "ft_retrain", "ft_retrain_soft",
            "grace", "grace_soft",
            "balancedit", "balancedit_soft",
            "ike", "ike_soft",
            "reasonedit", "reasonedit_soft",
            "softedit", "softedit_major", "softedit_task",
            "mend", "mend_soft", "mend_retrain", "mend_retrain_soft",
            "gt", "gt_soft",
            "ts_soft", "bc_soft", "ats_soft", "ts_ret_soft", "bc_ret_soft",
            "baseline",
        ],
        help="Editor method to use (*_soft = multilabel variants)",
    )
    parser.add_argument("--model_name", type=str, default=None, help="Short VLM name to map to full HF id (e.g., 'qwen3_4b', 'llava', 'blip')")
    parser.add_argument("--dataset_name", type=str, default="", help="Dataset name (overrides YAML if provided)")
    parser.add_argument("--label_strategy", type=str, default=None, choices=["text", "prob"], help="Single-QA label parsing (governs edit/locality split + acc/rank_acc across all datasets): 'text' (free-text answer) or 'prob' (argmax over support_prob). Default from config.yaml.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for edit dataloader")
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"], help="Split to search for edit examples")
    parser.add_argument("--edit_dir", type=str, default=None, help="Edit evaluation result directory (overrides config.yaml if provided)")
    parser.add_argument("--sequential", action="store_true", default=True, help="Run sequential editing/eval (enables wandb logging)")
    parser.add_argument("--no-sequential", dest="sequential", action="store_false", help="Run batch editing (non-sequential); required for retrain editors at scale")
    parser.add_argument("--eval_every", type=int, default=200, help="Eval checkpoint every N sequential edits (also always evals on the last batch)")
    parser.add_argument("--max_batches", type=int, default=None, help="Stop after N sequential edits (for testing/interruption simulation). Editor-state checkpoints still fire at eval_every, so a later resume continues from the last one.")

    # Args
    parser.add_argument("--subsample", type=int, default=0, help="Evaluate on a random subset of this many examples (0=all)")
    parser.add_argument("--pred_path", type=str, default=None, help="Optional path to saved edit dataset. If it exists the file is loaded, otherwise it is written after error discovery.")
    parser.add_argument("--pred_postedit_dir", type=str, default=None, help="Optional directory for saving post-edit predictions on the edit set.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results if they exist")
    parser.add_argument("--mode", type=str, default=None, choices=["vision", "language", "language_last", "dual_sbert"], help="Embedding mode for ReasonEdit")
    parser.add_argument("--pool_method", type=str, default=None, choices=["mean", "last"], help="Pooling method for ReasonEdit")
    # Soft-editor supervision knobs (only affect the *_soft weight editors). Default
    # from the editor YAML / code default (soft=loss). CLI overrides YAML.
    parser.add_argument("--soft", type=str, default=None, choices=["target", "loss", "rank_loss"],
                        help="Soft supervision mode: 'target' (CE on ranked string), 'loss' (KL vs vote-frequency q), 'rank_loss' (KL vs rank-group q; see rank_objective.md). Default: loss.")
    parser.add_argument("--rank_temp", type=float, default=None,
                        help="Temperature τ for rank_loss (q_r ∝ exp(-(r-1)/τ), then split equally inside ties). Default: 1.0.")
    parser.add_argument("--major_vote", action="store_true",
                        help="SoftEdit ablation: collapse the soft target to a one-hot on the majority label.")
    parser.add_argument("--subsample_edits", type=int, default=0, help="Subsample edit set to this many examples after error discovery (0=all)")
    parser.add_argument("--n_edit_cap", type=int, default=None, help="Cap number of edits (subsample edit set before training)")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Max new tokens for VLM generation (default from config.yaml)")
    parser.add_argument("--debug", action="store_true", help="Print per-sample gold/pred/soft_acc diagnostics for each eval set")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Isolate rank_loss ablation results under results/{ee,te,pred_postedit}/{editor}_rank/...
    # so they never overwrite the default count-q runs. No extra CLI flag.
    args.suffix = "_rank" if getattr(args, "soft", None) == "rank_loss" else ""
    # The edit pipeline always evaluates with the no-instruction postedit_qa task
    # (baseline.py keeps its own instructed softqa two-pass for the snapshot).
    args.task = "postedit_qa"
    config = configure_args(args, config_path=args.config)

    # Override editor settings if provided
    if args.mode:
        config.editor.mode = args.mode
    if args.pool_method:
        config.editor.pool_method = args.pool_method
    # Soft-editor supervision knobs (no-op for non-soft editors, which ignore them)
    if args.soft is not None:
        config.editor.soft = args.soft
    if args.rank_temp is not None:
        config.editor.rank_temp = args.rank_temp
    if args.major_vote:
        config.editor.major_vote = True

    # current run-specific settings
    config.subsample = args.subsample
    # The baseline (Step 1) is static and always written as softqa_<split>.json,
    # regardless of the editing eval task. Default to it when no explicit path given.
    config.pred_path = args.pred_path or os.path.join(
        config.pred_dir, f"softqa_{config.experiment.split}.json"
    )
    config.overwrite = args.overwrite
    config.debug = getattr(args, "debug", False)
    config.subsample_edits = args.subsample_edits
    config.n_edit_cap = args.n_edit_cap
    config.max_batches = args.max_batches
    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens
    run_edit(config, sequential=args.sequential, eval_every=args.eval_every)
