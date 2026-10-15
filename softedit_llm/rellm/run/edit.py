"""Editing-evaluation entrypoint for rankedit_llm (text analog of revlm/run/edit.py).

find_errors -> edit (sequential resume / batch) -> editeval. Sequential is the
default; retrain editors (ft_retrain / mend_retrain and their _soft variants) are
the ones that benefit from --no-sequential (joint retrain over the gathered edits).

Usage:
    python -m rellm.run.edit --editor baseline --model_name qwen --dataset_name protoqa
    python -m rellm.run.edit --editor ft_soft --model_name qwen --dataset_name protoqa
    python -m rellm.run.edit --editor ft_retrain_soft --model_name qwen --no-sequential
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from rellm.config_utils import configure_args
from .edit_utils import find_errors, edit_n_eval_seq, edit_n_eval_all


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
        from rellm.editors.softedit.auto_lambda import resolve_lambda_loc
        resolved_loc = resolve_lambda_loc(config, model, edit_ds.data,
                                          force_pilot=gen_auto)
        if loc_auto:
            config.editor.lambda_loc = float(resolved_loc)

    # ---- λ_gen (depends on pilot stats + resolved λ_loc) ----
    if gen_auto:
        from rellm.editors.softedit.auto_lambda import resolve_lambda_gen
        config.editor.lambda_gen = float(resolve_lambda_gen(
            config, float(config.editor.lambda_loc)))


def run_edit(config, sequential=True, eval_every=200, skip=()):
    """Universal edit runner: find errors, edit with chosen editor, run editeval.

    skip: metric sets to skip for quick sessions, default none. Aliases accepted,
    e.g. skip=["soft_gen", "t_gen", "loc", "rel"]; skipped sets are None in output.
    """
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
        # Batch path: a finished result JSON is the completion marker (resubmit is
        # a no-op). Incomplete runs resume inside edit_n_eval_all: editor weights
        # from `_resume/batch_ckpt.pt` (or a finished retrain ckpt), tokenized
        # history from `_resume/batch_history.pt`, and eval sets from existing
        # pred_postedit dumps. Sequential `_resume/ckpt.pt` is unused.
        if os.path.exists(out_path) and not config.overwrite:
            print(f"Edit evaluation result already exists at {out_path}. Skipping.", flush=True)
            print("-" * 50, flush=True)
            return

    model, edit_ds, loc_ds = find_errors(config)

    subsample_edits = getattr(config, "subsample_edits", 0)
    if subsample_edits and len(edit_ds.data) > subsample_edits:
        print(f"Subsampling edit set from {len(edit_ds.data)} to {subsample_edits} examples", flush=True)
        edit_ds.data = random.sample(edit_ds.data, subsample_edits)
        edit_ds.set_dataloader()

    # Step 1.5: resolve SoftEdit's λ_loc="auto" over the full edit set (a
    # standalone pre-step, before any editing). Rewrites config.editor.lambda_loc
    # in place so the editor built downstream sees a concrete float.
    _maybe_resolve_lambda_loc(config, model, edit_ds)

    if sequential:
        edit_n_eval_seq(config, model, edit_ds, out_path, eval_every=eval_every,
                        max_batches=getattr(config, "max_batches", None), skip=skip,
                        loc_ds=loc_ds)
    else:
        edit_n_eval_all(config, model, edit_ds, out_path, skip=skip, loc_ds=loc_ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Editing Evaluation")
    parser.add_argument("--config", type=str, default="rellm/config/config.yaml",
                        help="Path to YAML config file (CLI overrides YAML)")
    parser.add_argument(
        "--editor", type=str, required=True,
        choices=[
            "ft", "ft_soft", "ft_retrain", "ft_retrain_soft",
            "mend", "mend_soft", "mend_retrain", "mend_retrain_soft",
            "grace", "grace_soft",
            "ike", "ike_soft",
            "alphaedit", "alphaedit_soft",
            "softedit", "softedit_major",
            "ts_soft", "bc_soft", "ats_soft", "ts_ret_soft", "bc_ret_soft",
            "baseline",
        ],
        help="Editor method (*_soft = multilabel variants; *_retrain best with --no-sequential)",
    )
    parser.add_argument("--model_name", type=str, default=None, help="Short LLM name: qwen | llama | mistral")
    parser.add_argument("--dataset_name", type=str, default="", help="Dataset name (overrides YAML)")
    parser.add_argument("--label_strategy", type=str, default=None, choices=["text", "prob"],
                        help="Single-QA label parsing (edit/locality split + acc). Default from config.yaml.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for edit dataloader")
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"],
                        help="Split to search for edit examples")
    parser.add_argument("--edit_dir", type=str, default=None, help="Edit eval result directory (overrides config)")
    parser.add_argument("--sequential", action="store_true", default=True,
                        help="Run sequential editing/eval (default)")
    parser.add_argument("--no-sequential", dest="sequential", action="store_false",
                        help="Run batch editing (non-sequential); the effective mode for retrain editors")
    parser.add_argument("--eval_every", type=int, default=200,
                        help="Eval checkpoint every N sequential edits (also always evals on the last batch)")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Stop after N sequential edits (testing/interruption). Resume continues from last ckpt.")
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable wandb logging in sequential mode (off by default).")

    parser.add_argument("--subsample", type=int, default=0, help="Evaluate on a random subset (0=all)")
    parser.add_argument("--pred_path", type=str, default=None,
                        help="Baseline snapshot path. Defaults to results/pred/.../softqa_<split>.json")
    parser.add_argument("--pred_postedit_dir", type=str, default=None,
                        help="Directory for saving post-edit predictions on the edit set.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--soft", type=str, default=None, choices=["target", "loss", "rank_loss"],
                        help="Soft supervision mode: 'target' (CE on ranked string), 'loss' (KL vs vote-frequency q), 'rank_loss' (KL vs rank-group q; see rank_objective.md). Default: loss.")
    parser.add_argument("--rank_temp", type=float, default=None,
                        help="Temperature τ for rank_loss (q_r ∝ exp(-(r-1)/τ), then split equally inside ties). Default: 1.0.")
    parser.add_argument("--major_vote", action="store_true",
                        help="SoftEdit ablation: collapse the soft target to a one-hot on the majority label.")
    parser.add_argument("--subsample_edits", type=int, default=0,
                        help="Subsample edit set after error discovery (0=all)")
    parser.add_argument("--n_edit_cap", type=int, default=None,
                        help="Cap number of edits (subsample edit set before training)")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Max new tokens for generation (default from config.yaml)")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-sample gold/pred diagnostics for each eval set")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Isolate rank_loss ablation results under results/{ee,te,pred_postedit}/{editor}_rank/...
    # so they never overwrite the default count-q runs. No extra CLI flag.
    args.suffix = "_rank" if getattr(args, "soft", None) == "rank_loss" else ""
    # The edit pipeline always evaluates with the no-instruction postedit_qa task
    # (baseline.py keeps its own instructed softqa two-pass for the snapshot).
    args.task = "postedit_qa"
    config = configure_args(args, config_path=args.config)

    if args.soft is not None:
        config.editor.soft = args.soft
    if args.rank_temp is not None:
        config.editor.rank_temp = args.rank_temp
    if args.major_vote:
        config.editor.major_vote = True

    config.subsample = args.subsample
    config.pred_path = args.pred_path or os.path.join(
        config.pred_dir, f"softqa_{config.experiment.split}.json"
    )
    config.overwrite = args.overwrite
    config.debug = getattr(args, "debug", False)
    config.subsample_edits = args.subsample_edits
    config.n_edit_cap = args.n_edit_cap
    config.max_batches = args.max_batches
    # wandb off by default (only enabled when --wandb AND sequential).
    config.wandb = bool(args.wandb) and bool(args.sequential)
    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens

    run_edit(config, sequential=args.sequential, eval_every=args.eval_every)
