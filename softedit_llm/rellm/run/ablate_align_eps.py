"""align_eps sensitivity sweep (LLM).

Sweep ``align_eps`` (the early-stop convergence threshold on the alignment KL
term, which now gates align+cont) while keeping everything else at SoftEdit
defaults (lambda_gen=auto, lambda_loc=auto, locality_n=1, declarative="both").

Results go to ``results/ee/softedit_abl_aeps_<value>/...``.

Usage::

    python -m rellm.run.ablate_align_eps --align_eps 1e-6 --model_name llama --dataset_name protoqa --max_batches 200
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from rellm.config_utils import configure_args
from rellm.run.edit import run_edit


def _format_eps(v: float) -> str:
    """Compact string for the result-dir suffix: 1e-08, 1e-06, 1e-04, etc."""
    return f"{v:.0e}"


def _suffix(align_eps: float, lambda_gen) -> str:
    """Result-dir suffix.  Produces a *nested* path so all align_eps ablation
    results land under ``softedit_abl_aeps/<variant>_<eps>/``.

    Examples (after prepend of "softedit"):
      auto   → softedit_abl_aeps/1e-04
      gen1.0 → softedit_abl_aeps/gen1.0_1e-04
    """
    gen_tag = ""
    if lambda_gen is not None and str(lambda_gen).strip().lower() != "auto":
        gen_tag = f"gen{lambda_gen}_"
    return f"_abl_aeps/{gen_tag}{_format_eps(align_eps)}"


def _apply(config, align_eps: float, lambda_gen=None) -> None:
    config.editor.align_eps = align_eps
    if lambda_gen is not None and str(lambda_gen).strip().lower() != "auto":
        config.editor.lambda_gen = float(lambda_gen)
    print(
        f"[ablate_align_eps] align_eps={align_eps}  "
        f"lambda_gen={getattr(config.editor, 'lambda_gen', 'auto')}  "
        f"suffix={_suffix(align_eps, lambda_gen)}  "
        f"(all other config from defaults)",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="align_eps sensitivity sweep (LLM)")
    parser.add_argument("--align_eps", type=float, required=True,
                        help="Early-stop align convergence threshold (e.g. 1e-8, 1e-6, 1e-4)")
    parser.add_argument("--lambda_gen", type=str, default=None,
                        help="Override lambda_gen (e.g. 1.0). Default: leave as "
                             "config default (auto). Adds a _gen<val> suffix tag.")
    parser.add_argument("--skip", type=str, default="",
                        help="Comma-separated metric sets to skip in eval "
                             "(default: none). Pass '' or 'none' to skip nothing.")
    parser.add_argument("--config", type=str, default="rellm/config/config.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"])
    parser.add_argument("--sequential", action="store_true", default=True)
    parser.add_argument("--no-sequential", dest="sequential", action="store_false")
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--subsample", type=int, default=0)
    parser.add_argument("--subsample_edits", type=int, default=0)
    parser.add_argument("--pred_path", type=str, default=None)
    parser.add_argument("--pred_postedit_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.editor = "softedit"
    args.suffix = _suffix(args.align_eps, args.lambda_gen)
    args.task = "postedit_qa"
    config = configure_args(args, config_path=args.config)

    _apply(config, args.align_eps, args.lambda_gen)

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
    _skip = tuple(s.strip() for s in args.skip.split(",")
                  if s.strip() and s.strip().lower() != "none")
    print(f"[ablate_align_eps] skip={_skip if _skip else '()'}", flush=True)
    run_edit(config, sequential=args.sequential, eval_every=args.eval_every,
             skip=_skip)
