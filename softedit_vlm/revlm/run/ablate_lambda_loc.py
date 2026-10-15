"""λ_loc sensitivity sweep (VLM).

Sweep lambda_loc over a log scale while keeping everything else at the
"full" SoftEdit config (lambda_gen=1, locality_n=1, declarative="both").
Results go to ``results/ee/softedit_abl_lloc_<value>/...``.

Usage::

    python -m revlm.run.ablate_lambda_loc --lambda_loc 1e-2 --model_name llava --dataset_name vqav2 --max_batches 200
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

from revlm.config_utils import configure_args
from revlm.run.edit import run_edit


def _format_lambda(v: float) -> str:
    """Compact string for the result-dir suffix: 1e-04, 5e-03, 1e+00, etc."""
    return f"{v:.0e}"


def _apply(config, lam: float) -> None:
    config.editor.lambda_gen = 1.0
    config.editor.lambda_loc = lam
    config.editor.locality_n = 1
    config.editor.locality_declarative = "both"
    print(
        f"[ablate_lambda_loc] lambda_loc={lam}  suffix=_abl_lloc_{_format_lambda(lam)}  "
        f"lambda_gen={config.editor.lambda_gen}  "
        f"locality_n={config.editor.locality_n}  "
        f"locality_declarative={config.editor.locality_declarative}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="λ_loc sensitivity sweep (VLM)")
    parser.add_argument("--lambda_loc", type=float, required=True,
                        help="Locality loss weight (e.g. 1e-4, 5e-3, 1.0)")
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"])
    parser.add_argument("--sequential", action="store_true", default=True)
    parser.add_argument("--no-sequential", dest="sequential", action="store_false")
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--max_batches", type=int, default=None)
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
    args.suffix = f"_abl_lloc_{_format_lambda(args.lambda_loc)}"
    args.task = "postedit_qa"
    config = configure_args(args, config_path=args.config)

    _apply(config, args.lambda_loc)

    config.subsample = args.subsample
    config.pred_path = args.pred_path or os.path.join(
        config.pred_dir, f"softqa_{config.experiment.split}.json"
    )
    config.overwrite = args.overwrite
    config.debug = getattr(args, "debug", False)
    config.subsample_edits = args.subsample_edits
    config.max_batches = args.max_batches
    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens
    run_edit(config, sequential=args.sequential, eval_every=args.eval_every)
