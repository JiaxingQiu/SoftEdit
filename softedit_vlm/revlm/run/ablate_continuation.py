"""SoftEdit continuation-term ablation runner (VLM).

Tests whether the continuation NLL inside L_align is essential for text
generation accuracy, even when first-token distributional alignment is
already achieved by the KL term alone.

    no_cont: continuation=False  (first-token KL only, no ΔW_cont)

The "with continuation" baseline is the main ``softedit`` run — no separate
run needed.  Generality / locality are skipped for speed (only reliability
is evaluated).

Results land in ``results/ee/softedit_abl_no_cont/<model>/<dataset>/``.

Usage::

    python -m revlm.run.ablate_continuation --ablation no_cont --model_name blip --dataset_name vqav2 --sequential
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

ABLATIONS = {
    "no_cont": dict(continuation=False),
}

# Skip everything except reliability for speed.
_SKIP = ("t_gen", "i_gen", "soft_gen", "blend_gen", "loc")


def _apply_ablation(config, name: str) -> None:
    overrides = ABLATIONS[name]
    for key, value in overrides.items():
        setattr(config.editor, key, value)
    print(
        f"[ablate_continuation] ablation={name}  suffix=_abl_{name}  "
        f"continuation={getattr(config.editor, 'continuation', None)}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SoftEdit continuation-term ablation (VLM)")
    parser.add_argument(
        "--ablation", type=str, required=True, choices=list(ABLATIONS),
        help="Ablation config: no_cont",
    )
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
    args.suffix = f"_abl_{args.ablation}"
    args.task = "postedit_qa"
    config = configure_args(args, config_path=args.config)

    _apply_ablation(config, args.ablation)

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
    run_edit(config, sequential=args.sequential, eval_every=args.eval_every,
             skip=_SKIP)
