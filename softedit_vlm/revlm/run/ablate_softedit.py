"""SoftEdit loss-term ablation runner (VLM).

Configs isolate the gen–loc tradeoff of

    L = L_align + λ_gen · L_gen + λ_loc · L_loc

Results are isolated under ``results/{ee,te,pred_postedit}/softedit_abl_<name>/...``
via the existing suffix mechanism. Sequential JSONL peeks are left uncapped;
cap to N edits at analysis time.

The ``full`` config matches the default SoftEdit (main table), so its results can
be loaded directly from ``results/ee/softedit/...``. ``align_gen_loc_caption``
adds ``locality_caption="on"`` to match the prelim graph_smooth experiment setup.

Usage::

    python -m revlm.run.ablate_softedit --ablation align --model_name blip --dataset_name vqav2
    python -m revlm.run.ablate_softedit --ablation align_gen_loc_caption --model_name llava --dataset_name vqav2 --sequential
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

# (lambda_gen, lambda_loc, locality_n, locality_declarative)
# locality_declarative is unused when locality_n == 0.
ABLATIONS = {
    "align": dict(lambda_gen=0.0, lambda_loc=0.0, locality_n=0),
    "align_gen": dict(lambda_gen="auto", lambda_loc=0.0, locality_n=0),
    "align_gen_loc_rand": dict(
        lambda_gen=1.0, lambda_loc=1.0, locality_n=1,
        locality_declarative="random",
    ),
    "align_loc_gen0.5": dict(lambda_gen=0.5),
}


def _apply_ablation(config, name: str) -> None:
    overrides = ABLATIONS[name]
    for key, value in overrides.items():
        setattr(config.editor, key, value)
    print(
        f"[ablate_softedit] ablation={name}  suffix=_abl_{name}  "
        f"lambda_gen={getattr(config.editor, 'lambda_gen', None)}  "
        f"lambda_loc={getattr(config.editor, 'lambda_loc', None)}  "
        f"locality_n={getattr(config.editor, 'locality_n', None)}  "
        f"locality_declarative={getattr(config.editor, 'locality_declarative', None)}  "
        f"locality_caption={getattr(config.editor, 'locality_caption', None)}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SoftEdit loss-term ablation (VLM)")
    parser.add_argument(
        "--ablation", type=str, required=True, choices=list(ABLATIONS),
        help="Loss combo: align | align_gen | align_gen_loc_rand | align_loc_gen0.5",
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
    parser.add_argument("--loss_eps", type=float, default=None,
                        help="Early-stop convergence threshold (overrides editor default)")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.editor = "softedit"
    args.suffix = f"_abl_{args.ablation}"
    args.task = "postedit_qa"
    config = configure_args(args, config_path=args.config)

    _apply_ablation(config, args.ablation)

    if args.loss_eps is not None:
        config.editor.loss_eps = args.loss_eps

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
