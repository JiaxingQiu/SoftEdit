"""Offline: precompute AlphaEdit's wikipedia covariance + null-space projections.

One-time per backbone. Streams wikipedia through the model, accumulates the mom2
covariance of each edited ``mlp.down_proj``'s fan-in, and builds the null-space
projector ``P``; both are cached under ``data/stats/<model>/`` so edit jobs only
load them.

Usage (from the repo root):
    python -m rellm.editors.alphaedit.prep_alphaedit.compute_cov --model_name qwen
    python -m rellm.editors.alphaedit.prep_alphaedit.compute_cov --model_name llama --force

Requires a GPU (>=48G recommended for the 7-8B backbones). The covariance pass over
the wikipedia samples is the expensive part.
"""
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# .../rankedit_llm/rellm/editors/alphaedit/prep_alphaedit/compute_cov.py -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from rellm.config_utils import configure_args
from rellm.models import TQAModel
from rellm.editors.alphaedit.hparams import resolve_hparams
from rellm.editors.alphaedit.nullspace import build_projections, build_projection_grid


def main():
    ap = argparse.ArgumentParser(description="AlphaEdit covariance / null-space prep")
    ap.add_argument("--model_name", required=True, help="Short name: qwen | llama | mistral")
    ap.add_argument("--mom2_n_samples", type=int, default=None,
                    help="Override hparams.mom2_n_samples (default 100000).")
    ap.add_argument("--nullspace_threshold", type=float, default=None,
                    help="Override hparams.nullspace_threshold.")
    ap.add_argument("--threshold_grid", type=float, nargs="+", default=None,
                    help="Build+cache P for each threshold (SVD computed once, then "
                         "sliced). Overrides --nullspace_threshold when set.")
    ap.add_argument("--force", action="store_true",
                    help="Recompute covariance + projections even if cached.")
    args = ap.parse_args()

    cfg_args = SimpleNamespace(
        model_name=args.model_name, editor="alphaedit",
        dataset_name="protoqa", task="postedit_qa",
    )
    config = configure_args(cfg_args)
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Optional overrides flow through config.editor into resolve_hparams.
    if args.mom2_n_samples is not None:
        config.editor.mom2_n_samples = args.mom2_n_samples
    if args.nullspace_threshold is not None:
        config.editor.nullspace_threshold = args.nullspace_threshold

    model = TQAModel(config)
    hparams = resolve_hparams(config)
    grid_note = f"threshold_grid={args.threshold_grid}" if args.threshold_grid \
        else f"threshold={hparams.nullspace_threshold}"
    print(f"[prep] model={config.model.name} layers={hparams.layers} "
          f"mom2_dataset={hparams.mom2_dataset} n_samples={hparams.mom2_n_samples} "
          f"{grid_note}", flush=True)

    stats_dir = f"data/stats/{config.model.name.replace('/', '_')}"
    if args.threshold_grid:
        grid = build_projection_grid(
            model.model, model.tokenizer, hparams, args.threshold_grid,
            model_name=config.model.name, cache=True, force_recompute=args.force,
        )
        shapes = {t: tuple(P.shape) for t, P in grid.items()}
        print(f"[prep] done. grid P shapes={shapes} (cached under {stats_dir}/)", flush=True)
    else:
        P = build_projections(
            model.model, model.tokenizer, hparams,
            model_name=config.model.name, cache=True, force_recompute=args.force,
        )
        print(f"[prep] done. P shape={tuple(P.shape)} (cached under {stats_dir}/)", flush=True)


if __name__ == "__main__":
    main()
