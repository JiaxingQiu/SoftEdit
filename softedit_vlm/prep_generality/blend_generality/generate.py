"""Build the igen_blend set for a dataset (GPU / FLUX.2-klein-9B).

Only ``soft_gen_available`` samples are processed (same population as igen/tgen,
via ``iter_soft_gen_samples``). For each sample it builds N_ROUNDS blend images,
both anchored on the top-1 label but seeded differently (heterogeneous anchor
images) -> data/blend_generality/{dataset}/{uid}/blend{i}.png

Idempotent: skips blend images already on disk. Gold labels are NOT persisted —
they are reconstructed on the fly from the source parquet's answer_counts.

By default the dataset's native-domain refine (see ``refine.PRESETS``) is applied
to each blend before saving, so the ``{dataset}`` folder holds the final,
domain-matched images in one stage (no separate refine pass / folder swap). This
matches the old generate->refine two-stage output (presets use noise=0, so refine
is deterministic). Datasets without a preset (e.g. vqav2) are saved unrefined.
Pass ``--no_refine`` to save the raw 512 blends instead.

Usage (env: soft_gen):
    python -m prep_generality.blend_generality.generate --dataset vqav2          # one-stage, refined
    python -m prep_generality.blend_generality.generate --dataset vqav2 --limit 10  # smoke test (unrefined)
    python -m prep_generality.blend_generality.generate --dataset vqav2 --no_refine
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from prep_generality.soft_generality.backends.flux_klein import (
    DEFAULT_SEED,
    DEFAULT_SIZE,
)
from prep_generality.blend_generality.blend import BLEND_STEPS, DEFAULT_LABEL_SOURCE, build_blend_for_round
from prep_generality.blend_generality.paths import (
    BLEND_CAP,
    BLEND_ROOT,
    N_ROUNDS,
    igen_blend_path,
    iter_soft_gen_samples,
)

DATASETS = ("vqav2",)


def build(
    dataset: str,
    *,
    cap: int = BLEND_CAP,
    size: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
    num_inference_steps: int = BLEND_STEPS,
    label_source: str = DEFAULT_LABEL_SOURCE,
    overwrite: bool = False,
    offset: int = 0,
    limit: int = 0,
    refine: bool = True,
    refine_overrides: dict | None = None,
) -> None:
    samples = list(iter_soft_gen_samples(dataset))
    if offset:
        samples = samples[offset:]
    if limit:
        samples = samples[:limit]

    # One-stage refine (default): domain-match each blend before saving (see module doc).
    from prep_generality.blend_generality.refine import (
        PRESETS as REFINE_PRESETS,
        refine_image,
        _uid_seed as uid_refine_seed,
    )
    do_refine = refine and dataset in REFINE_PRESETS
    refine_params = {
        **REFINE_PRESETS[dataset],
        **{k: v for k, v in (refine_overrides or {}).items() if v is not None},
    } if do_refine else None

    print("=" * 60, flush=True)
    print(f"igen_blend build | dataset={dataset} | n={len(samples)} (soft_gen_available)", flush=True)
    print(f"out → {BLEND_ROOT / dataset}/<uid>/blend*.png", flush=True)
    print(f"cap={cap} size={size} steps={num_inference_steps} "
          f"seed={seed} label_source={label_source} overwrite={overwrite}", flush=True)
    if do_refine:
        print(f"refine=ON (one-stage) params={refine_params}", flush=True)
    elif refine:
        print(f"refine requested but no preset for {dataset!r} -> saving unrefined", flush=True)
    if offset:
        print(f"offset={offset} (skipped first {offset} samples)", flush=True)
    print("=" * 60, flush=True)

    from prep_generality.soft_generality.backends.flux_klein import FluxKleinEditor
    print("loading FluxKleinEditor (offload=False)...", flush=True)
    editor = FluxKleinEditor(offload=False)
    print("model loaded.", flush=True)

    t0 = time.time()
    n_new = n_skip = 0

    for i, s in enumerate(samples, 1):
        uid = s["uid"]
        for r in range(N_ROUNDS):
            out_path = igen_blend_path(dataset, uid, r)
            if out_path.exists() and not overwrite:
                n_skip += 1
                continue
            img = build_blend_for_round(
                editor, s, r,
                cap=cap, size=size, base_seed=seed,
                num_inference_steps=num_inference_steps,
                label_source=label_source,
            )
            if do_refine:
                img = refine_image(img, seed=uid_refine_seed(uid, r), **refine_params)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path)
            n_new += 1

        if i % 20 == 0:
            rate = i / max(time.time() - t0, 1e-6)
            print(f"[{dataset}] {i}/{len(samples)}  blend+{n_new} skip{n_skip}  "
                  f"{rate:.2f} samp/s", flush=True)

    print("=" * 60, flush=True)
    print(f"[done] {dataset}: blend images +{n_new} (skipped {n_skip})", flush=True)
    print(f"  igen_blend: {BLEND_ROOT / dataset}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build igen_blend set (FLUX)")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--cap", type=int, default=BLEND_CAP, help="blend over top-N labels")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--steps", type=int, default=BLEND_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--label_source", choices=("edit", "t2i"), default=DEFAULT_LABEL_SOURCE,
                        help="cifar10h only: 'edit' (default) or 't2i' fallback; ignored for other datasets")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--offset", type=int, default=0, help="skip first N samples (for parallel slicing)")
    parser.add_argument("--limit", type=int, default=0, help="process at most N samples (0=all)")
    parser.add_argument("--no_refine", dest="refine", action="store_false", default=True,
                        help="disable the default one-stage refine (save raw 512 blends)")
    parser.add_argument("--refine_crop_frac", type=float, default=None,
                        help="override refine crop_frac (central crop fraction)")
    parser.add_argument("--refine_crop_y", type=float, default=None,
                        help="override refine crop_y (vertical crop center)")
    parser.add_argument("--refine_noise", type=float, default=None,
                        help="override refine gaussian noise frac (0-1)")
    args = parser.parse_args()

    if args.label_source == "t2i" and args.dataset != "cifar10h":
        parser.error("--label_source t2i is only supported for cifar10h")

    build(
        args.dataset,
        cap=args.cap,
        size=args.size,
        seed=args.seed,
        num_inference_steps=args.steps,
        label_source=args.label_source,
        overwrite=args.overwrite,
        offset=args.offset,
        limit=args.limit,
        refine=args.refine,
        refine_overrides={
            "crop_frac": args.refine_crop_frac,
            "crop_y": args.refine_crop_y,
            "noise": args.refine_noise,
        },
    )
