"""igen_blend core: fresh T2I anchor + cross-edits + frequency-weighted latent blend.

Each round is anchored on the sample's top-1 label (rounds differ only by seed):
  1. generate a BRAND-NEW image (T2I, no source image) for the top-1 label;
  2. edit that anchor image toward every other blended label (top-``cap``);
  3. blend all label-images in FLUX VAE latent space, weighted by renormalized
     ``answer_counts`` frequency;
  4. decode -> one "ambiguous" blend image.

Reuses ``FluxKleinEditor`` (VAE encode/decode + edit) from soft_generality.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import torch
from PIL import Image, ImageFilter

from prep_generality.soft_generality.backends.flux_klein import (
    DEFAULT_SEED,
    DEFAULT_SIZE,
    DEFAULT_STEPS,
    FluxKleinEditor,
)

from .paths import BLEND_CAP, blend_weights, top_labels
from .prompts import build_edit_prompt, build_t2i_prompt

# FLUX.2-klein is step-distilled → guidance_scale is ignored (default 4.0 still fires
# a per-call log line). Drop just that noisy record; keep all other diffusers logs.
class _DropGuidanceIgnored(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "ignored for step-wise distilled" not in record.getMessage()


def _silence_guidance_warning() -> None:
    filt = _DropGuidanceIgnored()
    lg = logging.getLogger("diffusers")
    lg.addFilter(filt)
    # child module loggers propagate here; filter on the handler catches those too.
    for handler in lg.handlers + logging.getLogger().handlers:
        handler.addFilter(filt)


_silence_guidance_warning()


def _seed(uid: str, round_idx: int, base_seed: int = DEFAULT_SEED) -> int:
    h = hashlib.sha256(f"{uid}:{round_idx}:{base_seed}".encode()).hexdigest()
    return int(h[:8], 16)


# Per-dataset anchor degradation applied AFTER T2I, BEFORE cross-edits. Downscale
# (to ~CIFAR res) + Gaussian blur mimics the low-res/ambiguous source and loosens
# rigid structure so cross-edits (e.g. frog->cat) actually change the object.
ANCHOR_DEGRADE: dict[str, dict] = {
    "cifar10h": {"downscale": 24, "blur": 2.5},
}

# How each label's image is produced: "edit" (anchor + cross-edits) or "t2i"
# (independent fresh generation per label).
DEFAULT_LABEL_SOURCE = "edit"

# Blend uses the same step count as soft_generality (efficient).
BLEND_STEPS = 4


def degrade_anchor(img: Image.Image, dataset: str | None) -> Image.Image:
    """Downscale-then-upscale + blur per ``ANCHOR_DEGRADE[dataset]`` (no-op if absent)."""
    cfg = ANCHOR_DEGRADE.get(dataset or "")
    if not cfg:
        return img
    w, h = img.size
    ds = cfg.get("downscale")
    if ds:
        img = img.resize((ds, ds), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    blur = cfg.get("blur")
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return img


def generate_t2i(
    editor: FluxKleinEditor,
    prompt: str,
    *,
    size: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
    num_inference_steps: int = DEFAULT_STEPS,
) -> Image.Image:
    """Text-to-image generation (no source image) via Flux2KleinPipeline(image=None)."""
    generator = torch.Generator(device=editor.device).manual_seed(seed)
    with torch.inference_mode():
        out = editor.pipe(
            image=None,
            prompt=prompt,
            height=size,
            width=size,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            generator=generator,
        )
    return out.images[0]


def build_blend_for_round(
    editor: FluxKleinEditor,
    sample: dict[str, Any],
    round_idx: int,
    *,
    cap: int = BLEND_CAP,
    size: int = DEFAULT_SIZE,
    base_seed: int = DEFAULT_SEED,
    num_inference_steps: int = BLEND_STEPS,
    label_source: str = DEFAULT_LABEL_SOURCE,
    return_parts: bool = False,
) -> Image.Image | dict[str, Any]:
    """Build one blend image for the top-``cap`` labels of a sample.

    ``label_source`` controls how each label's image is produced:
      * ``"edit"`` — T2I the top-1 anchor, degrade it, then cross-edit toward the
        other labels (coherent; shared structure across labels).
      * ``"t2i"``  — generate each label independently via fresh T2I (+ degrade)
        (more label-faithful; less coherent). Use when edits won't flip the class.

    ``round_idx`` only changes the seed, so each round is a different draw. Set
    ``return_parts=True`` to also get intermediate per-label images and weights.
    """
    if label_source not in ("edit", "t2i"):
        raise ValueError(f"label_source must be 'edit' or 't2i', got {label_source!r}")

    dataset = sample.get("dataset")

    # t2i fallback is cifar10h-only; silently force "edit" for other datasets
    if label_source == "t2i" and dataset != "cifar10h":
        label_source = "edit"
    question = sample["question"]
    counts = sample["answer_counts"]

    # Ground the T2I anchor in the source image:
    #   * VQAv2  -> 1-sentence scene caption (context for the answer)
    #   * FERPlus -> 1-sentence expression-free face description (age/pose/lighting/...)
    # Only the anchor T2I uses this; cross-edits inherit identity from the anchor image.
    caption: str | None = None
    if dataset == "vqav2" and sample.get("image_path"):
        from .caption import caption_image
        caption = caption_image(sample["image_path"])
    elif dataset == "ferplus" and sample.get("image_path"):
        from .face_context import describe_face
        caption = describe_face(sample["image_path"])

    labels = top_labels(counts, cap)
    weights = blend_weights(counts, labels)
    anchor = labels[0]  # top-1 label (the anchor in "edit" mode)
    seed = _seed(sample["uid"], round_idx, base_seed)

    images_by_label: dict[str, Image.Image] = {}
    if label_source == "edit":
        # 1) fresh anchor image (T2I), then degrade (super-blurry / low-res per dataset)
        anchor_img = degrade_anchor(
            generate_t2i(
                editor,
                build_t2i_prompt(question, anchor, dataset=dataset, caption=caption),
                size=size, seed=seed, num_inference_steps=num_inference_steps,
            ),
            dataset,
        )
        images_by_label[anchor] = anchor_img
        # 2) cross-edit the anchor toward every other blended label
        for lab in labels:
            if lab == anchor:
                continue
            images_by_label[lab] = editor.edit(
                anchor_img,
                build_edit_prompt(question, lab, dataset=dataset),
                num_inference_steps=num_inference_steps,
                height=size, width=size, seed=seed,
            ).image
    else:  # "t2i" — independent fresh generation per label
        for k, lab in enumerate(labels):
            lab_seed = _seed(sample["uid"], f"{round_idx}:{k}", base_seed)
            images_by_label[lab] = degrade_anchor(
                generate_t2i(
                    editor,
                    build_t2i_prompt(question, lab, dataset=dataset, caption=caption),
                    size=size, seed=lab_seed, num_inference_steps=num_inference_steps,
                ),
                dataset,
            )

    # 3) frequency-weighted blend in VAE latent space (labels order == weights order)
    z_blend = None
    for lab, w in zip(labels, weights):
        z = editor._encode_image(images_by_label[lab], size, size)
        z_blend = z * w if z_blend is None else z_blend + z * w

    # 4) decode
    blended = editor._decode_latent(z_blend)

    if return_parts:
        return {
            "blend": blended,
            "anchor": anchor,
            "labels": labels,
            "weights": weights,
            "images_by_label": images_by_label,
            "label_source": label_source,
            "caption": caption,
            "seed": seed,
        }
    return blended
