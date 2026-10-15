"""Backend-agnostic generation pipeline for notebook demos and downstream jobs."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from PIL import Image

from .shared.prompts import build_prompt


class ImageEditor(Protocol):
    """Interface that all backends must satisfy."""
    name: str

    def strength_sweep(
        self,
        image: Image.Image,
        prompt: str,
        *,
        strengths: Sequence[float],
        height: int,
        width: int,
        seed: int,
        _log_prefix: str,
    ) -> list: ...

    def strength_titles(self, results: list) -> list[str]: ...


def prepare_image(sample: dict[str, Any], size: int) -> Image.Image:
    return sample["image"].resize((size, size), Image.Resampling.LANCZOS)


def generate_for_label(
    editor: ImageEditor,
    image: Image.Image,
    question: str,
    label: str,
    *,
    dataset: str | None = None,
    strengths: Sequence[float],
    num_inference_steps: int,
    size: int,
    seed: int,
    _log_prefix: str = "",
) -> dict[str, Any]:
    """Generate spectrum images for one label."""
    prompt = build_prompt(question, label, dataset=dataset)
    results = editor.strength_sweep(
        image,
        prompt,
        strengths=strengths,
        num_inference_steps=num_inference_steps,
        height=size,
        width=size,
        seed=seed,
        _log_prefix=_log_prefix,
    )
    titles = editor.strength_titles(results)
    return {
        "label": label,
        "prompt": prompt,
        "results": results,
        "titles": titles,
    }


def generate_for_sample(
    editor: ImageEditor,
    sample: dict[str, Any],
    *,
    strengths: Sequence[float] | None = None,
    num_inference_steps: int | None = None,
    labels: list[str] | None = None,
    size: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate for every label of a sample with the given editor.

    Defaults are pulled from the editor's backend config if not overridden.
    """
    from .backends.flux_klein.config import (
        DEFAULT_STEPS as FLUX_STEPS,
        DEFAULT_STRENGTH_SWEEP as FLUX_SWEEP,
        DEFAULT_SEED as FLUX_SEED,
        DEFAULT_SIZE as FLUX_SIZE,
    )

    _strengths = strengths or FLUX_SWEEP
    _steps = num_inference_steps or FLUX_STEPS
    _size = size or FLUX_SIZE
    _seed = seed if seed is not None else FLUX_SEED

    image = prepare_image(sample, size=_size)
    labels = labels or sample["ranked_labels"]
    n_labels = len(labels)
    print(f"[{editor.name}] sample={sample['uid']}  labels={n_labels}")

    by_label = {}
    for li, lab in enumerate(labels, 1):
        prefix = f"[{editor.name}] label {li}/{n_labels} \"{lab}\""
        print(prefix)
        by_label[lab] = generate_for_label(
            editor,
            image,
            sample["question"],
            lab,
            dataset=sample.get("dataset"),
            strengths=_strengths,
            num_inference_steps=_steps,
            size=_size,
            seed=_seed,
            _log_prefix=prefix,
        )
    print(f"[{editor.name}] done.")
    return {
        "sample": sample,
        "image": image,
        "backend": editor.name,
        "by_label": by_label,
        "size": _size,
        "seed": _seed,
    }
