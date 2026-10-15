"""Dataset-aware prompts for igen_blend.

Two prompt kinds are used:

  * T2I ("generation") — build a *brand-new* image (no source image) whose answer
    to the question is a target label. Used for the anchor image.
  * edit — reused from ``soft_generality`` to push the anchor image toward each
    other blended label (see ``build_prompt``).
"""

from __future__ import annotations

# Text-to-image generation templates (fresh image, instruction only).
T2I_TEMPLATES: dict[str, str] = {
    # cifar10h images are low-res / blurry (hence ambiguous) — don't push "realistic".
    "cifar10h": "a small blurry low-resolution photo of a {label}",
    # Direct expression clause; a one-sentence face context is prepended when available
    # (see T2I_FERPLUS_CONTEXT). Keep it terse so context isn't overwhelmed.
    "ferplus": "A human face. The face expresses {label}.",
    "vqav2": 'a realistic photo where the answer to the question "{question}" is "{label}"',
}
# VQAv2 with scene caption context (used when caption is available).
T2I_VQAV2_CAPTIONED = (
    'Scene: {caption}. A realistic photo where the answer to "{question}" is "{label}"'
)
# FERPlus with a one-sentence, expression-free face context prepended (see face_context).
T2I_FERPLUS_CONTEXT = "{caption} The face expresses {label}."

T2I_FALLBACK = 'a realistic photo where the answer to the question "{question}" is "{label}"'


def build_t2i_prompt(
    question: str,
    label: str,
    *,
    dataset: str | None = None,
    caption: str | None = None,
) -> str:
    """Build a text-to-image generation prompt for one target label.

    For VQAv2 a scene ``caption`` is prepended; for FERPlus a one-sentence,
    expression-free face ``caption`` (context) is prepended.
    """
    if dataset == "vqav2" and caption:
        return T2I_VQAV2_CAPTIONED.format(caption=caption, question=question, label=label)
    if dataset == "ferplus" and caption:
        return T2I_FERPLUS_CONTEXT.format(caption=caption, label=label)
    template = T2I_TEMPLATES.get(dataset, T2I_FALLBACK) if dataset else T2I_FALLBACK
    return template.format(question=question, label=label)


# Cross-edit templates. Only datasets that need a blend-specific override live here;
# everything else falls back to soft_generality's shared edit prompts (kept in sync).
from prep_generality.soft_generality.shared.prompts import build_prompt as _soft_edit_prompt

EDIT_TEMPLATES: dict[str, str] = {
    # Imperative content change; keep it terse so the model actually edits the object.
    "cifar10h": "Make this a {label}.",
}


def build_edit_prompt(
    question: str,
    label: str,
    *,
    dataset: str | None = None,
) -> str:
    """Build a cross-edit prompt (blend-specific override, else soft_generality's)."""
    if dataset in EDIT_TEMPLATES:
        return EDIT_TEMPLATES[dataset].format(question=question, label=label)
    return _soft_edit_prompt(question, label, dataset=dataset)
