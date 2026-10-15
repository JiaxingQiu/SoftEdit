"""Dataset-aware prompt templates for soft-generality image editing.

Strength is controlled by the editor (cfg_scale or strength param), not the prompt.
"""

from __future__ import annotations

TEMPLATES: dict[str, str] = {
    "vqav2": 'Edit this image so that the answer to "{question}" is "{label}".',
}

FALLBACK_TEMPLATE = 'Edit this image so that the answer to "{question}" is "{label}".'


def build_prompt(
    question: str,
    label: str,
    *,
    dataset: str | None = None,
) -> str:
    """Build an edit prompt using a dataset-specific template.

    Args:
        question: VQA / classification question for the image.
        label: Target answer to edit toward.
        dataset: Dataset name for template selection. Falls back to
            the generic VQA-style template if None or unrecognized.
    """
    template = TEMPLATES.get(dataset, FALLBACK_TEMPLATE) if dataset else FALLBACK_TEMPLATE
    return template.format(question=question, label=label)
