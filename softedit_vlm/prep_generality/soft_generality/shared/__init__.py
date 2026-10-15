"""Shared utilities (data loading, prompts) — backend-agnostic."""

from .prompts import FALLBACK_TEMPLATE, TEMPLATES, build_prompt
from .sample_io import list_datasets, load_sample

__all__ = [
    "FALLBACK_TEMPLATE",
    "TEMPLATES",
    "build_prompt",
    "list_datasets",
    "load_sample",
]
