"""Soft generality: strength-controlled image edits for SoftQA eval sets.

Usage:
    from prep_generality.soft_generality import load_sample, generate_for_sample
    from prep_generality.soft_generality.backends.flux_klein import FluxKleinEditor
"""

from .pipeline import generate_for_sample
from .shared import build_prompt, list_datasets, load_sample

__all__ = [
    "build_prompt",
    "generate_for_sample",
    "list_datasets",
    "load_sample",
]
