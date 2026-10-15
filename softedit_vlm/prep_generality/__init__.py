"""Preparation of generality evaluation sets.

Umbrella package for the generality-set builders:

- ``soft_generality``  — step 1: strength-controlled FLUX image edits + latent interpolation (GPU).
- ``generality``       — step 2: augmentation-based i-gen + t-gen (CPU).
- ``blend_generality`` — step 3: igen_blend: frequency-weighted FLUX latent blends (GPU).

Outputs are written under ``data/generality/``, ``data/blend_generality/``, and
``data/soft_generality/``.
"""
