"""igen_blend: ambiguous image-generality via frequency-weighted latent blends.

For each soft_gen_available sample, build 2 blend images, both anchored on the
top-1 label but seeded differently (heterogeneous anchor images): a fresh T2I
anchor, cross-edited toward the other top-``cap`` labels, then blended in FLUX VAE
latent space weighted by answer_counts frequency.

Outputs: data/blend_generality/{dataset}/{uid}/blend{i}.png
Gold labels are reconstructed on the fly from answer_counts (see ``igen_blend_gold``).
"""
