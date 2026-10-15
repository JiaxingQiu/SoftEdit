"""Generality set generation (augmentation-based).

Builds image-generality (i-gen) and text-generality (t-gen) evaluation sets by
augmenting the ORIGINAL <image, question> pairs of soft_gen_available samples.

- t-gen: original image + rephrased question (gpt-4o-mini)
- i-gen: label-safe augmented image + original question

Gold labels are the original sample's soft-label set, recovered on the fly at
eval time (identical to soft_generality).
"""
