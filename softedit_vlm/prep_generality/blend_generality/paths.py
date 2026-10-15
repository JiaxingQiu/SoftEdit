"""Output paths + on-the-fly label/weight helpers for igen_blend.

Built ONLY for ``soft_gen_available`` samples (reuse ``iter_soft_gen_samples``),
mirroring igen/tgen. Its own data root ``data/blend_generality/`` parallels the
module name (like ``data/soft_generality/``). Gold labels are reconstructable on
the fly from the source parquet's ``answer_counts`` — nothing extra is persisted.
"""

from __future__ import annotations

from pathlib import Path

from prep_generality.generality.shared.sample_io import (
    REPO_ROOT,
    iter_soft_gen_samples,
    ranked_labels,
)

# ---- output layout: data/blend_generality/{dataset}/{uid}/blend{i}.png ----
BLEND_ROOT = REPO_ROOT / "data" / "blend_generality"

BLEND_CAP = 3      # blend over the top-N labels by frequency
N_ROUNDS = 2       # two rounds -> blend{0,1}.png, both anchored on the top-1 label
                   # but seeded differently (heterogeneous anchor images)


def igen_blend_dir(dataset: str, uid: str) -> Path:
    return BLEND_ROOT / dataset / str(uid)


def igen_blend_path(dataset: str, uid: str, round_idx: int) -> Path:
    return igen_blend_dir(dataset, uid) / f"blend{round_idx}.png"


def top_labels(answer_counts: dict[str, int], cap: int = BLEND_CAP) -> list[str]:
    """The (capped) set of labels that are blended, ranked by frequency."""
    return ranked_labels(answer_counts)[:cap]


def blend_weights(answer_counts: dict[str, int], labels: list[str]) -> list[float]:
    """Frequency weights for ``labels``, renormalized to sum to 1."""
    freqs = [float(answer_counts[l]) for l in labels]
    total = sum(freqs)
    if total <= 0:
        return [1.0 / len(labels)] * len(labels)
    return [f / total for f in freqs]


def igen_blend_gold(answer_counts: dict[str, int], cap: int = BLEND_CAP) -> list[str]:
    """Gold label set for a blend image: the capped top labels as a tie set.

    Reconstructed on the fly at eval time from the source ``answer_counts`` — the
    same value for both blend{0,1}.png of a uid.
    """
    return top_labels(answer_counts, cap)


__all__ = [
    "REPO_ROOT",
    "BLEND_ROOT",
    "BLEND_CAP",
    "N_ROUNDS",
    "igen_blend_dir",
    "igen_blend_path",
    "top_labels",
    "blend_weights",
    "igen_blend_gold",
    "iter_soft_gen_samples",
]
