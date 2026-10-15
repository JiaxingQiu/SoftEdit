"""Turning a soft label *distribution* into supervision targets.

Two forms live here:

* ``ranked_label_str(gold)`` -- the comma-separated ranked label string used by
  the ``soft="target"`` weight editors and every retrieval editor (teacher-forced
  with ordinary next-token CE).
* ``build_soft_target(...)`` / :class:`SoftTarget` -- the target categorical ``q``
  over the edit's K labels plus their first-token ids, consumed by the
  distribution-aware KL loss (``soft in {"loss", "rank_loss"}``).

Kept separate from the loss math so the "what do we supervise toward" logic is one
small, testable unit independent of "how do we score it".
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from rellm.metrics.ranked_metrics import build_labels_true, norm


def ranked_label_str(gold: Dict) -> str:
    """Supervision target for the ``soft="target"`` mode (and retrieval editors):
    the full ranked multi-label list as a comma-separated single-word string::

        label1, label2, label3

    Comma-separated (not numbered) so it matches the postedit_qa format anchor and
    the edited model's first generated token is a real label.

    Label source, in priority order:
      1. ``gold['labels_ranked']`` (populated by the MultiQA/SoftQA task), else
      2. derived from ``gold['answer_counts']`` (raw annotator frequencies, present
         under every task) using the evaluator's frequency-tiered ordering.

    Falls back to the single majority-vote label when no distribution is available.
    """
    gold = gold or {}
    labels = list(gold.get("labels_ranked") or [])
    if not labels:
        counts = gold.get("answer_counts") or {}
        if counts:
            labels = [lbl for tier in build_labels_true(counts) for (lbl, _c) in tier]
    if labels:
        return ", ".join(labels)
    return str(gold.get("label_train") or gold.get("label") or "")


@dataclass
class SoftTarget:
    """Precomputed distribution supervision for one edit (batch_size=1)."""
    labels: List[str]          # K support labels (normalized)
    first_ids: torch.Tensor    # LongTensor [K]: each label's first token id
    q: torch.Tensor            # FloatTensor [K]: target distribution, sums to 1


def _tiers_from_ranked(labels_ranked) -> List[List]:
    """Coerce a `labels_ranked` field into tie-aware tiers.

    Accepts either an already-tiered list (list of lists -> preserved) or a flat
    ordered list (each label becomes its own singleton tier, i.e. no ties)."""
    tiers: List[List] = []
    for entry in labels_ranked:
        if isinstance(entry, (list, tuple)):
            tiers.append([(lbl, None) for lbl in entry])
        else:
            tiers.append([(entry, None)])
    return tiers


def _first_token_id(tokenizer, label: str) -> Optional[int]:
    ids = tokenizer.encode(label, add_special_tokens=False)
    return ids[0] if ids else None


def ranked_q_weights(gold: Dict, rank_temp: float = 1.0) -> Tuple[List[str], List[float]]:
    """Rank-group exponential weights for ``soft="rank_loss"``.

    Gold is an ordered partition R = [R_1, ..., R_M] (ties allowed).
    Decay is assigned at the rank, then split equally inside the group::

        q_r  ∝ exp(-(r-1)/τ)
        p(y) = q_r / |R_r|    for y in R_r

    Tiers come from ``build_labels_true(answer_counts)`` when counts exist
    (equal counts → one rank; count *values* are discarded), else from a nested
    or flat ``labels_ranked`` list. See ``readme/rank_objective.md``.
    """
    gold = gold or {}
    counts = gold.get("answer_counts") or {}
    rank_temp = max(float(rank_temp), 1e-6)
    tiers = build_labels_true(counts) if counts else _tiers_from_ranked(
        gold.get("labels_ranked") or [])
    labels: List[str] = []
    weights: List[float] = []
    for tier_idx, tier in enumerate(tiers):
        n = len(tier)
        if n == 0:
            continue
        mass = math.exp(-tier_idx / rank_temp)
        w = mass / n
        for lbl, _score in tier:
            labels.append(norm(lbl))
            weights.append(w)
    return labels, weights


def build_soft_target(gold: Dict, tokenizer, mode: str, rank_temp: float = 1.0,
                      device=None, major_vote: bool = False) -> SoftTarget:
    """Build the per-edit `SoftTarget` for `mode in {"loss", "rank_loss"}`.

    Labels whose first token cannot be resolved are dropped and `q` renormalized.
    Falls back to a one-hot on the single majority label when no usable
    distribution is available.

    ``major_vote`` (ablation): collapse the soft distribution to a one-hot on the
    majority label (argmax of `q`, first on a tie); the K labels and `first_ids`
    are unchanged."""
    gold = gold or {}
    counts = gold.get("answer_counts") or {}
    rank_temp = max(float(rank_temp), 1e-6)

    labels: List[str] = []
    weights: List[float] = []

    if mode == "rank_loss":
        labels, weights = ranked_q_weights(gold, rank_temp)
    else:  # "loss": use raw annotator frequencies
        positive = {lbl: float(c) for lbl, c in counts.items() if c and c > 0}
        for lbl, c in positive.items():
            labels.append(norm(lbl))
            weights.append(c)

    # Resolve first-token ids; drop labels the tokenizer can't map and align weights.
    ids: List[int] = []
    kept_labels: List[str] = []
    kept_weights: List[float] = []
    for lbl, w in zip(labels, weights):
        tid = _first_token_id(tokenizer, lbl)
        if tid is not None:
            ids.append(tid)
            kept_labels.append(lbl)
            kept_weights.append(w)

    # Fallback: one-hot on the single majority label (soft-CE).
    if not kept_labels:
        single = norm(gold.get("label_train") or gold.get("label") or "")
        tid = _first_token_id(tokenizer, single) if single else None
        if tid is None:
            raise ValueError(
                f"build_soft_target: no usable label for gold uid={gold.get('uid')!r} "
                f"(mode={mode}, counts={counts!r})")
        kept_labels, ids, kept_weights = [single], [tid], [1.0]

    q = torch.tensor(kept_weights, dtype=torch.float32, device=device)
    q = q / q.sum()
    if major_vote:
        q = torch.zeros_like(q)
        q[int(torch.argmax(torch.tensor(kept_weights)))] = 1.0
    first_ids = torch.tensor(ids, dtype=torch.long, device=device)
    return SoftTarget(labels=kept_labels, first_ids=first_ids, q=q)
