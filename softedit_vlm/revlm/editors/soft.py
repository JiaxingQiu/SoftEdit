"""Soft-label supervision for the `*_soft` editors (public facade).

A soft edit comes with a *distribution* over labels (annotator disagreement /
differential diagnosis) rather than a single gold label. This module is the stable
import surface editors use; the implementation is modularized under
``soft_utils/`` (targets / kl / continuation / loss). The distribution is turned
into a supervision signal in two forms:

Target-string form (``soft="target"`` weight editors, and all retrieval editors):
    `ranked_label_str(gold)` -> a comma-separated ranked label string that is
    teacher-forced with ordinary next-token cross-entropy.

Distribution form (``soft="loss"`` / ``soft="rank_loss"`` weight editors):
    `build_soft_target(...)` -> a target categorical `q` over the edit's K labels
    plus their first-token ids, matched to the model's first-token distribution
    `p_theta` by a forward-KL / soft cross-entropy loss:

        "loss"       q = normalize(answer_counts)          (vote frequencies)
        "rank_loss"  rank-group exp, ties split equally    (see readme/rank_objective.md)

`p_theta` is the softmax over the **full vocab** at the prompt-end position; `q`
lives on the labels' first token ids. `KL(q || p_theta)` is computed over the full
vocab (NOT renormalized over the K support tokens) so it anchors the support as
the argmax while matching proportions -- renormalizing would let the loss hit ~0
while the model still generates an off-support token, collapsing reliability. Its
per-logit gradient `p_j - q_j` is non-saturating, so it trains from a cold start,
and it reduces exactly to standard CE for a one-hot `q`. A missing/degenerate
distribution falls back to a one-hot `q` (KL then equals standard CE).

The first-token KL only supervises token 1, so multi-token answers are not
guaranteed to complete (the js_dist vs js_dist_gate gap). The distribution loss
therefore adds a **uniform continuation** term: for every branch with `q_k > 0`
and length `L_k > 1`, the length-normalized NLL of tokens `2:L_k` (teacher-forced
on the gold prefix), averaged uniformly over the M active branches. Both terms
are in nats and combine at natural scale (no weight). See
`readme/soft_objective.md` for the full spec and `readme/rank_objective.md` for
the rank-group \(q\).

Editors call `compute_edit_loss(self, outputs, tokens)` where they previously read
`outputs.loss`; it returns CE (target/default) or KL + continuation (distribution
modes). Each branch's answer token ids are stashed on `editor._soft_cont` (built
with `build_cont_label_ids`) and teacher-forced by reconstructing `prompt + y_k`
from the main edit `tokens` (so the image tensors are reused, not duplicated).
"""
from .soft_utils import (
    SOFT_LOSS_MODES,
    SoftTarget,
    build_cont_label_ids,
    build_soft_target,
    compute_edit_loss,
    continuation_nll,
    kl_loss,
    ranked_label_str,
    ranked_q_weights,
)

__all__ = [
    "SOFT_LOSS_MODES",
    "SoftTarget",
    "build_cont_label_ids",
    "build_soft_target",
    "compute_edit_loss",
    "continuation_nll",
    "kl_loss",
    "ranked_label_str",
    "ranked_q_weights",
]
