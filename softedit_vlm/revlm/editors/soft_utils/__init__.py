"""Soft-label supervision utilities for the ``*_soft`` editors.

Split into small, single-responsibility modules so the different editor families
(`ft`/`mend`, their `*_retrain` replays, and the hook-based `grace`/`balancedit`)
can share the same pieces without one monolith:

* ``targets``      -- distribution -> supervision targets (`SoftTarget`,
                      `build_soft_target`, `ranked_q_weights`, `ranked_label_str`).
* ``kl``           -- first-token distribution term `KL(q || p_theta)`.
* ``continuation`` -- multi-token completability term (uniform continuation NLL)
                      and the per-edit continuation-id builder.
* ``loss``         -- `compute_edit_loss` dispatch that combines the two terms.

See ``../readme/soft_objective.md`` for the objective spec and
``../readme/rank_objective.md`` for the ``rank_loss`` target \(q\).
"""
from .continuation import build_cont_label_ids, continuation_nll
from .kl import kl_loss
from .loss import SOFT_LOSS_MODES, compute_edit_loss
from .targets import SoftTarget, build_soft_target, ranked_label_str, ranked_q_weights

__all__ = [
    "SoftTarget",
    "build_soft_target",
    "ranked_label_str",
    "ranked_q_weights",
    "kl_loss",
    "continuation_nll",
    "build_cont_label_ids",
    "compute_edit_loss",
    "SOFT_LOSS_MODES",
]
