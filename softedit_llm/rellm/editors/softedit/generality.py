"""Generality term: graph-Laplacian Jacobian coupling over the label rows.

The affinity graph is built from the human soft target ``p`` itself::

    A_ij = p_i * p_j   (i != j),   A_ii = 0

so co-plausible labels get strong edges and labels the human excludes are
disconnected. The generality loss couples co-plausible label rows so they
respond *together* as the hidden state moves off the edit point::

    L_gen = sum_ij A_ij ||J_i - J_j||^2 = 2 tr(J^T L J),   L = diag(A 1) - A

where ``J_i = d q_i / d h`` are the softmax-Jacobian rows of the edited head at
``h0``. Freezing the relative ordering among plausible labels in a neighborhood
of ``h0`` is what makes the edit generalize instead of spiking one label.

Needs only the edit sample; lm_head (K label rows) only. Model-agnostic — ported
verbatim from the VLM repo.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_affinity(p: torch.Tensor) -> torch.Tensor:
    """Affinity ``A_ij = p_i p_j`` (i != j), zero diagonal. Raw, unnormalized."""
    p = p.detach().float().reshape(-1)
    A = p.unsqueeze(1) * p.unsqueeze(0)
    A.fill_diagonal_(0.0)
    return A


def _jac_rows(W: torch.Tensor, h: torch.Tensor, mode: str = "prob") -> torch.Tensor:
    """Per-label Jacobian rows [K, d] at ``h``.

    "prob":    J = S W   with S = diag(q) - q qᵀ   (softmax Jacobian)
    "logprob": J = (I - 1 qᵀ) W                     (log-softmax Jacobian)
    """
    q = F.softmax((W @ h.detach()).float(), dim=-1)  # [K]
    if mode == "logprob":
        return W - (q @ W).unsqueeze(0)  # [K, d]
    S = torch.diag(q) - q.unsqueeze(1) * q.unsqueeze(0)  # [K, K]
    return S @ W  # [K, d]


def graph_gen_at_point(
    W: torch.Tensor,
    h: torch.Tensor,
    A: torch.Tensor,
    mode: str = "prob",
) -> torch.Tensor:
    """``L_gen = sum_ij A_ij ||J_i - J_j||^2 = 2 tr(Jᵀ L J)`` at ``h``.

    Args:
        W: [K, d] edited label rows (W_sub + dW), in the computation graph.
        h: [d] hidden state (detached internally).
        A: [K, K] affinity from ``build_affinity``.
        mode: "prob" or "logprob".

    Returns:
        scalar tensor in the computation graph of ``W``.
    """
    J = _jac_rows(W, h, mode=mode)
    A = A.to(J.device, J.dtype)
    L = torch.diag(A.sum(dim=1)) - A
    return 2.0 * (J * (L @ J)).sum()
