"""First-token distribution term of the soft objective.

`KL(q || p_theta)` between the target categorical `q` (on the K support labels'
first token ids) and the model's first-token distribution `p_theta` (softmax over
the full vocab at the prompt-end position). This is the *proportions* half of the
two-term soft loss; the *completability* half lives in ``continuation.py``.
"""
import torch

from .targets import SoftTarget

_EPS = 1e-12


def _prompt_end_pos(labels: torch.Tensor) -> int:
    """Index of the logits row that predicts the first answer token: the position
    just before the first non-(-100) label. Robust to trailing answer padding."""
    row = labels[0] if labels.dim() > 1 else labels
    nz = (row != -100).nonzero(as_tuple=False)
    if nz.numel() == 0:
        return int(row.shape[0]) - 1
    return max(int(nz[0].item()) - 1, 0)


def kl_loss(logits: torch.Tensor, labels: torch.Tensor, target: SoftTarget) -> torch.Tensor:
    """Forward KL / soft cross-entropy `KL(q || p_theta)` over the full vocab.

    This is the distribution-aware objective. `p_theta` = softmax over the full
    vocab at the prompt-end position; `q` lives on the labels' first token ids. The
    per-logit gradient is `p_j - q_j`, which is **non-saturating**: even when the
    model starts with ~0 mass on the support labels (disjoint support), each support
    token gets a strong `-q_j` push, so the edit can actually move. It anchors the
    support as the argmax (reliability) and matches proportions (js_dist), and
    reduces exactly to standard CE when `q` is one-hot."""
    pos = _prompt_end_pos(labels)
    row = logits[0, pos, :].float()                      # [V]
    logp = torch.log_softmax(row, dim=-1)                # full-vocab [V]

    q = target.q.to(logp.device)
    logp_support = logp.index_select(0, target.first_ids.to(logp.device))  # [K]
    # KL(q||p) = sum_k q_k (log q_k - log p_k); off-support q is 0 so it drops out.
    kl = (q * (torch.log(q.clamp_min(_EPS)) - logp_support)).sum()
    return kl.clamp_min(0.0)
