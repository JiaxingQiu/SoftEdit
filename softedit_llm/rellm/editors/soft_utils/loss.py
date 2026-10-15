"""Per-iteration loss dispatch shared by every ``*_soft`` editor.

`compute_edit_loss(editor, outputs, tokens)` is the single entry point editors
call where they previously read `outputs.loss`. It combines the two soft terms
(`kl.kl_loss` + `continuation.continuation_nll`) for distribution modes, and falls
back to plain next-token CE for the target/default path.
"""
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from .continuation import continuation_nll
from .kl import kl_loss

SOFT_LOSS_MODES = ("target", "loss", "rank_loss")


def compute_edit_loss(editor, outputs, tokens) -> torch.Tensor:
    """Single source of truth for a soft editor's per-iteration loss.

    Distribution-aware modes (`soft in {"loss", "rank_loss"}` with a `SoftTarget`
    attached) return the two-term soft objective:

        KL(q || p_theta^(1))  +  (1/M) sum_k continuation_nll(y_k)

    i.e. the first-token KL (proportions) plus a uniform continuation term over
    the active multi-token branches (completability). The per-branch answer token
    ids are stashed on `editor._soft_cont` and teacher-forced against `tokens`
    (the main edit batch); when absent or all single-token, the loss reduces to
    the first-token KL alone. The "target"/default path returns the model's own
    next-token CE (unchanged)."""
    mode = getattr(editor, "soft_mode", "target")
    target = getattr(editor, "_soft_target", None)
    if mode in ("loss", "rank_loss") and target is not None:
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = kl_loss(logits, tokens["labels"], target)
        cont_label_ids = getattr(editor, "_soft_cont", None)
        if cont_label_ids:
            # Hook-based editors (grace) supply `cont_guard` to protect their
            # codebook: the extra continuation forwards must not re-enter the
            # add/split/merge branch, so retrieval stays exactly as the main
            # forward established it. Weight editors (ft/mend) supply none.
            guard = getattr(editor, "cont_guard", None)
            with (guard() if guard is not None else nullcontext()):
                cont = continuation_nll(editor.model, tokens, cont_label_ids)
            if cont is not None:
                loss = loss + cont
        return loss

    loss = getattr(outputs, "loss", None)
    if loss is not None:
        return loss
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    if "labels" in tokens:
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), tokens["labels"].view(-1), ignore_index=-100)
    raise RuntimeError("compute_edit_loss: model returned no loss and no labels to compute one")
