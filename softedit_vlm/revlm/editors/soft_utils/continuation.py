"""Continuation (completability) term of the soft objective.

The first-token KL only supervises token 1, so multi-token answers are not
guaranteed to complete (the js_dist vs js_dist_gate gap). For every branch with
`q_k > 0` and length `L_k > 1` we add the length-normalized NLL of tokens
`2:L_k` (teacher-forced on the gold prefix), averaged **uniformly** over the M
active branches -- mirroring the binary continuation gate.

Only the answer token ids are stored per edit (``build_cont_label_ids``): the
teacher-forced ``prompt + y_k`` forward is reconstructed on demand from the main
edit tokens (``_build_cont_inputs``) so the shared prompt/image is reused rather
than duplicated -- important for memory at large edit counts and for retrain
editors that persist edit history.
"""
from typing import List, Optional

import torch
import torch.nn.functional as F

from .targets import SoftTarget


def build_cont_label_ids(tokenizer, target: SoftTarget) -> Optional[List[torch.Tensor]]:
    """Per-branch ``answer + <eos>`` token ids for the continuation term (built once
    per edit).

    One 1-D CPU LongTensor per branch, encoded (add_special_tokens=False) with a
    trailing EOS appended. The EOS makes the continuation NLL supervise tokens
    ``2:L_k`` up to and including the stop token, so distribution-soft editors
    (which never see the full-answer CE) still learn to STOP after the answer --
    the soft analog of the EOS supervision `_tokenize_targets` adds to the hard
    path. It also means a single-token answer becomes a 1-step continuation
    (``token -> <eos>``) instead of being skipped, so stopping is trained for those
    too. Token 1 still matches the KL's first-token id (EOS is appended at the
    end). Only the ids are stored (tiny); the continuation forward is reconstructed
    from the main edit tokens at loss time, reusing the shared prompt/image. Returns
    None when no branch yields a continuation."""
    eos_id = getattr(tokenizer, "eos_token_id", None)
    out: List[torch.Tensor] = []
    for lbl in target.labels:
        ids = tokenizer.encode(lbl, add_special_tokens=False)
        if eos_id is not None and (not ids or ids[-1] != eos_id):
            ids = ids + [eos_id]
        if len(ids) < 2:
            continue
        out.append(torch.tensor(ids, dtype=torch.long))
    return out or None


def _build_cont_inputs(main_tokens, label_ids):
    """Reconstruct a teacher-forced ``prompt + y_k`` forward from the main edit
    tokens plus this branch's answer token ids.

    The continuation shares the exact prompt and image with the edit, so instead
    of storing a separate copy (with duplicated ``pixel_values``) we reuse the
    main tokens: slice off their answer, keep every image tensor as-is, and append
    this branch's ``label_ids``. Returns ``(model_inputs, cont_labels)`` where
    ``cont_labels`` masks the prompt **and the first answer token** to -100 (that
    token is owned by the first-token KL), leaving tokens ``2:L_k`` supervised.
    Returns None for single-token answers (no continuation)."""
    labels = main_tokens.get("labels")
    input_ids = main_tokens.get("input_ids")
    if labels is None or input_ids is None:
        return None
    nz = (labels[0] != -100).nonzero(as_tuple=True)[0]
    if nz.numel() == 0:
        return None
    prompt_len = int(nz[0].item())  # first answer position = prompt length

    device = input_ids.device
    label_ids = label_ids.to(device=device, dtype=input_ids.dtype).view(1, -1)
    if label_ids.shape[1] < 2:
        return None

    prompt_ids = input_ids[:, :prompt_len]
    full_ids = torch.cat([prompt_ids, label_ids], dim=1)

    attn = main_tokens.get("attention_mask")
    model_inputs = {k: v for k, v in main_tokens.items()
                    if k not in ("input_ids", "attention_mask", "labels",
                                 "_soft_target", "_soft_cont")}
    model_inputs["input_ids"] = full_ids
    if attn is not None:
        model_inputs["attention_mask"] = torch.cat(
            [attn[:, :prompt_len], torch.ones_like(label_ids)], dim=1)

    cont_labels = torch.full_like(full_ids, -100)
    cont_labels[:, prompt_len + 1:] = label_ids[:, 1:]  # keep tokens 2:L only
    return model_inputs, cont_labels


def continuation_nll(model, main_tokens, cont_label_ids) -> Optional[torch.Tensor]:
    """Uniform-average continuation loss over the active multi-token branches.

    `model` is any callable that maps model inputs to logits (an nn.Module, or a
    wrapper that guards a stateful adaptor for hook-based editors). `cont_label_ids`
    is a list of 1-D answer token-id tensors, one per branch with `q_k > 0` and
    length `L_k > 1`. Each is teacher-forced as `prompt + y_k` (reconstructed from
    `main_tokens` so the image is reused, not copied); we drop the first answer
    token (owned by the first-token KL) and take the length-normalized NLL of
    tokens `2:L_k`. The M per-branch losses are averaged **uniformly** (weight
    1/M). Returns None when no branch has a continuation, so the caller falls back
    to the first-token KL alone."""
    losses = []
    for label_ids in cont_label_ids:
        built = _build_cont_inputs(main_tokens, label_ids)
        if built is None:
            continue
        model_inputs, cont_labels = built
        out = model(**model_inputs)
        logits = out.logits if hasattr(out, "logits") else out
        shift_logits = logits[0, :-1, :].float()
        shift_labels = cont_labels[0, 1:]
        if int((shift_labels != -100).sum()) == 0:
            continue
        # reduction="mean" over the L_k-1 kept tokens = per-branch length norm.
        losses.append(F.cross_entropy(shift_logits, shift_labels, ignore_index=-100))
    if not losses:
        return None
    return torch.stack(losses).mean()
