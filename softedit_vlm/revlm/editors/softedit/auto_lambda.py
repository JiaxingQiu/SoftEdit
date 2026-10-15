"""Auto-tune λ_loc from the initial gen / loc loss ratio.

Self-contained plug-in utility.  Runs ONE forward pass per edit at ΔW = 0
(step 0 of the optimizer, before any training), reads off L_gen and L_loc,
then sets::

    λ_loc = round_half_decade( n_styles × mean(L_gen) / mean(L_loc) )

where *n_styles* is 2 for ``locality_declarative="both"`` and 1 otherwise.

Called as a pre-step before the main edit loop (similar to the auto-reject-
threshold calibration at 100 edits).

Cost: ~100 forward passes, 0 optimiser steps → typically < 1 min per backbone.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import generality, locality, train


# --------------------------------------------------------------------------- #
# half-decade rounding                                                         #
# --------------------------------------------------------------------------- #

def round_half_decade(x: float) -> float:
    """Snap *x* to the nearest value on the …, 1e-3, 5e-3, 1e-2, 5e-2, … grid.

    Decision boundaries are the geometric means of adjacent grid points:
        boundary(1, 5)  = sqrt(5)   → log₁₀ ≈ 0.349
        boundary(5, 10) = sqrt(50)  → log₁₀ ≈ 0.849
    """
    if x <= 0:
        return 1e-6
    log = math.log10(x)
    decade = math.floor(log)
    frac = log - decade
    if frac < math.log10(5) / 2:                     # < 0.349
        return 10.0 ** decade
    if frac < (1 + math.log10(5)) / 2:               # < 0.849
        return 5.0 * 10.0 ** decade
    return 10.0 ** (decade + 1)


# --------------------------------------------------------------------------- #
# step-0 loss computation (no optimisation, ΔW = 0)                            #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _step0_losses(
    head: nn.Linear,
    label_ids: torch.Tensor,
    hs: Sequence[torch.Tensor],
    targets_q: Sequence[torch.Tensor],
    n_edit: int,
    A: torch.Tensor,
    *,
    jac_mode: str = "prob",
) -> tuple:
    """Return ``(align_loss, gen_loss, loc_loss)`` at ΔW = 0.

    ``align_loss`` is the first-token KL on the edit sample(s).
    Continuation is not included (requires teacher-forcing states not
    available in this lightweight pilot).
    """
    device = head.weight.device
    W_full = head.weight.data.detach().float().to(device)
    label_ids_d = label_ids.to(device)
    W_sub = W_full[label_ids_d]                       # [K, d]  (= W_ed at ΔW=0)
    K = W_sub.shape[0]
    hs_d = [h.to(device).float().reshape(-1) for h in hs]
    qs_d = [q.to(device).float().reshape(-1) for q in targets_q]
    A_d = A.to(device)

    def _log_q(h):
        logits = (W_full @ h).clone()
        logits[label_ids_d] = W_sub @ h
        return F.log_softmax(logits, dim=-1)[label_ids_d]

    # alignment (edit point) ──────────────────────────────────────────────────
    align = 0.0
    for j in range(n_edit):
        align += float(F.kl_div(_log_q(hs_d[j]), qs_d[j],
                                reduction="sum", log_target=False))

    # generality ──────────────────────────────────────────────────────────────
    gen = 0.0
    if K > 1:
        gen = float(generality.graph_gen_at_point(W_sub, hs_d[0], A_d, mode=jac_mode))

    # locality ────────────────────────────────────────────────────────────────
    loc = 0.0
    for i in range(n_edit, len(hs_d)):
        loc += float(F.kl_div(_log_q(hs_d[i]), qs_d[i],
                               reduction="sum", log_target=False))
    return align, gen, loc


# --------------------------------------------------------------------------- #
# pipeline pre-step: resolve cfg.lambda_loc == "auto" -> float                 #
# --------------------------------------------------------------------------- #

def resolve_lambda_loc(config, model, edit_data, n_edits: int = 100,
                       force_pilot: bool = False):
    """Pre-step entry point (called from ``run_edit`` before any editing).

    If ``config.editor.lambda_loc == "auto"``, estimate λ_loc over the full
    edit set and return a concrete float; otherwise return the value unchanged.

    When ``force_pilot=True`` the 100-edit pilot runs even if λ_loc is fixed,
    so that ``resolve_lambda_gen`` can read the stashed pilot stats. The fixed
    λ_loc value is returned unchanged.
    """
    cfg = getattr(config, "editor", config)
    raw = getattr(cfg, "lambda_loc", "auto")
    is_auto = isinstance(raw, str) and raw.strip().lower() == "auto"
    if not is_auto and not force_pilot:
        return raw

    n_edits = int(getattr(cfg, "lambda_loc_n_edits", n_edits))
    wrapper = model if hasattr(model, "model") else None
    inner = model.model if hasattr(model, "model") else model
    tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None
    head = train.resolve_lm_head(inner)

    est = estimate_lambda_loc(
        model=inner, wrapper=wrapper, tokenizer=tokenizer, head=head,
        edit_data=edit_data, n_edits=n_edits,
        soft_mode=str(getattr(cfg, "soft", "loss")),
        rank_temp=float(getattr(cfg, "rank_temp", 1.0)),
        major_vote=bool(getattr(cfg, "major_vote", False)),
        locality_n=int(getattr(cfg, "locality_n", 1)),
        locality_caption=str(getattr(cfg, "locality_caption", "off")),
        locality_declarative=str(getattr(cfg, "locality_declarative", "both")),
        locality_blend=float(getattr(cfg, "locality_blend", 0.0)),
        jac_mode=str(getattr(cfg, "jac_mode", "prob")),
    )
    return est if is_auto else raw


def resolve_lambda_gen(config, lambda_loc: float) -> float:
    """Resolve ``lambda_gen="auto"`` using the pilot stats stashed by
    ``estimate_lambda_loc``.  Must be called AFTER ``resolve_lambda_loc``.

    If ``lambda_gen`` is not ``"auto"``, returns it unchanged.
    """
    cfg = getattr(config, "editor", config)
    raw = getattr(cfg, "lambda_gen", "auto")
    if not (isinstance(raw, str) and raw.strip().lower() == "auto"):
        return float(raw)

    pilot = estimate_lambda_loc._last_pilot
    if pilot is None:
        print("[auto_λ_gen] no pilot data (λ_loc was not auto?); "
              "defaulting λ_gen=1.0", flush=True)
        return 1.0

    return estimate_lambda_gen(
        mean_align=pilot["mean_align"],
        mean_gen=pilot["mean_gen"],
        mean_loc=pilot["mean_loc"],
        lambda_loc=lambda_loc,
        n_styles=pilot["n_styles"],
    )


# --------------------------------------------------------------------------- #
# main entry point                                                             #
# --------------------------------------------------------------------------- #

def estimate_lambda_loc(
    *,
    model,
    wrapper,
    tokenizer,
    head: nn.Linear,
    edit_data: list,
    n_edits: int = 100,
    soft_mode: str = "loss",
    rank_temp: float = 1.0,
    major_vote: bool = False,
    locality_n: int = 1,
    locality_caption: str = "off",
    locality_declarative: str = "both",
    locality_blend: float = 0.0,
    jac_mode: str = "prob",
    verbose: bool = True,
) -> float:
    """Estimate λ_loc from the initial gen / loc loss ratio.

    Samples up to *n_edits* entries from *edit_data*, runs one forward pass
    each at ΔW = 0, and returns::

        round_half_decade( n_styles × mean(L_gen) / mean(L_loc) )

    The returned value is **config-level** (the same unit as
    ``cfg.lambda_loc``); ``train_edit`` will divide by *n_styles* internally
    so that the *effective* λ_loc in the optimiser equals the raw ratio.
    """
    from revlm.dataset.utils.taskengineer import ANSWER_INSTRUCTION
    from revlm.editors.soft_utils import build_soft_target

    n_styles = 2 if locality_declarative == "both" else 1
    use = min(n_edits, len(edit_data))
    if use == 0:
        if verbose:
            print("[auto_λ_loc] no edit data; defaulting λ_loc=1.0", flush=True)
        return 1.0

    align_vals: List[float] = []
    gen_vals: List[float] = []
    loc_vals: List[float] = []

    for idx in range(use):
        ex = edit_data[idx]
        image = ex.get("image")
        question = ex.get("question", "")
        gold = ex.get("gold") or {}
        prompt = ex.get("prompt_orig") or ex.get("prompt") or question
        if image is None or not question:
            continue

        # ---- build soft target ----
        st = build_soft_target(gold, tokenizer, mode=soft_mode,
                               rank_temp=rank_temp, major_vote=major_vote)
        labels = st.labels
        first_ids = st.first_ids.to(head.weight.device)
        K = len(labels)
        if K == 0:
            continue

        # ---- capture edit-point hidden state ----
        edit_inputs = wrapper.encode([image], [prompt], tokenize=False)
        h0 = train.capture_head_input(model, head, edit_inputs)
        hs = [h0]
        targets_q = [st.q.detach().cpu().float()]

        # ---- locality anchors (skip when locality is off) ----
        anchors = []
        if locality_n > 0:
            anchors = locality.build_locality_anchors(
                model, wrapper, tokenizer, image, question, labels,
                answer_instruction=ANSWER_INSTRUCTION,
                n=locality_n, blend=locality_blend,
                caption=locality_caption, declarative=locality_declarative,
                verbose=False,
            )
        for anc in anchors:
            anc_inputs = wrapper.encode([anc.image], [anc.prompt], tokenize=False)
            hs.append(train.capture_head_input(model, head, anc_inputs))
            targets_q.append(torch.from_numpy(anc.target_q).float())

        # ---- step-0 losses ----
        A = generality.build_affinity(st.q.detach().cpu().float())
        align_0, gen_0, loc_0 = _step0_losses(
            head, first_ids, hs, targets_q, 1, A, jac_mode=jac_mode)
        align_vals.append(align_0)
        gen_vals.append(gen_0)
        loc_vals.append(loc_0)

        if verbose and (idx + 1) % 25 == 0:
            print(f"[auto_λ_loc] {idx + 1}/{use} edits sampled", flush=True)

    # ---- aggregate ----
    if not gen_vals:
        if verbose:
            print("[auto_λ_loc] no valid edits; defaulting λ_loc=1.0", flush=True)
        return 1.0

    mean_align = float(np.mean(align_vals))
    mean_gen = float(np.mean(gen_vals))
    mean_loc = float(np.mean(loc_vals))

    loc_off = mean_loc < 1e-12
    if loc_off:
        config_val = 1.0
    else:
        ratio = mean_gen / mean_loc
        config_val = round_half_decade(ratio * n_styles)

    # stash pilot stats for lambda_gen estimation (avoids a second 100-edit
    # pass). Always stash — even when loc≈0 — so λ_gen="auto" works
    # independently of λ_loc (e.g. locality-off ablations).
    estimate_lambda_loc._last_pilot = {
        "mean_align": mean_align,
        "mean_gen": mean_gen,
        "mean_loc": mean_loc,
        "lambda_loc": config_val,
        "n_styles": n_styles,
    }

    if verbose:
        if loc_off:
            print(f"[auto_λ_loc] mean(loc)={mean_loc:.4e} ≈ 0 (locality off); "
                  f"defaulting λ_loc={config_val}", flush=True)
        else:
            print(
                f"[auto_λ_loc] {len(gen_vals)} edits: "
                f"mean(align)={mean_align:.4e}  "
                f"mean(gen)={mean_gen:.4e}  mean(loc)={mean_loc:.4e}  "
                f"ratio={mean_gen / mean_loc:.4e}  n_styles={n_styles}  "
                f"→ λ_loc={config_val:.0e}",
                flush=True,
            )
    return config_val

# Stash for reuse by estimate_lambda_gen (avoids a second 100-edit pass).
estimate_lambda_loc._last_pilot = None


# --------------------------------------------------------------------------- #
# λ_gen auto-tuning: align-dominance ratio                                     #
# --------------------------------------------------------------------------- #

_LAMBDA_GEN_TARGET_RATIO = 100   # l_align ≈ 100 × λ_gen·(gen + λ_loc·loc)


def estimate_lambda_gen(
    *,
    mean_align: float,
    mean_gen: float,
    mean_loc: float,
    lambda_loc: float,
    n_styles: int = 1,
    target_ratio: float = _LAMBDA_GEN_TARGET_RATIO,
    verbose: bool = True,
) -> float:
    """Compute λ_gen so that ``l_align / (λ_gen·(l_gen + λ_loc·l_loc)) ≈ target_ratio``.

    Called AFTER λ_loc is resolved. Uses the same 100-edit pilot statistics.

    One-directional: scaling only kicks in when align already dominates the
    regulariser beyond ``target_ratio`` (i.e. ``l_align / l_reg > target_ratio``).
    When the ratio is already ``≤ target_ratio`` the regulariser is not
    over-shadowed, so λ_gen is left at ``1.0`` rather than shrinking gen+loc.
    Consequently the returned value is always in ``[1.0, 10.0]``.

    Note: ``lambda_loc`` is the *config-level* value; ``train_edit`` divides
    by *n_styles* internally, so the effective per-anchor weight in the
    optimiser is ``lambda_loc / n_styles``.  We use the same convention here.
    """
    effective_loc = lambda_loc / max(n_styles, 1)
    l_reg = mean_gen + effective_loc * mean_loc
    if l_reg < 1e-12:
        if verbose:
            print(f"[auto_λ_gen] l_reg={l_reg:.4e} ≈ 0; defaulting λ_gen=1.0",
                  flush=True)
        return 1.0

    raw_ratio = mean_align / l_reg
    # Only promote the gen+loc term when align already dominates beyond the
    # target (raw_ratio > target_ratio). Otherwise keep λ_gen=1.0 — never
    # shrink gen+loc below its natural weight.
    if raw_ratio <= target_ratio:
        if verbose:
            print(
                f"[auto_λ_gen] mean(align)={mean_align:.4e}  "
                f"l_reg(gen+λ_loc·loc)={l_reg:.4e}  "
                f"raw_ratio={raw_ratio:.1f}  "
                f"target={target_ratio}  "
                f"(≤ target → no scaling) → λ_gen=1.0",
                flush=True,
            )
        return 1.0

    raw = raw_ratio / target_ratio          # > 1 here (guard above)
    val = min(round_half_decade(raw), 10.0)  # cap — never amplify gen+loc beyond 10×

    if verbose:
        print(
            f"[auto_λ_gen] mean(align)={mean_align:.4e}  "
            f"l_reg(gen+λ_loc·loc)={l_reg:.4e}  "
            f"raw_ratio={raw_ratio:.1f}  "
            f"target={target_ratio}  "
            f"→ λ_gen={val:.2g}",
            flush=True,
        )
    return val
