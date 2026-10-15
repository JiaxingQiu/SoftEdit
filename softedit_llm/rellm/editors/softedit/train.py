"""Train the lm_head weight update ΔW (+ ΔW_cont) for one edit (text-only).

Three-term soft objective (see ``generality.py`` / ``locality.py``)::

    L = L_align + lambda_gen * L_gen + lambda_loc * L_loc + lambda_reg ||ΔW||²

  align: first-token KL(p ‖ q(h0)) + continuation NLL at the edit point h0.
  gen:   graph coupling of co-plausible label rows at h0 (generality).
  loc:   locality anchors — KL(one-hot ‖ q) + continuation.

``ΔW`` edits the K first-token label rows of the lm_head; ``ΔW_cont`` edits the
extra continuation-token rows so multi-token answers actually complete and stop.
Both are trained jointly (Adam) and written back by ``apply_delta``.

The lm_head is terminal and linear, so the whole objective is **algebraic**: the
hidden states (edit point + locality anchors, plus teacher-forced continuation
states) are captured once via forward hooks, and the loop only splices edited
rows into precomputed base logits — no in-loop model forwards.

Adapted from the VLM repo: dropped the image argument from every capture (the
text wrapper's ``encode`` takes prompts only); the algebraic optimizer is verbatim.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import generality, locality


# --------------------------------------------------------------------------- #
# lm_head resolution                                                           #
# --------------------------------------------------------------------------- #

def resolve_lm_head(model) -> nn.Linear:
    """Locate the lm_head nn.Linear across common decoder-only LLM architectures."""
    for path in ("lm_head", "language_model.lm_head", "model.lm_head"):
        mod = model
        for attr in path.split("."):
            mod = getattr(mod, attr, None)
            if mod is None:
                break
        if isinstance(mod, nn.Linear):
            return mod
    raise ValueError("model has no nn.Linear lm_head (searched model.lm_head "
                     "and model.language_model.lm_head)")


# --------------------------------------------------------------------------- #
# state capture (forward hooks on the lm_head)                                 #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def capture_head_input(model, head: nn.Linear, inputs) -> torch.Tensor:
    """Last-token input to the lm_head for one forward → [d] float (on device)."""
    captured: List[Optional[torch.Tensor]] = [None]

    def _pre_hook(module, args):
        captured[0] = args[0][:, -1, :].detach().squeeze(0)

    handle = head.register_forward_pre_hook(_pre_hook)
    try:
        model(**inputs)
    finally:
        handle.remove()
    if captured[0] is None:
        raise RuntimeError("no lm_head input captured")
    return captured[0].float().clone().reshape(-1)


@torch.no_grad()
def capture_continuation_states(
    model, head: nn.Linear, inputs, answer_ids: Sequence[torch.Tensor],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Teacher-force ``prompt + answer`` and capture lm_head-input states at the
    positions predicting tokens ``2:L`` (through ``<eos>``).

    Returns ``[(H, tgt), ...]`` per usable branch, where ``H`` is ``[L-1, d]`` cpu
    float and ``tgt`` is ``[L-1]`` cpu long (the target token ids ``a_1..a_{L-1}``).
    Token ``a_0`` is owned by the first-token KL and excluded.
    """
    input_ids = inputs.get("input_ids")
    if input_ids is None or not answer_ids:
        return []
    P = int(input_ids.shape[1])
    device = input_ids.device
    attn = inputs.get("attention_mask")
    base = {k: v for k, v in inputs.items()
            if k not in ("input_ids", "attention_mask")}

    captured: List[Optional[torch.Tensor]] = [None]

    def _pre_hook(module, args):
        captured[0] = args[0].detach()  # [1, seq, d]

    handle = head.register_forward_pre_hook(_pre_hook)
    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    try:
        for a in answer_ids:
            a = a.to(device=device, dtype=input_ids.dtype).view(1, -1)
            m = int(a.shape[1])
            if m < 2:
                continue
            model_inputs = dict(base)
            model_inputs["input_ids"] = torch.cat([input_ids, a], dim=1)
            if attn is not None:
                model_inputs["attention_mask"] = torch.cat(
                    [attn, torch.ones_like(a)], dim=1)
            captured[0] = None
            model(**model_inputs)
            x = captured[0]
            if x is None:
                raise RuntimeError("no continuation states captured")
            H = x[0, P:P + m - 1, :].float().cpu().clone()  # [m-1, d]
            tgt = a[0, 1:m].detach().cpu().clone()           # [m-1]
            out.append((H, tgt))
    finally:
        handle.remove()
    return out


def build_label_answer_ids(tokenizer, labels, first_ids):
    """Per-label EOS-terminated answer ids + the extra continuation rows.

    Returns ``(label2ans, cont_ids)`` where ``label2ans[label]`` is a 1-D cpu
    LongTensor ``(a_0, ..., a_{m-1}, <eos>)`` (m >= 2 only), and ``cont_ids`` is the
    sorted union of every branch's ``a_1:`` tokens minus the first-token ids (or
    None if empty).
    """
    from ..soft_utils.targets import SoftTarget
    from ..soft_utils.continuation import build_cont_label_ids

    first_list = [int(t) for t in first_ids.detach().cpu().tolist()]
    first_set = set(first_list)
    label2ans: Dict[str, torch.Tensor] = {}
    extra: set = set()
    for k, lbl in enumerate(labels):
        if k >= len(first_list):
            break
        st = SoftTarget(labels=[lbl], first_ids=first_ids[k:k + 1].detach().cpu(),
                        q=torch.tensor([1.0]))
        ids_list = build_cont_label_ids(tokenizer, st)
        if not ids_list:
            continue
        a = ids_list[0].detach().cpu().long()
        label2ans[lbl] = a
        for t in a[1:].tolist():
            if t not in first_set:
                extra.add(int(t))
    cont_ids = (torch.tensor(sorted(extra), dtype=torch.long) if extra else None)
    return label2ans, cont_ids


def _branches(target_q, labels, label2ans) -> List[torch.Tensor]:
    """Answer-id tensors for the labels this point puts mass on (q_k > 0)."""
    if not label2ans:
        return []
    tq = target_q.detach().cpu().tolist() if torch.is_tensor(target_q) \
        else list(np.asarray(target_q).reshape(-1))
    out = []
    for k, lbl in enumerate(labels):
        if k < len(tq) and float(tq[k]) > 0 and lbl in label2ans:
            out.append(label2ans[lbl])
    return out


# --------------------------------------------------------------------------- #
# apply / restore                                                             #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def apply_delta(head: nn.Linear, label_ids, dW, cont_ids=None, dW_cont=None) -> Dict:
    """Add ΔW (+ ΔW_cont) into the lm_head rows in place. Returns a restore backup."""
    w = head.weight.data
    label_ids = label_ids.to(w.device)
    backup: Dict = {"label_ids": label_ids, "label_rows": w[label_ids].clone()}
    w[label_ids] = (w[label_ids].float() + dW.to(w.device).float()).to(w.dtype)
    if cont_ids is not None and dW_cont is not None and cont_ids.numel() > 0:
        cont_ids = cont_ids.to(w.device)
        backup["cont_ids"] = cont_ids
        backup["cont_rows"] = w[cont_ids].clone()
        w[cont_ids] = (w[cont_ids].float() + dW_cont.to(w.device).float()).to(w.dtype)
    return backup


@torch.no_grad()
def restore_delta(head: nn.Linear, backup: Optional[Dict]) -> None:
    """Undo an ``apply_delta`` from its backup."""
    if not backup:
        return
    w = head.weight.data
    w[backup["label_ids"]] = backup["label_rows"]
    if "cont_ids" in backup:
        w[backup["cont_ids"]] = backup["cont_rows"]


# --------------------------------------------------------------------------- #
# optimization                                                                 #
# --------------------------------------------------------------------------- #

def _make_sched(opt, lr, n_steps, sched_mode, eta_min_frac,
                lr_patience, lr_factor, lr_threshold):
    if sched_mode == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=lr_factor, patience=lr_patience,
            threshold=lr_threshold, threshold_mode="rel", min_lr=float(lr) * 0.01,
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, int(n_steps)), eta_min=float(lr) * float(eta_min_frac),
    )


class _EarlyStop:
    """Best-state tracking + converge/patience early stop."""

    def __init__(self, params, loss_eps, stop_patience, align_eps=None):
        self.params = list(params)
        self.loss_eps = loss_eps
        self.align_eps = align_eps
        self.stop_patience = int(stop_patience)
        self.best = float("inf")
        self.best_state = [p.detach().clone() for p in self.params]
        self.patience = 0

    def update(self, monitor: float, align: float = 0.0) -> Optional[str]:
        if monitor < self.best:
            self.best = monitor
            self.best_state = [p.detach().clone() for p in self.params]
            self.patience = 0
        else:
            self.patience += 1
        converged = self.loss_eps is not None and monitor <= float(self.loss_eps)
        if converged and self.align_eps is not None:
            converged = align <= float(self.align_eps)
        if converged:
            return f"converged (monitor={monitor:.4g}, align+cont={align:.4g} ≤ {self.align_eps})"
        if self.patience >= self.stop_patience:
            return f"no improve ({self.patience}/{self.stop_patience})"
        return None

    def restore_best(self):
        with torch.no_grad():
            for p, b in zip(self.params, self.best_state):
                p.data.copy_(b)


def _optimize(
    head: nn.Linear,
    label_ids: torch.Tensor,
    hs: Sequence[torch.Tensor],
    targets_q: Sequence[torch.Tensor],
    n_edit: int,
    A: torch.Tensor,
    *,
    cont: Optional[Sequence] = None,
    cont_ids: Optional[torch.Tensor] = None,
    lambda_gen: float = 1.0,
    lambda_loc: float = 1.0,
    lambda_reg: float = 0.0,
    jac_mode: str = "prob",
    lr: float = 1e-3,
    n_steps: int = 5000,
    sched_mode: str = "plateau",
    eta_min_frac: float = 1e-3,
    loss_eps: Optional[float] = 1e-2,
    align_eps: Optional[float] = 1e-4,
    stop_patience: int = 100,
    lr_patience: int = 50,
    lr_factor: float = 0.5,
    lr_threshold: float = 1e-4,
    print_every: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    device = head.weight.device
    W_full = head.weight.data.detach().float().to(device)
    label_ids = label_ids.to(device)
    W_sub = W_full[label_ids]  # [K, d]
    K, d = W_sub.shape
    hs = [h.to(device).float().reshape(-1) for h in hs]
    qs = [q.to(device).float().reshape(-1) for q in targets_q]
    A = A.to(device)
    n_edit = max(int(n_edit), 1)
    print_every = print_every or max(1, n_steps // 10)

    dW = torch.zeros(K, d, device=device, requires_grad=True)
    params = [dW]

    # continuation blocks (algebraic): precompute base logits per branch/point
    has_cont = cont is not None and any(pt for pt in cont)
    dW_cont = None
    cont_row_ids = None
    cont_W0 = None
    cont_packed = None
    if has_cont:
        if cont_ids is not None and int(cont_ids.numel()) > 0:
            cont_row_ids = cont_ids.to(device)
            cont_W0 = W_full[cont_row_ids]
            dW_cont = torch.zeros_like(cont_W0, requires_grad=True)
            params.append(dW_cont)
        cont_packed = []
        for point in cont:
            packed = []
            for (H, tgt) in (point or []):
                Hd = H.to(device).float()
                Z0 = (Hd @ W_full.t()).detach()
                packed.append((Hd, tgt.to(device), Z0))
            cont_packed.append(packed)

    opt = torch.optim.Adam(params, lr=lr)
    sched = _make_sched(opt, lr, n_steps, sched_mode, eta_min_frac,
                        lr_patience, lr_factor, lr_threshold)
    stopper = _EarlyStop(params, loss_eps, stop_patience, align_eps=align_eps)
    if verbose:
        print(f"  [optim] λ_gen={lambda_gen:.3g}  λ_loc={lambda_loc:.3g}  "
              f"loss_eps={loss_eps}  align_eps={align_eps} (gates align+cont)", flush=True)
    t0 = time.perf_counter()

    def _log_q(h, W_ed):
        logits = (W_full @ h).clone()
        logits[label_ids] = W_ed @ h
        if dW_cont is not None:
            logits[cont_row_ids] = (cont_W0 + dW_cont) @ h
        return F.log_softmax(logits, dim=-1)[label_ids]

    def _cont_point(idx, W_ed):
        if cont_packed is None:
            return None
        packed = cont_packed[idx]
        if not packed:
            return None
        branch = []
        for (Hd, tgt, Z0) in packed:
            z = Z0.clone()
            z[:, label_ids] = Hd @ W_ed.t()
            if dW_cont is not None:
                z[:, cont_row_ids] = Hd @ (cont_W0 + dW_cont).t()
            branch.append(F.nll_loss(F.log_softmax(z, dim=-1), tgt, reduction="mean"))
        return torch.stack(branch).mean()

    for step in range(n_steps):
        opt.zero_grad()
        W_ed = W_sub + dW

        loss_align = torch.zeros((), device=device)
        loss_align_cont = torch.zeros((), device=device)
        for j in range(n_edit):
            loss_align = loss_align + F.kl_div(
                _log_q(hs[j], W_ed), qs[j], reduction="sum", log_target=False)
            c = _cont_point(j, W_ed)
            if c is not None:
                loss_align_cont = loss_align_cont + c

        loss_loc = torch.zeros((), device=device)
        loss_loc_cont = torch.zeros((), device=device)
        for i in range(n_edit, len(hs)):
            loss_loc = loss_loc + F.kl_div(
                _log_q(hs[i], W_ed), qs[i], reduction="sum", log_target=False)
            c = _cont_point(i, W_ed)
            if c is not None:
                loss_loc_cont = loss_loc_cont + c

        loss_gen = torch.zeros((), device=device)
        if lambda_gen > 0 and K > 1:
            loss_gen = generality.graph_gen_at_point(W_ed, hs[0], A, mode=jac_mode)

        loss_reg = dW.norm() ** 2
        if dW_cont is not None:
            loss_reg = loss_reg + dW_cont.norm() ** 2

        monitor = (loss_align + loss_align_cont
                   + lambda_gen * (loss_gen
                                   + lambda_loc * (loss_loc + loss_loc_cont)))
        loss = monitor + lambda_reg * loss_reg
        loss.backward()
        opt.step()
        if sched_mode == "plateau":
            sched.step(float(monitor.detach()))
        else:
            sched.step()

        mon = float(monitor.detach())
        reason = stopper.update(mon, align=float((loss_align + loss_align_cont).detach()))
        if verbose and (step % print_every == 0 or step == n_steps - 1 or reason):
            with torch.no_grad():
                q_fit = _log_q(hs[0], W_ed).exp().cpu().numpy()
            print(f"  {step:5d} | {mon:.4f} [align={float(loss_align):.4f}"
                  f"|cont={float(loss_align_cont):.4f} "
                  f"gen={float(loss_gen):.4f} "
                  f"loc={float(loss_loc):.4f}(λ_loc={lambda_loc:.3g})"
                  f"|cont_loc={float(loss_loc_cont):.4f}]"
                  f"(λ_gen={lambda_gen:.3g}) "
                  f"lr={opt.param_groups[0]['lr']:.2e} q={q_fit.round(3)}", flush=True)
        if reason:
            if verbose:
                print(f"  early stop at step {step}: {reason}", flush=True)
            break

    stopper.restore_best()
    dW_out = dW.detach()
    dW_cont_out = None if dW_cont is None else dW_cont.detach()
    if verbose:
        cont_tag = "" if dW_cont_out is None else \
            f"  ΔW_cont[{tuple(dW_cont_out.shape)}] norm={dW_cont_out.norm():.4f}"
        print(f"  Final ΔW[{tuple(dW_out.shape)}] norm={dW_out.norm():.4f}{cont_tag}  "
              f"best={stopper.best:.4g}  time={time.perf_counter() - t0:.1f}s", flush=True)
    return dW_out, dW_cont_out


# --------------------------------------------------------------------------- #
# high-level: capture supervision + optimize                                   #
# --------------------------------------------------------------------------- #

def train_edit(
    *,
    model,
    wrapper,
    tokenizer,
    head: nn.Linear,
    question: str,
    prompt: str,
    labels: List[str],
    first_ids: torch.Tensor,
    p: torch.Tensor,
    continuation: bool = True,
    lambda_gen: float = 1.0,
    lambda_loc: Optional[float] = None,
    lambda_reg: float = 0.0,
    locality_n: int = 1,
    locality_declarative: str = "both",
    jac_mode: str = "prob",
    lr: float = 1e-3,
    n_steps: int = 5000,
    sched_mode: str = "plateau",
    verbose: bool = True,
    **sched,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Capture the edit point + locality anchors, then optimize ΔW / ΔW_cont.

    Args:
        prompt: engineered generation prompt (question + answer instruction) —
            the edit point h0 is captured at its last token to match generation.
        question: bare question (used to compose locality declaratives).

    Returns:
        ``(dW, dW_cont, cont_ids)`` all on cpu (``dW_cont`` / ``cont_ids`` are None
        when continuation is off or all labels are single-token).
    """
    from ...dataset.utils.taskengineer import ANSWER_INSTRUCTION

    device = head.weight.device
    first_ids = first_ids.to(device)
    p_cpu = p.detach().cpu().float()

    # continuation setup
    label2ans: Dict[str, torch.Tensor] = {}
    cont_ids = None
    do_cont = bool(continuation)
    if do_cont:
        label2ans, cont_ids = build_label_answer_ids(tokenizer, labels, first_ids)
        do_cont = bool(label2ans)

    # ---- edit point ----
    edit_inputs = wrapper.encode([prompt], tokenize=False)
    h0 = capture_head_input(model, head, edit_inputs)
    hs: List[torch.Tensor] = [h0]
    targets_q: List[torch.Tensor] = [p_cpu]
    cont: List = []
    if do_cont:
        cont.append(capture_continuation_states(
            model, head, edit_inputs, _branches(p_cpu, labels, label2ans)))

    # ---- locality anchors ----
    anchors = []
    if locality_n > 0 and lambda_loc != 0:
        anchors = locality.build_locality_anchors(
            model, wrapper, tokenizer, question, labels,
            answer_instruction=ANSWER_INSTRUCTION,
            n=locality_n, declarative=locality_declarative,
            verbose=verbose,
        )
    for anc in anchors:
        anc_inputs = wrapper.encode([anc.prompt], tokenize=False)
        hs.append(capture_head_input(model, head, anc_inputs))
        tq = torch.from_numpy(anc.target_q).float()
        targets_q.append(tq)
        if do_cont:
            cont.append(capture_continuation_states(
                model, head, anc_inputs, _branches(tq, labels, label2ans)))

    n_edit = 1
    n_styles = 2 if locality_declarative == "both" else 1
    if lambda_loc is None:
        lambda_loc = 1.0 / max(len(anchors), 1)
    else:
        lambda_loc = lambda_loc / n_styles

    A = generality.build_affinity(p_cpu)
    dW, dW_cont = _optimize(
        head, first_ids, hs, targets_q, n_edit, A,
        cont=(cont if do_cont else None), cont_ids=cont_ids,
        lambda_gen=lambda_gen, lambda_loc=lambda_loc, lambda_reg=lambda_reg,
        jac_mode=jac_mode, lr=lr, n_steps=n_steps, sched_mode=sched_mode,
        verbose=verbose, **sched,
    )
    dW = dW.detach().cpu()
    dW_cont = None if dW_cont is None else dW_cont.detach().cpu()
    if dW_cont is None:
        cont_ids = None
    cont_ids = None if cont_ids is None else cont_ids.detach().cpu()
    return dW, dW_cont, cont_ids
