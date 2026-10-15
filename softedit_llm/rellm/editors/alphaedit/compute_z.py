"""Right-vector (z*) optimization for AlphaEdit, wired to our pipeline.

Faithful in spirit to ``AlphaEdit/compute_z.py``: optimize a delta added to the
edited block's output at the fact-lookup token so the model produces the target at
the final layer, then return ``z* = h_clean + delta`` (L2-ball clamped, delta
weight-decayed).

Adaptations (see ``alphaedit.md``):
  * fact-lookup token = last prompt token (``fact_token="last"``); no subject.
  * the objective is read from a real chat-templated ``model(**tokens)`` forward
    (the model's own head) rather than a manual ``ln_f @ lm_head`` projection:
      - hard  : next-token CE over the answer (``outputs.loss``);
      - soft  : the Stage-6 ``compute_edit_loss`` (first-token KL + continuation),
                identical to the other ``*_soft`` editors.
    The delta hook stays active through the continuation forwards (they run inside
    the same trace context and share the prompt length), so the soft term is
    measured on the delta-perturbed model.
  * the unrelated-prompt KL regularizer is dropped; the delta weight-decay and
    L2-ball clamp are kept.
"""
import torch

from .nethook import TraceDict, Trace, set_requires_grad
from ..soft import compute_edit_loss

_EPS = 1e-12


def prompt_end_pos(labels: torch.Tensor) -> int:
    """Row of the sequence that is the last prompt token (predicts the first answer
    token): one before the first non-(-100) label. Robust to answer padding."""
    row = labels[0] if labels.dim() > 1 else labels
    nz = (row != -100).nonzero(as_tuple=False)
    if nz.numel() == 0:
        return int(row.shape[0]) - 1
    return max(int(nz[0].item()) - 1, 0)


def _block_out(output):
    """Decoder block output is a tuple ``(hidden_states, ...)``; unwrap it."""
    return output[0] if isinstance(output, (tuple, list)) else output


def compute_z(model, tokens, hparams, layer, editor, soft: bool) -> torch.Tensor:
    """Optimize and return the target vector ``z*`` (block output, hidden dim) at
    the last prompt token for one edit (batch size 1)."""
    labels = tokens["labels"]
    # Fact-lookup token: last question-content token (set by the pipeline) so the
    # delta is optimized at the same position the key is read from in ``apply``.
    # Falls back to the last prompt token when unavailable.
    pos = getattr(editor, "_lookup_pos", None)
    if pos is None:
        pos = prompt_end_pos(labels)
    layer_name = hparams.layer_module_tmp.format(layer)
    fwd_inputs = {k: v for k, v in tokens.items()
                  if k in ("input_ids", "attention_mask")}

    # Clean block output at the lookup token = init value of the vector to nudge.
    with torch.no_grad():
        with Trace(model, layer_name, retain_output=True) as tr:
            model(**fwd_inputs)
    target_init = _block_out(tr.output)[0, pos, :].detach().clone().float()

    delta = torch.zeros(
        target_init.numel(),
        device=target_init.device,
        dtype=_block_out(tr.output).dtype,
        requires_grad=True,
    )
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    set_requires_grad(False, model)

    def edit_out(output, layer):  # invoke_with_optional_args -> (output=, layer=)
        h = _block_out(output)
        h[0, pos, :] = h[0, pos, :] + delta.to(h.dtype)
        return output

    init_norm = target_init.norm()
    verbose = bool(getattr(hparams, "z_verbose", True))
    n_steps = int(hparams.v_num_grad_steps)
    loss0 = None
    loss_last = 0.0
    for _ in range(n_steps):
        opt.zero_grad()
        with TraceDict(model, [layer_name], edit_output=edit_out,
                       retain_input=False, retain_output=False):
            outputs = model(**tokens)
            loss = compute_edit_loss(editor, outputs, tokens) if soft else outputs.loss
            wd = hparams.v_weight_decay * (delta.norm() / (init_norm ** 2 + _EPS))
            total = loss + wd.to(loss.device)
        total.backward()
        opt.step()
        with torch.no_grad():
            max_norm = hparams.clamp_norm_factor * init_norm
            if delta.norm() > max_norm:
                delta.mul_(max_norm / (delta.norm() + _EPS))
        loss_last = float(loss.detach())
        if loss0 is None:
            loss0 = loss_last

    if verbose:
        tag = "soft" if soft else "ce"
        print(f"[alphaedit.z] z* layer {layer} ({tag}): loss {loss0:.4f} -> "
              f"{loss_last:.4f} over {n_steps} steps, "
              f"|delta|={float(delta.norm()):.3f} (|z0|={float(init_norm):.3f})",
              flush=True)
    return (target_init + delta.detach().float())
