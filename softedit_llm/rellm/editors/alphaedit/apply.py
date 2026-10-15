"""Apply one AlphaEdit round (projected MEMIT solve + null-space cache update).

Faithful port of ``AlphaEdit/AlphaEdit_main.apply_AlphaEdit_to_model``. A round
edits a *chunk* of buffered edits at once (``hparams.num_edits``), so keys are
rank-N and ``cache_c`` accumulates rank-N per round — the regime upstream's
``L2`` / clamp were tuned for. Per edited ``mlp.down_proj`` layer, with stacked
keys ``K`` ``[d_in, N]``, residuals ``R = Z* − curZ`` ``[hidden, N]``, sequential
memory ``cache_c``, and null-space projector ``P``:

    ΔW = solve( P (K Kᵀ + cache_c) + L2·I ,  P K Rᵀ )
    W  += ΔW
    cache_c += K Kᵀ            # prior edits join the preserved set

Keys are **context-averaged** (upstream ``compute_ks`` averages over ~11 context
templates): each edit's key is the mean down_proj input, over ``hparams.n_contexts``
neutral wrappers of the question, at the last question-content token. Keys /
current outputs are recomputed per layer on the progressively edited model (as
upstream) so later layers see earlier layers' updates.
"""
import torch

from .compute_z import compute_z, prompt_end_pos, _block_out
from .nethook import TraceDict, get_parameter

# Neutral context wrappers for key averaging (identity first). Each prefixes the
# question so the last question-content token still varies per edit but its key
# is averaged over several surrounding contexts (de-correlation, upstream #3).
_CONTEXT_TEMPLATES = [
    "{}",
    "The following is a question. {}",
    "Here is a question. {}",
    "Consider the following question. {}",
    "Please answer the question. {}",
    "I have a question for you. {}",
    "Question: {}",
    "Answer the following. {}",
]


def _context_prompts(prompt, n):
    tpls = _CONTEXT_TEMPLATES[: max(1, int(n))]
    return [t.format(prompt) for t in tpls]


def _match_shape(matrix: torch.Tensor, shape) -> torch.Tensor:
    if matrix.shape == shape:
        return matrix
    if matrix.T.shape == shape:
        return matrix.T
    raise ValueError(
        f"AlphaEdit update {tuple(matrix.shape)} matches neither {tuple(shape)} nor its transpose."
    )


@torch.no_grad()
def _prep_ctx(editor, edits, hparams):
    """Precompute per-edit, per-context ``(input_ids, attn, pos)`` — these are
    layer-independent (tokenization + lookup position don't change as weights are
    edited), so we build them once per round and re-forward per layer."""
    wrapper = getattr(editor, "wrapper", None)
    n = int(getattr(hparams, "n_contexts", 1))
    can_ctx = wrapper is not None and hasattr(wrapper, "encode") and hasattr(wrapper, "question_lookup_pos")
    for e in edits:
        cache = []
        q, p = e.get("question"), e.get("prompt")
        if can_ctx and q and p:
            for wp in _context_prompts(p, n):
                enc = wrapper.encode([wp])
                ids = enc["input_ids"]
                attn = enc.get("attention_mask")
                pos = wrapper.question_lookup_pos(wp, q)
                if pos is None:
                    pos = int(ids.shape[1]) - 1
                cache.append((ids, attn, int(pos)))
        else:
            # Fallback: identity context from the buffered tokens.
            t = e["tokens"]
            pos = e.get("lookup_pos")
            if pos is None:
                pos = prompt_end_pos(t["labels"])
            cache.append((t["input_ids"], t.get("attention_mask"), int(pos)))
        e["_ctx"] = cache


@torch.no_grad()
def _layer_keys(model, edits, rewrite_name, zlayer_name, want_curz):
    """Context-averaged key per edit at ``rewrite_name`` (down_proj fan-in), and —
    when ``want_curz`` — the current z-layer block output per edit from the first
    (identity) context. Returns ``(K [d_in, N], curZ [hidden, N] | None)``."""
    Ks, Zs = [], []
    for e in edits:
        kctx = []
        cz = None
        for ci, (ids, attn, pos) in enumerate(e["_ctx"]):
            need_z = want_curz and ci == 0
            names = [rewrite_name] + ([zlayer_name] if need_z else [])
            with TraceDict(model, names, retain_input=True, retain_output=need_z) as tr:
                model(input_ids=ids, attention_mask=attn)
            kctx.append(tr[rewrite_name].input[0, pos, :].float())
            if need_z:
                cz = _block_out(tr[zlayer_name].output)[0, pos, :].float()
        Ks.append(torch.stack(kctx, 0).mean(0))
        if want_curz:
            Zs.append(cz)
    K = torch.stack(Ks, dim=1)                       # [d_in, N]
    Z = torch.stack(Zs, dim=1) if want_curz else None  # [hidden, N]
    return K, Z


def apply_alphaedit(model, edits, hparams, P, cache_c, editor):
    """Apply one AlphaEdit round over a list of buffered ``edits`` (chunk of up to
    ``hparams.num_edits``). Mutate ``model`` in place; update ``cache_c`` (list of
    CPU [d_in, d_in]). Each edit is a dict with ``tokens`` (for z*), ``lookup_pos``,
    ``soft`` / ``soft_target`` / ``soft_cont`` (soft variant), and ``question`` /
    ``prompt`` (for context-averaged keys)."""
    if not edits:
        return model
    layers = list(hparams.layers)
    z_layer = layers[-1]
    zlayer_name = hparams.layer_module_tmp.format(z_layer)
    rewrite_names = [hparams.rewrite_module_tmp.format(l) for l in layers]

    _prep_ctx(editor, edits, hparams)

    # z* (right vectors) for every edit, optimized on the pre-round model. Setting
    # the per-edit lookup/soft context lets the shared ``compute_z`` read them.
    z_list = []
    for e in edits:
        editor._lookup_pos = e.get("lookup_pos")
        editor._soft_target = e.get("soft_target")
        editor._soft_cont = e.get("soft_cont")
        z = compute_z(model, e["tokens"], hparams, z_layer, editor, bool(e.get("soft")))
        z_list.append(z.detach())
    Z = torch.stack(z_list, dim=1).to(device="cuda", dtype=torch.float32)  # [hidden, N]
    torch.cuda.empty_cache()

    with torch.no_grad():
        for i, layer in enumerate(layers):
            # Recompute (context-averaged) keys + current z on the progressively
            # edited model.
            K, curZ = _layer_keys(model, edits, rewrite_names[i], zlayer_name, want_curz=True)
            K1 = K.to(device="cuda", dtype=torch.float32)      # [d_in, N]
            curZ = curZ.to(device="cuda", dtype=torch.float32)  # [hidden, N]
            R = (Z - curZ) / (len(layers) - i)                  # [hidden, N]

            # A = P (K Kᵀ + cache_c) + L2·I, formed in place to avoid extra d×d
            # allocations (Qwen d_in=18944 -> 1.34 GiB per d×d float32).
            Cc = cache_c[i].to("cuda", torch.float32)
            Cc.addmm_(K1, K1.T)                                 # Cc := cache_c + K Kᵀ
            Pi = P[i].to("cuda", torch.float32)
            A = Pi @ Cc
            del Cc
            A.diagonal().add_(float(hparams.L2))
            B = (Pi @ K1) @ R.T                                 # [d_in, hidden]
            del Pi
            upd = torch.linalg.solve(A, B)                      # [d_in, hidden]
            del A, B

            W = get_parameter(model, rewrite_names[i] + ".weight")
            upd_w = _match_shape(upd, W.shape).to(W.dtype)
            W += upd_w
            del upd, upd_w, K1, curZ, R
            torch.cuda.empty_cache()

        # Fold this round's keys into the preserved-knowledge memory (rank-N),
        # recomputed on the fully edited model (as upstream).
        for i in range(len(layers)):
            K, _ = _layer_keys(model, edits, rewrite_names[i], zlayer_name, want_curz=False)
            k2 = K.float().cpu()                                # [d_in, N]
            cache_c[i].addmm_(k2, k2.T)

    for e in edits:
        e.pop("_ctx", None)
    return model
