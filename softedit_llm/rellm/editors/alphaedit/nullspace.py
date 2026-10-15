"""Null-space projector ``P`` for AlphaEdit.

The one AlphaEdit-specific object: for each edited layer, project parameter
perturbations onto the null space of the preserved-knowledge keys. Faithful to
``experiments/evaluate.py::get_project`` — SVD the wikipedia mom2 covariance and
keep the subspace spanned by its small singular values.
"""
import os
from dataclasses import replace

import torch

from .cov_stats import STATS_DIR, get_cov


def _load_cached_P(path):
    """Load a cached ``P`` tensor onto CPU.

    ``mmap=True`` avoids a second RAM copy of the (multi-GB) file during unpickle,
    which is what kills Jupyter kernels on Qwen (d_in=18944 → ~6.7 GiB per file).
    Sample count does not change this size — only ``d_in`` and the layer count do.
    """
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu")


def _svd_cov(model, tokenizer, layer, hparams):
    """Load/compute the layer's fan-in covariance and return its SVD ``(U, S)`` (CPU).

    The SVD is threshold-independent, so it can be reused to build ``P`` at any
    ``nullspace_threshold`` (see ``build_projection_grid``).
    """
    cov = get_cov(
        model,
        tokenizer,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples,
        hparams.mom2_dtype,
    ).cpu()
    # SVD on GPU when available (the fan-in dim can be ~19k -> CPU SVD is very slow).
    svd_in = cov.cuda() if torch.cuda.is_available() else cov
    U, S, _ = torch.linalg.svd(svd_in, full_matrices=False)
    U, S = U.cpu(), S.cpu()
    del svd_in
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return U, S


def _project_from_svd(U, S, threshold):
    """Null-space projector ``U_null U_nullᵀ`` for one layer at one threshold.

    Returns ``(P, ndim)`` where ``ndim`` is the null-space dimensionality (number of
    singular values below ``threshold``). Larger ``ndim`` -> the edit has more room
    (freer, higher reliability); smaller -> more constrained (better preservation).
    """
    small = (S < threshold).nonzero(as_tuple=True)[0]
    Un = U[:, small]
    return Un @ Un.T, len(small)


def get_project(model, tokenizer, layer, hparams):
    """Projector ``P = U_null U_nullᵀ`` for one edited ``down_proj`` layer.

    ``U_null`` = left singular vectors of the fan-in covariance whose singular
    value is below ``hparams.nullspace_threshold`` (the approximate null space of
    the preserved keys). Perturbations left-multiplied by ``P`` leave the preserved
    knowledge's output ~unchanged.
    """
    U, S = _svd_cov(model, tokenizer, layer, hparams)
    P, ndim = _project_from_svd(U, S, hparams.nullspace_threshold)
    print(f"[alphaedit.P] layer {layer}: null-space dim {ndim}/{S.numel()} "
          f"(threshold={hparams.nullspace_threshold})", flush=True)
    return P


def _cache_path(model_name, hparams, stats_dir=STATS_DIR):
    tag = "-".join(str(l) for l in hparams.layers)
    safe = model_name.replace("/", "_")
    return os.path.join(
        stats_dir, safe,
        f"P_nullspace_layers{tag}_n{hparams.mom2_n_samples}_thr{hparams.nullspace_threshold}.pt",
    )


def build_projections(model, tokenizer, hparams, model_name=None, cache=True,
                      force_recompute=False):
    """Stacked projector ``P`` of shape ``[len(layers), d, d]`` (d = down_proj fan-in).

    Cached to ``data/stats/<model>/P_nullspace_*.pt`` so edit jobs load instead of
    re-running the SVDs. Set ``force_recompute=True`` to rebuild.
    """
    model_name = model_name or model.config._name_or_path
    path = _cache_path(model_name, hparams)
    if cache and not force_recompute and os.path.exists(path):
        print(f"[alphaedit.P] loading cached projections from {path}", flush=True)
        return _load_cached_P(path)

    mats = [get_project(model, tokenizer, layer, hparams) for layer in hparams.layers]
    P = torch.stack(mats, dim=0)  # [L, d, d] on CPU (float32)
    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(P, path)
        print(f"[alphaedit.P] cached projections to {path}", flush=True)
    return P


def build_projection_grid(model, tokenizer, hparams, thresholds, model_name=None,
                          cache=True, force_recompute=False):
    """Build+cache ``P`` for several ``nullspace_threshold`` values in one pass.

    The per-layer SVD is threshold-independent, so it is computed **once** and then
    sliced for every threshold. The whole grid therefore costs one SVD pass (plus
    trivial slicing) instead of one SVD pass per threshold. Each threshold ``t`` is
    saved to its own ``P_nullspace_..._n<N>_thr<t>.pt`` (coexisting with the others).

    Returns ``{t: P_tensor}``.
    """
    model_name = model_name or model.config._name_or_path
    thresholds = list(thresholds)

    out = {}
    todo = []
    for t in thresholds:
        path = _cache_path(model_name, replace(hparams, nullspace_threshold=t))
        if cache and not force_recompute and os.path.exists(path):
            print(f"[alphaedit.P] loading cached projections from {path}", flush=True)
            out[t] = _load_cached_P(path)
        else:
            todo.append(t)
    if not todo:
        return out

    # One SVD per layer; slice it for every remaining threshold.
    mats = {t: [] for t in todo}
    for layer in hparams.layers:
        U, S = _svd_cov(model, tokenizer, layer, hparams)
        for t in todo:
            P_l, ndim = _project_from_svd(U, S, t)
            mats[t].append(P_l)
            print(f"[alphaedit.P] layer {layer}: null-space dim {ndim}/{S.numel()} "
                  f"(threshold={t})", flush=True)
        del U, S

    for t in todo:
        P = torch.stack(mats[t], dim=0)
        out[t] = P
        if cache:
            path = _cache_path(model_name, replace(hparams, nullspace_threshold=t))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(P, path)
            print(f"[alphaedit.P] cached projections to {path}", flush=True)
        mats[t] = None  # free as we go (each stacked P can be several GB)
    return out
