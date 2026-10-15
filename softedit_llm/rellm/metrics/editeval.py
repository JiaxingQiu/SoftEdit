"""Edit-evaluation driver (text, LLM analog of revlm/metrics/editeval.py).

Faithful port of the VLM edit-eval, scoped to the text setting: the image
generality (i-gen) and strength-ladder soft/blend metrics are dropped, and the
text soft-gen (one-hot per-target premise → ``sloc``) replaces the image
soft-gen. Metric sets:

  reliability      — edited model on the edit set vs human votes  → acc, align
  text_generality  — rephrased questions, same gold               → gen, sgen
  soft_generality  — per-target premise, one-hot gold             → sloc
  locality         — unrelated pool, gold = pre-edit model        → loc

The 6-cell report is ``[acc, gen, loc; align, sgen, sloc]`` (see
``prioritized_scores`` / ``print_score_summary``).
"""
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import copy
import gc
import json
import os
import random
import time

import pandas as pd
import torch


# --------------------------------------------------------------------------- #
# device / misc helpers (verbatim from VLM)
# --------------------------------------------------------------------------- #
def move_model_device(model: Any, device: str) -> None:
    dev = torch.device(device) if isinstance(device, str) else device
    if hasattr(model, "device"):
        try:
            model.device = dev
        except Exception:
            pass
    if hasattr(model, "config") and hasattr(model.config, "device"):
        try:
            model.config.device = dev
        except Exception:
            pass
    if hasattr(model, "model") and hasattr(model.model, "to"):
        model.model.to(dev)
    elif hasattr(model, "to"):
        model.to(dev)


def cuda_gc() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _hf_params(model: Any) -> Dict[str, torch.nn.Parameter]:
    inner = model.model if hasattr(model, "model") else model
    return dict(inner.named_parameters())


def _swap_edited_layers(model_new: Any, model_old: Any, names: Sequence[str]):
    """Copy pre-edit tensors from ``model_old`` onto ``model_new``; return a GPU stash of the edited tensors.

    Same locality gold (unedited forwards) without parking a second 8B copy on the GPU.
    """
    new_p = _hf_params(model_new)
    old_p = _hf_params(model_old)
    missing = [n for n in names if n not in new_p or n not in old_p]
    if missing:
        raise KeyError(f"locality layer-swap: missing params {missing[:3]}")
    stash = {n: new_p[n].detach().clone() for n in names}
    with torch.no_grad():
        for n in names:
            new_p[n].copy_(old_p[n].to(device=new_p[n].device, dtype=new_p[n].dtype))
    return stash


def _restore_edited_layers(model_new: Any, stash: Dict[str, torch.Tensor]) -> None:
    new_p = _hf_params(model_new)
    with torch.no_grad():
        for n, t in stash.items():
            new_p[n].copy_(t)


def _norm(label: Any) -> str:
    return str(label).strip().lower()


def _fmt(m):
    if isinstance(m, dict):
        return {k: _fmt(v) for k, v in m.items()}
    if isinstance(m, float):
        return round(m, 3)
    return m


_SKIP_ALIASES = {
    "reliability": "reliability", "rel": "reliability",
    "text_generality": "text_generality", "t_gen": "text_generality", "tgen": "text_generality",
    "soft_generality": "soft_generality", "soft_gen": "soft_generality", "sgen": "soft_generality",
    "locality": "locality", "loc": "locality",
}


def normalize_skip(skip: Sequence[str]) -> set:
    out = set()
    for s in skip or ():
        key = str(s).strip().lower()
        if key not in _SKIP_ALIASES:
            raise ValueError(f"unknown metric set {s!r}; choose from {sorted(set(_SKIP_ALIASES.values()))}")
        out.add(_SKIP_ALIASES[key])
    return out


def _primary_metric(metrics: Any) -> float:
    """Collapse a task-metric dict into a single scalar for printing.

    postedit_qa / qa -> soft accuracy ("acc"); softqa -> qa.acc; multiqa -> acc_at_1.
    """
    if isinstance(metrics, dict):
        if "acc" in metrics:
            return float(metrics["acc"])
        if "qa" in metrics and isinstance(metrics["qa"], dict):
            return float(metrics["qa"].get("acc", 0.0))
        if "acc_at_1" in metrics:
            return float(metrics["acc_at_1"])
        return 0.0
    return float(metrics)


# Retrieval editors that mutate prompts/state and re-apply per eval set.
_RETRIEVAL_EDITORS = ("ike", "softedit")


def _maybe_apply_ike(editor: Any, edit_ds: Any, train_ds: Any) -> None:
    if editor is None:
        return
    cfg = getattr(train_ds, "config", None)
    editor_cfg = getattr(cfg, "editor", None)
    editor_name = getattr(editor_cfg, "_name", None) if editor_cfg is not None else None
    if editor_name and editor_name.endswith("_soft"):
        editor_name = editor_name[:-5]
    if editor_name in _RETRIEVAL_EDITORS or (
            editor_name and editor_name.startswith("softedit_")):
        if hasattr(editor, "model") and hasattr(editor.model, "eval"):
            editor.model.eval()
        if hasattr(editor, "apply_to_dataset"):
            editor.apply_to_dataset(edit_ds)


# --------------------------------------------------------------------------- #
# metric sets
# --------------------------------------------------------------------------- #
def reliability(model_new: Any, edit_ds: Any, label: str = None, dump_path: str = None):
    """Task-metric dict for the edited model on ``edit_ds`` (side-effect free)."""
    ds = copy.deepcopy(edit_ds)
    ds.task_generate(model_new, use_cache=True)
    if label and getattr(edit_ds.config, "debug", False):
        from rellm.run.edit_utils import debug_samples  # lazy: avoid circular import
        debug_samples(ds, label)
    metrics = ds.task_engineer.eval(ds)
    if dump_path:
        try:
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            ds.snap(out_path=dump_path)
        except Exception as e:
            print(f"[dump] post-edit snapshot failed for {dump_path}: {e}", flush=True)
    return metrics


def text_generality(model_new: Any, edit_ds: Any,
                    related_texts: Dict[str, List[str]],
                    editor: Any = None, dump_path: str = None):
    """Accuracy/alignment on rephrased questions with the same (invariant) gold."""
    if not related_texts:
        return None
    df = edit_ds.load_df()
    ds = copy.deepcopy(edit_ds)
    related_df = pd.DataFrame(
        ((uid, variant) for uid, variants in related_texts.items() for variant in variants),
        columns=["uid", "question"],
    )
    related_df = related_df.merge(df.drop(columns=["question"]), on="uid", how="left")
    ds.data = ds.df2data(related_df)
    ds.set_dataloader()
    _maybe_apply_ike(editor, ds, edit_ds)
    return reliability(model_new, ds, label="text_generality", dump_path=dump_path)


def soft_generality(model_new: Any, edit_ds: Any,
                    soft_ctx: Dict[str, List[Tuple[str, str]]],
                    editor: Any = None, dump_path: str = None):
    """Alignment on per-target premises with **one-hot** gold on the target (sloc).

    ``soft_ctx``: {uid: [(target_label, "context question"), ...]}. Each becomes an
    eval row whose ``answer_counts`` is overridden to ``{target_label: 1}``.
    """
    if not soft_ctx:
        return None
    ds = copy.deepcopy(edit_ds)
    rows = []
    for uid, items in soft_ctx.items():
        for target, composed_q in items:
            rows.append({"uid": str(uid), "question": composed_q,
                         "answer_counts": json.dumps({target: 1})})
    if not rows:
        return None
    related_df = pd.DataFrame(rows, columns=["uid", "question", "answer_counts"])
    ds.data = ds.df2data(related_df)
    ds.set_dataloader()
    _maybe_apply_ike(editor, ds, edit_ds)
    return reliability(model_new, ds, label="soft_generality", dump_path=dump_path)


def locality(model_old: Any, model_new: Any, edit_ds: Any,
             unrelated_ds=None, sample_size=None, editor: Any = None,
             dump_path: str = None) -> Dict[str, float]:
    """Behaviour drift on unrelated inputs, scored with MODEL_OLD as the gold.

    When ``unrelated_ds`` is provided (the baseline-correct locality set built
    once by ``find_errors``), ``sample_size`` items are drawn from it.  Otherwise
    falls back to full parquet minus *all* edit uids on ``edit_ds``.
    """
    from rellm.metrics.ranked_metrics import (
        build_labels_true, js_distance, tau_readout, ndcg_readout,
    )

    if unrelated_ds is None:
        unrelated_ds = copy.deepcopy(edit_ds)
        full_df = edit_ds.load_df()
        used_uids = {str(ex.get("uid")) for ex in edit_ds.data}
        mask = ~full_df["uid"].astype(str).isin(used_uids)
        pool_df = full_df.loc[mask].reset_index(drop=True)
        if pool_df.empty:
            raise ValueError("No unrelated inputs found")
        if sample_size is not None:
            pool_df = pool_df.sample(
                n=min(sample_size, len(pool_df)),
                random_state=getattr(edit_ds.config, "seed", 333),
            )
        unrelated_ds.data = unrelated_ds.df2data(pool_df)
        unrelated_ds.set_dataloader()
    elif sample_size is not None and len(unrelated_ds.data) > sample_size:
        unrelated_ds = copy.deepcopy(unrelated_ds)
        base_seed = getattr(edit_ds.config, "seed", 333)
        rng = random.Random(base_seed + len(edit_ds.data))
        unrelated_ds.data = rng.sample(unrelated_ds.data, sample_size)
        unrelated_ds.set_dataloader()

    ds_old = copy.deepcopy(unrelated_ds)
    ds_new = copy.deepcopy(unrelated_ds)
    for d in (ds_old, ds_new):
        if getattr(d, "config", None) is not None:
            try:
                d.config.continuation_gate = False
            except Exception:
                pass

    target_device = getattr(edit_ds.config, "device", "cuda")
    if isinstance(target_device, str):
        target_device = torch.device(target_device)

    # Free KV-cache leftovers from reliability / t-gen / s-gen (those sets grow
    # with #edits; locality itself stays loc_sample_size). Then run the two
    # locality forwards. If the editor only mutates a named weight list
    # (AlphaEdit's down_proj range), swap those tensors in place so we never
    # hold two 8B copies on GPU/CPU at once. Scores are identical: unedited
    # vs edited generations on the same unrelated pool.
    cuda_gc()
    swap_names = (
        list(editor._edited_weight_names())
        if editor is not None and hasattr(editor, "_edited_weight_names")
        else None
    )

    if swap_names:
        ds_new.set_dataloader()
        move_model_device(model_new, target_device)
        _maybe_apply_ike(editor, ds_new, edit_ds)
        ds_new.task_generate(model_new, use_cache=True, compute_tf=False)
        cuda_gc()
        stash = _swap_edited_layers(model_new, model_old, swap_names)
        try:
            ds_old.set_dataloader()
            ds_old.task_generate(model_new, use_cache=True, compute_tf=False)
        finally:
            _restore_edited_layers(model_new, stash)
            del stash
            cuda_gc()
    else:
        # Edited pass first (already on GPU after s-gen), then the pre-edit copy.
        ds_new.set_dataloader()
        move_model_device(model_new, target_device)
        _maybe_apply_ike(editor, ds_new, edit_ds)
        ds_new.task_generate(model_new, use_cache=True, compute_tf=False)
        move_model_device(model_new, "cpu")
        cuda_gc()

        move_model_device(model_old, target_device)
        ds_old.set_dataloader()
        ds_old.task_generate(model_old, use_cache=True, compute_tf=False)
        move_model_device(model_old, "cpu")
        cuda_gc()

        move_model_device(model_new, target_device)

    def _open_readout(pred):
        spo = pred.get("support_prob_open") or {}
        return {_norm(k): float(v) for k, v in spo.items()
                if str(k) != "__OTHER__" and v is not None}

    acc_sum = js_sum = tau_sum = ndcg_sum = 0.0
    n = js_n = rank_n = 0
    for ex_old, ex_new in zip(ds_old.data, ds_new.data):
        pred_old = ex_old.get("pred") or {}
        pred_new = ex_new.get("pred") or {}
        if not pred_old or not pred_new:
            continue
        n += 1
        old_label = _norm(pred_old.get("label", ""))
        new_label = _norm(pred_new.get("label", ""))
        acc_sum += float(bool(old_label) and old_label == new_label)

        old_spo = _open_readout(pred_old)
        new_spo = _open_readout(pred_new)
        if old_spo and new_spo:
            js_sum += js_distance(old_spo, new_spo)
            js_n += 1
            gold_tiers = build_labels_true(old_spo)
            tau_sum += tau_readout(new_spo, gold_tiers, label2key=None)
            ndcg_sum += ndcg_readout(new_spo, gold_tiers, label2key=None)
            rank_n += 1

        ex_new["loc_gold"] = {
            "label": pred_old.get("label", ""),
            "answer": pred_old.get("answer", ""),
            "support_prob_open": pred_old.get("support_prob_open"),
            "open_label2key": pred_old.get("open_label2key"),
        }

    move_model_device(model_new, target_device)

    if n == 0:
        return {"simple_acc": 0.0, "hard_acc": 0.0, "rank_acc": 0.0, "acc": 0.0,
                "js_dist": None, "js_dist_gate": None,
                "tau_readout": None, "tau_readout_gate": None,
                "ndcg_readout": None, "ndcg_readout_gate": None, "n": 0}

    acc = acc_sum / n
    js = js_sum / js_n if js_n > 0 else None
    tau = tau_sum / rank_n if rank_n > 0 else None
    ndcg = ndcg_sum / rank_n if rank_n > 0 else None

    if dump_path:
        try:
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            ds_new.snap(out_path=dump_path)
        except Exception as e:
            print(f"[dump] locality snapshot failed for {dump_path}: {e}", flush=True)

    return {
        "simple_acc": acc, "hard_acc": acc, "rank_acc": acc, "acc": acc,
        "js_dist": js, "js_dist_gate": js,
        "tau_readout": tau, "tau_readout_gate": tau,
        "ndcg_readout": ndcg, "ndcg_readout_gate": ndcg,
        "n": n,
    }


# --------------------------------------------------------------------------- #
# scoreboard: [acc, gen, loc ; align, sgen, sloc]
# --------------------------------------------------------------------------- #
def prioritized_scores(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flat dict of the 6-cell metrics (+ loc align, soft-gen acc). Values are None
    for skipped sets; callers filter. Shared by the console summary and wandb."""
    nested = result.get("metrics") or {}
    rel = nested.get("reliability") or {}
    tgen = nested.get("text_generality") or {}
    soft = nested.get("soft_generality") or {}
    loc = nested.get("locality") or {}

    def _acc(block):
        v = block.get("simple_acc") if isinstance(block, dict) else None
        return float(v) if v is not None else None

    def _align(block):
        v = block.get("js_dist_gate") if isinstance(block, dict) else None
        return (1.0 - float(v)) if v is not None else None

    return {
        # row 1: accuracy
        "acc": _acc(rel),
        "gen": _acc(tgen),
        "loc": (float(result["locality"]) if result.get("locality") is not None else _acc(loc)),
        # row 2: alignment
        "align": _align(rel),
        "sgen": _align(tgen),
        "sloc": _align(soft),
        # extras
        "loc/align": _align(loc),
        "sloc/acc": _acc(soft),
        "n_edits": result.get("n_edits"),
        "time_per_edit": result.get("time_per_edit"),
    }


def print_score_summary(result: Dict[str, Any]) -> None:
    """Compact 2x4 scoreboard: reliability / t-gen / locality / s-gen (soft-gen).

    Rows are acc (simple_acc) and align (1 - js_dist_gate). The s-gen column is
    the one-hot soft-generality set (``sloc/acc``, ``sloc``).
    """
    s = prioritized_scores(result)

    def f(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "  -  "

    print("=" * 50, flush=True)
    print("[scores]  acc = simple_acc | align = 1 - js_dist_gate", flush=True)
    print(f"  {'':<8}{'reliability':>14}{'t-gen':>14}{'locality':>14}{'s-gen':>14}", flush=True)
    print(f"  {'acc':<8}{f(s.get('acc')):>14}{f(s.get('gen')):>14}{f(s.get('loc')):>14}{f(s.get('sloc/acc')):>14}", flush=True)
    print(f"  {'align':<8}{f(s.get('align')):>14}{f(s.get('sgen')):>14}{f(s.get('loc/align')):>14}{f(s.get('sloc')):>14}", flush=True)
    print("=" * 50, flush=True)


def _load_pred_dump(path: str):
    """Return a prediction-dump list, or None if missing/truncated/empty."""
    if not path or not os.path.isfile(path) or os.path.getsize(path) <= 2:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, list) or not data:
        return None
    return data


def _metrics_from_pred_dump(edit_ds: Any, dump_path: str):
    """Recompute task metrics from a saved prediction dump (no model forward)."""
    data = _load_pred_dump(dump_path)
    if data is None:
        return None
    try:
        ds = copy.deepcopy(edit_ds)
        ds.data = data
        if getattr(ds, "task_engineer", None) is None:
            ds.set_dataloader()
        else:
            for i, ex in enumerate(ds.data):
                ex["idx"] = i
                ds.task_engineer.eng_golds(ex)
        return ds.task_engineer.eval(ds)
    except Exception as e:
        print(f"[resume] failed to score dump {dump_path}: {e}", flush=True)
        return None


def _locality_from_dump(dump_path: str):
    """Recompute locality metrics from a dump that carries ``loc_gold`` + ``pred``."""
    from rellm.metrics.ranked_metrics import (
        build_labels_true, js_distance, tau_readout, ndcg_readout,
    )
    data = _load_pred_dump(dump_path)
    if data is None:
        return None
    if not any(isinstance(ex.get("loc_gold"), dict) for ex in data):
        return None

    def _open_readout(pred):
        spo = (pred or {}).get("support_prob_open") or {}
        return {_norm(k): float(v) for k, v in spo.items()
                if str(k) != "__OTHER__" and v is not None}

    n = 0
    acc_sum = js_sum = tau_sum = ndcg_sum = 0.0
    js_n = rank_n = 0
    try:
        for ex in data:
            pred_old = ex.get("loc_gold") or {}
            pred_new = ex.get("pred") or {}
            if not pred_old or not pred_new:
                continue
            n += 1
            old_label = _norm(pred_old.get("label", ""))
            new_label = _norm(pred_new.get("label", ""))
            acc_sum += float(bool(old_label) and old_label == new_label)
            old_spo = _open_readout(pred_old)
            new_spo = _open_readout(pred_new)
            if old_spo and new_spo:
                js_sum += js_distance(old_spo, new_spo)
                js_n += 1
                gold_tiers = build_labels_true(old_spo)
                tau_sum += tau_readout(new_spo, gold_tiers, label2key=None)
                ndcg_sum += ndcg_readout(new_spo, gold_tiers, label2key=None)
                rank_n += 1
    except Exception as e:
        print(f"[resume] failed to score locality dump {dump_path}: {e}", flush=True)
        return None
    if n == 0:
        return None
    acc = acc_sum / n
    js = js_sum / js_n if js_n > 0 else None
    tau = tau_sum / rank_n if rank_n > 0 else None
    ndcg = ndcg_sum / rank_n if rank_n > 0 else None
    return {
        "simple_acc": acc, "hard_acc": acc, "rank_acc": acc, "acc": acc,
        "js_dist": js, "js_dist_gate": js,
        "tau_readout": tau, "tau_readout_gate": tau,
        "ndcg_readout": ndcg, "ndcg_readout_gate": ndcg,
        "n": n,
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def editeval(
        model_old: Any,
        model_new: Any,
        edit_ds: Any,
        editor: Any,
        related_texts: Mapping[str, Sequence[str]] = None,
        soft_ctx: Mapping[str, Sequence[Tuple[str, str]]] = None,
        unrelated_ds=None,
        loc_sample_size: int = 100,
        edit_subsample_size: int = None,
        edit_time: float = None,
        dump_preds_dir: str = None,
        skip: Sequence[str] = (),
        resume_dumps: bool = False,
) -> Dict[str, Any]:
    """Run the post-edit metric sets and return a checkpoint dict.

    Returns ``{n_edits, locality, metrics:{reliability, text_generality,
    soft_generality, locality}, [edit_time, time_per_edit]}``.

    resume_dumps: batch-only. If True, skip a metric set when its prediction
    dump already exists and re-score from that dump. Sequential callers must
    leave this False so a redone peek cannot silently reuse a partial dump.
    """
    related_texts = related_texts or {}
    soft_ctx = soft_ctx or {}
    skip = normalize_skip(skip)
    if skip:
        print(f"[editeval] skipping: {sorted(skip)}", flush=True)
    if resume_dumps:
        print("[editeval] resume_dumps: reusing completed prediction dumps", flush=True)

    # Intermediate peeks: work on a subsampled deepcopy and re-apply the editor.
    train_ds = edit_ds
    if edit_subsample_size is not None:
        edit_ds = copy.deepcopy(edit_ds)
        if len(edit_ds.data) > edit_subsample_size:
            rng = random.Random(333)
            edit_ds.data = rng.sample(edit_ds.data, edit_subsample_size)
        edit_ds.set_dataloader()
        _maybe_apply_ike(editor, edit_ds, train_ds)
        uids = {str(ex["uid"]) for ex in edit_ds.data}
        related_texts = {k: v for k, v in related_texts.items() if str(k) in uids}
        soft_ctx = {k: v for k, v in soft_ctx.items() if str(k) in uids}

    def _dp(name: str):
        return os.path.join(dump_preds_dir, f"{name}.json") if dump_preds_dir else None

    rel_metrics = None
    if "reliability" not in skip:
        t = time.time()
        if resume_dumps:
            rel_metrics = _metrics_from_pred_dump(edit_ds, _dp("reliability"))
            if rel_metrics is not None:
                print(f"[resume] reliability from dump: {time.time() - t:.2f}s", flush=True)
        if rel_metrics is None:
            rel_metrics = reliability(model_new, edit_ds, label="reliability", dump_path=_dp("reliability"))
            print(f"[Timing] reliability: {time.time() - t:.2f}s", flush=True)
        print(f"Reliability: {_primary_metric(rel_metrics):.4f}  ({_fmt(rel_metrics)})", flush=True)
        cuda_gc()

    tgen_metrics = None
    if "text_generality" not in skip:
        t = time.time()
        if resume_dumps:
            tgen_metrics = _metrics_from_pred_dump(edit_ds, _dp("text_generality"))
            if tgen_metrics is not None:
                print(f"[resume] text_generality from dump: {time.time() - t:.2f}s", flush=True)
        if tgen_metrics is None:
            tgen_metrics = text_generality(model_new, edit_ds, related_texts, editor=editor, dump_path=_dp("text_generality"))
            if tgen_metrics is not None:
                print(f"[Timing] text_generality: {time.time() - t:.2f}s", flush=True)
        if tgen_metrics is not None:
            print(f"Text Generality: {_primary_metric(tgen_metrics):.4f}  ({_fmt(tgen_metrics)})", flush=True)
        cuda_gc()

    soft_metrics = None
    if "soft_generality" not in skip:
        t = time.time()
        if resume_dumps:
            soft_metrics = _metrics_from_pred_dump(edit_ds, _dp("soft_generality"))
            if soft_metrics is not None:
                print(f"[resume] soft_generality from dump: {time.time() - t:.2f}s", flush=True)
        if soft_metrics is None:
            soft_metrics = soft_generality(model_new, edit_ds, soft_ctx, editor=editor, dump_path=_dp("soft_generality"))
            if soft_metrics is not None:
                print(f"[Timing] soft_generality: {time.time() - t:.2f}s", flush=True)
        if soft_metrics is not None:
            print(f"Soft Generality (sloc): ({_fmt(soft_metrics)})", flush=True)
        cuda_gc()

    loc_metrics = None
    if "locality" not in skip:
        t = time.time()
        if resume_dumps:
            loc_metrics = _locality_from_dump(_dp("locality"))
            if loc_metrics is not None:
                print(f"[resume] locality from dump: {time.time() - t:.2f}s", flush=True)
        if loc_metrics is None:
            loc_metrics = locality(model_old, model_new, train_ds, unrelated_ds=unrelated_ds,
                                   sample_size=loc_sample_size, editor=editor, dump_path=_dp("locality"))
            print(f"[Timing] locality: {time.time() - t:.2f}s", flush=True)
        print(f"Locality: {loc_metrics['simple_acc']:.4f}  ({_fmt(loc_metrics)})", flush=True)
        cuda_gc()

    n_edits = len(edit_ds.data)
    result = {
        "n_edits": float(n_edits),
        "locality": float(loc_metrics["simple_acc"]) if loc_metrics else None,
        "metrics": {
            "reliability": rel_metrics,
            "text_generality": tgen_metrics,
            "soft_generality": soft_metrics,
            "locality": loc_metrics,
        },
    }
    if edit_time is not None and n_edits > 0:
        result["edit_time"] = float(edit_time)
        result["time_per_edit"] = float(edit_time / n_edits)
    print_score_summary(result)
    return result
