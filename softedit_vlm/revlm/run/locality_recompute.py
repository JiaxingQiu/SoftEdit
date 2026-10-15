"""Patch sequential JSONL locality scores onto the fixed baseline-correct loc_ds.

VQAv2-first offline fix. Does not re-edit. Strategies:

  exact slice   SoftEdit          slice codebook[:K], recompute Loc
  prop. slice    GRACE(+soft)     one global rekey, then slice round(M*K/N)
                  ReasonEdit(+soft)  slice codebook, no rekey
  shift         FT/MEND/BalancEdit (+soft)
                  complete   -> delta = full-eval Loc - last peek Loc  (no GPU)
                  incomplete -> one Loc eval on last ckpt, then shift
  skip          baseline, *retrain*, GT

Usage (one job per VLM; dataset defaults to vqav2)::

    python -m revlm.run.locality_recompute --model_name llava --dataset_name vqav2
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from revlm import *  # noqa: F401,F403
from revlm.config_utils import configure_args, update_config
from revlm.editors import get_editor
import random
from revlm.metrics.editeval import cuda_gc, locality
from revlm.metrics.ranked_metrics import build_labels_true, js_distance, tau_readout, ndcg_readout
from revlm.run.edit_utils import find_errors
from revlm.run.resume import load_ckpt


EXACT_SLICE = {"softedit", "softedit_task", "ike", "ike_soft"}
PROP_SLICE = {"reasonedit", "reasonedit_soft"}
SHIFT = {"ft", "ft_soft", "mend", "mend_soft", "balancedit", "balancedit_soft",
         "grace", "grace_soft"}
WEIGHT_ONLY = {"ft", "ft_soft", "mend", "mend_soft"}
SKIP = {
    "baseline",
    "ft_retrain", "ft_retrain_soft",
    "mend_retrain", "mend_retrain_soft",
    "gt", "gt_soft",
}
_ACC_KEYS = ("simple_acc", "hard_acc", "rank_acc", "acc")


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------

def _read_rows(path: Path) -> List[dict]:
    text = path.read_text().strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]
    except Exception:
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def _write_jsonl(path: Path, records: List[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _backup_jsonl(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".locfix.bak")
    if path.exists() and not bak.exists():
        shutil.copy2(path, bak)
        print(f"[locfix] backed up {path.name} -> {bak.name}", flush=True)


def _step(rec: dict) -> int:
    return int(rec.get("batch_idx") or rec.get("n_edits") or 0)


def _loc_acc(rec: dict) -> Optional[float]:
    v = rec.get("locality")
    if isinstance(v, (int, float)):
        return float(v)
    block = (rec.get("metrics") or {}).get("locality") or {}
    v = block.get("simple_acc", block.get("acc"))
    return float(v) if isinstance(v, (int, float)) else None


def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _patch_record(rec: dict, loc_metrics: dict, meta: dict) -> None:
    acc = float(loc_metrics.get("simple_acc", loc_metrics.get("acc", 0.0)))
    rec["locality"] = acc
    rec["locality_hard"] = float(loc_metrics.get("hard_acc", acc))
    rec.setdefault("metrics", {})["locality"] = loc_metrics
    rec["_loc_fix"] = meta


def _shift_record(rec: dict, delta: float, meta: dict) -> None:
    old = _loc_acc(rec)
    if old is not None:
        rec["locality"] = _clamp01(old + delta)
    hard = rec.get("locality_hard")
    if isinstance(hard, (int, float)):
        rec["locality_hard"] = _clamp01(float(hard) + delta)
    block = (rec.get("metrics") or {}).get("locality")
    if isinstance(block, dict):
        for k in _ACC_KEYS:
            if isinstance(block.get(k), (int, float)):
                block[k] = _clamp01(float(block[k]) + delta)
    rec["_loc_fix"] = {**meta, "old_loc": old}


# ---------------------------------------------------------------------------
# codebook slicing
# ---------------------------------------------------------------------------

def _n_keep(M: int, K: int, N: int) -> int:
    if M <= 0 or K <= 0:
        return 0
    if N <= 0:
        return M
    if M == N:
        return min(K, M)
    return max(0, min(M, int(round(M * K / float(N)))))


def _slice_list_or_tensor(obj, n: int):
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj[:n].clone() if obj.shape[0] >= n else obj.clone()
    if isinstance(obj, list):
        return list(obj[:n])
    return obj


def _codebook_len(state: dict, editor_name: str) -> int:
    if not state:
        return 0
    if editor_name in ("grace", "grace_soft"):
        if not state.get("has_codebook"):
            return 0
        keys = state.get("keys")
        return int(keys.shape[0]) if isinstance(keys, torch.Tensor) else 0
    cb = state.get("codebook")
    if isinstance(cb, list):
        return len(cb)
    return 0


def slice_state(state: dict, editor_name: str, K: int, N: int) -> dict:
    """Return a shallow-copied resume state with the codebook cut to peek K."""
    out = dict(state or {})
    M = _codebook_len(state, editor_name)
    n = _n_keep(M, K, N)
    if editor_name in ("grace", "grace_soft"):
        if not state.get("has_codebook") or n <= 0:
            return {"has_codebook": False}
        out["has_codebook"] = True
        for key in ("keys", "values", "epsilons"):
            out[key] = _slice_list_or_tensor(state.get(key), n)
        labels = state.get("key_labels") or []
        out["key_labels"] = list(labels[:n])
        return out

    out["codebook"] = list((state.get("codebook") or [])[:n])
    out["key_embs"] = _slice_list_or_tensor(state.get("key_embs"), n)
    if "key_radii" in state:
        out["key_radii"] = _slice_list_or_tensor(state.get("key_radii"), n)
    if "key_sigmas" in state:
        out["key_sigmas"] = None  # recompute on the sliced set
    out["key_dist_threshold"] = None
    uids = []
    for e in out["codebook"]:
        if isinstance(e, dict) and e.get("uid") is not None:
            uids.append(e.get("uid"))
    added = state.get("_added_uids")
    if added is not None:
        if uids:
            out["_added_uids"] = set(uids)
        elif isinstance(added, (set, list, tuple)):
            out["_added_uids"] = set(list(added)[:n])
        else:
            out["_added_uids"] = set()
    if "_edit_count" in state:
        out["_edit_count"] = n
    return out


# ---------------------------------------------------------------------------
# editor / model helpers
# ---------------------------------------------------------------------------

def _bind_config(config, edit_ds, loc_ds, model) -> None:
    edit_ds.config = config
    loc_ds.config = config
    if hasattr(model, "config"):
        try:
            model.config = config
        except Exception:
            pass


def _gpu_device(config=None):
    """Always CUDA when available. Never follow a config.device that was
    accidentally overwritten to CPU by move_model_device()."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if config is not None:
        return torch.device(getattr(config, "device", "cpu"))
    return torch.device("cpu")


def _move_weights(model, device) -> None:
    """Move tensors only. Do not write model.config.device — that NestedConfig
    is shared with the experiment and locality() reads it as the GPU target."""
    dev = torch.device(device) if isinstance(device, str) else device
    if hasattr(model, "device"):
        try:
            model.device = dev
        except Exception:
            pass
    inner = model.model if hasattr(model, "model") else model
    if hasattr(inner, "to"):
        inner.to(dev)


def _cpu_deepcopy(model, config):
    """Clone the model on CPU using torch serialization (much faster than
    Python ``copy.deepcopy`` for large models — avoids traversing the full
    Python object graph and uses optimized tensor I/O instead)."""
    import io
    gpu = _gpu_device(config)
    if hasattr(model, "device"):
        model.device = torch.device("cpu")
    _move_weights(model, "cpu")
    cuda_gc()
    t0 = time.time()
    buf = io.BytesIO()
    torch.save(model, buf)
    buf.seek(0)
    clone = torch.load(buf, map_location="cpu", weights_only=False)
    del buf
    print(f"[locfix]   clone via torch.save/load: {time.time() - t0:.1f}s", flush=True)
    config.device = gpu
    if hasattr(model, "device"):
        model.device = torch.device("cpu")
    return clone


def _to_device(model, device) -> None:
    _move_weights(model, device)


def _eval_mode(model) -> None:
    inner = model.model if hasattr(model, "model") else model
    if hasattr(inner, "eval"):
        inner.eval()


def _edit_stub(edit_ds, n: int):
    stub = copy.copy(edit_ds)
    n = max(0, min(int(n), len(edit_ds.data)))
    stub.data = list(edit_ds.data[:n]) if n else list(edit_ds.data[:1])
    return stub


def _freeze_adapters(editor, editor_name: str) -> None:
    """Make codebook-carrying editors read-only so inference never mutates keys.

    BalancEdit and GRACE both have an ``iter == 0`` branch that adds/splits keys
    on every forward.  Setting ``iter = -1`` skips it; ``training = False`` and
    ``other_is_training = False`` keep the non-edit forward path.
    """
    if editor_name in ("balancedit", "balancedit_soft"):
        for layer in getattr(editor, "layers", []):
            lm = editor._get_layer_module(layer)
            lm.training = False
            lm.other_is_training = False
            lm.iter = -1
            lm.calculate_eps = False
            lm.cal_rephrase_eps = False
        print(f"[locfix] {editor_name}: froze {len(editor.layers)} adapter layer(s)", flush=True)
    if editor_name in ("grace", "grace_soft"):
        tl = getattr(editor, "target_layer", None)
        if tl is not None:
            tl.training = False
            setattr(tl, "iter", -1)
            setattr(tl, "key_id", -1)
        print(f"[locfix] {editor_name}: froze GRACE adaptor", flush=True)


def _pin_cuda(*objs) -> torch.device:
    """Force NestedConfig.device back to CUDA.

    ``editeval.locality`` / ``move_model_device`` write model.config.device
    when parking a model on CPU. That NestedConfig is shared, so the *next*
    peek would then eval on CPU (K=200 fast, K=400+ slow).
    """
    gpu = _gpu_device()
    for obj in objs:
        if obj is None:
            continue
        for cfg in (obj, getattr(obj, "config", None)):
            if cfg is not None and hasattr(cfg, "device"):
                try:
                    cfg.device = gpu
                except Exception:
                    pass
    return gpu


def eval_loc(model_old, model_new, editor, edit_ds, loc_ds, K: int,
             loc_sample_size: int) -> dict:
    _pin_cuda(edit_ds, loc_ds, model_old, model_new)
    stub = _edit_stub(edit_ds, K)
    t0 = time.time()
    metrics = locality(
        model_old, model_new, stub,
        unrelated_ds=loc_ds,
        sample_size=loc_sample_size,
        editor=editor,
    )
    _pin_cuda(edit_ds, loc_ds, model_old, model_new)
    print(f"[locfix]   loc@{K}: simple_acc={metrics.get('simple_acc'):.4f}  "
          f"n={metrics.get('n')}  ({time.time() - t0:.1f}s)", flush=True)
    return metrics


# ---------------------------------------------------------------------------
# inplace weight-only locality eval  (avoids deepcopy for FT / MEND)
# ---------------------------------------------------------------------------

def _resolve_ft_module(model, config):
    """Resolve the FT target module using the same logic as the FT editor."""
    from revlm.editors.utils import brackets_to_periods, parent_module
    inner = model.model if hasattr(model, "model") else model
    layer_spec = config.inner_params[0]
    suffixes = [".weight", ".bias"]
    layer = layer_spec.rsplit(".", 1)[0] if any(layer_spec.endswith(s) for s in suffixes) else layer_spec
    layer_path = brackets_to_periods(layer)
    parent = parent_module(inner, layer_path)
    layer_name = layer_path.rsplit(".", 1)[-1]
    return getattr(parent, layer_name)


def _backup_and_apply_ft(model, ckpt_state, config):
    """Apply FT checkpoint weights to the target module, return backup."""
    target = _resolve_ft_module(model, config)
    backup = {k: v.clone() for k, v in target.state_dict().items()}
    ref_param = next(target.parameters())
    dev, dtype = ref_param.device, ref_param.dtype
    target.load_state_dict(
        {k: v.to(dev, dtype) for k, v in ckpt_state.get("layer", {}).items()}
    )
    return ("ft", target, backup)


def _backup_and_apply_mend(model, ckpt_state):
    """Apply MEND checkpoint weights, return backup of originals."""
    inner = model.model if hasattr(model, "model") else model
    pd = dict(inner.named_parameters())
    backup = {}
    for pname, ckpt_val in ckpt_state.get("weights", {}).items():
        if pname in pd:
            backup[pname] = pd[pname].data.clone()
            pd[pname].data.copy_(ckpt_val.to(pd[pname].device, pd[pname].dtype))
    return ("mend", inner, backup)


def _restore_weights(handle):
    """Restore original weights from a backup handle."""
    kind, target, backup = handle
    if kind == "ft":
        target.load_state_dict(
            {k: v.to(next(target.parameters()).device) for k, v in backup.items()}
        )
    else:
        pd = dict(target.named_parameters())
        for pname, orig_val in backup.items():
            if pname in pd:
                pd[pname].data.copy_(orig_val.to(pd[pname].device, pd[pname].dtype))


def _norm_label(s):
    return str(s).strip().lower() if s else ""


def eval_loc_inplace(model_base, ckpt_editor_state, editor_name: str,
                     config, edit_ds, loc_ds, K: int,
                     loc_sample_size: int) -> dict:
    """Locality eval using weight swap on a single model — no deepcopy.

    1. Gold pass: generate with original weights
    2. Apply checkpoint weights in-place
    3. Edited pass: generate with edited weights
    4. Restore original weights
    5. Score
    """
    stub = _edit_stub(edit_ds, K)
    t0 = time.time()

    # Sample locality probes (same logic as locality())
    unrelated = copy.deepcopy(loc_ds)
    if loc_sample_size and len(unrelated.data) > loc_sample_size:
        base_seed = getattr(config, "seed", 333)
        rng = random.Random(base_seed + len(stub.data))
        unrelated.data = rng.sample(unrelated.data, loc_sample_size)

    ds_old = copy.deepcopy(unrelated)
    ds_new = copy.deepcopy(unrelated)
    for d in (ds_old, ds_new):
        if getattr(d, "config", None) is not None:
            try:
                d.config.continuation_gate = False
            except Exception:
                pass

    target_device = _gpu_device(config)
    config.device = target_device

    # 1) Gold pass — original weights, on GPU
    print(f"[locfix]   gold pass ({len(ds_old.data)} samples) on {target_device}...", flush=True)
    _to_device(model_base, target_device)
    ds_old.set_dataloader()
    ds_old.task_generate(model_base, use_cache=True, compute_tf=False)

    # 2) Apply checkpoint weights in-place
    if editor_name in ("ft", "ft_soft"):
        handle = _backup_and_apply_ft(model_base, ckpt_editor_state, config)
    else:
        handle = _backup_and_apply_mend(model_base, ckpt_editor_state)
    n_backed = len(handle[2])
    print(f"[locfix]   applied {n_backed} weight(s), edited pass...", flush=True)

    # 3) Edited pass — checkpoint weights, same GPU
    ds_new.set_dataloader()
    ds_new.task_generate(model_base, use_cache=True, compute_tf=False)

    # 4) Restore original weights
    _restore_weights(handle)
    _to_device(model_base, "cpu")
    cuda_gc()

    # 5) Score (mirrors locality() scoring)
    def _open_readout(pred):
        spo = pred.get("support_prob_open") or {}
        return {_norm_label(k): float(v) for k, v in spo.items()
                if str(k) != "__OTHER__" and v is not None}

    n = 0
    acc_sum = js_sum = tau_sum = ndcg_sum = 0.0
    js_n = rank_n = 0
    for ex_old, ex_new in zip(ds_old.data, ds_new.data):
        pred_old = ex_old.get("pred") or {}
        pred_new = ex_new.get("pred") or {}
        if not pred_old or not pred_new:
            continue
        n += 1
        old_label = _norm_label(pred_old.get("label", ""))
        new_label = _norm_label(pred_new.get("label", ""))
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

    if n == 0:
        metrics = {"simple_acc": 0.0, "hard_acc": 0.0, "n": 0}
    else:
        acc = acc_sum / n
        metrics = {
            "simple_acc": acc, "hard_acc": acc, "rank_acc": acc, "acc": acc,
            "js_dist": js_sum / js_n if js_n > 0 else None,
            "tau_readout": tau_sum / rank_n if rank_n > 0 else None,
            "ndcg_readout": ndcg_sum / rank_n if rank_n > 0 else None,
            "n": n,
        }

    print(f"[locfix]   loc@{K}: simple_acc={metrics.get('simple_acc'):.4f}  "
          f"n={metrics.get('n')}  ({time.time() - t0:.1f}s)", flush=True)
    return metrics


def _reset_gold(model_base, config=None) -> None:
    """Park the never-wrapped gold model on CPU and free GPU before a new editor.

    Never write config.device here — locality() uses it as the GPU target.
    """
    if hasattr(model_base, "device"):
        model_base.device = torch.device("cpu")
    _move_weights(model_base, "cpu")
    if config is not None:
        config.device = _gpu_device(config)
        if hasattr(model_base, "config"):
            try:
                model_base.config = config
            except Exception:
                pass
    cuda_gc()


def make_editor(config, model_base, editor_name: str):
    """Clone the gold model, then construct the editor on the clone.

    ``model_base`` is never wrapped or weight-edited. Every editor starts from
    that CPU snapshot so GRACE adapters / FT weights / SoftEdit hooks cannot
    leak into the next editor.
    """
    print(f"[locfix] {editor_name}: fresh clone from gold model", flush=True)
    model_new = _cpu_deepcopy(model_base, config)
    gpu = _gpu_device(config)
    config.device = gpu
    print(f"[locfix] {editor_name}: moving clone to {gpu}", flush=True)
    _to_device(model_new, gpu)
    editor = get_editor(config, model_new)
    if hasattr(model_new, "model"):
        editor.generate = model_new.model.generate
    _eval_mode(model_new)
    return editor, model_new


def _release(editor, model_new, model_base) -> None:
    """Drop this editor's clone completely; gold stays on CPU, unwrapped."""
    editor_name = getattr(editor, "log_name", type(editor).__name__) if editor is not None else "?"
    if editor is not None:
        del editor
    if model_new is not None and model_new is not model_base:
        del model_new
    _reset_gold(model_base)
    print(f"[locfix] {editor_name}: released working copy, gold on CPU", flush=True)


# ---------------------------------------------------------------------------
# per-editor strategies
# ---------------------------------------------------------------------------

def _records_complete(records: List[dict], edit_dir: Path, n_errors: int) -> bool:
    """True only when the last JSONL line is the full-set eval (batch_idx >= #errors).

    The COMPLETE marker is not sufficient: an incomplete run's last line is still a
    peek (buggy loc pool). Using it as the shift anchor would be wrong.
    """
    if not records:
        return False
    last_id = _step(max(records, key=_step))
    return last_id >= int(n_errors)


def run_shift(records: List[dict], edit_dir: Path, complete: bool,
              config, model_base, editor_name: str, edit_ds, loc_ds,
              loc_sample_size: int, dry_run: bool) -> List[dict]:
    last = max(records, key=_step)
    last_id = _step(last)
    peeks = [r for r in records if _step(r) < last_id]

    if complete:
        if not peeks:
            print(f"[locfix] {editor_name}: complete but no peek lines; skip", flush=True)
            return records
        last_peek = max(peeks, key=_step)
        full_loc = _loc_acc(last)
        peek_loc = _loc_acc(last_peek)
        if full_loc is None or peek_loc is None:
            print(f"[locfix] {editor_name}: missing Loc on full/last-peek; skip", flush=True)
            return records
        delta = full_loc - peek_loc
        meta = {"strategy": "shift_complete", "delta": delta,
                "full_loc": full_loc, "last_peek_loc": peek_loc,
                "last_peek_k": _step(last_peek)}
        print(f"[locfix] {editor_name}: complete shift  delta={delta:+.4f}  "
              f"(full@{last_id}={full_loc:.4f} - peek@{_step(last_peek)}={peek_loc:.4f})",
              flush=True)
        if dry_run:
            return records
        for rec in peeks:
            _shift_record(rec, delta, meta)
        last["_loc_fix"] = {**meta, "role": "full_eval_unchanged"}
        return records

    # incomplete: one Loc eval on last ckpt
    ckpt = load_ckpt(str(edit_dir))
    if ckpt is None or ckpt.get("editor") is None:
        print(f"[locfix] {editor_name}: incomplete but no ckpt.pt; skip", flush=True)
        return records
    old_loc = _loc_acc(last)
    if dry_run:
        print(f"[locfix] {editor_name}: would eval Loc@{last_id} then shift "
              f"(old_last={old_loc})", flush=True)
        return records

    if editor_name in WEIGHT_ONLY:
        # Fast path: swap weights in-place on gold model, no deepcopy
        print(f"[locfix] {editor_name}: inplace weight-swap eval (no deepcopy)",
              flush=True)
        _eval_mode(model_base)
        metrics = eval_loc_inplace(
            model_base, ckpt.get("editor"), editor_name,
            config, edit_ds, loc_ds, last_id, loc_sample_size,
        )
        _to_device(model_base, "cpu")
        cuda_gc()
    else:
        # Adapter editors (BalancEdit etc.) need the full editor wrapper
        editor, model_new = make_editor(config, model_base, editor_name)
        try:
            editor.load_resume_state(ckpt.get("editor"))
            _freeze_adapters(editor, editor_name)
            _eval_mode(model_new)
            metrics = eval_loc(model_base, model_new, editor, edit_ds, loc_ds,
                                last_id, loc_sample_size)
        finally:
            _release(editor, model_new, model_base)
    new_loc = float(metrics.get("simple_acc", 0.0))
    if old_loc is None:
        print(f"[locfix] {editor_name}: no old Loc to shift from; skip", flush=True)
        return records
    delta = new_loc - old_loc
    meta = {"strategy": "shift_incomplete", "delta": delta,
            "new_loc": new_loc, "old_last_loc": old_loc, "k": last_id}
    print(f"[locfix] {editor_name}: incomplete shift  delta={delta:+.4f}  "
          f"(new={new_loc:.4f} - old={old_loc:.4f})", flush=True)
    for rec in records:
        _shift_record(rec, delta, meta)
    return records


def run_slice(records: List[dict], edit_dir: Path, editor_name: str,
              config, model_base, edit_ds, loc_ds, n_errors: int,
              loc_sample_size: int, overwrite_loc: bool, dry_run: bool) -> List[dict]:
    ckpt = load_ckpt(str(edit_dir))
    if ckpt is None or ckpt.get("editor") is None:
        print(f"[locfix] {editor_name}: no ckpt.pt; skip", flush=True)
        return records
    full_state = ckpt.get("editor") or {}
    last_id = _step(max(records, key=_step))
    N = n_errors if last_id >= n_errors else last_id
    M = _codebook_len(full_state, editor_name)
    print(f"[locfix] {editor_name}: codebook M={M}  N={N}  lines={len(records)}",
          flush=True)
    if dry_run:
        for rec in records:
            K = _step(rec)
            print(f"[locfix]   would slice keep={_n_keep(M, K, N)} at K={K}", flush=True)
        return records

    editor, model_new = make_editor(config, model_base, editor_name)
    try:
        editor.load_resume_state(full_state)
        if hasattr(editor, "rekey") and editor_name in ("grace", "grace_soft"):
            _to_device(model_new, config.device)
            print(f"[locfix] {editor_name}: global rekey on {last_id} samples",
                  flush=True)
            editor.rekey(model_new, edit_ds, n_samples=last_id)
            full_state = editor.save_resume_state()
            M = _codebook_len(full_state, editor_name)
            print(f"[locfix] {editor_name}: rekeyed codebook M={M}", flush=True)

        for rec in records:
            K = _step(rec)
            if rec.get("_loc_fix") and not overwrite_loc:
                print(f"[locfix]   skip K={K} (already patched)", flush=True)
                continue
            sliced = slice_state(full_state, editor_name, K, N)
            editor.load_resume_state(sliced)
            if editor_name in ("reasonedit", "reasonedit_soft"):
                editor.key_dist_threshold = None
                if hasattr(editor, "key_sigmas"):
                    editor.key_sigmas = None
            if editor_name in ("softedit", "softedit_task"):
                editor.key_dist_threshold = None
            _freeze_adapters(editor, editor_name)
            _eval_mode(model_new)
            keep = _n_keep(M, K, N)
            print(f"[locfix] {editor_name}: K={K} keep={keep}/{M}", flush=True)
            metrics = eval_loc(model_base, model_new, editor, edit_ds, loc_ds,
                                K, loc_sample_size)
            _patch_record(rec, metrics, {
                "strategy": "slice", "k": K, "keep": keep, "M": M, "N": N,
            })
            _write_jsonl(edit_dir / "postedit_qa_all.json", records)
    finally:
        _release(editor, model_new, model_base)
    return records


def strategy_of(name: str) -> str:
    if name in SKIP:
        return "skip"
    if name in EXACT_SLICE:
        return "exact_slice"
    if name in PROP_SLICE:
        return "prop_slice"
    if name in SHIFT:
        return "shift"
    return "skip"


def discover_editors(results_root: Path, model_tag: str, dataset: str,
                     only: Optional[List[str]]) -> List[Tuple[str, Path]]:
    found = []
    ee = results_root / "ee"
    if not ee.is_dir():
        return found
    for d in sorted(p for p in ee.iterdir() if p.is_dir()):
        name = d.name.lower()
        if only is not None and name not in only:
            continue
        jsonl = d / model_tag / dataset / "postedit_qa_all.json"
        if jsonl.is_file():
            found.append((name, jsonl.parent))
    return found


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Recompute / shift sequential Loc in place")
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Short VLM name: llava | blip | qwen3_4b")
    parser.add_argument("--dataset_name", type=str, default="vqav2")
    parser.add_argument("--editors", type=str, default=None,
                        help="Comma-separated editor names (default: all discovered except skip)")
    parser.add_argument("--loc_sample_size", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite_loc", action="store_true",
                        help="Re-patch lines that already have _loc_fix")
    parser.add_argument("--split", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.suffix = ""
    args.task = "postedit_qa"
    args.editor = "baseline"
    args.overwrite = False
    args.n_iter = None
    args.max_n_edits = None
    args.max_new_tokens = None
    args.seed = None
    args.ckpt_dir = None
    args.dropout = None
    args.subsample = 0
    args.label_strategy = None

    config = configure_args(args, config_path=args.config)
    config.pred_path = os.path.join(config.pred_dir, f"softqa_{config.experiment.split}.json")
    config.debug = False
    model_tag = (config.model.name.split("/")[-1] or "model").replace(" ", "_")
    dataset = args.dataset_name
    only = [x.strip().lower() for x in args.editors.split(",")] if args.editors else None

    print("=" * 50, flush=True)
    print(f"[locfix] model={args.model_name} ({model_tag})  dataset={dataset}",
          flush=True)
    print(f"[locfix] pred_path={config.pred_path}", flush=True)

    gpu = _gpu_device(config)
    config.device = gpu
    print(f"[locfix] eval device={gpu}  cuda={torch.cuda.is_available()}", flush=True)

    model, edit_ds, loc_ds = find_errors(config)
    n_errors = len(edit_ds.data)
    print(f"[locfix] n_errors={n_errors}  loc_pool={len(loc_ds.data)}", flush=True)

    # Park gold on CPU without touching config.device (shared NestedConfig).
    _move_weights(model, "cpu")
    config.device = gpu
    cuda_gc()
    model_base = model

    results_root = PROJECT_ROOT / "results"
    jobs = discover_editors(results_root, model_tag, dataset, only)
    print(f"[locfix] discovered {len(jobs)} result dirs", flush=True)

    for editor_name, edit_dir in jobs:
        strat = strategy_of(editor_name)
        jsonl = edit_dir / "postedit_qa_all.json"
        records = _read_rows(jsonl)
        complete = _records_complete(records, edit_dir, n_errors)
        last_id = _step(max(records, key=_step)) if records else 0
        print("=" * 50, flush=True)
        print(f"[locfix] {editor_name}: strategy={strat}  complete={complete}  "
              f"lines={len(records)}  last={last_id}  dir={edit_dir}", flush=True)
        _reset_gold(model_base, config)

        if strat == "skip":
            print(f"[locfix] {editor_name}: skip", flush=True)
            continue
        if not records:
            print(f"[locfix] {editor_name}: empty JSONL; skip", flush=True)
            continue
        if not args.overwrite_loc and records and all(r.get("_loc_fix") for r in records):
            print(f"[locfix] {editor_name}: already patched; skip "
                  f"(--overwrite_loc to redo)", flush=True)
            continue

        ed_config = update_config(
            config, editor=editor_name, model_name=args.model_name,
            dataset_name=dataset, task="postedit_qa", split=args.split,
            config_path=args.config,
        )
        ed_config.pred_path = config.pred_path
        ed_config.debug = False
        _bind_config(ed_config, edit_ds, loc_ds, model_base)

        _backup_jsonl(jsonl)

        if strat == "shift":
            records = run_shift(
                records, edit_dir, complete, ed_config, model_base, editor_name,
                edit_ds, loc_ds, args.loc_sample_size, args.dry_run,
            )
            if not args.dry_run:
                _write_jsonl(jsonl, records)
        else:
            records = run_slice(
                records, edit_dir, editor_name, ed_config, model_base,
                edit_ds, loc_ds, n_errors, args.loc_sample_size,
                args.overwrite_loc, args.dry_run,
            )
            if not args.dry_run:
                _write_jsonl(jsonl, records)
        _reset_gold(model_base, config)
        print(f"[locfix] {editor_name}: done (gold reset)", flush=True)

    print("=" * 50, flush=True)
    print("[locfix] all editors finished", flush=True)


if __name__ == "__main__":
    main()
