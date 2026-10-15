"""Patch sequential JSONL locality scores onto the fixed baseline-correct loc_ds.

  SoftEdit     slice codebook[:K] from ckpt.pt, recompute Loc
  IKE(+soft)   slice corpus[:K] from ckpt.pt when present; otherwise rebuild
               the append-only corpus from edit_ds.data[:K] (legacy runs that
               finished before IKE grew save_resume_state)

Skipped: baseline, retrain, FT/MEND/GRACE/AlphaEdit (those reruns already
use loc_ds).

Usage (one job per LLM)::

    python -m rellm.run.locality_recompute --model_name qwen --dataset_name protoqa
"""
import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from rellm import *  # noqa: F401,F403
from rellm.config_utils import configure_args, update_config
from rellm.editors import get_editor
from rellm.metrics.editeval import cuda_gc, locality
from rellm.run.edit_utils import find_errors
from rellm.run.resume import load_ckpt


EXACT_SLICE = {"softedit", "ike", "ike_soft"}


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


def _patch_record(rec: dict, loc_metrics: dict, meta: dict) -> None:
    acc = float(loc_metrics.get("simple_acc", loc_metrics.get("acc", 0.0)))
    rec["locality"] = acc
    rec["locality_hard"] = float(loc_metrics.get("hard_acc", acc))
    rec.setdefault("metrics", {})["locality"] = loc_metrics
    rec["_loc_fix"] = meta


# ---------------------------------------------------------------------------
# codebook slicing (SoftEdit)
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


def _codebook_len(state: dict, editor_name: str = "softedit") -> int:
    state = state or {}
    if editor_name in ("ike", "ike_soft"):
        sents = state.get("corpus_sentences")
        return len(sents) if isinstance(sents, list) else 0
    cb = state.get("codebook")
    return len(cb) if isinstance(cb, list) else 0


def slice_state(state: dict, K: int, N: int, editor_name: str = "softedit") -> dict:
    """Shallow-copied resume state with the codebook/corpus cut to peek K."""
    out = dict(state or {})
    M = _codebook_len(state, editor_name)
    n = _n_keep(M, K, N)
    added = state.get("_added_uids")
    uid_list = list(added) if isinstance(added, (set, list, tuple)) else []

    if editor_name in ("ike", "ike_soft"):
        out["corpus_sentences"] = list((state.get("corpus_sentences") or [])[:n])
        out["corpus_embeddings"] = _slice_list_or_tensor(
            state.get("corpus_embeddings"), n)
        out["_added_uids"] = uid_list[:n]
        return out

    out["codebook"] = list((state.get("codebook") or [])[:n])
    out["key_embs"] = _slice_list_or_tensor(state.get("key_embs"), n)
    out["key_dist_threshold"] = None
    uids = [e.get("uid") for e in out["codebook"]
            if isinstance(e, dict) and e.get("uid") is not None]
    if added is not None:
        out["_added_uids"] = set(uids) if uids else set(uid_list[:n])
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
    if torch.cuda.is_available():
        return torch.device("cuda")
    if config is not None:
        return torch.device(getattr(config, "device", "cpu"))
    return torch.device("cpu")


def _move_weights(model, device) -> None:
    """Move tensors only. Do not write model.config.device (shared NestedConfig)."""
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
    """Not used: TQAModel is unpicklable (``get_preprocess`` local fn)."""
    raise RuntimeError(
        "TQAModel cannot be torch.save'd. Retrieval locfix wraps the gold model."
    )


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


def _pin_cuda(*objs) -> torch.device:
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


def _reset_gold(model_base, config=None) -> None:
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
    """Wrap the gold TQAModel in place.

    IKE/SoftEdit do not keep weight edits (SoftEdit restores ΔW via
    ``pregen_arm(None)``), matching sequential ``_preedit_snapshot``.
    ``torch.save(TQAModel)`` fails on ``get_preprocess``'s local function.
    """
    gpu = _gpu_device(config)
    config.device = gpu
    print(f"[locfix] {editor_name}: wrap gold model on {gpu} (no clone)",
          flush=True)
    _to_device(model_base, gpu)
    editor = get_editor(config, model_base)
    if hasattr(model_base, "model"):
        editor.generate = model_base.model.generate
    _eval_mode(model_base)
    return editor, model_base


def _release(editor, model_new, model_base) -> None:
    editor_name = getattr(editor, "log_name", type(editor).__name__) if editor is not None else "?"
    if editor is not None and getattr(editor, "_patch_backup", None) is not None:
        try:
            from rellm.editors.softedit import train as se_train
            se_train.restore_delta(editor.head, editor._patch_backup)
            editor._patch_backup = None
        except Exception:
            pass
    if editor is not None:
        del editor
    if model_new is not None and model_new is not model_base:
        del model_new
    _reset_gold(model_base)
    print(f"[locfix] {editor_name}: released wrapper, gold on CPU", flush=True)


def _ike_reset(editor) -> None:
    editor.corpus_sentences = []
    editor.corpus_embeddings = None
    editor._added_uids = set()


def _ike_add_prefix(editor, edit_ds, k_from: int, k_to: int) -> None:
    stub = copy.copy(edit_ds)
    stub.data = list(edit_ds.data[k_from:k_to])
    if stub.data:
        editor.edit(editor.config, edit_ds=stub)


# ---------------------------------------------------------------------------
# per-editor slice
# ---------------------------------------------------------------------------

def run_softedit_slice(records: List[dict], edit_dir: Path, editor_name: str,
                       config, model_base, edit_ds, loc_ds, n_errors: int,
                       loc_sample_size: int, overwrite_loc: bool,
                       dry_run: bool) -> List[dict]:
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
            print(f"[locfix]   would slice keep={_n_keep(M, K, N)} at K={K}",
                  flush=True)
        return records

    editor, model_new = make_editor(config, model_base, editor_name)
    try:
        jsonl = edit_dir / "postedit_qa_all.json"
        for rec in records:
            K = _step(rec)
            if rec.get("_loc_fix") and not overwrite_loc:
                print(f"[locfix]   skip K={K} (already patched)", flush=True)
                continue
            sliced = slice_state(full_state, K, N, editor_name)
            editor.load_resume_state(sliced)
            if editor_name == "softedit":
                editor.key_dist_threshold = None
            keep = _n_keep(M, K, N)
            print(f"[locfix] {editor_name}: K={K} keep={keep}/{M}", flush=True)
            metrics = eval_loc(model_base, model_new, editor, edit_ds, loc_ds,
                               K, loc_sample_size)
            _patch_record(rec, metrics, {
                "strategy": "slice", "k": K, "keep": keep, "M": M, "N": N,
            })
            _write_jsonl(jsonl, records)
    finally:
        _release(editor, model_new, model_base)
    return records


def _ike_ckpt_state(edit_dir: Path):
    ckpt = load_ckpt(str(edit_dir))
    if ckpt is None:
        return None
    state = ckpt.get("editor") or {}
    if state.get("corpus_sentences"):
        return state
    return None


def run_ike_slice(records: List[dict], edit_dir: Path, editor_name: str,
                  config, model_base, edit_ds, loc_ds, n_errors: int,
                  loc_sample_size: int, overwrite_loc: bool,
                  dry_run: bool) -> List[dict]:
    """Slice IKE corpus from ckpt.pt, or rebuild from edit_ds[:K] if no ckpt."""
    full_state = _ike_ckpt_state(edit_dir)
    if full_state is not None:
        print(f"[locfix] {editor_name}: slicing corpus from ckpt.pt", flush=True)
        return run_softedit_slice(
            records, edit_dir, editor_name, config, model_base, edit_ds, loc_ds,
            n_errors, loc_sample_size, overwrite_loc, dry_run,
        )

    ordered = sorted(records, key=_step)
    print(f"[locfix] {editor_name}: no ckpt.pt; rebuild corpus from edit prefix  "
          f"lines={len(ordered)}", flush=True)
    if dry_run:
        for rec in ordered:
            print(f"[locfix]   would rebuild corpus[:{_step(rec)}]", flush=True)
        return records

    editor, model_new = make_editor(config, model_base, editor_name)
    try:
        jsonl = edit_dir / "postedit_qa_all.json"
        filled_to = 0
        _ike_reset(editor)
        for rec in ordered:
            K = _step(rec)
            if K > filled_to:
                _ike_add_prefix(editor, edit_ds, filled_to, K)
                filled_to = K
            keep = len(getattr(editor, "corpus_sentences", []) or [])
            if rec.get("_loc_fix") and not overwrite_loc:
                print(f"[locfix]   skip K={K} (already patched, corpus={keep})",
                      flush=True)
                continue
            print(f"[locfix] {editor_name}: K={K} corpus={keep}", flush=True)
            metrics = eval_loc(model_base, model_new, editor, edit_ds, loc_ds,
                               K, loc_sample_size)
            _patch_record(rec, metrics, {
                "strategy": "prefix_rebuild", "k": K, "keep": keep,
            })
            _write_jsonl(jsonl, records)
    finally:
        _release(editor, model_new, model_base)
    return records


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


def _cli_ns(args):
    """Fill fields configure_args reads besides the argparse ones."""
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
    args.task_dir = None
    args.edit_dir = None
    args.pred_dir = None
    args.pred_postedit_dir = None
    return args


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Recompute sequential Loc in place (SoftEdit + IKE)")
    parser.add_argument("--config", type=str, default="rellm/config/config.yaml")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Short LLM name: qwen | llama | mistral")
    parser.add_argument("--dataset_name", type=str, default="protoqa")
    parser.add_argument("--editors", type=str, default="softedit,ike,ike_soft",
                        help="Comma-separated editor names")
    parser.add_argument("--loc_sample_size", type=int, default=100)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite_loc", action="store_true",
                        help="Re-patch lines that already have _loc_fix")
    parser.add_argument("--split", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--reject_threshold_pct", type=float, default=10.0,
        help="SoftEdit L2 reject percentile. Pin to the original run "
             "(jobs used 10; the live default in softedit.py may have drifted).",
    )
    args = _cli_ns(parser.parse_args())

    config = configure_args(args, config_path=args.config)
    config.pred_path = os.path.join(
        config.pred_dir, f"softqa_{config.experiment.split}.json")
    config.debug = False
    model_tag = (config.model.name.split("/")[-1] or "model").replace(" ", "_")
    dataset = args.dataset_name
    only = [x.strip().lower() for x in args.editors.split(",") if x.strip()]

    print("=" * 50, flush=True)
    print(f"[locfix] model={args.model_name} ({model_tag})  dataset={dataset}",
          flush=True)
    print(f"[locfix] editors={only}  pred_path={config.pred_path}", flush=True)

    gpu = _gpu_device(config)
    config.device = gpu
    print(f"[locfix] eval device={gpu}  cuda={torch.cuda.is_available()}",
          flush=True)

    model, edit_ds, loc_ds = find_errors(config)
    n_errors = len(edit_ds.data)
    print(f"[locfix] n_errors={n_errors}  loc_pool={len(loc_ds.data)}",
          flush=True)

    _move_weights(model, "cpu")
    config.device = gpu
    cuda_gc()
    model_base = model

    results_root = PROJECT_ROOT / "results"
    jobs = discover_editors(results_root, model_tag, dataset, only)
    print(f"[locfix] discovered {len(jobs)} result dirs", flush=True)

    for editor_name, edit_dir in jobs:
        jsonl = edit_dir / "postedit_qa_all.json"
        records = _read_rows(jsonl)
        last_id = _step(max(records, key=_step)) if records else 0
        print("=" * 50, flush=True)
        print(f"[locfix] {editor_name}: lines={len(records)}  last={last_id}  "
              f"dir={edit_dir}", flush=True)
        _reset_gold(model_base, config)

        if editor_name not in EXACT_SLICE:
            print(f"[locfix] {editor_name}: skip (not exact-slice)", flush=True)
            continue
        if not records:
            print(f"[locfix] {editor_name}: empty JSONL; skip", flush=True)
            continue
        if not args.overwrite_loc and all(r.get("_loc_fix") for r in records):
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
        if editor_name == "softedit":
            ed_config.editor.reject_threshold_pct = float(
                args.reject_threshold_pct)
        _bind_config(ed_config, edit_ds, loc_ds, model_base)
        _backup_jsonl(jsonl)

        if editor_name == "softedit":
            records = run_softedit_slice(
                records, edit_dir, editor_name, ed_config, model_base,
                edit_ds, loc_ds, n_errors, args.loc_sample_size,
                args.overwrite_loc, args.dry_run,
            )
        else:
            records = run_ike_slice(
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
