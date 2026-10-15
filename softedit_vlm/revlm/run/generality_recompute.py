"""Recompute sequential SoftEdit metrics after changing the apply-time gate.

Does not re-edit. Slices ``codebook[:K]`` from ``ckpt.pt`` (same as locfix) and
re-runs the full ``editeval`` board (Rel, T-Gen, I-Gen, Soft-Gen / SLoc, Blend,
Loc). Apply-time knobs come from the live ``SoftEdit.__init__`` defaults in
``revlm/editors/softedit/softedit.py`` (no CLI overrides).

Peek protocol matches sequential: Acc/Gen/S-Gen on subsample 50 (seed 333);
Loc from the frozen baseline-correct pool (100). Last line uses the full so-far
set only when ``K >= n_errors``.

Default writes the sequential JSONL in place (``_gen_fix`` + ``.genfix.bak``).
``--dry_run`` prints the peek plan and does not touch GPU or files.

Usage::

    python -m revlm.run.generality_recompute --model_name blip --dataset_name vqav2
    python -m revlm.run.generality_recompute --model_name blip --dataset_name vqav2 --dry_run
    python -m revlm.run.generality_recompute --model_name blip --ks 200,2000
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from revlm import *  # noqa: F401,F403
from revlm.config_utils import configure_args, update_config
from revlm.editors import get_editor
from revlm.metrics.editeval import cuda_gc, editeval, normalize_skip
from revlm.metrics.utils.t_gen import get_t_gen_input
from revlm.metrics.utils.i_gen import get_i_gen_input
from revlm.metrics.utils.soft_gen import get_soft_gen_input_by_strength
from revlm.metrics.utils.blend_gen import (
    ALL_STRENGTHS as BLEND_ALL_STRENGTHS,
    get_blend_gen_input_by_strength,
)
from revlm.run.edit_utils import find_errors
from revlm.run.resume import load_ckpt
from revlm.run.locality_recompute import (
    _bind_config,
    _codebook_len,
    _edit_stub,
    _eval_mode,
    _gpu_device,
    _move_weights,
    _n_keep,
    _pin_cuda,
    _read_rows,
    _reset_gold,
    _step,
    _to_device,
    _write_jsonl,
    slice_state,
)


EDITOR_NAME = "softedit"
DEFAULT_SKIP = ()
METRIC_KEYS = (
    "reliability",
    "text_generality",
    "image_generality",
    "soft_generality",
    "blend_generality",
    "locality",
)
PEEK_SUBSAMPLE = 50
PEEK_SEED = 333
LOC_SAMPLE_SIZE = 100


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------

def _backup_jsonl(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".genfix.bak")
    if path.exists() and not bak.exists():
        shutil.copy2(path, bak)
        print(f"[genfix] backed up {path.name} -> {bak.name}", flush=True)


def _acc(block: Any) -> Optional[float]:
    if isinstance(block, (int, float)):
        return float(block)
    if not isinstance(block, dict):
        return None
    v = block.get("simple_acc", block.get("acc"))
    return float(v) if isinstance(v, (int, float)) else None


def _hard_acc(block: Any) -> Optional[float]:
    if not isinstance(block, dict):
        return None
    v = block.get("hard_acc")
    return float(v) if isinstance(v, (int, float)) else None


def _gen_block(rec: dict, name: str) -> dict:
    return (rec.get("metrics") or {}).get(name) or {}


def _patch_record(rec: dict, result: dict, meta: dict) -> None:
    nested = result.get("metrics") or {}
    rec.setdefault("metrics", {})
    for key in METRIC_KEYS:
        block = nested.get(key)
        if block is not None:
            rec["metrics"][key] = block
    if result.get("locality") is not None:
        rec["locality"] = result["locality"]
        rec["locality_hard"] = result.get("locality_hard")
    rec["_gen_fix"] = meta


def _summarize_block(block: Any) -> dict:
    return {
        "acc": _acc(block),
        "hard_acc": _hard_acc(block),
        "n": (block.get("n") if isinstance(block, dict) else None),
    }


def _align(block: Any) -> Optional[float]:
    if not isinstance(block, dict):
        return None
    v = block.get("js_dist_gate")
    return (1.0 - float(v)) if isinstance(v, (int, float)) else None


def _sloc_align(soft: Any) -> Optional[float]:
    if not isinstance(soft, dict):
        return None
    return _align(soft.get("1.0") or {})


# ---------------------------------------------------------------------------
# editor wrap (gold in place — SoftEdit restores ΔW after each generate)
# ---------------------------------------------------------------------------

def wrap_softedit(config, model_base):
    """Wrap the gold VQAModel in place. SoftEdit does not keep weight edits."""
    gpu = _gpu_device(config)
    config.device = gpu
    print(f"[genfix] softedit: wrap gold model on {gpu} (no clone)", flush=True)
    _to_device(model_base, gpu)
    editor = get_editor(config, model_base)
    if hasattr(model_base, "model"):
        editor.generate = model_base.model.generate
    _eval_mode(model_base)
    return editor, model_base


def _unwrap_softedit(editor) -> None:
    if editor is None:
        return
    try:
        from revlm.editors.softedit import train as se_train
        if getattr(editor, "_patch_backup", None) is not None:
            se_train.restore_delta(editor.head, editor._patch_backup)
            editor._patch_backup = None
    except Exception:
        pass
    hook = getattr(editor, "_key_hook", None)
    if hook is not None:
        try:
            hook.remove()
        except Exception:
            pass
        editor._key_hook = None


def close(ctx: dict) -> None:
    """Drop the SoftEdit wrapper and park gold on CPU. Safe to call twice."""
    editor = ctx.get("editor")
    model_base = ctx.get("model_base")
    _unwrap_softedit(editor)
    if editor is not None:
        ctx["editor"] = None
        del editor
    if model_base is not None:
        _reset_gold(model_base, ctx.get("config"))
    print("[genfix] released wrapper, gold on CPU", flush=True)


def _live_gate(editor) -> dict:
    return {
        "reject_threshold_pct": float(getattr(editor, "reject_threshold_pct", 100.0)),
        "judge_paraphrase": bool(getattr(editor, "judge_paraphrase", True)),
        "key_mode": str(getattr(editor, "key_mode", "")),
        "pool_method": str(getattr(editor, "pool_method", "")),
    }


def _log_gate(editor) -> None:
    g = _live_gate(editor)
    print(
        f"[genfix] apply-time from SoftEdit: "
        f"reject_threshold_pct={g['reject_threshold_pct']}  "
        f"judge_paraphrase={g['judge_paraphrase']}  "
        f"key_mode={g['key_mode']!r}  pool_method={g['pool_method']!r}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# peek eval
# ---------------------------------------------------------------------------

def _is_last_full(K: int, n_errors: int) -> bool:
    return int(K) >= int(n_errors)


def _find_rec(records: List[dict], K: int) -> Optional[dict]:
    for rec in records:
        if _step(rec) == int(K):
            return rec
    return None


def _ordered_peeks(records: List[dict], ks: Optional[Sequence[int]] = None,
                   max_peeks: Optional[int] = None) -> List[dict]:
    if ks:
        want = {int(k) for k in ks}
        out = [r for r in records if _step(r) in want]
        out.sort(key=_step)
        missing = want - {_step(r) for r in out}
        if missing:
            print(f"[genfix] no JSONL line for K={sorted(missing)}", flush=True)
        return out
    out = sorted(records, key=_step)
    if max_peeks is not None:
        out = out[: int(max_peeks)]
    return out


def _snapshot(rec: Optional[dict], nested: Optional[dict] = None) -> dict:
    src = nested if nested is not None else ((rec or {}).get("metrics") or {})
    rel = src.get("reliability") if isinstance(src, dict) else None
    tgen = src.get("text_generality") if isinstance(src, dict) else None
    igen = src.get("image_generality") if isinstance(src, dict) else None
    loc = src.get("locality") if isinstance(src, dict) else None
    soft = src.get("soft_generality") if isinstance(src, dict) else None
    return {
        "rel": _summarize_block(rel),
        "tgen": _summarize_block(tgen),
        "igen": _summarize_block(igen),
        "loc": _summarize_block(loc),
        "sgen_t": {"align": _align(tgen)},
        "sgen_i": {"align": _align(igen)},
        "sloc": {"align": _sloc_align(soft)},
    }


def eval_peek(
    ctx: dict,
    rec_or_k: Union[int, dict],
    skip: Sequence[str] = DEFAULT_SKIP,
    dump_preds_dir: Optional[str] = None,
) -> dict:
    """Full sequential editeval at one peek. Does not write JSONL."""
    editor = ctx.get("editor")
    if editor is None:
        raise RuntimeError("eval_peek: call setup() first (or ctx was closed)")
    rec = rec_or_k if isinstance(rec_or_k, dict) else _find_rec(ctx["records"], int(rec_or_k))
    K = _step(rec) if rec is not None else int(rec_or_k)
    n_errors = int(ctx["n_errors"])
    M = int(ctx["M"])
    N = int(ctx["N"])
    keep = _n_keep(M, K, N)
    is_last = _is_last_full(K, n_errors)
    subsample = None if is_last else PEEK_SUBSAMPLE
    skip_set = normalize_skip(skip)
    loc_sample_size = int(ctx.get("loc_sample_size", LOC_SAMPLE_SIZE))

    sliced = slice_state(ctx["full_state"], EDITOR_NAME, K, N)
    editor.load_resume_state(sliced)
    editor.key_dist_threshold = None

    config = ctx["config"]
    edit_ds = ctx["edit_ds"]
    loc_ds = ctx["loc_ds"]
    model = ctx["model_base"]
    _pin_cuda(edit_ds, loc_ds, model, config)
    _eval_mode(model)

    stub = _edit_stub(edit_ds, K)
    dataset_name = config.experiment.dataset_name
    related_texts = (
        get_t_gen_input(dataset_name, stub) if "text_generality" not in skip_set else {}
    )
    related_images = (
        get_i_gen_input(dataset_name, stub, k=2) if "image_generality" not in skip_set else {}
    )
    soft_gen_images = (
        get_soft_gen_input_by_strength(dataset_name, stub)
        if "soft_generality" not in skip_set else None
    )
    blend_gen_images = (
        get_blend_gen_input_by_strength(
            dataset_name, stub, strengths=BLEND_ALL_STRENGTHS,
        )
        if "blend_generality" not in skip_set else None
    )

    old = _snapshot(rec) if rec else None
    gate = _live_gate(editor)
    print("=" * 50, flush=True)
    print(
        f"[genfix] K={K} keep={keep}/{M}  last_full={is_last}  "
        f"subsample={subsample}  loc_n={loc_sample_size}  "
        f"gate={gate['reject_threshold_pct']}",
        flush=True,
    )
    if old is not None:
        print(
            f"[genfix]   old rel={old['rel']['acc']}  t-gen={old['tgen']['acc']}  "
            f"i-gen={old['igen']['acc']}  loc={old['loc']['acc']}  "
            f"sloc={old['sloc']['align']}",
            flush=True,
        )

    # Full-set peek: pre-apply the editor and pre-generate so that reliability
    # (which relies on cached predictions via use_cache=True inside editeval)
    # sees the editor-armed outputs.  Subsampled peeks are handled inside
    # editeval's own subsample branch.  This mirrors the pre-apply in
    # edit_utils.edit_n_eval_seq for the final batch.
    if is_last and hasattr(editor, "apply_to_dataset"):
        editor.apply_to_dataset(stub)
        stub.task_generate(model, use_cache=False)

    t0 = time.time()
    result = editeval(
        model, model, stub, editor,
        related_texts, related_images,
        soft_gen_images=soft_gen_images,
        blend_gen_images=blend_gen_images,
        unrelated_ds=loc_ds,
        loc_sample_size=loc_sample_size,
        edit_subsample_size=subsample,
        dump_preds_dir=dump_preds_dir,
        skip=skip,
    )
    _pin_cuda(edit_ds, loc_ds, model, config)
    dt = time.time() - t0

    new = _snapshot(None, result.get("metrics") or {})
    meta = {
        "k": K,
        "keep": keep,
        "M": M,
        "N": N,
        "edit_subsample_size": subsample,
        "loc_sample_size": loc_sample_size,
        "peek_seed": PEEK_SEED,
        "seconds": round(dt, 1),
        "skip": sorted(skip_set),
        **gate,
    }
    out = {
        "k": K,
        "keep": keep,
        "is_last_full": is_last,
        "old": old,
        "new": new,
        "old_tgen": (old or {}).get("tgen"),
        "old_igen": (old or {}).get("igen"),
        "new_tgen": new.get("tgen"),
        "new_igen": new.get("igen"),
        "meta": meta,
        "result": result,
        "rec": rec,
    }
    print(
        f"[genfix]   new rel={new['rel']['acc']}  t-gen={new['tgen']['acc']}  "
        f"i-gen={new['igen']['acc']}  loc={new['loc']['acc']}  "
        f"sloc={new['sloc']['align']}  ({dt:.1f}s)",
        flush=True,
    )
    ctx.setdefault("results", []).append(out)
    return out


def peek_table(ctx: dict) -> List[dict]:
    """Flat rows from ``ctx['results']`` for a notebook DataFrame."""
    rows = []
    for item in ctx.get("results") or []:
        old = item.get("old") or {}
        new = item.get("new") or {}
        rows.append({
            "k": item["k"],
            "keep": item["keep"],
            "gate": item["meta"].get("reject_threshold_pct"),
            "rel_old": (old.get("rel") or {}).get("acc"),
            "rel_new": (new.get("rel") or {}).get("acc"),
            "tgen_old": (old.get("tgen") or {}).get("acc"),
            "tgen_new": (new.get("tgen") or {}).get("acc"),
            "igen_old": (old.get("igen") or {}).get("acc"),
            "igen_new": (new.get("igen") or {}).get("acc"),
            "loc_old": (old.get("loc") or {}).get("acc"),
            "loc_new": (new.get("loc") or {}).get("acc"),
            "sgen_t_old": (old.get("sgen_t") or {}).get("align"),
            "sgen_t_new": (new.get("sgen_t") or {}).get("align"),
            "sgen_i_old": (old.get("sgen_i") or {}).get("align"),
            "sgen_i_new": (new.get("sgen_i") or {}).get("align"),
            "sloc_old": (old.get("sloc") or {}).get("align"),
            "sloc_new": (new.get("sloc") or {}).get("align"),
            "seconds": item["meta"].get("seconds"),
        })
    return rows


def run_peeks(
    ctx: dict,
    ks: Optional[Sequence[int]] = None,
    max_peeks: Optional[int] = None,
    write_inplace: bool = True,
    overwrite_gen: bool = False,
    out_jsonl: Optional[Union[str, Path]] = None,
    skip: Sequence[str] = DEFAULT_SKIP,
    dump_preds_dir: Optional[str] = None,
    dry_run: bool = False,
) -> List[dict]:
    """Eval selected peeks. ``write_inplace`` patches results/ee unless dry_run."""
    records = ctx["records"]
    peeks = _ordered_peeks(records, ks=ks, max_peeks=max_peeks)
    jsonl = Path(ctx["edit_dir"]) / "postedit_qa_all.json"
    if dry_run:
        M, N = int(ctx["M"]), int(ctx["N"])
        for rec in peeks:
            K = _step(rec)
            print(f"[genfix]   would eval K={K} keep={_n_keep(M, K, N)}/{M}",
                  flush=True)
        return []
    wrote_backup = False
    outs = []
    for rec in peeks:
        K = _step(rec)
        if rec.get("_gen_fix") and not overwrite_gen:
            print(f"[genfix]   skip K={K} (already patched; overwrite_gen=True to redo)",
                  flush=True)
            continue
        item = eval_peek(ctx, rec, skip=skip, dump_preds_dir=dump_preds_dir)
        outs.append(item)
        if write_inplace and item["rec"] is not None:
            if not wrote_backup:
                _backup_jsonl(jsonl)
                wrote_backup = True
            _patch_record(item["rec"], item["result"], item["meta"])
            _write_jsonl(jsonl, records)
            print(f"[genfix]   wrote {jsonl} (K={K})", flush=True)
    if out_jsonl:
        dest = Path(out_jsonl)
        dest.parent.mkdir(parents=True, exist_ok=True)
        patched = [dict(r) for r in records]
        by_k = {_step(r): r for r in patched}
        for item in outs:
            rec = by_k.get(item["k"])
            if rec is not None:
                _patch_record(rec, item["result"], item["meta"])
        _write_jsonl(dest, patched)
        print(f"[genfix] wrote copy {dest}", flush=True)
    return outs


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

def _cli_ns(args):
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


def setup(
    model_name: str,
    dataset_name: str = "vqav2",
    config_path: str = "revlm/config/config.yaml",
    split: str = "all",
    batch_size: int = 1,
) -> dict:
    """Load gold model, loc/edit pools, SoftEdit codebook. No JSONL writes.

    Apply-time knobs follow ``softedit.py``.
    """
    args = argparse.Namespace(
        config=config_path,
        model_name=model_name,
        dataset_name=dataset_name,
        split=split,
        batch_size=batch_size,
    )
    args = _cli_ns(args)
    config = configure_args(args, config_path=config_path)
    config.pred_path = os.path.join(
        config.pred_dir, f"softqa_{config.experiment.split}.json"
    )
    config.debug = False
    model_tag = (config.model.name.split("/")[-1] or "model").replace(" ", "_")
    gpu = _gpu_device(config)
    config.device = gpu

    print("=" * 50, flush=True)
    print(f"[genfix] model={model_name} ({model_tag})  dataset={dataset_name}",
          flush=True)
    print(f"[genfix] pred_path={config.pred_path}", flush=True)
    print(f"[genfix] eval device={gpu}  cuda={torch.cuda.is_available()}",
          flush=True)

    model, edit_ds, loc_ds = find_errors(config)
    n_errors = len(edit_ds.data)
    print(f"[genfix] n_errors={n_errors}  loc_pool={len(loc_ds.data)}", flush=True)

    _move_weights(model, "cpu")
    config.device = gpu
    cuda_gc()

    ed_config = update_config(
        config, editor=EDITOR_NAME, model_name=model_name,
        dataset_name=dataset_name, task="postedit_qa", split=split,
        config_path=config_path,
    )
    ed_config.pred_path = config.pred_path
    ed_config.debug = False
    _bind_config(ed_config, edit_ds, loc_ds, model)

    edit_dir = Path(ed_config.edit_dir)
    jsonl = edit_dir / "postedit_qa_all.json"
    if not jsonl.is_file():
        raise FileNotFoundError(f"[genfix] no sequential JSONL at {jsonl}")
    ckpt = load_ckpt(str(edit_dir))
    if ckpt is None or ckpt.get("editor") is None:
        raise FileNotFoundError(f"[genfix] no ckpt.pt under {edit_dir / '_resume'}")

    records = _read_rows(jsonl)
    if not records:
        raise RuntimeError(f"[genfix] empty JSONL: {jsonl}")
    full_state = ckpt.get("editor") or {}
    last_id = _step(max(records, key=_step))
    N = n_errors if last_id >= n_errors else last_id
    M = _codebook_len(full_state, EDITOR_NAME)
    print(
        f"[genfix] codebook M={M}  N={N}  lines={len(records)}  last={last_id}  "
        f"dir={edit_dir}",
        flush=True,
    )

    editor, model_new = wrap_softedit(ed_config, model)
    editor.load_resume_state(full_state)
    editor.key_dist_threshold = None
    _log_gate(editor)

    return {
        "config": ed_config,
        "model_base": model,
        "model_new": model_new,
        "editor": editor,
        "edit_ds": edit_ds,
        "loc_ds": loc_ds,
        "records": records,
        "edit_dir": edit_dir,
        "jsonl": jsonl,
        "full_state": full_state,
        "n_errors": n_errors,
        "M": M,
        "N": N,
        "last_id": last_id,
        "model_name": model_name,
        "model_tag": model_tag,
        "dataset_name": dataset_name,
        "loc_sample_size": LOC_SAMPLE_SIZE,
        "results": [],
    }


def run_genfix(
    model_name: str,
    dataset_name: str = "vqav2",
    write_inplace: bool = True,
    overwrite_gen: bool = False,
    max_peeks: Optional[int] = None,
    ks: Optional[Sequence[int]] = None,
    out_jsonl: Optional[Union[str, Path]] = None,
    skip: Sequence[str] = DEFAULT_SKIP,
    dump_preds_dir: Optional[str] = None,
    dry_run: bool = False,
    config_path: str = "revlm/config/config.yaml",
    split: str = "all",
    batch_size: int = 1,
) -> dict:
    """End-to-end: setup, eval peeks, close. Returns the ctx (results filled)."""
    ctx = setup(
        model_name, dataset_name=dataset_name,
        config_path=config_path, split=split, batch_size=batch_size,
    )
    try:
        run_peeks(
            ctx, ks=ks, max_peeks=max_peeks,
            write_inplace=write_inplace, overwrite_gen=overwrite_gen,
            out_jsonl=out_jsonl, skip=skip, dump_preds_dir=dump_preds_dir,
            dry_run=dry_run,
        )
    finally:
        close(ctx)
    return ctx


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Recompute sequential SoftEdit full editeval (apply-time gate)"
    )
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Short VLM name: llava | blip | qwen3_4b")
    parser.add_argument("--dataset_name", type=str, default="vqav2")
    parser.add_argument("--max_peeks", type=int, default=None,
                        help="Only the first N JSONL peeks (by K).")
    parser.add_argument("--ks", type=str, default=None,
                        help="Comma-separated peek Ks, e.g. 200,400")
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print peek plan only; no GPU eval and no JSONL writes.",
    )
    parser.add_argument("--overwrite_gen", action="store_true",
                        help="Re-patch lines that already have _gen_fix")
    parser.add_argument(
        "--out_jsonl", type=str, default=None,
        help="Also write a patched copy here (still writes results/ee unless "
             "--dry_run).",
    )
    parser.add_argument(
        "--dump_dir", type=str, default=None,
        help="Optional pred dump dir (default: none, so original dumps stay).",
    )
    parser.add_argument(
        "--skip", type=str, default="",
        help="Comma-separated metric sets to skip (default: none / full eval).",
    )
    parser.add_argument("--split", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    ks = [int(x) for x in args.ks.split(",")] if args.ks else None
    skip = tuple(x.strip() for x in args.skip.split(",") if x.strip())
    if args.dry_run:
        print("[genfix] dry_run: no GPU, no JSONL writes", flush=True)
    else:
        print("[genfix] WRITE_INPLACE: will patch results/ee JSONL", flush=True)
    run_genfix(
        args.model_name,
        dataset_name=args.dataset_name,
        write_inplace=not args.dry_run,
        overwrite_gen=args.overwrite_gen,
        max_peeks=args.max_peeks,
        ks=ks,
        out_jsonl=args.out_jsonl,
        skip=skip,
        dump_preds_dir=args.dump_dir,
        dry_run=args.dry_run,
        config_path=args.config,
        split=args.split,
        batch_size=args.batch_size,
    )
    print("=" * 50, flush=True)
    print("[genfix] done", flush=True)


if __name__ == "__main__":
    main()
