"""Recompute sequential SoftEdit editeval with auto ``reject_threshold_pct``.

Does not re-edit. Never writes the original JSONL. Scores go to a sidecar::

    <edit_dir>/postedit_qa_all.auto.json

so you can compare against the current ``postedit_qa_all.json`` and decide
whether to keep auto.

Protocol: load complete peeks from the original JSONL. The sidecar holds
only finished auto peeks (JSONL, one K per line). After each peek, append
that line. Incomplete last lines are skipped. A later resubmit continues
from original Ks that are not yet in the sidecar.

Usage::

    python -m revlm.run.auto_threshold_recompute --model_name blip --dataset_name vqav2
    python -m revlm.run.auto_threshold_recompute --model_name blip --dry_run
    python -m revlm.run.auto_threshold_recompute --model_name blip --ks 200,400
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from revlm.editors.softedit.softedit import _AUTO_FIRST, _auto_calib_n
from revlm.run import generality_recompute as genfix
from revlm.run.generality_recompute import (  # noqa: E402
    close,
    eval_peek,
    setup as _setup,
    _ordered_peeks,
    _patch_record,
    _step,
)
from revlm.run.locality_recompute import _n_keep, _write_jsonl


OUT_NAME = "postedit_qa_all.auto.json"
TAG = "autofix"


def _read_complete_rows(path: Path) -> list:
    """Parse JSONL/JSON, skipping a half-written last line (live sequential jobs)."""
    if not path.is_file():
        return []
    text = path.read_text().strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]
    except Exception:
        pass
    rows, skipped = [], 0
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except Exception:
            skipped += 1
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    if skipped:
        print(f"[{TAG}] skipped {skipped} incomplete line(s) in {path.name}",
              flush=True)
    return rows


genfix._read_rows = _read_complete_rows


def _auto_by_k(rows: list) -> dict:
    return {_step(r): r for r in rows if r.get("_auto_fix")}


def _append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _prepare_sidecar(original: list, dest: Path, overwrite: bool = False,
                     compact: bool = True) -> set:
    """Keep only ``_auto_fix`` rows in the sidecar. Return done Ks."""
    sidecar = _read_complete_rows(dest)
    done_by_k = _auto_by_k(sidecar)
    n_unfixed = sum(1 for r in sidecar if not r.get("_auto_fix"))
    if overwrite:
        done_by_k = {}
        if dest.is_file():
            dest.unlink()
        n_unfixed = len(sidecar)
    elif compact and dest.is_file() and (n_unfixed or len(sidecar) != len(done_by_k)):
        if done_by_k:
            _write_jsonl(dest, [done_by_k[k] for k in sorted(done_by_k)])
        elif dest.is_file():
            dest.unlink()
        print(f"[{TAG}] compacted sidecar: dropped {n_unfixed} unfixed peek(s)",
              flush=True)

    pending = [r for r in original if _step(r) not in done_by_k]
    start_k = _step(pending[0]) if pending else None
    print(
        f"[{TAG}] original complete peeks={len(original)}  "
        f"auto-done={len(done_by_k)}  pending={len(pending)}"
        + (f"  start K={start_k}" if start_k is not None else "  nothing pending"),
        flush=True,
    )
    return set(done_by_k)


def _jsonable(obj):
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _enable_auto(ctx: dict, auto_max_drop: Optional[int] = None,
                 auto_mode: str = "auto100") -> None:
    """Flip the live editor into auto / auto100; re-bind images after each slice."""
    mode = str(auto_mode or "auto100").strip().lower()
    if mode not in ("auto", "auto100"):
        raise ValueError(f"auto_mode must be 'auto' or 'auto100', got {auto_mode!r}")
    editor = ctx["editor"]
    editor._reject_auto = True
    editor._auto_mode = mode
    editor.reject_threshold_pct = 99.0
    editor._auto_tuned_at = None
    editor.key_dist_threshold = None
    if auto_max_drop is not None:
        editor.auto_max_drop = max(0, int(auto_max_drop))
    drop = int(getattr(editor, "auto_max_drop", 5))
    orig = editor.load_resume_state

    def _load(state):
        at, pct, saved_mode = (
            editor._auto_tuned_at, editor.reject_threshold_pct, editor._auto_mode)
        orig(state)
        editor._reject_auto = True
        editor._auto_mode = saved_mode
        C = _auto_calib_n(len(editor.codebook), saved_mode)
        if saved_mode == "auto100":
            keep = at is not None
        else:
            keep = at is not None and at == C
        if keep:
            editor._auto_tuned_at = at
            editor.reject_threshold_pct = pct
        else:
            editor._auto_tuned_at = None
            editor.reject_threshold_pct = 99.0
        editor.key_dist_threshold = None
        bind = getattr(editor, "bind_edit_sources", None)
        if callable(bind):
            bind(getattr(ctx.get("edit_ds"), "data", []) or [])

    editor.load_resume_state = _load
    bind = getattr(editor, "bind_edit_sources", None)
    if callable(bind):
        bind(getattr(ctx.get("edit_ds"), "data", []) or [])
    if mode == "auto100":
        maybe = getattr(editor, "_maybe_auto_tune", None)
        if callable(maybe):
            maybe()
        sched = f"tune once at {_AUTO_FIRST} edits, then freeze pct"
    else:
        sched = f"tune at {_AUTO_FIRST}, then every 1000"
    print(f"[{TAG}] reject_threshold_pct={mode} enabled "
          f"({sched}; floor 10; max_drop={drop})", flush=True)


def _stamp(item: dict, editor) -> None:
    meta = dict(item.get("meta") or {})
    meta["mode"] = str(getattr(editor, "_auto_mode", None) or "auto")
    meta["reject_threshold_pct"] = float(getattr(editor, "reject_threshold_pct", 99.0))
    meta["auto_max_drop"] = int(getattr(editor, "auto_max_drop", 5))
    log = getattr(editor, "last_auto_log", None)
    if log:
        meta["auto_log"] = _jsonable(log)
    item["meta"] = meta
    rec = item.get("rec")
    if rec is not None:
        _patch_record(rec, item["result"], meta)
        rec["_auto_fix"] = rec.pop("_gen_fix", meta)


def run_peeks(
    ctx: dict,
    dest: Path,
    ks: Optional[Sequence[int]] = None,
    max_peeks: Optional[int] = None,
    overwrite: bool = False,
    skip: Sequence[str] = (),
    dump_preds_dir: Optional[str] = None,
    dry_run: bool = False,
) -> list:
    records = ctx["records"]
    done_ks = set(ctx.get("auto_done_ks") or ())
    peeks = _ordered_peeks(records, ks=ks, max_peeks=max_peeks)
    if dry_run:
        M, N = int(ctx["M"]), int(ctx["N"])
        print(f"[{TAG}] sidecar={dest}  (will not touch {ctx['jsonl']})", flush=True)
        for rec in peeks:
            K = _step(rec)
            if K in done_ks and not overwrite:
                print(f"[{TAG}]   skip K={K} (already in sidecar)", flush=True)
            else:
                print(f"[{TAG}]   would eval K={K} keep={_n_keep(M, K, N)}/{M}",
                      flush=True)
        return []

    dest.parent.mkdir(parents=True, exist_ok=True)
    outs = []
    editor = ctx["editor"]
    for rec in peeks:
        K = _step(rec)
        if K in done_ks and not overwrite:
            print(f"[{TAG}]   skip K={K} (already in sidecar; --overwrite to redo)",
                  flush=True)
            continue
        item = eval_peek(ctx, copy.deepcopy(rec), skip=skip,
                         dump_preds_dir=dump_preds_dir)
        _stamp(item, editor)
        stamped = item.get("rec")
        if stamped is None:
            continue
        _append_jsonl(dest, stamped)
        done_ks.add(K)
        outs.append(item)
        pct = item["meta"].get("reject_threshold_pct")
        print(f"[{TAG}]   appended {dest} (K={K} auto_pct={pct})", flush=True)
    ctx["auto_done_ks"] = done_ks
    return outs


def run_autofix(
    model_name: str,
    dataset_name: str = "vqav2",
    overwrite: bool = False,
    max_peeks: Optional[int] = None,
    ks: Optional[Sequence[int]] = None,
    out_jsonl: Optional[Union[str, Path]] = None,
    skip: Sequence[str] = (),
    dump_preds_dir: Optional[str] = None,
    dry_run: bool = False,
    config_path: str = "revlm/config/config.yaml",
    split: str = "all",
    batch_size: int = 1,
    auto_max_drop: Optional[int] = None,
    auto_mode: str = "auto100",
) -> dict:
    ctx = _setup(
        model_name, dataset_name=dataset_name,
        config_path=config_path, split=split, batch_size=batch_size,
    )
    dest = Path(out_jsonl) if out_jsonl else Path(ctx["edit_dir"]) / OUT_NAME
    print(f"[{TAG}] original JSONL: {ctx['jsonl']}  (untouched)", flush=True)
    print(f"[{TAG}] sidecar JSONL:  {dest}", flush=True)
    ctx["auto_done_ks"] = _prepare_sidecar(
        ctx["records"], dest, overwrite=overwrite, compact=not dry_run,
    )
    _enable_auto(ctx, auto_max_drop=auto_max_drop, auto_mode=auto_mode)
    try:
        run_peeks(
            ctx, dest, ks=ks, max_peeks=max_peeks, overwrite=overwrite,
            skip=skip, dump_preds_dir=dump_preds_dir, dry_run=dry_run,
        )
    finally:
        close(ctx)
    return ctx


def main():
    parser = argparse.ArgumentParser(
        description="SoftEdit auto-threshold recompute (sidecar JSONL, original untouched)"
    )
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="vqav2")
    parser.add_argument("--max_peeks", type=int, default=None)
    parser.add_argument("--ks", type=str, default=None,
                        help="Comma-separated peek Ks, e.g. 200,400")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Redo peeks already present in the sidecar")
    parser.add_argument("--out_jsonl", type=str, default=None,
                        help="Sidecar path (default: <edit_dir>/postedit_qa_all.auto.json)")
    parser.add_argument("--skip", type=str, default="")
    parser.add_argument("--split", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--auto_mode", type=str, default="auto100",
                        choices=("auto100", "auto"),
                        help="auto100: freeze pct after first 100 edits. "
                             "auto: retune at 100, then every 1000.")
    parser.add_argument("--auto_max_drop", type=int, default=None,
                        help="Max hit drop vs pct=99 (0 = no drop). Default: editor cfg / 5")
    args = parser.parse_args()
    ks = [int(x) for x in args.ks.split(",")] if args.ks else None
    skip = tuple(x.strip() for x in args.skip.split(",") if x.strip())
    if args.dry_run:
        print(f"[{TAG}] dry_run: no GPU, no writes", flush=True)
    else:
        print(f"[{TAG}] sidecar only — original JSONL will not be written",
              flush=True)
    run_autofix(
        args.model_name,
        dataset_name=args.dataset_name,
        overwrite=args.overwrite,
        max_peeks=args.max_peeks,
        ks=ks,
        out_jsonl=args.out_jsonl,
        skip=skip,
        dry_run=args.dry_run,
        config_path=args.config,
        split=args.split,
        batch_size=args.batch_size,
        auto_max_drop=args.auto_max_drop,
        auto_mode=args.auto_mode,
    )
    print("=" * 50, flush=True)
    print(f"[{TAG}] done", flush=True)


if __name__ == "__main__":
    main()
