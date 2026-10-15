"""Edit + eval core (LLM analog of revlm/run/edit_utils.py).

find_errors -> edit_n_eval_seq (sequential, resume+wandb) / edit_n_eval_all
(batch; the effective mode for the ft_retrain / mend_retrain joint-retrain
editors). Text-scoped: no images; generality = t-gen (->sgen) + soft-gen (->sloc).
"""
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from rellm.models import TQAModel
from rellm.dataset.datasets import TQADataset
from rellm.editors import get_editor
from rellm.metrics.editeval import (
    editeval, move_model_device, cuda_gc, normalize_skip, prioritized_scores,
)
from rellm.metrics.utils import get_t_gen_input, get_soft_gen_input
from .resume import (
    edit_fingerprint, load_ckpt, save_ckpt, backup_ckpt, mark_complete,
    truncate_jsonl, capture_rng, restore_rng,
    load_batch_ckpt, save_batch_ckpt, clear_batch_ckpt,
    load_batch_history, save_batch_history, clear_batch_history,
)

try:
    import wandb
except Exception:  # wandb optional (off by default)
    wandb = None


def _fmt_dhms(total_seconds: float) -> str:
    total_seconds = int(round(float(total_seconds)))
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{days}d {hours:02d}h {minutes:02d}m {seconds:02d}s"


def _base_editor_name(editor_name: str) -> str:
    """Strip a trailing `_soft` so pipeline dispatch treats a soft variant like its base."""
    editor_name = editor_name or ""
    return editor_name[:-5] if editor_name.endswith("_soft") else editor_name


# Fit once on the whole edit set, then hook. Retrain editors train weights;
# ts, bc, and ats fit a calibrator. History stays on CPU so the fit sees every edit.
_BATCH_FIT_EDITORS = {"ft_retrain", "mend_retrain", "ts", "bc", "ats"}


# Retrieval editors take edit_ds and apply_to_dataset (no weight-training path).
# SoftEdit is retrieval-class: it patches lm_head rows per-sample at generation
# (via dataset.pregen_arm) and restores them, so it needs no weight-copy of the model.
_RETRIEVAL_EDITORS = ("ike", "softedit")


def _is_retrieval_editor(editor_name: str) -> bool:
    base = _base_editor_name(editor_name)
    return any(base == n or base.startswith(n + "_") for n in _RETRIEVAL_EDITORS)


def _preedit_snapshot(config, model, editor_name):
    """CPU copy of the pre-edit model for locality gold.

    Sequential used to ``deepcopy`` while the live model was still on GPU, which
    parked a second 8B replica in VRAM until the first locality peek moved it.
    Same snapshot as ``edit_n_eval_all``: clone on CPU, put the live model back.
    """
    if _is_retrieval_editor((editor_name or "").lower()):
        return model
    orig_device = getattr(config, "device", None)
    if torch.cuda.is_available() and orig_device is not None:
        config.device = torch.device("cpu")
        if hasattr(model, "device"):
            model.device = config.device
        move_model_device(model, config.device)
        cuda_gc()
    model_old = copy.deepcopy(model)
    if torch.cuda.is_available() and orig_device is not None:
        config.device = orig_device
        if hasattr(model, "device"):
            model.device = config.device
        move_model_device(model, orig_device)
        cuda_gc()
    return model_old


def _soft_batch(editor, batch):
    """For `*_soft` weight editors in ``soft="target"`` mode, swap each gold's
    supervision target (`label_train`) to the ranked multi-label string. No-op for
    original editors and for the distribution-aware modes (loss / rank_loss)."""
    if not getattr(editor, "soft", False):
        return batch
    if getattr(editor, "soft_mode", "target") != "target":
        return batch
    from rellm.editors.soft import ranked_label_str  # Stage 6
    out = dict(batch)
    out["golds"] = [dict(g, label_train=ranked_label_str(g)) for g in batch.get("golds", [])]
    return out


def _set_soft_target(editor, batch, model):
    """Attach per-edit soft supervision for distribution-aware soft modes; cleared
    for original editors and the ``soft="target"`` mode (they keep CE)."""
    if not getattr(editor, "soft", False) or getattr(editor, "soft_mode", "target") == "target":
        editor._soft_target = None
        editor._soft_cont = None
        return
    from rellm.editors.soft import build_cont_label_ids, build_soft_target  # Stage 6
    golds = batch.get("golds", [])
    if not golds:
        editor._soft_target = None
        editor._soft_cont = None
        return
    target = build_soft_target(
        golds[0], model.tokenizer,
        mode=getattr(editor, "soft_mode", "loss"),
        rank_temp=getattr(editor, "rank_temp", 1.0),
    )
    editor._soft_target = target
    editor._soft_cont = build_cont_label_ids(model.tokenizer, target)


def _set_edit_lookup(editor, batch, model):
    """Attach the fact-lookup ("key") token position for the current single-example
    edit. AlphaEdit keys on the last question-content token (which varies per edit)
    instead of the fixed instruction/template suffix. Also stashes the raw question /
    prompt so a *deferred* batched solve can rebuild the context-averaged keys (the
    live ``_lookup_pos`` is overwritten by the next edit). No-op for editors that
    don't opt in via ``_uses_lookup_pos``."""
    if not getattr(editor, "_uses_lookup_pos", False):
        return
    prompts = batch.get("prompts") or []
    questions = batch.get("questions") or []
    editor._edit_prompt = prompts[0] if prompts else None
    editor._edit_question = questions[0] if questions else None
    if prompts and questions and hasattr(model, "question_lookup_pos"):
        editor._lookup_pos = model.question_lookup_pos(prompts[0], questions[0])
    else:
        editor._lookup_pos = None


def _flush_editor(editor, config=None):
    """Commit any buffered edits (AlphaEdit batched-solve) so the model reflects all
    edits before an eval / checkpoint. No-op for editors without a buffer."""
    if editor is not None and hasattr(editor, "flush"):
        editor.flush(config)


def debug_samples(dataset, label, n=10):
    """Per-sample gold/pred diagnostics (only under config.debug)."""
    from rellm.metrics.ranked_metrics import (
        build_labels_true, flatten_labels, align_probs, rank_acc, js_distance, norm,
    )

    def _fp(x):
        s = f"{x:.2f}"
        return s[1:] if s.startswith("0.") else s

    def _probs(vals):
        return "[" + " ".join(_fp(v) for v in vals) + "]" if vals else "[]"

    sample_size = min(n, len(dataset.data))
    print(f"[{label}] per-sample (first {sample_size}):", flush=True)
    for ex in dataset.data[:sample_size]:
        gold = ex.get("gold", {})
        answer_counts = gold.get("answer_counts") or ex.get("answer_counts", {}) or {}
        tiers = build_labels_true(answer_counts)
        lt = flatten_labels(tiers)
        pred = ex.get("pred") or ex.get("pred_qa") or {}
        sp_pred = pred.get("support_prob")
        labels_pred = [norm(l) for l in pred.get("labels_ranked", [])] or (
            [norm(pred.get("label", ""))] if pred.get("label") else [])
        print(f"  gold  uid={ex.get('uid')} true=[{', '.join(lt)}] "
              f"probs={_probs(align_probs(answer_counts, lt))}", flush=True)
        js_pred = _fp(js_distance(answer_counts, sp_pred)) if sp_pred else "--"
        print(f"  pred  labels=[{', '.join(labels_pred)}] "
              f"probs={_probs(align_probs(sp_pred, labels_pred))}  js\u2016gt={js_pred}", flush=True)
        print("  " + "-" * 68, flush=True)


def print10(dataset, label):
    if getattr(dataset.config, "debug", False):
        debug_samples(dataset, label)
    print(f"[{label}] metrics: {dataset.task_engineer.eval(dataset)}", flush=True)


def _wandb_job_name_from_edit_dir(edit_dir: str) -> str:
    if not edit_dir:
        return None
    results_root = (PROJECT_ROOT / "results").resolve()
    try:
        rel = Path(edit_dir).resolve().relative_to(results_root)
    except Exception:
        rel = Path(edit_dir)
    job = str(rel).strip("/\\").replace("/", "_").replace("\\", "_")
    return job or None


def _wandb_log_payload(batch_out: dict) -> dict:
    return {k: v for k, v in prioritized_scores(batch_out).items() if v is not None}


def find_errors(config):
    """Step 0-1: Load baseline predictions from snapshot, build the edit set (errors)."""
    seed = getattr(config, "seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = TQAModel(config)
    pred_snapshot = getattr(config, "pred_path", None)
    if not pred_snapshot:
        pred_snapshot = os.path.join(config.pred_dir, config.fname)
    ds = TQADataset(config)

    print("=" * 50, flush=True)
    print("Step 1 (predictions)", flush=True)
    t1 = time.time()
    if not os.path.exists(pred_snapshot):
        raise FileNotFoundError(
            f"No prediction snapshot at {pred_snapshot}. Run baseline.py first:\n"
            f"  python -m rellm.run.baseline --model_name {config.model.name} "
            f"--dataset_name {config.experiment.dataset_name}"
        )
    with open(pred_snapshot, "r") as f:
        ds.data = json.load(f)
    ds.reparse_snapshot()
    valid_uids = {str(r["uid"]) for r in ds.df2data(ds.load_df())}
    n_before = len(ds.data)
    ds.data = [ex for ex in ds.data if str(ex["uid"]) in valid_uids]
    print(f"Total samples {len(ds.data)}/{n_before} loaded from {pred_snapshot} "
          f"(filtered to current parquet)", flush=True)

    all_data = list(ds.data)
    edit_ds = ds.get_edits()
    edit_ds.set_dataloader()
    print10(edit_ds, label="model_old")
    print(f"Edit subset (errors): {len(edit_ds.data)}", flush=True)

    edit_uids = {str(ex.get("uid")) for ex in edit_ds.data}
    loc_ds = copy.deepcopy(edit_ds)
    loc_ds.data = [ex for ex in all_data if str(ex.get("uid")) not in edit_uids]
    loc_ds.set_dataloader()
    print(f"Locality pool (baseline-correct): {len(loc_ds.data)}", flush=True)
    print(f"Total time: {time.time() - t1:.2f}s", flush=True)
    return model, edit_ds, loc_ds


def _gen_inputs(config, edit_ds, skip):
    """Build generality inputs (t-gen + soft-gen) for the edit set."""
    dataset_name = config.experiment.dataset_name
    related_texts = get_t_gen_input(dataset_name, edit_ds) if "text_generality" not in skip else {}
    soft_ctx = get_soft_gen_input(dataset_name, edit_ds) if "soft_generality" not in skip else {}
    return related_texts, soft_ctx


def _is_eval_dump_name(name: str) -> bool:
    return name in {
        "reliability.json", "text_generality.json", "soft_generality.json", "locality.json",
    }


def _clear_eval_dumps(dump_dir: str) -> None:
    """Drop stale per-set eval dumps so a fresh batch edit cannot resume them."""
    if not dump_dir or not os.path.isdir(dump_dir):
        return
    n = 0
    for name in os.listdir(dump_dir):
        if _is_eval_dump_name(name):
            os.remove(os.path.join(dump_dir, name))
            n += 1
    if n:
        print(f"[resume] cleared {n} stale eval dumps in {dump_dir}", flush=True)


def _train_ckpt_to_resume_state(tckpt: dict):
    """Map a retrain ``train_ckpt_*.pt`` payload onto ``load_resume_state``."""
    if not tckpt:
        return None
    if "layer" in tckpt:
        return {"layer": tckpt["layer"]}
    if "weights" in tckpt:
        return {"weights": tckpt["weights"], "hypernet": tckpt.get("hypernet", {})}
    return None


def _save_batch_editor_ckpt(config, editor, fingerprint, edit_time, n_edits):
    if editor is not None and not hasattr(editor, "save_resume_state"):
        return
    editor_state = editor.save_resume_state() if editor is not None else None
    save_batch_ckpt(
        config.edit_dir,
        {
            "mode": "batch",
            "phase": "eval",
            "fingerprint": fingerprint,
            "edit_time": edit_time,
            "n_edits": n_edits,
        },
        editor_state,
    )


def _collect_retrain_history(config, model, editor, edit_ds, base_editor):
    """Tokenize every edit into CPU ``batch_history`` (no training)."""
    if hasattr(model, "model"):
        model.model.train()
    print(f"Starting edits with editor='{getattr(config.editor, '_name', '')}'...", flush=True)
    batch_history = []
    for batch_idx, batch in enumerate(edit_ds.loader):
        tokens = model.prepare_training_batch(_soft_batch(editor, batch))
        _set_soft_target(editor, batch, model)
        _set_edit_lookup(editor, batch, model)
        if base_editor not in _BATCH_FIT_EDITORS:
            editor.edit(config, tokens, batch_history=batch_history)
        _cpu = base_editor in _BATCH_FIT_EDITORS
        tokens_copy = {k: (v.detach().cpu().clone() if (isinstance(v, torch.Tensor) and _cpu) else
                           (v.clone() if isinstance(v, torch.Tensor) else v))
                       for k, v in tokens.items()}
        if getattr(editor, "_soft_target", None) is not None:
            tokens_copy["_soft_target"] = editor._soft_target
        if _cpu and getattr(editor, "_soft_cont", None) is not None:
            tokens_copy["_soft_cont"] = editor._soft_cont
        batch_history.append(tokens_copy)
        del tokens
        if (batch_idx + 1) % 10 == 0:
            print(f"Edited {batch_idx + 1} batches", flush=True)
    return batch_history


def edit_n_eval_all(config, model, edit_ds, out_path, skip=(), loc_ds=None):
    """Apply all edits, then evaluate once (batch / non-sequential).

    Incomplete batch runs resume by default:
      - mid-training: ``train_ckpt_*.pt`` (epochs) + optional ``batch_history.pt``
      - post-training: ``_resume/batch_ckpt.pt`` or a finished retrain ckpt
      - mid-eval: existing ``pred_postedit`` dumps
    ``--overwrite`` starts fresh. Sequential ``ckpt.pt`` / COMPLETE is unused.
    """
    skip = normalize_skip(skip)
    n_edit_cap = getattr(config, "n_edit_cap", None)
    if n_edit_cap is not None and n_edit_cap > 0 and len(edit_ds.data) > n_edit_cap:
        print(f"Capping edit set from {len(edit_ds.data)} to {n_edit_cap} examples", flush=True)
        edit_ds.data = random.sample(edit_ds.data, n_edit_cap)
        edit_ds.set_dataloader()

    # Snapshot the pre-edit model on CPU (avoid doubling VRAM), keep model_old on CPU.
    editor_name = getattr(config.editor, "_name", "")
    base_editor = _base_editor_name(editor_name)
    model_old = _preedit_snapshot(config, model, editor_name)

    t_job = time.time()
    print("=" * 50, flush=True)
    print("Step 2 (editing)", flush=True)
    t2 = time.time()

    editor = None
    if editor_name != "baseline":
        editor = get_editor(config, model)
        editor.generate = model.model.generate if hasattr(model, "model") else model.generate

    edit_dir = getattr(config, "edit_dir", os.path.dirname(out_path))
    fingerprint = edit_fingerprint(config, edit_ds)
    pred_postedit_snapshot = os.path.join(config.pred_postedit_dir, config.fname)
    overwrite = getattr(config, "overwrite", False)
    if overwrite:
        clear_batch_ckpt(edit_dir)
        clear_batch_history(edit_dir)

    skip_edit = False
    skip_tokenize = False
    batch_history = None
    edit_time = None
    batch_ckpt = None if overwrite else load_batch_ckpt(edit_dir)
    if batch_ckpt is not None:
        meta = batch_ckpt.get("meta", {})
        if meta.get("fingerprint") != fingerprint:
            raise RuntimeError(
                "[resume] batch edit-set fingerprint mismatch - refusing to resume. "
                "The baseline snapshot, seed, subsample, editor, model, or dataset "
                "changed. Re-run with --overwrite to start fresh."
            )
        if editor is not None and batch_ckpt.get("editor") is not None:
            editor.load_resume_state(batch_ckpt["editor"])
            if hasattr(editor, "rekey"):
                editor.rekey(model, edit_ds, n_samples=len(edit_ds.data))
        skip_edit = True
        edit_time = meta.get("edit_time")
        print(f"[resume] batch: restored editor from {edit_dir}/_resume/batch_ckpt.pt "
              f"(skipping Step 2 edits)", flush=True)
    elif (
        not overwrite
        and editor is not None
        and hasattr(editor, "_load_training_ckpt")
        and hasattr(editor, "load_resume_state")
        and os.path.isfile(pred_postedit_snapshot)
    ):
        tckpt = editor._load_training_ckpt(config)
        n_iter = int(getattr(getattr(config, "editor", config), "n_iter", 20) or 20)
        state = _train_ckpt_to_resume_state(tckpt)
        if tckpt is not None and state is not None and n_iter and int(tckpt.get("epoch", -1)) + 1 >= n_iter:
            editor.load_resume_state(state)
            skip_edit = True
            print(f"[resume] batch: restored completed training ckpt "
                  f"(epoch {int(tckpt['epoch']) + 1}/{n_iter}), skipping Step 2 edits", flush=True)

    if not skip_edit and not overwrite and base_editor in {"ft_retrain", "mend_retrain"}:
        saved_hist = load_batch_history(edit_dir, fingerprint)
        if saved_hist:
            batch_history = saved_hist
            skip_tokenize = True
            print(f"[resume] batch: loaded {len(batch_history)} tokenized edits "
                  f"from {edit_dir}/_resume/batch_history.pt", flush=True)

    if not skip_edit:
        if editor_name == "baseline":
            if hasattr(model, "model"):
                model.model.eval()
        elif _is_retrieval_editor(editor_name):
            if hasattr(model, "model"):
                model.model.eval()
            editor.edit(config, edit_ds=edit_ds)
        else:
            if not skip_tokenize:
                batch_history = _collect_retrain_history(config, model, editor, edit_ds, base_editor)
                if base_editor in {"ft_retrain", "mend_retrain"} and batch_history:
                    save_batch_history(edit_dir, fingerprint, batch_history)
                    print(f"[resume] batch: saved tokenized history ({len(batch_history)} edits)", flush=True)
            if base_editor in _BATCH_FIT_EDITORS and batch_history:
                if hasattr(model, "model"):
                    model.model.train()
                editor.edit(config, batch_history[-1], batch_history=batch_history[:-1])
            # Commit buffered edits (AlphaEdit batched solve) before eval.
            _flush_editor(editor, config)
            if hasattr(model, "model"):
                model.model.eval()
        edit_time = time.time() - t2
        _save_batch_editor_ckpt(config, editor, fingerprint, edit_time, len(edit_ds.data))
        if overwrite or not _is_retrieval_editor(editor_name):
            _clear_eval_dumps(getattr(config, "pred_postedit_dir", None))
    else:
        if hasattr(model, "model"):
            model.model.eval()
        if edit_time is None:
            edit_time = 0.0
        _save_batch_editor_ckpt(config, editor, fingerprint, edit_time, len(edit_ds.data))

    reuse_postedit = (
        os.path.isfile(pred_postedit_snapshot)
        and not overwrite
        and (skip_edit or _is_retrieval_editor(editor_name))
    )
    if reuse_postedit:
        with open(pred_postedit_snapshot, "r") as f:
            edit_ds.data = json.load(f)
        edit_ds.set_dataloader()
        print(f"[resume] loaded post-edit predictions from {pred_postedit_snapshot}", flush=True)
    else:
        if _is_retrieval_editor(editor_name) and editor is not None and hasattr(editor, "apply_to_dataset"):
            editor.apply_to_dataset(edit_ds)
        edit_ds.task_generate(model, use_cache=False)
        edit_ds.snap(out_path=pred_postedit_snapshot)
        print(f"Post-edit predictions saved to {pred_postedit_snapshot}", flush=True)
    print10(edit_ds, label="model_new")
    print(f"Edit time: {edit_time:.2f}s", flush=True)

    print("=" * 50, flush=True)
    print("Step 3 (evaluation)", flush=True)
    t3 = time.time()
    related_texts, soft_ctx = _gen_inputs(config, edit_ds, skip)
    out_dict = editeval(
        model_old, model, edit_ds, editor,
        related_texts=related_texts, soft_ctx=soft_ctx,
        unrelated_ds=loc_ds,
        edit_time=edit_time,
        dump_preds_dir=getattr(config, "pred_postedit_dir", None),
        skip=skip,
        resume_dumps=True,
    )
    out_dict["finish_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    out_dict["job_total_time"] = _fmt_dhms(time.time() - t_job)
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"Total time: {time.time() - t3:.2f}s", flush=True)
    print(f"Saved edit-eval metrics to {out_path}", flush=True)
    print("=" * 50, flush=True)
    return out_dict


def edit_n_eval_seq(config, model, edit_ds, out_path, max_batches=None, eval_every: int = 200, skip=(), loc_ds=None):
    """Edit one sample at a time, evaluating every `eval_every` edits (resume-safe)."""
    skip = normalize_skip(skip)
    editor_name = getattr(config.editor, "_name", "")
    base_editor = _base_editor_name(editor_name)
    model_old = _preedit_snapshot(config, model, editor_name)
    dataset_name = config.experiment.dataset_name

    # Sequential editing forces single-sample batches regardless of config.batch_size.
    edit_ds.set_dataloader(batch_size=1)
    total_batches = len(edit_ds.loader) if hasattr(edit_ds.loader, "__len__") else None

    use_wandb = getattr(config, "wandb", False) and wandb is not None
    if use_wandb and wandb.run is None:
        job_name = _wandb_job_name_from_edit_dir(getattr(config, "edit_dir", None))
        wandb.init(project=os.environ.get("WANDB_PROJECT", "rankedit_llm"),
                   config=config, name=job_name)

    editor = None
    if editor_name != "baseline":
        editor = get_editor(config, model)
        editor.generate = model.model.generate if hasattr(model, "model") else model.generate

    batch_history = []
    all_out_dicts = []
    seen_idxs = []
    seen_idx_set = set()

    edit_dir = getattr(config, "edit_dir", os.path.dirname(out_path))
    fingerprint = edit_fingerprint(config, edit_ds)
    resume_from = 0
    cumulative_edit_time = 0.0
    n_written = 0
    ckpt = None if getattr(config, "overwrite", False) else load_ckpt(edit_dir)
    if ckpt is not None:
        meta = ckpt.get("meta", {})
        if meta.get("fingerprint") != fingerprint:
            raise RuntimeError(
                "[resume] edit-set fingerprint mismatch - refusing to resume. Re-run with --overwrite."
            )
        resume_from = int(meta.get("batch_idx", 0))
        cumulative_edit_time = float(meta.get("cumulative_edit_time", 0.0))
        n_written = int(meta.get("jsonl_lines", 0))
        truncate_jsonl(out_path, n_written)
        restore_rng(meta.get("rng", {}))
        if editor is not None:
            editor.load_resume_state(ckpt.get("editor"))
            if hasattr(editor, "rekey"):
                backup_ckpt(edit_dir)
                editor.rekey(model, edit_ds, n_samples=resume_from)
                save_ckpt(edit_dir, meta, editor.save_resume_state())
        print(f"[resume] resuming from batch {resume_from} "
              f"(jsonl_lines={n_written}, cum_edit_time={cumulative_edit_time:.1f}s)", flush=True)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("")

    completed = True
    for batch_idx, batch in enumerate(edit_ds.loader):
        if max_batches is not None and batch_idx >= max_batches:
            print(f"Early stop at batch {batch_idx}", flush=True)
            completed = False
            break
        for i in batch.get("idxs", []):
            if i not in seen_idx_set:
                seen_idx_set.add(i)
                seen_idxs.append(int(i))
        if batch_idx < resume_from:
            continue

        edit_ds_sofar = copy.deepcopy(edit_ds)
        edit_ds_sofar.data = [edit_ds_sofar.data[i] for i in seen_idxs]
        edit_ds_sofar.set_dataloader()

        print("=" * 50, flush=True)
        print(f"Batch {batch_idx + 1}: editing", flush=True)
        t2 = time.time()
        if editor_name == "baseline":
            pass
        elif _is_retrieval_editor(editor_name):
            editor.edit(config, edit_ds=edit_ds_sofar)
        else:
            if hasattr(model, "model"):
                model.model.train()
            tokens = model.prepare_training_batch(_soft_batch(editor, batch))
            _set_soft_target(editor, batch, model)
            _set_edit_lookup(editor, batch, model)
            if base_editor not in _BATCH_FIT_EDITORS:
                editor.edit(config, tokens, batch_history=batch_history)
            _cpu = base_editor in _BATCH_FIT_EDITORS
            tokens_copy = {k: (v.detach().cpu().clone() if (isinstance(v, torch.Tensor) and _cpu) else
                               (v.clone() if isinstance(v, torch.Tensor) else v))
                           for k, v in tokens.items()}
            if getattr(editor, "_soft_target", None) is not None:
                tokens_copy["_soft_target"] = editor._soft_target
            if _cpu and getattr(editor, "_soft_cont", None) is not None:
                tokens_copy["_soft_cont"] = editor._soft_cont
            batch_history.append(tokens_copy)
            del tokens

        batch_edit_time = time.time() - t2
        cumulative_edit_time += batch_edit_time
        is_last = ((total_batches is not None and (batch_idx + 1) == total_batches)
                   or (max_batches is not None and (batch_idx + 1) == max_batches))
        should_eval = ((eval_every is None) or (eval_every <= 0)
                       or ((batch_idx + 1) % int(eval_every) == 0) or is_last)
        if should_eval:
            if base_editor in _BATCH_FIT_EDITORS and batch_history:
                editor.edit(config, batch_history[-1], batch_history=batch_history[:-1])
            # Commit buffered edits (AlphaEdit batched solve) before generating /
            # evaluating / checkpointing, so the model reflects every edit so far.
            _flush_editor(editor, config)
            if hasattr(model, "model"):
                model.model.eval()
            if is_last:
                if _is_retrieval_editor(editor_name) and editor is not None and hasattr(editor, "apply_to_dataset"):
                    editor.apply_to_dataset(edit_ds_sofar)
                edit_ds_sofar.task_generate(model, use_cache=False)
                if hasattr(editor, "print_stats"):
                    editor.print_stats()
                if hasattr(editor, "reset_counters"):
                    editor.reset_counters()
                print10(edit_ds_sofar, label="model_new")
            print(f"Edit time (batch): {batch_edit_time:.2f}s, (cumulative): {cumulative_edit_time:.2f}s", flush=True)

            print("=" * 50, flush=True)
            print(f"Batch {batch_idx + 1}: evaluation", flush=True)
            t3 = time.time()
            related_texts, soft_ctx = _gen_inputs(config, edit_ds_sofar, skip)
            pred_dir = getattr(config, "pred_postedit_dir", None)
            if is_last or not pred_dir:
                dump_preds_dir = pred_dir
            else:
                dump_preds_dir = os.path.join(pred_dir, "seq", f"batch_{batch_idx + 1:06d}")
            batch_out_dict = editeval(
                model_old, model, edit_ds_sofar, editor,
                related_texts=related_texts, soft_ctx=soft_ctx,
                unrelated_ds=loc_ds,
                edit_subsample_size=None if is_last else 50,
                edit_time=cumulative_edit_time,
                dump_preds_dir=dump_preds_dir,
                skip=skip,
            )
            batch_out_dict["batch_idx"] = batch_idx + 1
            batch_out_dict["finish_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            all_out_dicts.append(batch_out_dict)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(batch_out_dict, ensure_ascii=False, sort_keys=True) + "\n")
            n_written += 1

            if editor is None or hasattr(editor, "save_resume_state"):
                editor_state = editor.save_resume_state() if editor is not None else None
                meta = {
                    "batch_idx": batch_idx + 1,
                    "jsonl_lines": n_written,
                    "fingerprint": fingerprint,
                    "cumulative_edit_time": cumulative_edit_time,
                    "rng": capture_rng(),
                }
                save_ckpt(edit_dir, meta, editor_state)

            if use_wandb and wandb.run is not None:
                wandb.log(_wandb_log_payload(batch_out_dict), step=batch_idx + 1)
            print(f"Eval time: {time.time() - t3:.2f}s | Saved to {out_path}", flush=True)

    if completed:
        mark_complete(edit_dir)
        print(f"[resume] run COMPLETE -> {edit_dir}/_resume/COMPLETE", flush=True)
    return all_out_dicts
