import copy
import json
import os
import random
import sys
from pathlib import Path

import torch
import time
import numpy as np
import wandb

# Add project root to path so we can run as a module or script
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from revlm import *  # noqa: F401,F403
from revlm.editors import get_editor
from revlm.metrics.editeval import move_model_device, cuda_gc, normalize_skip, prioritized_scores
from revlm.metrics.utils.blend_gen import ALL_STRENGTHS as BLEND_ALL_STRENGTHS
from .resume import (
    edit_fingerprint, load_ckpt, save_ckpt, backup_ckpt, mark_complete,
    truncate_jsonl, capture_rng, restore_rng,
    load_batch_ckpt, save_batch_ckpt, clear_batch_ckpt,
)


def _fmt_dhms(total_seconds: float) -> str:
    total_seconds = int(round(float(total_seconds)))
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{days}d {hours:02d}h {minutes:02d}m {seconds:02d}s"


def _base_editor_name(editor_name: str) -> str:
    """Strip a trailing `_soft` so pipeline dispatch (retrieval vs weight,
    balancedit token prep, retrain) treats a soft variant like its base editor."""
    editor_name = editor_name or ""
    return editor_name[:-5] if editor_name.endswith("_soft") else editor_name


# Retrieval editors (and their variants, e.g. softedit_task) take
# edit_ds and apply_to_dataset; they must NOT fall into the weight-editor training path.
_RETRIEVAL_EDITORS = ("ike", "reasonedit", "gt", "softedit")

# Editors that accumulate every edit, then fit once. ts, bc, and ats fit a calibrator.
_BATCH_FIT_EDITORS = {"ft_retrain", "mend_retrain", "ts", "bc", "ats"}


def _is_retrieval_editor(editor_name: str) -> bool:
    base = _base_editor_name(editor_name)
    return any(base == n or base.startswith(n + "_") for n in _RETRIEVAL_EDITORS)


def _soft_batch(editor, batch):
    """For `*_soft` weight editors in the ``soft="target"`` mode, swap each gold's
    supervision target (`label_train`) to the ranked multi-label string. Returns
    `batch` unchanged for original editors and for the distribution-aware modes
    (``soft in {"loss", "rank_loss"}``), which shape the loss instead of the target
    string. The edit prompt `x` is always the bare question (set by the postedit_qa
    task engineer), so nothing else needs swapping here."""
    if not getattr(editor, "soft", False):
        return batch
    if getattr(editor, "soft_mode", "target") != "target":
        return batch
    from revlm.editors.soft import ranked_label_str
    out = dict(batch)
    out["golds"] = [dict(g, label_train=ranked_label_str(g)) for g in batch.get("golds", [])]
    return out


def _set_soft_target(editor, batch, model):
    """Attach the per-edit soft supervision for distribution-aware soft modes.

    Soft weight editors run one sample per batch (sequential editing), so we build
    the `SoftTarget` from `batch["golds"][0]` and stash it on the editor for its
    `edit()` loss loop to read via `compute_edit_loss`. We also prebuild the
    per-branch continuation batches (`editor._soft_cont`) for the continuation
    term. Both are cleared (set to None) for original editors and the
    ``soft="target"`` mode so they keep using CE."""
    if not getattr(editor, "soft", False) or getattr(editor, "soft_mode", "target") == "target":
        editor._soft_target = None
        editor._soft_cont = None
        return
    from revlm.editors.soft import build_cont_label_ids, build_soft_target
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


def debug_samples(dataset, label, n=10):
    """Per-sample 3-row block for the first n samples (diagnosis only; call sites
    gate this behind config.debug).

    Everything is shown over the SAME column order `labels_true` (the tie-tiered
    ground truth from answer_counts), so the three distributions line up:
      gold  -> normalized answer_counts (human votes)
      read  -> SoftEdit codebook readout q (support_prob_readout) + retrieval
               diagnostics (d1 = scaled retrieval distance, hit/MISS = whether
               the retrieved entry is the example's own edit)
      pred  -> the VLM's own support_prob from the actuated generation
    Each of read/pred also reports its label-ranking scores vs labels_true and its
    JS distance to the gold votes (js‖gt)."""
    from revlm.metrics.ranked_metrics import (
        build_labels_true, flatten_labels, align_probs,
        rank_acc, acc_at_1, set_f1, ndcg_at_k, js_distance, norm,
    )

    def _fp(x):
        # 2-decimal, no leading zero: 0.60 -> ".60", 1.00 -> "1.00"
        s = f"{x:.2f}"
        return s[1:] if s.startswith("0.") else s

    def _probs(vals):
        return "[" + " ".join(_fp(v) for v in vals) + "]" if vals else "[]"

    def _rank(labels, tiers):
        if not labels:
            return "racc=--   acc@1=--   f1=--   ndcg=--  "
        top1 = labels[0]
        return (f"racc={rank_acc(top1, tiers):.2f} acc@1={acc_at_1(labels, tiers):.2f} "
                f"f1={set_f1(labels, tiers):.2f} ndcg={ndcg_at_k(labels, tiers):.2f}")

    sample_size = min(n, len(dataset.data))
    print(f"[{label}] per-sample (first {sample_size}):", flush=True)
    for ex in dataset.data[:sample_size]:
        gold = ex.get("gold", {})
        answer_counts = gold.get("answer_counts") or ex.get("answer_counts", {}) or {}
        tiers = build_labels_true(answer_counts)
        lt = flatten_labels(tiers)                       # ordered column labels

        pred = ex.get("pred") or ex.get("pred_qa") or {}
        sp_read = pred.get("support_prob_readout")
        sp_pred = pred.get("support_prob")
        dbg = pred.get("readout_dbg") or {}

        labels_read = sorted(sp_read, key=sp_read.get, reverse=True) if sp_read else []
        labels_pred = [norm(l) for l in pred.get("labels_ranked", [])] or (
            [norm(pred.get("label", ""))] if pred.get("label") else [])

        # gold row (header): uid + tie-tiered ground truth + human-vote probs
        print(f"  gold  uid={ex.get('uid')} true=[{', '.join(lt)}] "
              f"probs={_probs(align_probs(answer_counts, lt))}", flush=True)
        # read row (codebook readout) + retrieval diagnostics.
        # v2 dbg: {src, d1, hit, entry_uid, ...}; v1 dbg: {dist, bw, gate, ...}.
        parts = []
        if "d1" in dbg:
            parts.append(f"d1={dbg['d1']:.2f}")
        if "hit" in dbg:
            parts.append("hit" if dbg["hit"] else f"MISS->{dbg.get('entry_uid')}")
        if dbg.get("src") == "p":
            parts.append("K=1->p")
        if "dist" in dbg and dbg.get("dist") is not None and dbg.get("bw") is not None:  # v1
            parts.append(f"dist={dbg['dist']:.2f} bw={dbg['bw']:.2f}")
        if dbg.get("gate"):  # v1
            parts.append("GATE->p")
        geo = " ".join(parts)
        if sp_read:
            print(f"  read  labels=[{', '.join(labels_read)}] probs={_probs(align_probs(sp_read, labels_read))} "
                  f"{geo}  {_rank(labels_read, tiers)}  js\u2016gt={_fp(js_distance(answer_counts, sp_read))}",
                  flush=True)
        else:
            print(f"  read  (no retrieval)  {geo}".rstrip(), flush=True)
        # pred row (VLM text -> support_prob)
        js_pred = _fp(js_distance(answer_counts, sp_pred)) if sp_pred else "--"
        print(f"  pred  labels=[{', '.join(labels_pred)}] probs={_probs(align_probs(sp_pred, labels_pred))} "
              f"{_rank(labels_pred, tiers)}  js\u2016gt={js_pred}", flush=True)
        print("  " + "-" * 68, flush=True)


def print10(dataset, label):
    """Print the aggregate task-metric dict for `dataset` under `label`
    (e.g. model_old = baseline errors, model_new = after editing). When
    config.debug is set, also dump per-sample gold/pred diagnostics."""
    if getattr(dataset.config, "debug", False):
        debug_samples(dataset, label)
    print(f"[{label}] metrics: {dataset.task_engineer.eval(dataset)}", flush=True)


def _wandb_job_name_from_edit_dir(edit_dir: str) -> str:
    """Derive a stable wandb job name from edit_dir by stripping the root and replacing slashes."""
    if not edit_dir:
        return None
    results_root = (PROJECT_ROOT / "results").resolve()
    try:
        rel = Path(edit_dir).resolve().relative_to(results_root)
    except Exception:
        rel = Path(edit_dir)
    job = str(rel).strip("/\\")
    job = job.replace("/", "_").replace("\\", "_")
    return job or None


def _wandb_log_payload(batch_out: dict) -> dict:
    """Thin wandb dashboard payload from a full editeval checkpoint.

    Reuses ``prioritized_scores`` (the same acc + align scoreboard printed to the
    console) so wandb and stdout never drift. Per group: acc = ``simple_acc``,
    align = ``1 - js_dist_gate`` (higher == better); the loc families use
    ``mono``. ``/`` namespaces auto-group the panels in wandb. None-valued keys
    (skipped sets) are dropped.
    """
    return {k: v for k, v in prioritized_scores(batch_out).items() if v is not None}


def find_errors(config):
    """Step 0-1: Load predictions from snapshot, build edit set.

    Requires baseline predictions to already exist (run baseline.py first).
    """
    seed = getattr(config, "seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = VQAModel(config)
    pred_snapshot = getattr(config, "pred_path", None)
    if not pred_snapshot:
        pred_snapshot = os.path.join(config.pred_dir, config.fname)
    ds = VQADataset(config)

    print("="*50, flush=True)
    print("Step 1 (predictions)", flush=True)
    t1 = time.time()
    if not os.path.exists(pred_snapshot):
        raise FileNotFoundError(
            f"No prediction snapshot at {pred_snapshot}. "
            f"Run baseline.py first:\n"
            f"  python -m revlm.run.baseline --model_name {config.model.name} "
            f"--dataset_name {config.experiment.dataset_name}"
        )
    with open(pred_snapshot, "r") as f:
        ds.data = json.load(f)
    # The label is a derived field: re-parse it from the stored raw answer via the
    # task's own parser so a cached snapshot honors the current parsing/strategy.
    ds.reparse_snapshot()
    valid_uids = {str(r["uid"]) for r in ds.df2data(ds.load_df())}
    n_before = len(ds.data)
    ds.data = [ex for ex in ds.data if str(ex["uid"]) in valid_uids]
    print(f"Total samples {len(ds.data)}/{n_before} loaded from {pred_snapshot} "
          f"(filtered to current parquet)", flush=True)

    # Build edit subset (only errors) and locality set (baseline-correct)
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



def _is_eval_dump_name(name: str) -> bool:
    if name in {"reliability.json", "text_generality.json", "image_generality.json", "locality.json"}:
        return True
    if name.startswith("soft_generality_t") and name.endswith(".json"):
        return True
    if name.startswith("blend_generality_t") and name.endswith(".json"):
        return True
    return False


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


def edit_n_eval_all(config, model, edit_ds, out_path, skip=(), loc_ds=None):
    """Edit and evaluate - apply edits, then evaluate the edited model.

    skip: metric sets to skip for quick sessions (e.g. ["soft_gen", "i_gen"]).
    Incomplete batch runs resume by default: editor weights from
    ``_resume/batch_ckpt.pt`` (or a finished retrain ``train_ckpt_*.pt``) and
    eval sets from existing ``pred_postedit`` dumps. ``--overwrite`` starts fresh.
    Sequential resume (``ckpt.pt`` / COMPLETE) is not used here.
    """
    skip = normalize_skip(skip)
    # Cap the number of edits if specified
    n_edit_cap = getattr(config, "n_edit_cap", None)
    if n_edit_cap is not None and n_edit_cap > 0 and len(edit_ds.data) > n_edit_cap:
        print(f"Capping edit set from {len(edit_ds.data)} to {n_edit_cap} examples", flush=True)
        edit_ds.data = random.sample(edit_ds.data, n_edit_cap)
        edit_ds.set_dataloader()

    # Create a snapshot of the model before editing for comparison.
    # NOTE: deepcopy(model) on GPU can OOM for large VLMs (it temporarily doubles VRAM),
    # so we snapshot on CPU and keep model_old on CPU for metrics.
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
        move_model_device(model, config.device)

    # Keep an unmodified copy so reliability(model_old) uses original prompts
    # (IKE-style editors mutate prompts in-place).
    pristine_edit_ds = copy.deepcopy(edit_ds)
    t_job = time.time()

    # Step 2: apply edits on edit_ds with chosen editor
    print("="*50, flush=True)
    print("Step 2 (editing)", flush=True)
    t2 = time.time()
    editor_name = getattr(config.editor, "_name", "")
    base_editor = _base_editor_name(editor_name)

    # For baseline, we skip constructing an editor and performing any edits; the model and prompts stay unchanged.
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

    skip_edit = False
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

    if not skip_edit:
        if editor_name == "baseline":
            if hasattr(model, "model"):
                model.model.eval()
        elif _is_retrieval_editor(editor_name):
            if hasattr(model, "model"):
                model.model.eval()
            editor.edit(config, edit_ds=edit_ds)
        else:
            if hasattr(model, "model"):
                model.model.train()
            print(f"Starting edits with editor='{config.editor._name}'...", flush=True)
            batch_history = []
            for batch_idx, batch in enumerate(edit_ds.loader):
                tokens = model.prepare_training_batch(_soft_batch(editor, batch))
                _set_soft_target(editor, batch, model)
                extra = {}
                if base_editor == "balancedit":
                    questions = [edit_ds.data[i].get("question", "") for i in batch.get("idxs", [])]
                    rephrase_tokens, locality_tokens = editor.prepare_balancedit_tokens(batch, questions=questions)
                    extra = dict(rephrase_tokens=rephrase_tokens, locality_tokens=locality_tokens)
                if base_editor not in _BATCH_FIT_EDITORS:
                    editor.edit(config, tokens, batch_history=batch_history, **extra)

                # Keep a lightweight history copy for methods that need replay/regularization.
                # Retrain editors consume history in one big training loop with their own
                # _move_to_device, so keep tensors on CPU to avoid OOM at 10k+ edits.
                _cpu = base_editor in _BATCH_FIT_EDITORS
                tokens_copy = {k: (v.detach().cpu().clone() if (isinstance(v, torch.Tensor) and _cpu) else
                                  (v.clone() if isinstance(v, torch.Tensor) else v))
                               for k, v in tokens.items()}
                if getattr(editor, "_soft_target", None) is not None:
                    tokens_copy["_soft_target"] = editor._soft_target
                # Retrain editors consume the continuation term from history; the ids
                # are tiny CPU tensors (no image data), so store them directly.
                if _cpu and getattr(editor, "_soft_cont", None) is not None:
                    tokens_copy["_soft_cont"] = editor._soft_cont
                batch_history.append(tokens_copy)

                del tokens
                if (batch_idx + 1) % 10 == 0:
                    print(f"Edited {batch_idx + 1} batches", flush=True)

            if base_editor in _BATCH_FIT_EDITORS and batch_history:
                editor.edit(config, batch_history[-1], batch_history=batch_history[:-1])
            if hasattr(model, "model"):
                model.model.eval()
        edit_time = time.time() - t2  # capture edit time before generation
        _save_batch_editor_ckpt(config, editor, fingerprint, edit_time, len(edit_ds.data))
        # Weight-editor reruns produce new weights; drop stale eval dumps so they
        # cannot be resumed. Retrieval corpus rebuild is deterministic from the
        # edit set, so keep dumps unless --overwrite.
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

    # Step 3: evaluate the edited model
    print("="*50, flush=True)
    print("Step 3 (evaluation)", flush=True)
    t3 = time.time()
    model_new = model
    dataset_name = config.experiment.dataset_name
    model_name = config.model.name
    related_texts = get_t_gen_input(dataset_name, edit_ds) if "text_generality" not in skip else {}
    related_images = get_i_gen_input(dataset_name, edit_ds, k=2) if "image_generality" not in skip else {}
    soft_gen_images = get_soft_gen_input_by_strength(dataset_name, edit_ds) if "soft_generality" not in skip else {}
    blend_gen_images = get_blend_gen_input_by_strength(dataset_name, edit_ds, strengths=BLEND_ALL_STRENGTHS) if "blend_generality" not in skip else {}
    out_dict = editeval(
        model_old,
        model_new,
        edit_ds,
        editor,
        related_texts,
        related_images,
        soft_gen_images=soft_gen_images,
        blend_gen_images=blend_gen_images,
        unrelated_ds=loc_ds,
        edit_time=edit_time,
        dump_preds_dir=getattr(config, "pred_postedit_dir", None),
        skip=skip,
        resume_dumps=True,
    )
    # add a job finish time
    out_dict['finish_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    out_dict["job_total_time"] = _fmt_dhms(time.time() - t_job)
    
    # # reliability() is side-effect free on edit_ds (operates on a deepcopy)
    # out_dict['reliability_old'] =  reliability(model_old, pristine_edit_ds)
    # print(f"Reliability (model_old, on edit set): {out_dict['reliability_old']:.4f}", flush=True)
    # print(f"Reliability (model_new, on edit set): {out_dict['reliability']:.4f}", flush=True)
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"Total time: {time.time() - t3:.2f}s", flush=True)
    print(f"Saved edit-eval metrics to {out_path}", flush=True)
    print("="*50, flush=True)
    
    return out_dict



def edit_n_eval_seq(config, model, edit_ds, out_path, max_batches=None, eval_every: int = 200, skip=(), loc_ds=None):
    """Edit sequentially, evaluating every `eval_every` batchess.

    skip: metric sets to skip for quick sessions (e.g. ["soft_gen", "i_gen"]).
    """
    skip = normalize_skip(skip)
    editor_name_check = getattr(config.editor, "_name", "").lower()
    # Retrieval-based editors don't modify weights - skip expensive deepcopy
    # (gt = retrieval-free oracle: also weight-free, prompt injection only).
    if _is_retrieval_editor(editor_name_check):
        model_old = model  # Same reference, no copy needed
    else:
        model_old = copy.deepcopy(model)
    pristine_edit_ds = copy.deepcopy(edit_ds)
    editor_name = getattr(config.editor, "_name", "")
    base_editor = _base_editor_name(editor_name)
    dataset_name = config.experiment.dataset_name
    model_name = config.model.name
    # Sequential editing edits one sample at a time: force the editing loop to
    # yield single-sample batches regardless of config.batch_size. (Weight
    # editors like GRACE assume one edit per key; per-batch eval below still uses
    # the configured batch size via edit_ds_sofar.set_dataloader().)
    edit_ds.set_dataloader(batch_size=1)
    total_batches = len(edit_ds.loader) if hasattr(edit_ds.loader, "__len__") else None

    # Initialize wandb if enabled
    use_wandb = getattr(config, "wandb", False)
    if use_wandb and wandb.run is None:
        job_name = _wandb_job_name_from_edit_dir(getattr(config, "edit_dir", None))
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "rankedit"),
            config=config,
            name=job_name,
        )

    editor = None
    if editor_name != "baseline":
        editor = get_editor(config, model)
        editor.generate = model.model.generate if hasattr(model, "model") else model.generate

    batch_history = []
    all_out_dicts = []
    seen_idxs = []
    seen_idx_set = set()

    # ---- resume / fresh-start bookkeeping ----
    # Editor + model_old are already constructed above; model_old holds the
    # pre-edit (base) weights, so restoring editor state below only mutates `model`.
    edit_dir = getattr(config, "edit_dir", os.path.dirname(out_path))
    fingerprint = edit_fingerprint(config, edit_ds)
    resume_from = 0
    cumulative_edit_time = 0.0  # total edit time across all batches (persisted across resumes)
    n_written = 0               # committed JSONL line count
    ckpt = None if getattr(config, "overwrite", False) else load_ckpt(edit_dir)
    if ckpt is not None:
        meta = ckpt.get("meta", {})
        if meta.get("fingerprint") != fingerprint:
            raise RuntimeError(
                "[resume] edit-set fingerprint mismatch - refusing to resume. The baseline "
                "snapshot, seed, subsample, editor, model, or dataset changed. Re-run with "
                "--overwrite to start fresh."
            )
        resume_from = int(meta.get("batch_idx", 0))
        cumulative_edit_time = float(meta.get("cumulative_edit_time", 0.0))
        n_written = int(meta.get("jsonl_lines", 0))
        truncate_jsonl(out_path, n_written)
        restore_rng(meta.get("rng", {}))
        if editor is not None:
            editor.load_resume_state(ckpt.get("editor"))
            # GRACE (and any editor with rekey): rewrite activation keys for the
            # current GPU, after backing up the original ckpt so we never lose it.
            if hasattr(editor, "rekey"):
                backup_ckpt(edit_dir)
                editor.rekey(model, edit_ds, n_samples=resume_from)
                save_ckpt(edit_dir, meta, editor.save_resume_state())
        print(f"[resume] resuming from batch {resume_from} "
              f"(jsonl_lines={n_written}, cum_edit_time={cumulative_edit_time:.1f}s)", flush=True)
    else:
        # Fresh start: truncate/create the JSONL once, then append one object per eval.
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

        # Fast-forward already-applied+checkpointed edits on resume (no model work):
        # seen_idxs is rebuilt above; editor/model state was restored from the checkpoint.
        if batch_idx < resume_from:
            continue

        edit_ds_sofar = copy.deepcopy(edit_ds)
        pristine_ds_sofar = copy.deepcopy(pristine_edit_ds)
        edit_ds_sofar.data = [edit_ds_sofar.data[i] for i in seen_idxs]
        pristine_ds_sofar.data = [pristine_ds_sofar.data[i] for i in seen_idxs]
        edit_ds_sofar.set_dataloader()

        # Edit
        print("="*50, flush=True)
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
            extra = {}
            if base_editor == "balancedit":
                questions = [edit_ds.data[i].get("question", "") for i in batch.get("idxs", [])]
                rephrase_tokens, locality_tokens = editor.prepare_balancedit_tokens(batch, questions=questions)
                extra = dict(rephrase_tokens=rephrase_tokens, locality_tokens=locality_tokens)
            if base_editor not in _BATCH_FIT_EDITORS:
                editor.edit(config, tokens, batch_history=batch_history, **extra)
            _cpu = base_editor in _BATCH_FIT_EDITORS
            tokens_copy = {k: (v.detach().cpu().clone() if (isinstance(v, torch.Tensor) and _cpu) else
                              (v.clone() if isinstance(v, torch.Tensor) else v))
                           for k, v in tokens.items()}
            if getattr(editor, "_soft_target", None) is not None:
                tokens_copy["_soft_target"] = editor._soft_target
            # Retrain editors consume the continuation term from history; the ids
            # are tiny CPU tensors (no image data), so store them directly.
            if _cpu and getattr(editor, "_soft_cont", None) is not None:
                tokens_copy["_soft_cont"] = editor._soft_cont
            batch_history.append(tokens_copy)
            del tokens

        batch_edit_time = time.time() - t2  # this batch's edit time
        cumulative_edit_time += batch_edit_time  # accumulate total edit time
        is_last = ( (total_batches is not None and (batch_idx + 1) == total_batches) or (max_batches is not None and (batch_idx + 1) == max_batches) )
        should_eval = ( (eval_every is None) or (eval_every <= 0) or ((batch_idx + 1) % int(eval_every) == 0) or is_last )
        if should_eval:
            # ft_retrain/mend_retrain: do one retrain on all accumulated history before eval
            if base_editor in _BATCH_FIT_EDITORS and batch_history:
                editor.edit(config, batch_history[-1], batch_history=batch_history[:-1])
            if hasattr(model, "model"):
                model.model.eval()
            # Full-set apply + generate only on the final eval: every peek metric is
            # computed inside editeval on its own (subsampled) copies with the editor
            # re-applied there, so this O(#edits) pass would only feed the model_new
            # print and grows linearly with the number of edits.
            if is_last:
                if _is_retrieval_editor(editor_name) and editor is not None and hasattr(editor, "apply_to_dataset"):
                    editor.apply_to_dataset(edit_ds_sofar)
                edit_ds_sofar.task_generate(model, use_cache=False)
                # Print edit application stats inline (for GRACE/BalancEdit)
                if hasattr(editor, "print_stats"):
                    editor.print_stats()
                if hasattr(editor, "reset_counters"):
                    editor.reset_counters()
                print10(edit_ds_sofar, label="model_new")
            print(f"Edit time (batch): {batch_edit_time:.2f}s, (cumulative): {cumulative_edit_time:.2f}s", flush=True)

            # Evaluate
            print("="*50, flush=True)
            print(f"Batch {batch_idx + 1}: evaluation", flush=True)
            t3 = time.time()
            related_texts = get_t_gen_input(dataset_name, edit_ds_sofar) if "text_generality" not in skip else {}
            related_images = get_i_gen_input(dataset_name, edit_ds_sofar, k=2) if "image_generality" not in skip else {}
            soft_gen_images = get_soft_gen_input_by_strength(dataset_name, edit_ds_sofar) if "soft_generality" not in skip else {}
            blend_gen_images = get_blend_gen_input_by_strength(dataset_name, edit_ds_sofar, strengths=BLEND_ALL_STRENGTHS) if "blend_generality" not in skip else {}
            # Every eval dumps per-sample post-edit predictions (edit + generality
            # sets) -- a near-free by-product for offline metric recomputation.
            # Final eval -> pred_postedit_dir itself (unchanged layout); intermediate
            # peeks (subsampled 50 edits) -> seq/batch_{N}/ so the sequential curve
            # can be re-scored offline too. Resume-safe: a redone block overwrites
            # its own batch folder.
            pred_dir = getattr(config, "pred_postedit_dir", None)
            if is_last or not pred_dir:
                dump_preds_dir = pred_dir
            else:
                dump_preds_dir = os.path.join(pred_dir, "seq", f"batch_{batch_idx + 1:06d}")
            batch_out_dict = editeval(
                model_old, model, edit_ds_sofar, editor,
                related_texts, related_images,
                soft_gen_images=soft_gen_images,
                blend_gen_images=blend_gen_images,
                unrelated_ds=loc_ds,
                edit_subsample_size=None if is_last else 50,
                edit_time=cumulative_edit_time,
                dump_preds_dir=dump_preds_dir,
                skip=skip,
            )
            batch_out_dict['batch_idx'] = batch_idx + 1
            batch_out_dict['finish_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            # batch_out_dict['reliability_old'] = reliability(model_old, pristine_ds_sofar)
            # print(f"Reliability (old): {batch_out_dict['reliability_old']:.4f}, (new): {batch_out_dict['reliability']:.4f}", flush=True)
            all_out_dicts.append(batch_out_dict)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(batch_out_dict, ensure_ascii=False, sort_keys=True) + "\n")
            n_written += 1

            # ---- editor-state checkpoint, aligned to this eval (crash-safe) ----
            # The JSONL line is appended above FIRST; the checkpoint is saved AFTER it
            # and records the current line count. A crash between the two leaves the
            # checkpoint one block behind, so on resume we truncate the JSONL back to
            # meta.jsonl_lines and redo just that block. Editors that don't yet
            # implement save_resume_state simply skip the save (they restart fresh).
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

            # Log a thin flat payload to wandb (full nested dicts stay in JSONL)
            if use_wandb and wandb.run is not None:
                wandb.log(_wandb_log_payload(batch_out_dict), step=batch_idx + 1)
            
            print(f"Eval time: {time.time() - t3:.2f}s | Saved to {out_path}", flush=True)

    if completed:
        mark_complete(edit_dir)
        print(f"[resume] run COMPLETE -> {edit_dir}/_resume/COMPLETE", flush=True)

    return all_out_dicts


