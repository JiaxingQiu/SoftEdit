"""Resume helpers for sequential and batch editing (LLM analog).

Sequential: `_resume/ckpt.pt` + `_resume/COMPLETE`. Resume is the default;
`--overwrite` wipes state. The sequential path never reads `batch_ckpt.pt`.

Batch: `_resume/batch_ckpt.pt` holds editor weights after Step 2 so a resubmit
can skip re-editing. `_resume/batch_history.pt` holds the tokenized retrain
history so a mid-training crash can skip re-tokenizing. Per-set eval dumps
under `pred_postedit_dir` let Step 3 skip completed metric sets. The batch
path never reads sequential `ckpt.pt`.

All editor-specific (de)serialization lives on the editors themselves via
`save_resume_state` / `load_resume_state`; this module is editor-agnostic.
"""

import hashlib
import json
import os
import random
import shutil

import numpy as np
import torch

_RESUME_SUBDIR = "_resume"
_CKPT_NAME = "ckpt.pt"
_BATCH_CKPT_NAME = "batch_ckpt.pt"
_BATCH_HISTORY_NAME = "batch_history.pt"
_COMPLETE_NAME = "COMPLETE"


def resume_dir(edit_dir: str) -> str:
    return os.path.join(edit_dir, _RESUME_SUBDIR)


def _ckpt_path(edit_dir: str) -> str:
    return os.path.join(resume_dir(edit_dir), _CKPT_NAME)


def _complete_path(edit_dir: str) -> str:
    return os.path.join(resume_dir(edit_dir), _COMPLETE_NAME)


def is_complete(edit_dir: str) -> bool:
    return os.path.exists(_complete_path(edit_dir))


def mark_complete(edit_dir: str) -> None:
    os.makedirs(resume_dir(edit_dir), exist_ok=True)
    with open(_complete_path(edit_dir), "w", encoding="utf-8") as f:
        f.write("")


def clear_resume(edit_dir: str) -> None:
    """Wipe all resume state (used by --overwrite)."""
    d = resume_dir(edit_dir)
    if os.path.isdir(d):
        shutil.rmtree(d)


def save_ckpt(edit_dir: str, meta: dict, editor_state) -> None:
    """Atomically write the checkpoint (tmp + os.replace)."""
    os.makedirs(resume_dir(edit_dir), exist_ok=True)
    path = _ckpt_path(edit_dir)
    tmp = path + ".tmp"
    torch.save({"meta": meta, "editor": editor_state}, tmp)
    os.replace(tmp, path)


def load_ckpt(edit_dir: str):
    """Return the checkpoint dict, or None if there is none."""
    path = _ckpt_path(edit_dir)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _batch_ckpt_path(edit_dir: str) -> str:
    return os.path.join(resume_dir(edit_dir), _BATCH_CKPT_NAME)


def save_batch_ckpt(edit_dir: str, meta: dict, editor_state) -> None:
    """Atomically write the batch-edit checkpoint (tmp + os.replace).

    Separate from sequential ``ckpt.pt`` so the two paths cannot clobber each other.
    """
    os.makedirs(resume_dir(edit_dir), exist_ok=True)
    path = _batch_ckpt_path(edit_dir)
    tmp = path + ".tmp"
    torch.save({"meta": meta, "editor": editor_state}, tmp)
    os.replace(tmp, path)


def load_batch_ckpt(edit_dir: str):
    """Return the batch checkpoint dict, or None if there is none."""
    path = _batch_ckpt_path(edit_dir)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def clear_batch_ckpt(edit_dir: str) -> None:
    """Remove only the batch checkpoint (leaves sequential ``ckpt.pt`` alone)."""
    path = _batch_ckpt_path(edit_dir)
    if os.path.exists(path):
        os.remove(path)


def _batch_history_path(edit_dir: str) -> str:
    return os.path.join(resume_dir(edit_dir), _BATCH_HISTORY_NAME)


def save_batch_history(edit_dir: str, fingerprint: str, history) -> None:
    """Persist the tokenized retrain history so a crash mid-training can skip re-tokenizing."""
    os.makedirs(resume_dir(edit_dir), exist_ok=True)
    path = _batch_history_path(edit_dir)
    tmp = path + ".tmp"
    torch.save({"fingerprint": fingerprint, "history": history}, tmp)
    os.replace(tmp, path)


def load_batch_history(edit_dir: str, fingerprint: str):
    """Return the saved tokenized history, or None if missing / fingerprint mismatch."""
    path = _batch_history_path(edit_dir)
    if not os.path.exists(path):
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("fingerprint") != fingerprint:
        print("[resume] batch_history fingerprint mismatch - ignoring saved history", flush=True)
        return None
    return blob.get("history")


def clear_batch_history(edit_dir: str) -> None:
    path = _batch_history_path(edit_dir)
    if os.path.exists(path):
        os.remove(path)


def backup_ckpt(edit_dir: str):
    """Copy ``ckpt.pt`` to ``ckpt.pt.bak.<N>`` before an in-place rewrite."""
    path = _ckpt_path(edit_dir)
    if not os.path.exists(path):
        return None
    n = 0
    while True:
        bak = f"{path}.bak.{n}"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
            print(f"[resume] backed up ckpt -> {bak}", flush=True)
            return bak
        n += 1


def edit_fingerprint(config, edit_ds) -> str:
    """Hash everything that determines the edit set and its order."""
    payload = {
        "uids": [str(ex.get("uid")) for ex in getattr(edit_ds, "data", [])],
        "seed": getattr(config, "seed", None),
        "subsample_edits": getattr(config, "subsample_edits", 0),
        "n_edit_cap": getattr(config, "n_edit_cap", None),
        "editor": getattr(getattr(config, "editor", None), "_name", ""),
        "model": getattr(getattr(config, "model", None), "name", ""),
        "dataset": getattr(getattr(config, "experiment", None), "dataset_name", ""),
        "split": getattr(getattr(config, "experiment", None), "split", ""),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def truncate_jsonl(path: str, n_lines: int) -> None:
    """Keep only the first n_lines of a JSONL file."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) <= n_lines:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines[:n_lines])


def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as e:
        print(f"[resume] RNG restore skipped: {e}", flush=True)
