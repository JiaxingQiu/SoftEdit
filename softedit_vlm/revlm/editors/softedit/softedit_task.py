"""SoftEditTask: outcome-steered retrieval key for single-task datasets.

Wraps :class:`SoftEdit` with a lightweight online probe that maps the key-layer
activations to task-outcome probabilities. The probe output is concatenated with
the base key at retrieval time::

    e(x) = [ base_key(x) ;  gamma * z(x) ]

where ``z(x) = softmax(W @ scaler(key_act(x)) + b)`` is trained on the edits
collected so far (each edit carries its gold label via ``entry["p"]``).

Editing / actuation are identical to SoftEdit (weight-update codebook). Only the
retrieval key is augmented — the probe is trained and keys re-concatenated inside
``apply_to_dataset``, before retrieval.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from typing import Dict, List

from .softedit import SoftEdit


class SoftEditTask(SoftEdit):
    """SoftEdit + online outcome probe for the retrieval key."""

    def __init__(self, config, model):
        super().__init__(config, model)
        cfg = getattr(config, "editor", config)

        # Task variant default is tighter; honor auto / auto100 from the parent parse.
        raw = getattr(cfg, "reject_threshold_pct", 10.0)
        if not (isinstance(raw, str) and raw.strip().lower() in ("auto", "auto100")):
            self._reject_auto = False
            self._auto_mode = None
            self.reject_threshold_pct = float(raw)
        self.probe_gamma = float(getattr(cfg, "probe_gamma", 200.0))
        self.probe_lr = float(getattr(cfg, "probe_lr", 1e-2))
        self.probe_n_iter = int(getattr(cfg, "probe_n_iter", 2000))
        self.probe_wd = float(getattr(cfg, "probe_weight_decay", 1e-2))

        self._probe: nn.Linear = None
        self._probe_scaler: StandardScaler = None
        self._probe_classes: List[str] = []
        self._last_key_act = None  # pooled key-layer activation from the most recent _encode

    # ------------------------------------------------------------------
    # encode / edit: base keys only; stash activation for probe training
    # ------------------------------------------------------------------

    def _encode(self, image, question):
        key = super()._encode(image, question)
        # Capture activation from THIS forward (the key hook fired during super()).
        self._last_key_act = self._pool_key(self._key_act, 1)[0].cpu().float()
        if self._probe is not None and self.probe_gamma > 0:
            z = self._probe_z(self._last_key_act)
            key = torch.cat([key, self.probe_gamma * z])
        return key

    def _build_entry(self, image, question, gold, uid, tag="", prompt=None):
        entry = super()._build_entry(image, question, gold, uid, tag=tag, prompt=prompt)
        if entry is None:
            return None
        entry["vis"] = self._last_key_act
        return entry

    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None):
        # Editing must store dual keys only. A previous apply may have left a
        # composite key_embs / an active probe — restore dual before SoftEdit
        # appends, otherwise dimensions mismatch.
        self._probe = None
        if self.codebook:
            self.key_embs = torch.stack([e["key"] for e in self.codebook])
            self.key_dist_threshold = None
        return super().edit(config, tokens=tokens, batch_history=batch_history,
                            edit_ds=edit_ds, train_ds=train_ds)

    def _maybe_auto_tune(self):
        # Retrieval keys are the probe-augmented ones; wait until apply rekeys.
        if self._probe is None and self.probe_gamma > 0:
            return
        super()._maybe_auto_tune()

    # ------------------------------------------------------------------
    # probe
    # ------------------------------------------------------------------

    def _train_probe(self):
        """Train a linear probe: key-layer activations -> gold label index."""
        if self.probe_gamma <= 0 or len(self.codebook) < 2:
            self._probe = None
            return

        gold_labels = [e["labels"][int(e["p"].argmax())] for e in self.codebook]
        classes = sorted(set(gold_labels))
        if len(classes) < 2:
            print("[SoftEditTask] <2 classes in codebook, probe disabled", flush=True)
            self._probe = None
            return
        cls_to_idx = {c: i for i, c in enumerate(classes)}

        X = torch.stack([e["vis"] for e in self.codebook]).numpy()
        y = np.array([cls_to_idx[l] for l in gold_labels])

        self._probe_scaler = StandardScaler().fit(X)
        X_sc = self._probe_scaler.transform(X).astype(np.float32)

        d_in, n_cls = X_sc.shape[1], len(classes)
        probe = nn.Linear(d_in, n_cls)
        nn.init.zeros_(probe.bias)
        nn.init.normal_(probe.weight, std=0.01)

        X_t = torch.tensor(X_sc)
        y_t = torch.tensor(y, dtype=torch.long)
        opt = optim.AdamW(probe.parameters(), lr=self.probe_lr, weight_decay=self.probe_wd)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.probe_n_iter)

        probe.train()
        for _ in range(self.probe_n_iter):
            loss = nn.functional.cross_entropy(probe(X_t), y_t)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        probe.eval()

        acc = (probe(X_t).argmax(1) == y_t).float().mean()
        print(f"[SoftEditTask] probe trained: {len(self.codebook)} edits, "
              f"{n_cls} classes, acc={acc:.3f}, gamma={self.probe_gamma}", flush=True)

        self._probe = probe
        self._probe_classes = classes

    @torch.no_grad()
    def _probe_z(self, act: torch.Tensor) -> torch.Tensor:
        """act: [d] cpu float -> z: [n_classes] cpu float."""
        x = self._probe_scaler.transform(act.unsqueeze(0).numpy()).astype(np.float32)
        return torch.softmax(self._probe(torch.tensor(x)), dim=-1)[0]

    def _rekey_codebook(self):
        """Rebuild key_embs as [base_key ; gamma * z] from stored base keys."""
        keys = []
        for entry in self.codebook:
            z = self._probe_z(entry["vis"])
            keys.append(torch.cat([entry["key"], self.probe_gamma * z]))
        self.key_embs = torch.stack(keys)
        self.key_dist_threshold = None
        self._auto_tuned_at = None  # composite keys changed; apply will retune

    # ------------------------------------------------------------------
    # apply: retrain probe, re-key, then SoftEdit retrieval / patch
    # ------------------------------------------------------------------

    def apply_to_dataset(self, dataset):
        if self._probe is None and self.probe_gamma > 0:
            self._train_probe()
            if self._probe is not None:
                self._rekey_codebook()
        super().apply_to_dataset(dataset)

    # ------------------------------------------------------------------
    # resume IO
    # ------------------------------------------------------------------

    def save_resume_state(self) -> Dict:
        state = super().save_resume_state()
        state["schema"] = "softedit_task_weight_v1"
        state["probe_classes"] = self._probe_classes
        return state

    def load_resume_state(self, state: Dict) -> None:
        state = state or {}
        schema = state.get("schema", "")
        if schema.startswith("softedit_task"):
            self._probe_classes = state.get("probe_classes", [])
            state["schema"] = "softedit_weight_v1"  # parent schema check
        super().load_resume_state(state)
        self._probe = None  # retrained on next apply_to_dataset
        if self.codebook:
            self.key_embs = torch.stack([e["key"] for e in self.codebook])
