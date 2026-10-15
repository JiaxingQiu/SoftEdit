"""GTEdit: ground-truth oracle editor (retrieval-free ReasonEdit).

`reasonedit` stores one fact per edit and *retrieves* the nearest fact for each
query. `gt` removes the retrieval entirely: it records the ground-truth fact for
every edit keyed by uid, and at eval time it *always* prepends that edit's fact to
the edit itself and to all of its generality queries (text/image/soft), with zero
retrieval error.

This is the metric ceiling for a "prepend the answer" editor:
- the gap between a real editor and `gt` is retrieval/optimization loss;
- the gap between `gt` (single majority label) and `gt_soft` (ranked multi-label
  string) isolates the effect of *what answer string we inject* — the experiment:
  does providing multilabel improve JS distance / soft generality, independent of
  the editing method.

The injected string is keyed by the EDIT's uid and carries the EDIT's own gold, so
it applies uniformly to all of that edit's generality variants. It is deliberately
NOT the eval example's own gold (soft-generality overrides each variant's gold to a
single synthetic label; using it would make `gt` and `gt_soft` identical there).

Locality: unrelated queries have no uid in the fact map, so nothing is injected and
the base behavior is preserved — exactly what a perfect oracle should do.
"""

from typing import Dict

import torch
import torch.nn as nn


class GTEdit(nn.Module):
    """Ground-truth oracle: always inject each edit's own gold fact (single label)."""

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.device = getattr(config, "device", torch.device("cpu"))
        self.soft = False
        self._gt_facts: Dict[str, str] = {}  # str(uid) -> "New Fact: ... is <answer>."
        self._added_uids = set()

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    def generate(self, *a, **kw):
        return (self.model if hasattr(self.model, "generate") else self.wrapper).generate(*a, **kw)

    # ---- fact construction -------------------------------------------------
    def _edit_answer(self, ex) -> str:
        """Single majority-vote label (overridden by GTEditSoft for multilabel)."""
        return (ex.get("gold") or {}).get("label") or ex.get("answer") or ex.get("target") or ""

    @staticmethod
    def _fact(question: str, answer: str) -> str:
        # Same wording ReasonEdit stores as the codebook value, for a fair ceiling.
        return f"New Fact: The answer to '{question}' is {answer}." if answer else ""

    # ---- edit / apply (mirror ReasonEdit's external call sites) ------------
    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None):
        """Record the ground-truth fact for each new edit (no embedding, no codebook)."""
        if edit_ds is None:
            return self.model
        n_added = 0
        for ex in getattr(edit_ds, "data", []):
            uid = ex.get("uid") or (ex.get("image"), ex.get("question"))
            key = str(uid)
            if key in self._added_uids:
                continue
            fact = self._fact(ex.get("question", ""), self._edit_answer(ex))
            if not fact:
                continue
            self._gt_facts[key] = fact
            self._added_uids.add(key)
            n_added += 1
        print(f"[GTEdit] +{n_added} gt facts ({len(self._gt_facts)} total, soft={self.soft})", flush=True)
        return self.model

    def apply_to_dataset(self, dataset):
        """Prepend each example's edit-fact (looked up by uid) to its prompt.

        Called on the edit set (edit_utils, pre-eval) and on every generality/
        locality set (editeval._maybe_apply_ike). Unknown uids (locality) get no fact.
        """
        applied = 0
        data = getattr(dataset, "data", [])
        for ex in data:
            prompt_orig = ex.get("prompt_orig") or ex.get("prompt", "")
            fact = self._gt_facts.get(str(ex.get("uid")))
            if fact and prompt_orig:
                ex["prompt_orig"] = prompt_orig
                ex["prompt"] = f"{fact} {prompt_orig}"
                applied += 1
            else:
                ex["prompt"] = prompt_orig
        print(f"[GTEdit] applied gt facts to {applied}/{len(data)} examples", flush=True)

    # ---- resume ------------------------------------------------------------
    def save_resume_state(self) -> Dict:
        return {"_gt_facts": self._gt_facts, "_added_uids": self._added_uids}

    def load_resume_state(self, state: Dict) -> None:
        self._gt_facts = dict((state or {}).get("_gt_facts") or {})
        self._added_uids = set((state or {}).get("_added_uids") or set())
        print(f"[GTEdit] resume: restored {len(self._gt_facts)} gt facts", flush=True)


class GTEditSoft(GTEdit):
    """Multilabel oracle: inject the full ranked multi-label string per edit."""

    def __init__(self, config, model):
        super().__init__(config, model)
        self.soft = True

    def _edit_answer(self, ex) -> str:
        from .soft import ranked_label_str
        gold = ex.get("gold") or {}
        return ranked_label_str(gold) or gold.get("label") or ex.get("answer") or ""
