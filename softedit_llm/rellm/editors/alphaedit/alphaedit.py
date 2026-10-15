"""AlphaEdit editor (LLM), wired to the rankedit_llm editor interface.

Sequential, null-space-constrained locate-then-edit. Each ``edit()`` call applies
one edit round in place on the backbone's ``mlp.down_proj`` weights (a contiguous
early-middle layer range) and folds the edit's keys into ``cache_c`` so prior edits
join the preserved-knowledge set that the null-space projector ``P`` protects.

``P`` is deterministic from the wikipedia covariance and loaded from
``data/stats/`` (precompute with ``prep_alphaedit/compute_cov.py``); only the
edited weights + ``cache_c`` are checkpointed for resume.
"""
import torch

from .apply import apply_alphaedit
from .hparams import resolve_hparams
from .nethook import get_parameter, set_requires_grad
from .nullspace import build_projections


class AlphaEdit(torch.nn.Module):
    log_name = "alphaedit"  # overridden by the factory (e.g. "alphaedit_soft")

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None
        self.device = config.device
        self.soft = False
        self.soft_mode = "target"

        # Fact-lookup ("key") token = last question-content token, set per-edit by
        # the pipeline (rellm.run.edit_utils._set_edit_lookup). ``None`` -> fall
        # back to the last prompt token. Keying on the question (not the fixed
        # instruction/template suffix) keeps the null-space cache well-conditioned.
        self._uses_lookup_pos = True
        self._lookup_pos = None
        self._edit_question = None
        self._edit_prompt = None

        self.hparams = resolve_hparams(config)
        self.model_name = getattr(config.model, "name", self.model.config._name_or_path)

        # AlphaEdit is closed-form: no autograd on model params (only the z* delta).
        set_requires_grad(False, self.model)

        # Fan-in dim of the edited down_proj (shared across the layer range).
        w0 = get_parameter(self.model, self.hparams.rewrite_module_tmp.format(
            self.hparams.layers[0]) + ".weight")
        self.d_in = int(w0.shape[1])

        # Null-space projections P (loaded from cache; heavy to compute — run
        # prep_alphaedit/compute_cov.py first). Kept on CPU; sliced to GPU per solve.
        print(f"[alphaedit] layers={self.hparams.layers} d_in={self.d_in} "
              f"nullspace_threshold={self.hparams.nullspace_threshold} "
              f"mom2_n_samples={self.hparams.mom2_n_samples}", flush=True)
        self.P = build_projections(
            self.model, self.tokenizer, self.hparams, model_name=self.model_name)
        p_gb = self.P.numel() * self.P.element_size() / (1024 ** 3)
        cc_gb = (self.d_in * self.d_in * 4 * len(self.hparams.layers)) / (1024 ** 3)
        print(f"[alphaedit] P {tuple(self.P.shape)} = {p_gb:.1f} GiB CPU; "
              f"cache_c another {cc_gb:.1f} GiB. n_samples does not change this "
              f"(size is n_layers × d_in²).", flush=True)

        # Sequential preservation memory (accumulates edited keys), CPU float32.
        self.cache_c = [torch.zeros(self.d_in, self.d_in, dtype=torch.float32)
                        for _ in self.hparams.layers]

        # Batched sequential editing: buffer edits and solve a chunk of up to
        # ``num_edits`` jointly (rank-N keys / cache_c fold). The pipeline flushes
        # the buffer before every eval / checkpoint (edit_utils), so the model
        # always reflects all buffered edits at those points.
        self.num_edits = max(1, int(getattr(self.hparams, "num_edits", 1)))
        self._buffer = []

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    def generate(self, *a, **kw):
        gen = self.model if hasattr(self.model, "generate") else self.wrapper
        return gen.generate(*a, **kw)

    def edit(self, config, tokens, batch_history=None):
        """Buffer one edit (batch size 1); solve a chunk when the buffer is full.

        The buffer snapshots everything the deferred solve needs — tokens for z*,
        the per-edit lookup position, the soft-target context, and the raw
        question/prompt for context-averaged keys — because the editor's live
        ``_lookup_pos`` / ``_soft_*`` attributes are overwritten by the next edit.
        """
        soft = bool(self.soft) and getattr(self, "soft_mode", "target") in ("loss", "rank_loss")
        entry = {
            "tokens": {k: tokens[k].detach() for k in ("input_ids", "attention_mask", "labels")
                       if k in tokens},
            "lookup_pos": getattr(self, "_lookup_pos", None),
            "soft": soft,
            "soft_target": getattr(self, "_soft_target", None),
            "soft_cont": getattr(self, "_soft_cont", None),
            "question": getattr(self, "_edit_question", None),
            "prompt": getattr(self, "_edit_prompt", None),
        }
        self._buffer.append(entry)
        if len(self._buffer) >= self.num_edits:
            self.flush(config)
        return self.model

    def flush(self, config=None):
        """Solve and commit all buffered edits (one rank-N AlphaEdit round). No-op
        when the buffer is empty. Called on chunk-full and by the pipeline before
        every eval / checkpoint so the model reflects all buffered edits."""
        if not self._buffer:
            return self.model
        buf, self._buffer = self._buffer, []
        apply_alphaedit(self.model, buf, self.hparams, self.P, self.cache_c, self)
        return self.model

    def _edited_weight_names(self):
        return [self.hparams.rewrite_module_tmp.format(l) + ".weight"
                for l in self.hparams.layers]

    def save_resume_state(self):
        """Edited down_proj weights + the sequential cache (P is recomputed/loaded)."""
        # Defensive: never checkpoint with buffered-but-unsolved edits (the pipeline
        # also flushes before saving, but a stale buffer would silently drop edits).
        self.flush()
        weights = {name: get_parameter(self.model, name).detach().cpu().clone()
                   for name in self._edited_weight_names()}
        return {
            "weights": weights,
            "cache_c": [c.detach().cpu().clone() for c in self.cache_c],
        }

    def load_resume_state(self, state):
        if not state:
            return
        with torch.no_grad():
            for name, w in state.get("weights", {}).items():
                p = get_parameter(self.model, name)
                p.copy_(w.to(p.device, p.dtype))
        cc = state.get("cache_c", None)
        if cc is not None:
            self.cache_c = [c.detach().cpu().clone().float() for c in cc]
        print(f"[alphaedit] resume: restored {len(state.get('weights', {}))} edited "
              f"layers + cache_c", flush=True)


class AlphaEditSoft(AlphaEdit):
    """Multilabel AlphaEdit: z* is optimized toward the ranked multi-label
    distribution using the shared Stage-6 soft objective (first-token KL +
    continuation), identical to the other ``*_soft`` editors. The pipeline attaches
    ``_soft_target`` / ``_soft_cont`` (distribution modes) or swaps the target to
    the ranked string (``soft="target"``) before calling ``edit``.
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        self.soft = True
        self.soft_mode = getattr(config.editor, "soft", "loss")
        self.rank_temp = float(getattr(config.editor, "rank_temp", 1.0))
