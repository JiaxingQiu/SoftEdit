"""SoftEdit (LLM): retrieval codebook of small lm_head weight updates.

Text-only adaptation of the VLM SoftEdit. Front-end is an SBERT nearest-neighbor
over the question (same retrieval space as our IKE) with a leave-one-out
calibrated reject gate and an optional paraphrase judge. The **codebook value**
is a tiny weight update ``ΔW`` (+ ``ΔW_cont``) on the lm_head rows of the edit's
candidate answers. Fully self-contained.

    edit:  train ΔW so softmax((W + ΔW) h0) matches the human proportions p*,
           regularized by the generality (graph) + locality anchor terms
           (see ``train.py`` / ``generality.py`` / ``locality.py``).
    apply: encode the query key, retrieve the nearest edit (reject gate + judge),
           and — on a hit — write ΔW into the lm_head **before** generation for
           that sample, restoring the original rows afterwards. No prompt change,
           no inject: the edit is realized in the model's own forward.

Each codebook entry::

    key        : SBERT(question)  [d_key] cpu float32
    labels/first_ids/p : the K candidates + human gold distribution
    dW         : [K, d]  first-token label-row update
    dW_cont    : [C, d]  continuation-row update (None if continuation off / single-token)
    cont_ids   : [C]     rows dW_cont edits

Note on tied embeddings: with ``tie_word_embeddings=True`` (Phi/Qwen) the lm_head
weight is shared with the input embeddings, so patching a row also nudges that
token's embedding. Because the patch is applied only for the current sample and
restored right after generation, and answer tokens rarely appear in the prompt,
the effect on the forward is negligible.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional

from ..soft_utils import build_soft_target
from ..utils import parent_module, brackets_to_periods
from . import train


# reject_threshold_pct:
#   "auto100" (default) — tune once at 100 edits, then freeze pct
#   "auto"              — tune at 100, then every 1000 as the codebook grows
#   <float>             — fixed percentile
_AUTO_FIRST = 100
_AUTO_EVERY = 1000
_AUTO_START = 99          # sweep 99, 95, 90, ..., floor
_AUTO_FLOOR = 10
# Max fewer hits vs pct=99. 0 = no drop (strict). Default 5 allows a small window.
_AUTO_MAX_DROP = 5
_AUTO_N_SAMPLE = 100
_AUTO_N_PERTURB = 5    # independent rephrase probes per sampled edit
_AUTO_PCT_STEP = 5
_AUTO_SEED = 0


def _auto_pcts():
    """Descending pct grid: 99, 95, 90, ..., floor."""
    rest = list(range(95, _AUTO_FLOOR - 1, -_AUTO_PCT_STEP))
    return [_AUTO_START] + [p for p in rest if p != _AUTO_START]


def _parse_reject_threshold(cfg, default="auto100"):
    """Return ``(mode, pct)``. ``mode`` is ``"auto"``, ``"auto100"``, or ``None``."""
    raw = getattr(cfg, "reject_threshold_pct", default)
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("auto", "auto100"):
            return s, float(_AUTO_START)
        raise ValueError(
            f"reject_threshold_pct string must be 'auto' or 'auto100', got {raw!r}")
    return None, float(raw)


def _auto_calib_n(n: int, mode: str = "auto100") -> int:
    """Codebook prefix for the next calibration, or 0 if none yet."""
    n = int(n)
    if n < _AUTO_FIRST:
        return 0
    if mode == "auto100":
        return _AUTO_FIRST
    return _AUTO_FIRST + ((n - _AUTO_FIRST) // _AUTO_EVERY) * _AUTO_EVERY


def _make_arm(editor):
    """Dataset-level pre-generation hook: patch lm_head for each batch's edit.

    A plain function (not a bound method) so ``copy.deepcopy`` — which editeval
    runs on datasets — shares this closure instead of deep-copying the whole
    editor (and its LLM).
    """
    def pregen_arm(examples):
        editor._arm(examples)
    return pregen_arm


class SoftEdit(nn.Module):
    """Retrieval codebook of lm_head weight updates for multilabel edits."""

    log_name = "softedit"

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        cfg = getattr(config, "editor", config)

        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None
        self.device = getattr(config, "device", torch.device("cpu"))

        # ---- retrieval config ----
        # key_mode: how the retrieval key is built (retrieval is always L2 NN).
        #   "sbert"         : SBERT(question)                    [default; same as IKE]
        #   "language"      : config.model.inner_params[0] activation
        #                     (first proj of the last decoder block)
        #   "language_last" : input to lm_head (the final hidden state)
        # pool_method (for the two language modes): "mean" over the prompt or
        #   "last" (the generation position).
        self.key_mode = str(getattr(cfg, "key_mode", "sbert"))
        if self.key_mode not in ("sbert", "language", "language_last"):
            raise ValueError(f"SoftEdit key_mode must be 'sbert', 'language', or "
                             f"'language_last', got {self.key_mode!r}")
        self.pool_method = str(getattr(cfg, "pool_method", "mean"))
        if self.pool_method not in ("mean", "last"):
            raise ValueError(f"SoftEdit pool_method must be 'mean' or 'last', "
                             f"got {self.pool_method!r}")
        self.sentence_model_name = getattr(
            cfg, "sentence_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self._auto_mode, self.reject_threshold_pct = _parse_reject_threshold(cfg)
        self._reject_auto = self._auto_mode is not None
        self._auto_tuned_at = None       # codebook size last calibrated at (auto)
        self.last_auto_log = None
        try:
            self.auto_max_drop = max(0, int(getattr(cfg, "auto_max_drop", _AUTO_MAX_DROP)))
        except (TypeError, ValueError):
            self.auto_max_drop = int(_AUTO_MAX_DROP)
        if self._auto_mode == "auto100":
            print("[SoftEdit] reject_threshold_pct=auto100 "
                  f"(tune once at {_AUTO_FIRST} edits; floor {_AUTO_FLOOR}; "
                  f"max_drop={self.auto_max_drop})",
                  flush=True)
        elif self._auto_mode == "auto":
            print("[SoftEdit] reject_threshold_pct=auto "
                  f"(tune at {_AUTO_FIRST}, then every {_AUTO_EVERY}; "
                  f"floor {_AUTO_FLOOR}; max_drop={self.auto_max_drop})",
                  flush=True)
        self.judge_paraphrase = bool(getattr(cfg, "judge_paraphrase", True))

        # ---- soft target config ----
        self.soft_mode = getattr(cfg, "soft", "loss")
        self.rank_temp = float(getattr(cfg, "rank_temp", 1.0))
        # major_vote (ablation): collapse target distribution to one-hot on the
        # majority label; everything else (training, eval) held the same.
        self.major_vote = bool(getattr(cfg, "major_vote", False))
        self.debug = bool(getattr(config, "debug", False))

        # ---- weight-update (training) config ----
        self.continuation = bool(getattr(cfg, "continuation", True))
        _raw_gen = getattr(cfg, "lambda_gen", "auto")
        if isinstance(_raw_gen, str) and _raw_gen.strip().lower() == "auto":
            self._lambda_gen_auto = True
            self.lambda_gen = None          # set by resolve_lambda_gen
        else:
            self._lambda_gen_auto = False
            self.lambda_gen = float(_raw_gen)
        _raw_loc = getattr(cfg, "lambda_loc", "auto")
        if isinstance(_raw_loc, str) and _raw_loc.strip().lower() == "auto":
            self._lambda_loc_auto = True
            self.lambda_loc = None          # set before first edit
        else:
            self._lambda_loc_auto = False
            self.lambda_loc = float(_raw_loc) if _raw_loc is not None else None
        # #edits sampled by the auto-λ_loc pre-step (resolve_lambda_loc); read
        # from config here for visibility/ablation even though the pre-step runs
        # before this editor is constructed.
        self.lambda_loc_n_edits = int(getattr(cfg, "lambda_loc_n_edits", 100))
        self.lambda_reg = float(getattr(cfg, "lambda_reg", 0.0))
        self.locality_n = int(getattr(cfg, "locality_n", 1))
        self.locality_declarative = str(getattr(cfg, "locality_declarative", "both"))
        self.jac_mode = str(getattr(cfg, "jac_mode", "prob"))
        self.edit_lr = float(getattr(cfg, "edit_lr", 1e-3))
        self.n_steps = int(getattr(cfg, "n_steps", 5000))
        self.loss_eps = float(getattr(cfg, "loss_eps", 1e-2))
        self.align_eps = float(getattr(cfg, "align_eps", 1e-4))
        self.sched_mode = str(getattr(cfg, "sched_mode", "plateau"))
        self.edit_verbose = bool(getattr(cfg, "edit_verbose", True))

        # ---- codebook + caches ----
        self.codebook: List[Dict] = []
        self.key_embs = None            # [N, d_key] cpu float32
        self.key_dist_threshold = None
        self._added_uids = set()
        self._sbert = None
        self._judge = None

        # ---- hooks / state ----
        self._patch_backup = None       # active lm_head restore backup (apply time)
        self.last_retrieval_log = None
        self._key_act = None            # last captured activation (hidden/dual key)

        # lm_head we edit
        self.head = train.resolve_lm_head(self.model)

        # Retrieval-key hook (only for hidden / dual modes).
        self._key_hook = self._register_key_hook()

    # ==================== forward / generate ====================

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    def generate(self, *a, **kw):
        return (self.model if hasattr(self.model, "generate") else self.wrapper).generate(*a, **kw)

    # ==================== encoding / retrieval key ====================

    def _get_sbert(self):
        if self._sbert is None:
            print(f"[SoftEdit] loading SBERT ({self.sentence_model_name}) ...", flush=True)
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(self.sentence_model_name)
            self._sbert.to(self.device)
        return self._sbert

    def _get_judge(self):
        if self._judge is None:
            from .judge import ParaphraseJudge
            self._judge = ParaphraseJudge()
        return self._judge

    def _register_key_hook(self):
        """Forward hook capturing the activation used as the language key.

        Stores the raw activation in ``self._key_act`` on every model forward;
        ``_encode_hidden`` pools the one from its own forward. ``None`` for the
        SBERT key (no model forward needed).
        """
        if self.key_mode == "sbert":
            return None

        if self.key_mode == "language_last":
            # Input to lm_head = the final hidden state.
            def _pre(module, args):
                self._key_act = args[0].detach() if isinstance(args[0], torch.Tensor) else None
            return self.head.register_forward_pre_hook(_pre)

        # "language": hook config.model.inner_params[0] (first proj of the last block).
        model_cfg = getattr(self.config, "model", self.config)
        inner = getattr(model_cfg, "inner_params", [])
        if not inner:
            raise ValueError("SoftEdit key_mode='language' requires config.model.inner_params")
        name = inner[0]
        name = name.rsplit(".", 1)[0] if name.endswith((".weight", ".bias")) else name
        mod = parent_module(self.model, brackets_to_periods(name))
        layer = getattr(mod, name.rsplit(".", 1)[-1])

        def _fwd(module, inp, output):
            self._key_act = inp[0].detach() if isinstance(inp[0], torch.Tensor) else None
        return layer.register_forward_hook(_fwd)

    def _pool_key(self, act: torch.Tensor) -> torch.Tensor:
        """Pool a captured activation [.., seq, d] → [d] (mean or last over seq)."""
        a = act.to(self.device, torch.float32)
        if a.dim() == 3:
            a = a[0]
        if a.dim() == 1:
            return a
        return a[-1] if self.pool_method == "last" else a.mean(dim=0)

    def _encode_sbert(self, question: str) -> torch.Tensor:
        emb = self._get_sbert().encode([question], convert_to_tensor=True,
                                       show_progress_bar=False)
        return emb[0].to("cpu", torch.float32)

    @torch.no_grad()
    def _encode_hidden(self, question: str) -> torch.Tensor:
        """One LLM forward → pooled key-layer activation [d] (cpu)."""
        self._key_act = None
        inputs = self.wrapper.encode([question], tokenize=False)
        self.model(**inputs)
        if self._key_act is None:
            raise RuntimeError("SoftEdit: no key activation captured "
                               f"(key_mode={self.key_mode!r}); check the hook.")
        return self._pool_key(self._key_act).to("cpu", torch.float32)

    @torch.no_grad()
    def _encode(self, question: str) -> torch.Tensor:
        """Build the retrieval key [d_key] (cpu); retrieval is L2 nearest-neighbor."""
        if self.key_mode == "sbert":
            return self._encode_sbert(question)
        return self._encode_hidden(question)

    # ==================== edit (build codebook) ====================

    def _target(self, gold: Dict):
        st = build_soft_target(gold or {}, self.tokenizer, mode=self.soft_mode,
                               rank_temp=self.rank_temp, major_vote=self.major_vote)
        return st.labels, st.first_ids.to(self.device), st.q

    def _build_entry(self, question, gold, uid, tag="", prompt=None):
        labels, first_ids, p = self._target(gold)
        K = len(labels)
        if K == 0:
            return None

        key = self._encode(question)
        print(f"[SoftEdit]{tag} building entry: K={K} labels={labels}", flush=True)

        eng_prompt = prompt or question
        dW, dW_cont, cont_ids = train.train_edit(
            model=self.model, wrapper=self.wrapper, tokenizer=self.tokenizer,
            head=self.head, question=question, prompt=eng_prompt,
            labels=labels, first_ids=first_ids, p=p,
            continuation=self.continuation,
            lambda_gen=self.lambda_gen, lambda_loc=self.lambda_loc,
            lambda_reg=self.lambda_reg,
            locality_n=self.locality_n,
            locality_declarative=self.locality_declarative, jac_mode=self.jac_mode,
            lr=self.edit_lr, n_steps=self.n_steps, sched_mode=self.sched_mode,
            loss_eps=self.loss_eps, align_eps=self.align_eps,
            verbose=self.edit_verbose,
        )
        return {
            "uid": uid, "question": question, "labels": labels,
            "first_ids": first_ids.cpu(), "p": p.cpu(), "key": key,
            "dW": dW, "dW_cont": dW_cont, "cont_ids": cont_ids,
        }

    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None):
        """Add codebook entries for each new edit (retrieval-editor signature)."""
        if edit_ds is None:
            return self.model
        for ex in getattr(edit_ds, "data", []):
            uid = ex.get("uid") or ex.get("question")
            key = str(uid)
            if key in self._added_uids:
                continue
            q = ex.get("question", "")
            if not q:
                continue
            tag = f" edit {len(self.codebook) + 1}"
            prompt = ex.get("prompt_orig") or ex.get("prompt") or q
            entry = self._build_entry(q, ex.get("gold") or {}, key, tag=tag,
                                      prompt=prompt)
            if entry is None:
                continue
            self.codebook.append(entry)
            emb = entry["key"].unsqueeze(0)
            self.key_embs = emb if self.key_embs is None else torch.cat([self.key_embs, emb], dim=0)
            self._added_uids.add(key)

        self.key_dist_threshold = None
        self._maybe_auto_tune()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.model

    # ==================== auto reject_threshold_pct ====================

    def _maybe_auto_tune(self):
        """Tune ``reject_threshold_pct`` on the auto / auto100 schedule."""
        if not self._reject_auto:
            return
        C = _auto_calib_n(len(self.codebook), self._auto_mode or "auto100")
        if C == 0:
            return
        if self._auto_mode == "auto100":
            if self._auto_tuned_at is not None:
                return
        elif C == self._auto_tuned_at:
            return
        self.recalibrate_reject_threshold_pct(upto=C)
        self._auto_tuned_at = C

    def recalibrate_reject_threshold_pct(self, upto: Optional[int] = None) -> float:
        """Pick the smallest pct (floor 10) whose hit drop vs pct=99 is ≤ ``auto_max_drop``.

        Samples 100 codebook edits (or all, if fewer), builds 5 rephrase
        perturbations each, and scores them with the live retriever (L2 gate
        + judge). ``upto`` restricts the codebook prefix.
        """
        n = len(self.codebook)
        C = n if upto is None else max(0, min(int(upto), n))
        if C < 2 or self.key_embs is None:
            self.reject_threshold_pct = float(_AUTO_START)
            self.key_dist_threshold = None
            print(f"[SoftEdit] auto: codebook too small, pct={_AUTO_START}", flush=True)
            return self.reject_threshold_pct

        keys = self.key_embs[:C].float()
        entries = self.codebook[:C]
        rng = np.random.RandomState(_AUTO_SEED)
        n_sample = min(_AUTO_N_SAMPLE, C)
        sample_idx = rng.choice(C, size=n_sample, replace=False)

        from .perturb import Augmenter
        aug = Augmenter.shared()
        q_keys, q_text, parents = [], [], []
        for i, pidx in enumerate(sample_idx):
            pidx = int(pidx)
            q0 = entries[pidx].get("question") or ""
            if not q0:
                continue
            for k in range(_AUTO_N_PERTURB):
                try:
                    q = aug.perturb(q0, _AUTO_SEED + 1000 * i + k)
                    q_keys.append(self._encode(q))
                    q_text.append(q)
                    parents.append(pidx)
                except Exception as exc:
                    print(f"[SoftEdit] auto: perturb failed (i={i}, k={k}): {exc}",
                          flush=True)
                    continue

        if not q_keys:
            print(f"[SoftEdit] auto: no generality probes, pct={_AUTO_START}", flush=True)
            self.reject_threshold_pct = float(_AUTO_START)
            self.key_dist_threshold = None
            return self.reject_threshold_pct

        Q = torch.stack(q_keys).float()
        dists = torch.cdist(Q, keys)                       # [n_q, C]
        dmin, jstar = dists.min(dim=1)
        dmin = dmin.numpy()
        jstar = jstar.numpy()
        parents = np.asarray(parents, dtype=np.int64)

        judge_ok = np.ones(len(jstar), dtype=bool)
        if self.judge_paraphrase:
            judge = self._get_judge()
            for i, (q, j) in enumerate(zip(q_text, jstar)):
                keep, _ = judge.accept(q, entries[int(j)].get("question") or "")
                judge_ok[i] = bool(keep)

        correct = (jstar == parents) & judge_ok
        d = torch.cdist(keys, keys)
        d.fill_diagonal_(float("inf"))
        loo_nn = d.min(dim=1).values.numpy()

        n_q = max(len(correct), 1)
        pcts = _auto_pcts()
        sweep = []
        chosen = _AUTO_START
        base_n = None
        for pct in pcts:
            tau = float(np.percentile(loo_nn, pct))
            n_hit = int((correct & (dmin <= tau)).sum())
            if base_n is None:
                base_n = n_hit
            if n_hit >= base_n - self.auto_max_drop:
                chosen = pct
            sweep.append({
                "pct": pct, "tau": tau, "n_hit": n_hit,
                "acc": n_hit / n_q, "drop": n_hit - base_n,
            })

        self.reject_threshold_pct = float(chosen)
        self.key_dist_threshold = None
        acc0 = sweep[0]["acc"]
        acc_ch = next(r["acc"] for r in sweep if r["pct"] == chosen)
        self.last_auto_log = {
            "upto": C, "n_probes": len(correct), "n_hit_99": base_n,
            "max_drop": self.auto_max_drop, "pct": chosen,
            "acc_99": acc0, "acc": acc_ch, "sweep": sweep,
        }
        drop_ch = next(r["drop"] for r in sweep if r["pct"] == chosen)
        print(
            f"[SoftEdit] auto: N={n} calib_at={C} probes={len(correct)} "
            f"acc@99={acc0:.3f} ({base_n}/{len(correct)}) max_drop={self.auto_max_drop} "
            f"-> pct={chosen} (dHits={drop_ch:+d})",
            flush=True,
        )
        print(f"{'pct':>5}  {'tau':>10}  {'hits':>12}  {'acc':>6}  {'dHits':>6}",
              flush=True)
        for r in sweep:
            mark = "  <-" if r["pct"] == chosen else ""
            print(
                f"{r['pct']:5d}  {r['tau']:10.4f}  "
                f"{r['n_hit']:4d}/{len(correct):<6d}  {r['acc']:6.3f}  "
                f"{r['drop']:+6d}{mark}",
                flush=True,
            )
        return self.reject_threshold_pct

    # ==================== retrieval ====================

    def _compute_caches(self):
        """Leave-one-edit-out reject threshold from raw L2 pairwise distances."""
        if (self.key_embs is None or len(self.key_embs) < 2
                or self.reject_threshold_pct <= 0):
            self.key_dist_threshold = None
            return
        d = torch.cdist(self.key_embs.float(), self.key_embs.float())
        d.fill_diagonal_(float("inf"))
        loo_nn = d.min(dim=1).values.numpy()
        self.key_dist_threshold = float(np.percentile(loo_nn, self.reject_threshold_pct))

    def _nearest(self, key_new: torch.Tensor):
        """Return (idx, dist) of the nearest key; idx None when rejected/empty."""
        if self.key_embs is None or len(self.codebook) == 0:
            return None, None
        if self.reject_threshold_pct > 0 and self.key_dist_threshold is None:
            self._compute_caches()
        d = torch.norm(self.key_embs.float() - key_new.float()[None, :], dim=1)
        j = int(torch.argmin(d))
        dmin = float(d[j])
        if self.key_dist_threshold is not None and dmin > self.key_dist_threshold:
            return None, dmin
        return j, dmin

    # ==================== apply / generation-time patch ====================

    def _arm(self, examples):
        """Patch (or restore with None/[]) the lm_head for a generation batch.

        Restores any active patch first, then — if the batch's example matched a
        codebook entry — writes that entry's ΔW into the lm_head. Editing is done
        one sample at a time (sequential editing), so each batch carries a single
        retrieved entry.
        """
        train.restore_delta(self.head, self._patch_backup)
        self._patch_backup = None
        if not examples:
            return
        entries = {ex.get("_se_entry") for ex in examples if ex.get("_se_entry") is not None}
        if not entries:
            return
        if len(entries) > 1:
            print(f"[SoftEdit] WARNING: batch mixes {len(entries)} retrieved edits; "
                  f"weight patch applies one. Use batch_size=1 for exact locality.",
                  flush=True)
        entry = self.codebook[next(iter(entries))]
        self._patch_backup = train.apply_delta(
            self.head, entry["first_ids"], entry["dW"],
            cont_ids=entry.get("cont_ids"), dW_cont=entry.get("dW_cont"),
        )

    def apply_to_dataset(self, dataset):
        self._maybe_auto_tune()
        applied = 0
        judge_rej = 0
        log = []
        train.restore_delta(self.head, self._patch_backup)
        self._patch_backup = None
        data = getattr(dataset, "data", [])
        for ex in data:
            q_str = ex.get("question", "")
            j, dmin, judge_yes = None, None, None
            if q_str:
                key_new = self._encode(q_str)
                j, dmin = self._nearest(key_new)
                if j is not None and self.judge_paraphrase:
                    keep, judge_yes = self._get_judge().accept(
                        q_str, self.codebook[j]["question"])
                    if not keep:
                        j = None
                        judge_rej += 1

            ex["_se_entry"] = j
            if j is not None:
                entry = self.codebook[j]
                dbg = {"src": "weight", "d1": round(float(dmin), 4),
                       "entry_uid": entry.get("uid"),
                       "hit": str(entry.get("uid")) == str(ex.get("uid"))}
                if judge_yes is not None:
                    dbg["judge_yes"] = round(float(judge_yes), 4)
                ex["readout_dbg"] = dbg
                applied += 1
            else:
                dbg = {"src": None}
                if dmin is not None:
                    dbg["d1"] = round(float(dmin), 4)
                if judge_yes is not None:
                    dbg["judge_yes"] = round(float(judge_yes), 4)
                ex["readout_dbg"] = dbg
            log.append({"uid": ex.get("uid"), "entry": j})

        dataset.pregen_arm = _make_arm(self)
        self.last_retrieval_log = log
        extra = f" | judge_rej={judge_rej}" if self.judge_paraphrase else ""
        print(f"[SoftEdit] applied (weight) to {applied}/{len(data)} examples{extra}",
              flush=True)

    # ==================== resume IO ====================

    def save_resume_state(self) -> Dict:
        out = {
            "schema": "softedit_weight_v1",
            "codebook": self.codebook,
            "key_embs": self.key_embs,
            "_added_uids": self._added_uids,
            "_auto_tuned_at": self._auto_tuned_at,
            "_auto_mode": self._auto_mode,
        }
        if self._reject_auto:
            out["reject_threshold_pct"] = float(self.reject_threshold_pct)
        return out

    def load_resume_state(self, state: Dict) -> None:
        state = state or {}
        codebook = state.get("codebook") or []
        if codebook and state.get("schema") != "softedit_weight_v1":
            raise ValueError(
                "SoftEdit (weight) cannot resume from a non-weight checkpoint "
                "(entries lack ΔW). Re-run the edits.")
        self.codebook = codebook
        self.key_embs = state.get("key_embs")
        self._added_uids = set(state.get("_added_uids") or set())
        self.key_dist_threshold = None
        self._auto_tuned_at = state.get("_auto_tuned_at")
        if state.get("_auto_mode") in ("auto", "auto100"):
            self._auto_mode = state["_auto_mode"]
            self._reject_auto = True
        if self._reject_auto and state.get("reject_threshold_pct") is not None:
            self.reject_threshold_pct = float(state["reject_threshold_pct"])
        print(f"[SoftEdit] resume: restored {len(self.codebook)} entries", flush=True)
