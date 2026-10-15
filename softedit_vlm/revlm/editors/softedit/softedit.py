"""SoftEdit: retrieval codebook of small lm_head weight updates.

Front-end (retrieval) is a ReasonEdit-style dual-embedding key with a
leave-one-out calibrated reject gate and an optional paraphrase judge. The
**codebook value** is a tiny weight update ``ΔW`` (+ ``ΔW_cont``) on the lm_head
rows of the edit's candidate answers. The package is fully self-contained.

    edit:  train ΔW so softmax((W + ΔW) h0) matches the human proportions p*,
           regularized by the generality (graph) + locality anchor terms
           (see ``train.py`` / ``generality.py`` / ``locality.py``).
    apply: encode the query key, retrieve the nearest edit (reject gate + judge),
           and — on a hit — write ΔW into the lm_head **before** generation for
           that sample, restoring the original rows afterwards. No prompt change,
           no inject: the edit is realized in the model's own forward.

Each codebook entry::

    key        : retrieval key ([vision activation ; scaled SBERT(question)])
    labels/first_ids/p : the K candidates + human gold distribution
    dW         : [K, d]  first-token label-row update
    dW_cont    : [C, d]  continuation-row update (None if continuation off / single-token)
    cont_ids   : [C]     rows dW_cont edits
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional

from ..utils import parent_module, brackets_to_periods
from revlm.editors.soft_utils import build_soft_target
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
_AUTO_N_PERTURB = 5    # joint image+text probes per sampled edit
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
    runs on datasets after ``apply_to_dataset`` — shares this closure instead of
    deep-copying the whole editor (and its VLM).
    """
    def pregen_arm(examples):
        editor._arm(examples)
    return pregen_arm


class SoftEdit(nn.Module):
    """Retrieval codebook of lm_head weight updates for multilabel edits."""

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        cfg = getattr(config, "editor", config)

        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None
        self.device = getattr(config, "device", torch.device("cpu"))

        # ---- retrieval config ----
        self.key_mode = getattr(cfg, "key_mode", "dual_sbert")
        if self.key_mode not in ("vision", "language", "language_last", "dual_sbert"):
            raise ValueError(f"SoftEdit key_mode must be one of "
                             f"('vision','language','language_last','dual_sbert'), "
                             f"got {self.key_mode!r}")
        self.pool_method = getattr(cfg, "pool_method", "mean")
        self.lang_scaler = float(getattr(config.model, "lang_scaler_sbert", 0.0))
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
        # continuation: on/off switch (default on) — off = first-token KL only.
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
        self.locality_caption = str(getattr(cfg, "locality_caption", "off"))
        self.locality_declarative = str(getattr(cfg, "locality_declarative", "both"))
        self.locality_blend = float(getattr(cfg, "locality_blend", 0.0))
        self.jac_mode = str(getattr(cfg, "jac_mode", "prob"))
        self.edit_lr = float(getattr(cfg, "edit_lr", 1e-3))
        self.n_steps = int(getattr(cfg, "n_steps", 5000))
        self.loss_eps = float(getattr(cfg, "loss_eps", 1e-2))
        self.align_eps = float(getattr(cfg, "align_eps", 1e-4))
        self.sched_mode = str(getattr(cfg, "sched_mode", "plateau"))
        # per-edit training logs (step losses, locality anchors, final ΔW norm)
        self.edit_verbose = bool(getattr(cfg, "edit_verbose", True))

        # ---- codebook + caches ----
        self.codebook: List[Dict] = []
        self.key_embs = None            # [N, d_key] cpu float32
        self.key_dist_threshold = None
        self._added_uids = set()
        self._sbert = None
        self._judge = None

        # ---- hooks / state ----
        self._key_act = None
        self._patch_backup = None       # active lm_head restore backup (apply time)
        self.last_retrieval_log = None

        # lm_head we edit + the key-layer hook for retrieval
        self.head = train.resolve_lm_head(self.model)
        model_cfg = getattr(config, "model", config)
        KEY_HOOK_PARAM = {
            "vision": "inner_params_vision",
            "dual_sbert": "inner_params_vision",
            "language": "inner_params_lang",
            "language_last": "inner_params",
        }
        key_params = getattr(model_cfg, KEY_HOOK_PARAM[self.key_mode], [])
        if not key_params:
            raise ValueError(f"SoftEdit key_mode={self.key_mode!r} requires "
                             f"config.model.{KEY_HOOK_PARAM[self.key_mode]}")
        self._key_hook = self._register(key_params[0], self._key_hook_fn)

    # ==================== hooks ====================

    def _register(self, param_name, fn):
        name = param_name.rsplit(".", 1)[0] if param_name.endswith((".weight", ".bias")) else param_name
        mod = parent_module(self.model, brackets_to_periods(name))
        layer = getattr(mod, name.rsplit(".", 1)[-1])
        return layer.register_forward_hook(fn)

    def _key_hook_fn(self, module, inp, output):
        self._key_act = inp[0].detach() if isinstance(inp[0], torch.Tensor) else None

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    def generate(self, *a, **kw):
        return (self.model if hasattr(self.model, "generate") else self.wrapper).generate(*a, **kw)

    # ==================== encoding / retrieval key ====================

    def _get_sbert(self):
        if self._sbert is None:
            print("[SoftEdit] loading SBERT (paraphrase-mpnet-base-v2) ...", flush=True)
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")
            self._sbert.to(self.device)
        return self._sbert

    def _get_judge(self):
        if self._judge is None:
            from .judge import ParaphraseJudge
            self._judge = ParaphraseJudge()
        return self._judge

    def _encode_sbert(self, texts: List[str]) -> torch.Tensor:
        emb = self._get_sbert().encode(texts, convert_to_tensor=True)
        return emb.to(self.device, torch.float32)

    def _pool_key(self, act, B):
        act = act.to(self.device, torch.float32)
        last = self.pool_method == "last"
        if act.dim() == 3:
            return act[:, -1, :] if last else act.mean(dim=1)
        if act.dim() == 2:
            if act.shape[0] == B:
                return act
            if act.shape[0] % B == 0:
                reshaped = act.view(B, act.shape[0] // B, -1)
                return reshaped[:, -1, :] if last else reshaped.mean(dim=1)
            if last:
                return act[-1:].expand(B, -1)
            return act.mean(dim=0, keepdim=True).expand(B, -1)
        raise RuntimeError(f"Unexpected key activation shape {tuple(act.shape)}")

    @torch.no_grad()
    def _encode(self, image, question) -> torch.Tensor:
        """One forward → retrieval key [d_key] (cpu)."""
        self.model.eval()
        self._key_act = None
        inputs = self.wrapper.encode([image], [question], tokenize=False)
        self.model(**inputs)
        act = self._pool_key(self._key_act, 1)  # [1, d_act]
        if self.key_mode == "dual_sbert":
            lang = self._encode_sbert([question]) * self.lang_scaler  # [1, 768]
            key = torch.cat([act, lang], dim=-1)
        else:
            key = act
        return key[0].cpu()

    # ==================== edit (build codebook) ====================

    def _target(self, gold: Dict):
        st = build_soft_target(gold or {}, self.tokenizer, mode=self.soft_mode,
                               rank_temp=self.rank_temp, major_vote=self.major_vote)
        return st.labels, st.first_ids.to(self.device), st.q

    def _build_entry(self, image, question, gold, uid, tag="", prompt=None):
        labels, first_ids, p = self._target(gold)
        K = len(labels)
        if K == 0:
            return None

        key = self._encode(image, question)
        print(f"[SoftEdit]{tag} building entry: K={K} labels={labels}", flush=True)

        eng_prompt = prompt or question
        dW, dW_cont, cont_ids = train.train_edit(
            model=self.model, wrapper=self.wrapper, tokenizer=self.tokenizer,
            head=self.head, image=image, question=question, prompt=eng_prompt,
            labels=labels, first_ids=first_ids, p=p,
            continuation=self.continuation,
            lambda_gen=self.lambda_gen, lambda_loc=self.lambda_loc,
            lambda_reg=self.lambda_reg,
            locality_n=self.locality_n, locality_caption=self.locality_caption,
            locality_declarative=self.locality_declarative,
            locality_blend=self.locality_blend, jac_mode=self.jac_mode,
            lr=self.edit_lr, n_steps=self.n_steps, sched_mode=self.sched_mode,
            loss_eps=self.loss_eps, align_eps=self.align_eps,
            verbose=self.edit_verbose,
        )
        return {
            "uid": uid, "question": question, "labels": labels,
            "first_ids": first_ids.cpu(), "p": p.cpu(), "key": key,
            "dW": dW, "dW_cont": dW_cont, "cont_ids": cont_ids,
            "_image": image,  # in-memory only; stripped from ckpt
        }

    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None):
        """Add codebook entries for each new edit (retrieval-editor signature)."""
        if edit_ds is None:
            return self.model
        for ex in getattr(edit_ds, "data", []):
            uid = ex.get("uid") or (ex.get("image"), ex.get("question"))
            key = str(uid)
            if key in self._added_uids:
                continue
            img, q = ex.get("image"), ex.get("question", "")
            if img is None:
                continue
            tag = f" edit {len(self.codebook) + 1}"
            prompt = ex.get("prompt_orig") or ex.get("prompt") or q
            entry = self._build_entry(img, q, ex.get("gold") or {}, key, tag=tag,
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

    def bind_edit_sources(self, data):
        """Attach images from a dataset by uid (needed after loading a ckpt)."""
        by_uid = {}
        for ex in (data or []):
            uid = ex.get("uid") or (ex.get("image"), ex.get("question"))
            by_uid[str(uid)] = ex
        n = 0
        for e in self.codebook:
            src = by_uid.get(str(e.get("uid")))
            if src is None:
                continue
            img = src.get("image")
            if img is not None:
                e["_image"] = img
                n += 1
        print(f"[SoftEdit] bind_edit_sources: attached {n}/{len(self.codebook)} images",
              flush=True)

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

        Samples 100 codebook edits (or all, if fewer), builds 5 joint
        image+text perturbations each, and scores them with the live retriever
        (L2 gate + judge). ``upto`` restricts the codebook prefix.
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
        skipped = 0
        for i, pidx in enumerate(sample_idx):
            pidx = int(pidx)
            q0 = entries[pidx].get("question") or ""
            img0 = entries[pidx].get("_image") or entries[pidx].get("image")
            if not q0:
                skipped += 1
                continue
            for k in range(_AUTO_N_PERTURB):
                try:
                    img, q = aug.perturb(img0, q0, _AUTO_SEED + 1000 * i + k)
                    if img is None:
                        skipped += 1
                        continue
                    q_keys.append(self._encode(img, q))
                    q_text.append(q)
                    parents.append(pidx)
                except Exception as exc:
                    skipped += 1
                    print(f"[SoftEdit] auto: perturb failed (i={i}, k={k}): {exc}",
                          flush=True)
                    continue

        if not q_keys:
            print("[SoftEdit] auto: no generality probes "
                  f"(skipped={skipped}); pct={_AUTO_START}. bind_edit_sources() if images "
                  "were dropped at resume.", flush=True)
            self.reject_threshold_pct = float(_AUTO_START)
            self.key_dist_threshold = None
            return self.reject_threshold_pct

        Q = torch.stack(q_keys).float()
        dists = torch.cdist(Q, keys)
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
            q_str, img = ex.get("question", ""), ex.get("image")
            j, dmin, judge_yes = None, None, None
            if img is not None and q_str:
                key_new = self._encode(img, q_str)
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
        codebook = [{k: v for k, v in e.items() if k != "_image"}
                    for e in self.codebook]
        out = {
            "schema": "softedit_weight_v1",
            "codebook": codebook,
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
