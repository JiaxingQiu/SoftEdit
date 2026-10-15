# TaskEngineer: task-dependent I/O engineering for each example.
# ex is a binding to the dict stored in self.data, mutable in place.
# (1) prompt
# (2) gold  — label (answer text), label_train (supervision target)
# (3) pred  — label, answer (raw text), plus task-specific fields
#
# Supported tasks:
#   qa      — single-answer QA with soft_acc / rank_acc eval
#   multiqa — ranked multilabel QA with Acc@1, RankAcc@1, F1, nDCG@K eval
#   softqa  — joint protocol: QA + MultiQA on same sample (two passes)


# Shared single-label QA instruction. Used by BOTH the prediction pass
# (QATaskEngineer, via softqa in baseline.py) and the editing/eval pass
# (PostEditQATaskEngineer). Keeping a single source of truth guarantees the
# prompt that selects the edit set (rank_acc==0) is identical to the prompt used
# for editing + reliability, so the two never drift. An explicit "single-word"
# instruction constrains the model to emit one label token (vs. free-form
# synonyms/phrases), which is what the fixed-vocabulary soft metrics expect.
ANSWER_INSTRUCTION = "Return single-word answers." # "Short answer:"


def norm(s):
    return str(s).strip().lower() if s is not None else None


def _parse_single_label(answer, support_prob=None, strategy="text"):
    """Shared top-1 label policy for single-label QA tasks (QA + PostEditQA).

    - strategy="prob": argmax over the support distribution (guaranteed-valid label).
    - strategy="text": robust single-answer parse (strips punctuation/markdown, splits
      hedging connectors like "X or Y", returns the first word), else "". This never
      discards a free-text single answer the way the numbered-list parser would.
    """
    from revlm.metrics.ranked_metrics import parse_single_answer, parse_qa_label
    if strategy == "prob" and support_prob:
        return parse_qa_label(answer, support_prob, "prob")
    return parse_single_answer(answer or "")


def support_labels(ex):
    """Candidate label set for support_prob: explicit ``ex["support_labels"]``
    override when present (soft_generality scores over the edit sample's
    original label set -- a one-label support renormalizes to {label: 1.0} and
    makes cross-strength probability comparisons meaningless), else
    answer_counts keys."""
    if ex.get("support_labels"):
        return list(ex["support_labels"])
    ac = ex.get("answer_counts") or (ex.get("gold") or {}).get("answer_counts") or {}
    return list(ac.keys())


def human_labels(ex):
    """Human answer labels (answer_counts keys) for the open readout.

    Unlike ``support_labels`` this ignores any ``support_labels`` override: the
    open readout's only external knowledge is HOW MANY labels the annotators
    gave (k) — the model reads out its own top-k — never which candidate
    strings to probe. See revlm/metrics/rank_metrics.md."""
    ac = ex.get("answer_counts") or (ex.get("gold") or {}).get("answer_counts") or {}
    return list(ac.keys())


class _RetrievalCounter:
    """Aggregate retrieval diagnostics for retrieval editors (SoftEdit v2).

    The editor stamps every example's ``readout_dbg`` (``src=None`` when the
    reject gate fired; ``hit`` = retrieved entry's uid == example's uid, valid
    because every edit-derived eval set keeps the source edit's uid). Editors
    that don't set ``readout_dbg`` produce no counts and the metrics dict stays
    unchanged, so this is a no-op for everything but SoftEdit.

    Reported (over the scored examples):
      retr_rate: fraction with a retrieved (non-rejected) entry
      retr_acc : fraction whose retrieved entry is the RIGHT edit — the number
                 that isolates "wrong retrieval" from "bad steering/readout"
    """

    def __init__(self):
        self.n = 0
        self.retrieved = 0
        self.hits = 0

    def add(self, pred):
        dbg = pred.get("readout_dbg")
        # "src" is the v2 marker; v1's dbg ({dist, bw, gate, surface}) lacks it
        # and would otherwise be miscounted as 0% retrieval.
        if not isinstance(dbg, dict) or "src" not in dbg:
            return
        self.n += 1
        if dbg.get("src"):
            self.retrieved += 1
            if dbg.get("hit"):
                self.hits += 1

    def metrics(self):
        if self.n == 0:
            return {}
        return {"retr_rate": self.retrieved / self.n,
                "retr_acc": self.hits / self.n}


def _prep_model_core(model, use_cache):
    """Best-effort per-batch model prep (use_cache + Qwen3-VL rope reset)."""
    try:
        core = getattr(getattr(model, "model", model), "model", getattr(model, "model", model))
        cfg = getattr(core, "config", None)
        if cfg is not None:
            cfg.use_cache = use_cache
        if hasattr(core, "gradient_checkpointing_disable"):
            core.gradient_checkpointing_disable()
        rope_owner = core if hasattr(core, "rope_deltas") else getattr(core, "model", None)
        if rope_owner is not None and hasattr(rope_owner, "rope_deltas"):
            rope_owner.rope_deltas = None
    except Exception:
        pass


def _carry_open_readout(ex, open_readout):
    """Store the open-vocabulary readout on the pred.

    An editor may pre-set ``ex["support_prob_open"]`` (SoftEdit logit mode,
    where the generation logits never reflect the edit); that takes precedence
    over the distribution computed from this generation pass."""
    spo, l2k = ex.get("support_prob_open"), ex.get("open_label2key")
    if spo is None and open_readout is not None:
        spo, l2k = open_readout
    if spo is not None:
        ex['pred']['support_prob_open'] = spo
        ex['pred']['open_label2key'] = l2k


def _carry_readout_raw(ex, readout_raw):
    """Store the raw readout record on the pred for offline rescoring.

    SoftEdit logit mode pre-sets ``ex["readout_raw"]`` to None to signal that
    the generation logits don't reflect the edit (the editor's claimed
    distribution lives in ``support_prob_readout``). In that case, no raw
    record is stored — the generation-pass record would mislead offline
    analysis. For all other modes (hidden_state, prompt, no editor) the
    generation logits faithfully reflect the model state."""
    rr = ex.get("readout_raw", readout_raw)
    if rr is not None:
        ex['pred']['readout_raw'] = rr


def _scored_qa_pass(dataset, model, engineer, use_cache, max_new_tokens):
    """QA generation that also records first-token support probs (one forward).

    By default (``continuation_gate=True``), a binary continuation check
    is applied per-example after the generation pass: for each first-token
    branch matching a multi-token human answer, the model is teacher-forced
    on the remaining tokens. Branches that fail are zeroed and the mass
    flows into OTHER. Both first-token-only and gated readouts are stored
    on the pred so downstream eval reports both score variants
    (``js_dist`` / ``js_dist_gate``, etc.). Set ``continuation_gate=False``
    to skip the extra teacher-forced forwards.
    """
    arm = getattr(dataset, "pregen_arm", None)
    do_gate = getattr(getattr(dataset, "config", None), "continuation_gate", True)
    for batch in dataset.loader:
        if not batch["images"] or not batch["prompts"]:
            continue
        _prep_model_core(model, use_cache)
        if arm is not None:
            arm([dataset.data[i] for i in batch["idxs"]])
        labels_batch = [support_labels(dataset.data[i]) for i in batch["idxs"]]
        open_batch = [human_labels(dataset.data[i]) for i in batch["idxs"]]
        answers, support_probs, open_readouts, raw_readouts = model.generate_scored(
            batch["images"], batch["prompts"], labels_batch, open_labels=open_batch,
            max_new_tokens=max_new_tokens, use_cache=use_cache,
        )
        for idx, a, sp, oro, rr in zip(batch["idxs"], answers, support_probs,
                                        open_readouts, raw_readouts):
            ex = dataset.data[idx]
            engineer.eng_preds(ex, a, model, support_prob=sp,
                               open_readout=oro, readout_raw=rr)
            # Continuation gate: skip when the editor overrides the readout
            # (e.g. SoftEdit logit mode sets readout_raw=None on ex).
            if do_gate and oro is not None and ex.get("pred", {}).get("readout_raw") is not None:
                _apply_gate_to_pred(model, ex)
    if arm is not None:
        arm(None)


def _apply_gate_to_pred(model, ex):
    """Run the continuation gate for one example and store gated readout on pred.

    The ungated ``support_prob_open`` / ``open_label2key`` remain unchanged;
    gated versions are stored as ``support_prob_open_gate`` /
    ``continuation_gate_log``.
    """
    from revlm.metrics.readout import apply_continuation_gate

    pred = ex.get("pred") or {}
    spo = pred.get("support_prob_open")
    l2k = pred.get("open_label2key")
    if not spo or not l2k:
        return

    h_labels = human_labels(ex)
    if not h_labels:
        return

    image = ex.get("image")
    prompt = ex.get("prompt", "")

    gate = apply_continuation_gate(
        model, image, prompt, spo, h_labels, l2k)

    pred["support_prob_open_gate"] = gate.support_prob_open
    pred["continuation_gate_log"] = gate.gate_log
    pred["continuation_gate_key2answer"] = gate.key2answer


class TaskIOEngineer:
    def __init__(self):
        self.name = "default"
    def eng_prompt(self, ex):
        raise NotImplementedError("Subclasses must implement eng_prompt")
    def eng_golds(self, ex):
        raise NotImplementedError("Subclasses must implement eng_golds")
    def eng_preds(self, ex):
        raise NotImplementedError("Subclasses must implement eng_preds")

    def parse_answer(self, answer, support_prob=None):
        """Task-specific: raw generation text -> derived pred fields.

        Returns a dict of the *derived* prediction fields (e.g. ``label`` and,
        for ranked tasks, ``labels_ranked``). Co-locating this with ``eng_prompt``
        keeps each task's prompt and its answer parsing in lockstep, so they can
        never drift. Callers add the raw ``answer``/``support_prob`` themselves.
        """
        raise NotImplementedError("Subclasses must implement parse_answer")

    def reparse(self, data):
        """Re-derive predicted labels in-place from each example's stored raw
        ``answer``, using this task's ``parse_answer``. Used when loading a cached
        prediction snapshot so it honors the current parsing/label_strategy."""
        for ex in data:
            p = ex.get("pred")
            if isinstance(p, dict) and "answer" in p:
                p.update(self.parse_answer(p.get("answer", ""), p.get("support_prob")))


class QATaskEngineer(TaskIOEngineer):
    def __init__(self, seed=333, model_name="", label_strategy="text", **kwargs):
        super().__init__()
        self.name = "qa"
        self.seed = seed
        self.label_strategy = label_strategy

    def eng_golds(self, ex):
        ex['gold'] = {}
        answer = ex.get('answer') or (ex.get('labels_ranked', [None])[0])
        ex['gold']['label'] = str(answer).lower().strip()
        ex['gold']['label_train'] = ex['gold']['label']
        ex['gold']['answer_counts'] = ex.get('answer_counts', {})

    def eng_prompt(self, ex):
        ex["prompt"] = f"{ex['question']} {ANSWER_INSTRUCTION}".strip() if ANSWER_INSTRUCTION else str(ex['question']).strip()

    def parse_answer(self, answer, support_prob=None):
        return {"label": _parse_single_label(answer, support_prob, self.label_strategy)}

    def eng_preds(self, ex, a: str, model, support_prob=None, open_readout=None, readout_raw=None):
        # SoftEdit v2 logit mode: the editor pre-computed the answer from its
        # readout (distribution overwrite); prefer it over the generated text.
        # No-op when the field is absent. Stored as pred['answer'] so snapshot
        # save/reparse round-trips consistently.
        a = ex.get("answer_override") or a
        ex['pred'] = {'answer': a}
        if support_prob is not None:
            ex['pred']['support_prob'] = support_prob
        # A retrieval editor (e.g. SoftEdit) may attach a readout distribution on
        # the example; carry it into the pred so js_dist can prefer it. Label and
        # ranking metrics still come from the generated text below.
        if ex.get("support_prob_readout") is not None:
            ex['pred']['support_prob_readout'] = ex['support_prob_readout']
        if ex.get("readout_dbg") is not None:
            ex['pred']['readout_dbg'] = ex['readout_dbg']
        _carry_open_readout(ex, open_readout)
        _carry_readout_raw(ex, readout_raw)
        ex['pred'].update(self.parse_answer(a, support_prob))

    def run_generate(self, dataset, model, use_cache=True, max_new_tokens=None, **kwargs):
        """Single scored QA pass (greedy answer + first-token support probs)."""
        if max_new_tokens is None:
            max_new_tokens = getattr(dataset.config, "max_new_tokens", 10)
        _scored_qa_pass(dataset, model, self, use_cache, max_new_tokens)

    def eval(self, vlmdataset):
        """Evaluate QA task: soft accuracy, rank accuracy, JS distance.

        Reads the already-parsed `pred["label"]` (text-to-text); the label
        strategy is applied upstream at generation / snapshot load.

        When continuation-gated readouts are present on the preds
        (``support_prob_open_gate``), the three readout-dependent metrics
        are also computed from the gated distribution and reported as
        ``js_dist_gate``, ``ndcg_readout_gate``, ``tau_readout_gate``.
        """
        from revlm.metrics.ranked_metrics import (
            build_labels_true, soft_acc, rank_acc, simple_acc, js_distance,
            ndcg_readout, tau_readout, counts_on_token_keys,
        )
        acc_sum = 0.0
        hacc_sum = 0.0
        racc_sum = 0.0
        sacc_sum = 0.0
        js_sum = 0.0
        js_n = 0
        ndcg_r_sum = 0.0
        ndcg_r_n = 0
        tau_r_sum = 0.0
        tau_r_n = 0
        # Gated accumulators
        js_g_sum = 0.0
        js_g_n = 0
        ndcg_g_sum = 0.0
        ndcg_g_n = 0
        tau_g_sum = 0.0
        tau_g_n = 0
        n = 0
        retr = _RetrievalCounter()
        for ex in vlmdataset.data:
            gold = ex.get("gold", {})
            pred = ex.get("pred", {})
            if pred.get("label") is None:
                continue
            p_label = norm(pred.get("label", ""))
            n += 1
            answer_counts = gold.get("answer_counts", {})
            labels_true = build_labels_true(answer_counts)
            acc_sum += soft_acc(p_label, labels_true)
            hacc_sum += float(p_label == norm(gold.get("label", "")))
            racc_sum += rank_acc(p_label, labels_true)
            sacc_sum += simple_acc(p_label, labels_true)
            spo = pred.get("support_prob_open")
            support_prob = spo or pred.get("support_prob_readout") or pred.get("support_prob")
            if support_prob and answer_counts:
                target = (counts_on_token_keys(answer_counts, pred.get("open_label2key"))
                          if spo else answer_counts)
                js_sum += js_distance(target, support_prob)
                js_n += 1
            if spo and labels_true:
                l2k = pred.get("open_label2key")
                ndcg_r_sum += ndcg_readout(spo, labels_true, l2k)
                tau_r_sum += tau_readout(spo, labels_true, l2k)
                ndcg_r_n += 1
                tau_r_n += 1
            # Gated readout scores
            spo_g = pred.get("support_prob_open_gate")
            if spo_g and answer_counts:
                l2k = pred.get("open_label2key")
                target_g = counts_on_token_keys(answer_counts, l2k)
                js_g_sum += js_distance(target_g, spo_g)
                js_g_n += 1
            if spo_g and labels_true:
                l2k = pred.get("open_label2key")
                ndcg_g_sum += ndcg_readout(spo_g, labels_true, l2k)
                tau_g_sum += tau_readout(spo_g, labels_true, l2k)
                ndcg_g_n += 1
                tau_g_n += 1
            retr.add(pred)
        out = {"simple_acc": sacc_sum / n if n > 0 else 0.0,
               "hard_acc": hacc_sum / n if n > 0 else 0.0,
               "js_dist": js_sum / js_n if js_n > 0 else None,
               "js_dist_gate": js_g_sum / js_g_n if js_g_n > 0 else None,
               "tau_readout": tau_r_sum / tau_r_n if tau_r_n > 0 else None,
               "tau_readout_gate": tau_g_sum / tau_g_n if tau_g_n > 0 else None,
               "acc": acc_sum / n if n > 0 else 0.0,
               "rank_acc": racc_sum / n if n > 0 else 0.0,
               "ndcg_readout": ndcg_r_sum / ndcg_r_n if ndcg_r_n > 0 else None,
               "ndcg_readout_gate": ndcg_g_sum / ndcg_g_n if ndcg_g_n > 0 else None,
               "n": n}
        out.update(retr.metrics())
        return out


class MultiQATaskEngineer(TaskIOEngineer):
    """Task engineer for ranked multilabel QA.

    Each example is expected to have:
      - labels_ranked: list of labels ordered by annotator frequency (descending)
      - answer_counts: dict {label: count}
    """

    INSTRUCTION = (
        "List all plausible single-word answers ranked from most to least likely. "
        "Return a numbered list:\n1. answer1\n2. answer2\n3. ..."
    )
    SUFFIX_CONCISE = "\nReply ONLY with a numbered list, no reasoning or formatting."

    def __init__(self, seed=333, model_name="", **kwargs):
        super().__init__()
        self.name = "multiqa"
        self.seed = seed
        self.concise = "gemma" in model_name.lower()

    def eng_golds(self, ex):
        labels_ranked = ex.get("labels_ranked", [])
        ex['gold'] = {
            'label': str(labels_ranked[0]).strip().lower() if labels_ranked else "",
            'labels_ranked': [str(l).strip().lower() for l in labels_ranked],
            'k': len(labels_ranked),
            'label_train': "\n".join(f"{i+1}. {l}" for i, l in enumerate(labels_ranked)),
            'answer_counts': ex.get('answer_counts', {}),
        }

    def eng_prompt(self, ex):
        prompt = f"{ex['question']} {self.INSTRUCTION}"
        if self.concise:
            prompt += self.SUFFIX_CONCISE
        ex['prompt'] = prompt

    def parse_answer(self, answer, support_prob=None):
        from revlm.metrics.ranked_metrics import parse_numbered_list
        parsed = parse_numbered_list(answer or "")
        return {"labels_ranked": parsed, "label": parsed[0] if parsed else ""}

    def eng_preds(self, ex, a: str, model):
        ex['pred'] = {'answer': a, **self.parse_answer(a)}

    def eval(self, vlmdataset):
        from revlm.metrics.ranked_metrics import build_labels_true, soft_acc, rank_acc, set_f1, ndcg_at_k

        acc1_sum = racc1_sum = f1_sum = ndcg_sum = 0.0
        n = 0
        for ex in vlmdataset.data:
            gold = ex.get("gold", {})
            pred = ex.get("pred", {})
            pred_ranked = pred.get("labels_ranked", [])
            labels_true = build_labels_true(gold.get("answer_counts", {}))
            if not labels_true:
                continue
            n += 1
            p1 = pred_ranked[0] if pred_ranked else ""
            acc1_sum += soft_acc(p1, labels_true)
            racc1_sum += rank_acc(p1, labels_true)
            f1_sum += set_f1(pred_ranked, labels_true)
            ndcg_sum += ndcg_at_k(pred_ranked, labels_true)

        if n == 0:
            return {"acc_at_1": 0.0, "rank_acc_at_1": 0.0, "f1": 0.0, "ndcg_at_k": 0.0, "n": 0}
        return {
            "acc_at_1": acc1_sum / n,
            "rank_acc_at_1": racc1_sum / n,
            "f1": f1_sum / n,
            "ndcg_at_k": ndcg_sum / n,
            "n": n,
        }


class SoftQATaskEngineer(TaskIOEngineer):
    """Joint evaluation protocol: QA (free single answer) + MultiQA (ranked list).

    Two forward passes per sample, producing pred_qa and pred_multiqa.
    Generic pred aliases to pred_qa (single-label) for downstream edit code.
    Eval returns nested {"qa": {...}, "multiqa": {...}}.
    """

    def __init__(self, seed=333, model_name="", label_strategy="text", **kwargs):
        super().__init__()
        self.name = "softqa"
        self.seed = seed
        self.label_strategy = label_strategy
        self._qa = QATaskEngineer(seed=seed, model_name=model_name, label_strategy=label_strategy, **kwargs)
        self._multiqa = MultiQATaskEngineer(seed=seed, model_name=model_name, **kwargs)

    def eng_golds(self, ex):
        # Build the ranked (multiqa) gold so eval can score the full 6 metrics,
        # but supervise editing on the single majority-vote label. label_train is
        # the *edit target* (not the eval format); the ranked list is a separate
        # generation pass at eval time. Soft variants override label_train.
        self._multiqa.eng_golds(ex)
        ex['gold']['label_train'] = ex['gold']['label']

    def eng_prompt(self, ex):
        self._qa.eng_prompt(ex)
        ex['prompt_qa'] = ex['prompt']
        self._multiqa.eng_prompt(ex)
        ex['prompt_multiqa'] = ex['prompt']
        ex['prompt'] = ex['prompt_qa']

    def eng_preds(self, ex, a: str, model):
        raise NotImplementedError(
            "SoftQA uses run_generate for two-pass generation, not eng_preds"
        )

    def parse_answer(self, answer, support_prob=None):
        # Generic single-pred alias mirrors the QA pass (see run_generate).
        return self._qa.parse_answer(answer, support_prob)

    def reparse(self, data):
        """Re-derive both passes: pred_qa via the QA parser, pred_multiqa via the
        MultiQA parser, then realias the generic ``pred`` to pred_qa."""
        for ex in data:
            pq = ex.get("pred_qa")
            if isinstance(pq, dict) and "answer" in pq:
                pq.update(self._qa.parse_answer(pq.get("answer", ""), pq.get("support_prob")))
            pm = ex.get("pred_multiqa")
            if isinstance(pm, dict) and "answer" in pm:
                pm.update(self._multiqa.parse_answer(pm.get("answer", "")))
            if isinstance(ex.get("pred_qa"), dict):
                ex["pred"] = ex["pred_qa"]

    def run_generate(self, dataset, model, use_cache=True, max_new_tokens=None, **kwargs):
        """Two-pass generation: QA then MultiQA."""
        from torch.utils.data import DataLoader

        if max_new_tokens is None:
            max_new_tokens = getattr(dataset.config, "max_new_tokens", 10)

        # Pass 1: QA (short answer + first-token support probs, one forward)
        for ex in dataset.data:
            ex['prompt'] = ex['prompt_qa']
        dataset.loader = DataLoader(
            dataset, batch_size=dataset.config.batch_size,
            shuffle=False, num_workers=0, pin_memory=True,
            collate_fn=dataset.image_collate,
        )
        _scored_qa_pass(dataset, model, self._qa, use_cache, max_new_tokens)
        for ex in dataset.data:
            ex['pred_qa'] = ex.pop('pred', {})

        # Pass 2: MultiQA (ranked list — allow more tokens)
        multiqa_max = max(max_new_tokens, 100)
        for ex in dataset.data:
            ex['prompt'] = ex['prompt_multiqa']
        dataset.loader = DataLoader(
            dataset, batch_size=dataset.config.batch_size,
            shuffle=False, num_workers=0, pin_memory=True,
            collate_fn=dataset.image_collate,
        )
        self._run_pass(dataset, model, self._multiqa, use_cache, multiqa_max)
        for ex in dataset.data:
            ex['pred_multiqa'] = ex.pop('pred', {})
            ex['pred'] = ex['pred_qa']

    @staticmethod
    def _run_pass(dataset, model, engineer, use_cache, max_new_tokens):
        for batch in dataset.loader:
            if not batch["images"] or not batch["prompts"]:
                continue
            _prep_model_core(model, use_cache)
            outs = model.generate(
                batch["images"], batch["prompts"],
                max_new_tokens=max_new_tokens, use_cache=use_cache,
            )
            for idx, a in zip(batch["idxs"], outs):
                engineer.eng_preds(dataset.data[idx], a, model)

    def eval(self, vlmdataset):
        from revlm.metrics.ranked_metrics import (
            build_labels_true, soft_acc, rank_acc, simple_acc, set_f1, ndcg_at_k,
            js_distance, ndcg_readout, tau_readout, counts_on_token_keys,
        )

        qa_sacc_sum = 0.0
        qa_hacc_sum = 0.0
        qa_acc_sum = 0.0
        qa_racc_sum = 0.0
        qa_js_sum = 0.0
        qa_js_n = 0
        qa_ndcg_r_sum = 0.0
        qa_ndcg_r_n = 0
        qa_tau_r_sum = 0.0
        qa_tau_r_n = 0
        # Gated accumulators
        qa_js_g_sum = 0.0
        qa_js_g_n = 0
        qa_ndcg_g_sum = 0.0
        qa_ndcg_g_n = 0
        qa_tau_g_sum = 0.0
        qa_tau_g_n = 0
        qa_total = 0
        acc1_sum = racc1_sum = f1_sum = ndcg_sum = 0.0
        multi_n = 0

        for ex in vlmdataset.data:
            gold = ex.get("gold", {})
            pred_qa = ex.get("pred_qa", {})
            pred_multi = ex.get("pred_multiqa", {})
            answer_counts = gold.get("answer_counts", {})
            labels_true = build_labels_true(answer_counts)

            if not labels_true:
                continue

            if pred_qa.get("label") is not None:
                p_qa = norm(pred_qa.get("label", ""))
                qa_total += 1
                qa_sacc_sum += simple_acc(p_qa, labels_true)
                qa_hacc_sum += float(p_qa == norm(gold.get("label", "")))
                qa_acc_sum += soft_acc(p_qa, labels_true)
                qa_racc_sum += rank_acc(p_qa, labels_true)
                spo = pred_qa.get("support_prob_open")
                support_prob = spo or pred_qa.get("support_prob")
                if support_prob and answer_counts:
                    target = (counts_on_token_keys(answer_counts, pred_qa.get("open_label2key"))
                              if spo else answer_counts)
                    qa_js_sum += js_distance(target, support_prob)
                    qa_js_n += 1
                if spo:
                    l2k = pred_qa.get("open_label2key")
                    qa_ndcg_r_sum += ndcg_readout(spo, labels_true, l2k)
                    qa_tau_r_sum += tau_readout(spo, labels_true, l2k)
                    qa_ndcg_r_n += 1
                    qa_tau_r_n += 1
                # Gated readout scores
                spo_g = pred_qa.get("support_prob_open_gate")
                if spo_g and answer_counts:
                    l2k = pred_qa.get("open_label2key")
                    target_g = counts_on_token_keys(answer_counts, l2k)
                    qa_js_g_sum += js_distance(target_g, spo_g)
                    qa_js_g_n += 1
                if spo_g and labels_true:
                    l2k = pred_qa.get("open_label2key")
                    qa_ndcg_g_sum += ndcg_readout(spo_g, labels_true, l2k)
                    qa_tau_g_sum += tau_readout(spo_g, labels_true, l2k)
                    qa_ndcg_g_n += 1
                    qa_tau_g_n += 1

            pred_ranked = pred_multi.get("labels_ranked", [])
            p1 = pred_ranked[0] if pred_ranked else ""
            multi_n += 1
            acc1_sum += soft_acc(p1, labels_true)
            racc1_sum += rank_acc(p1, labels_true)
            f1_sum += set_f1(pred_ranked, labels_true)
            ndcg_sum += ndcg_at_k(pred_ranked, labels_true)

        return {
            "qa": {
                "simple_acc": qa_sacc_sum / qa_total if qa_total > 0 else 0.0,
                "hard_acc": qa_hacc_sum / qa_total if qa_total > 0 else 0.0,
                "js_dist": qa_js_sum / qa_js_n if qa_js_n > 0 else None,
                "js_dist_gate": qa_js_g_sum / qa_js_g_n if qa_js_g_n > 0 else None,
                "tau_readout": qa_tau_r_sum / qa_tau_r_n if qa_tau_r_n > 0 else None,
                "tau_readout_gate": qa_tau_g_sum / qa_tau_g_n if qa_tau_g_n > 0 else None,
                "acc": qa_acc_sum / qa_total if qa_total > 0 else 0.0,
                "rank_acc": qa_racc_sum / qa_total if qa_total > 0 else 0.0,
                "ndcg_readout": qa_ndcg_r_sum / qa_ndcg_r_n if qa_ndcg_r_n > 0 else None,
                "ndcg_readout_gate": qa_ndcg_g_sum / qa_ndcg_g_n if qa_ndcg_g_n > 0 else None,
                "n": qa_total,
            },
            "multiqa": {
                "acc_at_1": acc1_sum / multi_n if multi_n > 0 else 0.0,
                "rank_acc_at_1": racc1_sum / multi_n if multi_n > 0 else 0.0,
                "f1": f1_sum / multi_n if multi_n > 0 else 0.0,
                "ndcg_at_k": ndcg_sum / multi_n if multi_n > 0 else 0.0,
                "n": multi_n,
            },
        }


class PostEditQATaskEngineer(TaskIOEngineer):
    """Post-edit evaluation: one anchored pass, scored like QA.

    The edited model is prompted with the question plus a single global format
    instruction (:attr:`INSTRUCTION`) and scored in one pass (same scored
    generation as QA). The instruction only anchors the *format* (single-word
    answers), so editing is easy to learn and the output is trivially parseable;
    it never varies by editor type. Setting ``INSTRUCTION = ""`` reverts to the
    bare-question (no-instruction) behavior. From that pass:
      - first-token distribution over the label support -> js_dist (as in QA), and
      - the text output parsed into a ranked label list (single answer -> a
        1-element list; comma list -> the parsed ranked list).

    Because the instruction lives here (and set_dataloader calls eng_prompt for
    every set), editing and every eval set — edit / locality / generality — share
    the identical prompt, for both the pre- and post-edit models.

    Eval reports the single-QA metrics (acc, rank_acc, tf_acc, js_dist) and the
    multiqa metrics (acc@1, rank_acc@1, f1, ndcg@k) from that one prediction.
    """

    # Answer-slot anchor, shared with the prediction pass (QATaskEngineer) via the
    # module-level ANSWER_INSTRUCTION constant so the edit-set selection prompt and
    # the editing/eval prompt are guaranteed identical. Set the constant to "" for
    # the bare question.
    INSTRUCTION = ANSWER_INSTRUCTION

    def __init__(self, seed=333, model_name="", label_strategy="text", **kwargs):
        super().__init__()
        self.name = "postedit_qa"
        self.seed = seed
        self.label_strategy = label_strategy
        # Reuse the multiqa gold builder (label / labels_ranked / answer_counts),
        # then supervise editing on the single majority-vote label (soft variants
        # override label_train with the ranked string via edit_utils).
        self._multiqa = MultiQATaskEngineer(seed=seed, model_name=model_name, **kwargs)

    def eng_golds(self, ex):
        self._multiqa.eng_golds(ex)
        ex['gold']['label_train'] = ex['gold']['label']

    def eng_prompt(self, ex):
        q = ex['question']
        ex['prompt'] = f"{q} {self.INSTRUCTION}" if self.INSTRUCTION else q

    def parse_answer(self, answer, support_prob=None):
        from revlm.metrics.ranked_metrics import parse_numbered_list
        # Ranked list is always text-parsed (multilabel metrics are list-based);
        # top-1 uses the shared single-label policy (same as QA), so the edit-set
        # selection and reliability read an identically-derived label.
        parsed = parse_numbered_list(answer or "")
        label = _parse_single_label(answer, support_prob, self.label_strategy)
        return {"labels_ranked": parsed, "label": label}

    def eng_preds(self, ex, a: str, model, support_prob=None, open_readout=None, readout_raw=None):
        # SoftEdit v2 logit mode: prefer the editor's pre-computed readout answer
        # (a ranked comma list; both parsers handle it). No-op when absent.
        a = ex.get("answer_override") or a
        ex['pred'] = {'answer': a}
        if support_prob is not None:
            ex['pred']['support_prob'] = support_prob
        # A retrieval editor (e.g. SoftEdit) may attach a readout distribution on
        # the example; carry it into the pred so js_dist can prefer it. Label and
        # ranking metrics still come from the generated text below.
        if ex.get("support_prob_readout") is not None:
            ex['pred']['support_prob_readout'] = ex['support_prob_readout']
        if ex.get("readout_dbg") is not None:
            ex['pred']['readout_dbg'] = ex['readout_dbg']
        _carry_open_readout(ex, open_readout)
        _carry_readout_raw(ex, readout_raw)
        ex['pred'].update(self.parse_answer(a, support_prob))

    def run_generate(self, dataset, model, use_cache=True, max_new_tokens=None, compute_tf=None):
        """Single scored pass; enough room for a short comma-separated label list.

        Also records a teacher-forced argmax accuracy on the majority-vote label
        (prior-work "rewrite_acc" style) unless disabled. Locality uses
        ``compute_tf=False`` — it only needs string labels and the unrelated pool
        is large, so the extra forward would dominate runtime."""
        if max_new_tokens is None:
            max_new_tokens = getattr(dataset.config, "max_new_tokens", 40)
        max_new_tokens = max(int(max_new_tokens), 40)
        _scored_qa_pass(dataset, model, self, use_cache, max_new_tokens)

        if compute_tf is None:
            compute_tf = getattr(dataset.config, "tf_eval", True)
        if compute_tf:
            _prep_model_core(model, use_cache)
            for ex in dataset.data:
                pred = ex.get("pred")
                target = ex.get("gold", {}).get("label", "")
                if pred is not None and target:
                    pred["tf_acc"] = model.teacher_forced_acc(ex["image"], ex["prompt"], target)

    def eval(self, vlmdataset):
        from revlm.metrics.ranked_metrics import (
            build_labels_true, soft_acc, rank_acc, simple_acc, set_f1, ndcg_at_k,
            js_distance, ndcg_readout, tau_readout, counts_on_token_keys,
        )
        acc_sum = hacc_sum = racc_sum = sacc_sum = js_sum = 0.0
        acc1_sum = racc1_sum = f1_sum = ndcg_sum = 0.0
        tf_sum = 0.0
        js_n = 0
        ndcg_r_sum = 0.0
        ndcg_r_n = 0
        tau_r_sum = 0.0
        tau_r_n = 0
        # Gated accumulators
        js_g_sum = 0.0
        js_g_n = 0
        ndcg_g_sum = 0.0
        ndcg_g_n = 0
        tau_g_sum = 0.0
        tau_g_n = 0
        tf_n = 0
        n = 0
        retr = _RetrievalCounter()
        for ex in vlmdataset.data:
            gold = ex.get("gold", {})
            pred = ex.get("pred", {})
            answer_counts = gold.get("answer_counts", {})
            labels_true = build_labels_true(answer_counts)
            if not labels_true:
                continue
            n += 1
            parsed = pred.get("labels_ranked", []) or []
            top1 = norm(pred.get("label", "")) or (parsed[0] if parsed else "")
            acc_sum += soft_acc(top1, labels_true)
            hacc_sum += float(top1 == norm(gold.get("label", "")))
            racc_sum += rank_acc(top1, labels_true)
            sacc_sum += simple_acc(top1, labels_true)
            acc1_sum += soft_acc(parsed[0] if parsed else "", labels_true)
            racc1_sum += rank_acc(parsed[0] if parsed else "", labels_true)
            f1_sum += set_f1(parsed, labels_true)
            ndcg_sum += ndcg_at_k(parsed, labels_true)
            spo = pred.get("support_prob_open")
            support_prob = spo or pred.get("support_prob_readout") or pred.get("support_prob")
            if support_prob and answer_counts:
                target = (counts_on_token_keys(answer_counts, pred.get("open_label2key"))
                          if spo else answer_counts)
                js_sum += js_distance(target, support_prob)
                js_n += 1
            if spo:
                l2k = pred.get("open_label2key")
                ndcg_r_sum += ndcg_readout(spo, labels_true, l2k)
                tau_r_sum += tau_readout(spo, labels_true, l2k)
                ndcg_r_n += 1
                tau_r_n += 1
            # Gated readout scores
            spo_g = pred.get("support_prob_open_gate")
            if spo_g and answer_counts:
                l2k = pred.get("open_label2key")
                target_g = counts_on_token_keys(answer_counts, l2k)
                js_g_sum += js_distance(target_g, spo_g)
                js_g_n += 1
            if spo_g and labels_true:
                l2k = pred.get("open_label2key")
                ndcg_g_sum += ndcg_readout(spo_g, labels_true, l2k)
                tau_g_sum += tau_readout(spo_g, labels_true, l2k)
                ndcg_g_n += 1
                tau_g_n += 1
            tf = pred.get("tf_acc")
            if tf is not None:
                tf_sum += tf
                tf_n += 1
            retr.add(pred)
        if n == 0:
            return {"simple_acc": 0.0, "hard_acc": 0.0,
                    "js_dist": None, "js_dist_gate": None,
                    "tau_readout": None, "tau_readout_gate": None,
                    "acc": 0.0, "rank_acc": 0.0, "tf_acc": None,
                    "ndcg_readout": None, "ndcg_readout_gate": None,
                    "acc_at_1": 0.0, "rank_acc_at_1": 0.0, "f1": 0.0, "ndcg_at_k": 0.0,
                    "n": 0}
        out = {
            "simple_acc": sacc_sum / n,
            "hard_acc": hacc_sum / n,
            "js_dist": js_sum / js_n if js_n > 0 else None,
            "js_dist_gate": js_g_sum / js_g_n if js_g_n > 0 else None,
            "tau_readout": tau_r_sum / tau_r_n if tau_r_n > 0 else None,
            "tau_readout_gate": tau_g_sum / tau_g_n if tau_g_n > 0 else None,
            "acc": acc_sum / n,
            "rank_acc": racc_sum / n,
            "tf_acc": tf_sum / tf_n if tf_n > 0 else None,
            "ndcg_readout": ndcg_r_sum / ndcg_r_n if ndcg_r_n > 0 else None,
            "ndcg_readout_gate": ndcg_g_sum / ndcg_g_n if ndcg_g_n > 0 else None,
            "acc_at_1": acc1_sum / n,
            "rank_acc_at_1": racc1_sum / n,
            "f1": f1_sum / n,
            "ndcg_at_k": ndcg_sum / n,
            "n": n,
        }
        out.update(retr.metrics())
        return out


def get_taskengineer(task: str, **kwargs):
    task = task.lower()
    if task == "qa":
        return QATaskEngineer(**kwargs)
    elif task == "multiqa":
        return MultiQATaskEngineer(**kwargs)
    elif task == "softqa":
        return SoftQATaskEngineer(**kwargs)
    elif task == "postedit_qa":
        return PostEditQATaskEngineer(**kwargs)
    else:
        raise ValueError(f"Unknown task engineer: {task}. Supported: qa, multiqa, softqa, postedit_qa")
