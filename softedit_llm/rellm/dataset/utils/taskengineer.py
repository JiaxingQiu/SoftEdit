# TaskEngineer: task-dependent I/O engineering for each example (text-only / LLM).
# ex is a binding to the dict stored in self.data, mutable in place.
# (1) prompt
# (2) gold  — label (answer text), label_train (supervision target)
# (3) pred  — label, answer (raw text), plus task-specific fields
#
# Supported tasks:
#   qa      — single-answer QA with soft_acc / rank_acc eval
#   multiqa — ranked multilabel QA with Acc@1, RankAcc@1, F1, nDCG@K eval
#   softqa  — joint protocol: QA + MultiQA on same sample (two passes)
#   postedit_qa — post-edit eval: one anchored pass scored like QA


# Shared single-answer QA instruction. Used by BOTH the prediction pass
# (QATaskEngineer, via softqa) and the editing/eval pass (PostEditQATaskEngineer),
# so the prompt that selects the edit set (rank_acc==0) is identical to the prompt
# used for editing + reliability. ProtoQA answers are words or short phrases.
ANSWER_INSTRUCTION = "Answer with a word or a short phrase."


def norm(s):
    return str(s).strip().lower() if s is not None else None


def _parse_single_label(answer, support_prob=None, strategy="text"):
    """Shared top-1 label policy for single-label QA tasks (QA + PostEditQA)."""
    from rellm.metrics.ranked_metrics import parse_single_answer, parse_qa_label
    if strategy == "prob" and support_prob:
        return parse_qa_label(answer, support_prob, "prob")
    return parse_single_answer(answer or "")


def support_labels(ex):
    """Candidate label set for support_prob: explicit ``ex["support_labels"]``
    override when present (soft_generality scores over the edit sample's original
    label set), else answer_counts keys."""
    if ex.get("support_labels"):
        return list(ex["support_labels"])
    ac = ex.get("answer_counts") or (ex.get("gold") or {}).get("answer_counts") or {}
    return list(ac.keys())


def human_labels(ex):
    """Human answer labels (answer_counts keys) for the open readout."""
    ac = ex.get("answer_counts") or (ex.get("gold") or {}).get("answer_counts") or {}
    return list(ac.keys())


class _RetrievalCounter:
    """Aggregate retrieval diagnostics for retrieval editors (no-op for others)."""

    def __init__(self):
        self.n = 0
        self.retrieved = 0
        self.hits = 0

    def add(self, pred):
        dbg = pred.get("readout_dbg")
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
    """Best-effort per-batch model prep (use_cache toggle)."""
    try:
        core = getattr(model, "model", model)
        cfg = getattr(core, "config", None)
        if cfg is not None:
            cfg.use_cache = use_cache
        if hasattr(core, "gradient_checkpointing_disable"):
            core.gradient_checkpointing_disable()
    except Exception:
        pass


def _carry_open_readout(ex, open_readout):
    """Store the open-vocabulary readout on the pred (editor override wins)."""
    spo, l2k = ex.get("support_prob_open"), ex.get("open_label2key")
    if spo is None and open_readout is not None:
        spo, l2k = open_readout
    if spo is not None:
        ex['pred']['support_prob_open'] = spo
        ex['pred']['open_label2key'] = l2k


def _carry_readout_raw(ex, readout_raw):
    """Store the raw readout record on the pred for offline rescoring."""
    rr = ex.get("readout_raw", readout_raw)
    if rr is not None:
        ex['pred']['readout_raw'] = rr


def _scored_qa_pass(dataset, model, engineer, use_cache, max_new_tokens):
    """QA generation that also records first-token support probs (one forward).

    A binary continuation gate is applied per-example after generation (unless
    disabled via `config.continuation_gate=False`): for each first-token branch
    matching a multi-token human answer, the model is teacher-forced on the
    remaining tokens; failing branches are zeroed and their mass flows into
    OTHER. Both first-token and gated readouts are stored so eval reports both
    (`js_dist` / `js_dist_gate`).
    """
    arm = getattr(dataset, "pregen_arm", None)
    do_gate = getattr(getattr(dataset, "config", None), "continuation_gate", True)
    for batch in dataset.loader:
        if not batch["prompts"]:
            continue
        _prep_model_core(model, use_cache)
        if arm is not None:
            arm([dataset.data[i] for i in batch["idxs"]])
        labels_batch = [support_labels(dataset.data[i]) for i in batch["idxs"]]
        open_batch = [human_labels(dataset.data[i]) for i in batch["idxs"]]
        answers, support_probs, open_readouts, raw_readouts = model.generate_scored(
            batch["prompts"], labels_batch, open_labels=open_batch,
            max_new_tokens=max_new_tokens, use_cache=use_cache,
        )
        for idx, a, sp, oro, rr in zip(batch["idxs"], answers, support_probs,
                                        open_readouts, raw_readouts):
            ex = dataset.data[idx]
            engineer.eng_preds(ex, a, model, support_prob=sp,
                               open_readout=oro, readout_raw=rr)
            if do_gate and oro is not None and ex.get("pred", {}).get("readout_raw") is not None:
                _apply_gate_to_pred(model, ex)
    if arm is not None:
        arm(None)


def _apply_gate_to_pred(model, ex):
    """Run the continuation gate for one example and store gated readout on pred."""
    from rellm.metrics.readout import apply_continuation_gate

    pred = ex.get("pred") or {}
    spo = pred.get("support_prob_open")
    l2k = pred.get("open_label2key")
    if not spo or not l2k:
        return

    h_labels = human_labels(ex)
    if not h_labels:
        return

    prompt = ex.get("prompt", "")
    gate = apply_continuation_gate(model, prompt, spo, h_labels, l2k)

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
        raise NotImplementedError("Subclasses must implement parse_answer")

    def reparse(self, data):
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
        a = ex.get("answer_override") or a
        ex['pred'] = {'answer': a}
        if support_prob is not None:
            ex['pred']['support_prob'] = support_prob
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
            max_new_tokens = getattr(dataset.config, "max_new_tokens", 32)
        _scored_qa_pass(dataset, model, self, use_cache, max_new_tokens)

    def eval(self, tqadataset):
        from rellm.metrics.ranked_metrics import (
            build_labels_true, soft_acc, rank_acc, simple_acc, js_distance,
            ndcg_readout, tau_readout, counts_on_token_keys,
        )
        acc_sum = hacc_sum = racc_sum = sacc_sum = js_sum = 0.0
        js_n = 0
        ndcg_r_sum = ndcg_r_n = 0.0
        tau_r_sum = tau_r_n = 0.0
        js_g_sum = js_g_n = 0.0
        ndcg_g_sum = ndcg_g_n = 0.0
        tau_g_sum = tau_g_n = 0.0
        n = 0
        retr = _RetrievalCounter()
        for ex in tqadataset.data:
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
    """Task engineer for ranked multilabel QA."""

    INSTRUCTION = (
        "List all plausible answers (a word or short phrase each) ranked from "
        "most to least likely. Return a numbered list:\n1. answer1\n2. answer2\n3. ..."
    )
    SUFFIX_CONCISE = "\nReply ONLY with a numbered list, no reasoning or formatting."

    def __init__(self, seed=333, model_name="", **kwargs):
        super().__init__()
        self.name = "multiqa"
        self.seed = seed
        self.concise = False

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
        from rellm.metrics.ranked_metrics import parse_numbered_list
        parsed = parse_numbered_list(answer or "")
        return {"labels_ranked": parsed, "label": parsed[0] if parsed else ""}

    def eng_preds(self, ex, a: str, model):
        ex['pred'] = {'answer': a, **self.parse_answer(a)}

    def eval(self, tqadataset):
        from rellm.metrics.ranked_metrics import build_labels_true, soft_acc, rank_acc, set_f1, ndcg_at_k

        acc1_sum = racc1_sum = f1_sum = ndcg_sum = 0.0
        n = 0
        for ex in tqadataset.data:
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
    """Joint protocol: QA (free single answer) + MultiQA (ranked list). Two passes."""

    def __init__(self, seed=333, model_name="", label_strategy="text", **kwargs):
        super().__init__()
        self.name = "softqa"
        self.seed = seed
        self.label_strategy = label_strategy
        self._qa = QATaskEngineer(seed=seed, model_name=model_name, label_strategy=label_strategy, **kwargs)
        self._multiqa = MultiQATaskEngineer(seed=seed, model_name=model_name, **kwargs)

    def eng_golds(self, ex):
        self._multiqa.eng_golds(ex)
        ex['gold']['label_train'] = ex['gold']['label']

    def eng_prompt(self, ex):
        self._qa.eng_prompt(ex)
        ex['prompt_qa'] = ex['prompt']
        self._multiqa.eng_prompt(ex)
        ex['prompt_multiqa'] = ex['prompt']
        ex['prompt'] = ex['prompt_qa']

    def eng_preds(self, ex, a: str, model):
        raise NotImplementedError("SoftQA uses run_generate for two-pass generation")

    def parse_answer(self, answer, support_prob=None):
        return self._qa.parse_answer(answer, support_prob)

    def reparse(self, data):
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
            max_new_tokens = getattr(dataset.config, "max_new_tokens", 32)

        # Pass 1: QA (short answer + first-token support probs, one forward)
        for ex in dataset.data:
            ex['prompt'] = ex['prompt_qa']
        dataset.loader = DataLoader(
            dataset, batch_size=dataset.config.batch_size,
            shuffle=False, num_workers=0, pin_memory=True,
            collate_fn=dataset.text_collate,
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
            collate_fn=dataset.text_collate,
        )
        self._run_pass(dataset, model, self._multiqa, use_cache, multiqa_max)
        for ex in dataset.data:
            ex['pred_multiqa'] = ex.pop('pred', {})
            ex['pred'] = ex['pred_qa']

    @staticmethod
    def _run_pass(dataset, model, engineer, use_cache, max_new_tokens):
        for batch in dataset.loader:
            if not batch["prompts"]:
                continue
            _prep_model_core(model, use_cache)
            outs = model.generate(
                batch["prompts"], max_new_tokens=max_new_tokens, use_cache=use_cache,
            )
            for idx, a in zip(batch["idxs"], outs):
                engineer.eng_preds(dataset.data[idx], a, model)

    def eval(self, tqadataset):
        qa_metrics = self._qa.eval(_View(tqadataset, "pred_qa"))
        multi_metrics = self._multiqa.eval(_View(tqadataset, "pred_multiqa"))
        return {"qa": qa_metrics, "multiqa": multi_metrics}


class _View:
    """Lightweight view exposing `.data` with a chosen pred field aliased to `pred`.

    Lets SoftQA reuse QA/MultiQA `.eval` (which read `ex["pred"]`) on the
    per-pass predictions without duplicating the scoring code.
    """
    def __init__(self, dataset, pred_key):
        self.data = [dict(ex, pred=ex.get(pred_key, {})) for ex in dataset.data]


class PostEditQATaskEngineer(TaskIOEngineer):
    """Post-edit evaluation: one anchored pass, scored like QA.

    The edited model is prompted with the question plus the shared format
    instruction (:attr:`INSTRUCTION`) and scored in one pass. Because the
    instruction lives here (and set_dataloader calls eng_prompt for every set),
    editing and every eval set — edit / locality / generality — share the
    identical prompt for both the pre- and post-edit models.
    """

    INSTRUCTION = ANSWER_INSTRUCTION

    def __init__(self, seed=333, model_name="", label_strategy="text", **kwargs):
        super().__init__()
        self.name = "postedit_qa"
        self.seed = seed
        self.label_strategy = label_strategy
        self._multiqa = MultiQATaskEngineer(seed=seed, model_name=model_name, **kwargs)

    def eng_golds(self, ex):
        self._multiqa.eng_golds(ex)
        ex['gold']['label_train'] = ex['gold']['label']

    def eng_prompt(self, ex):
        q = ex['question']
        ex['prompt'] = f"{q} {self.INSTRUCTION}" if self.INSTRUCTION else q

    def parse_answer(self, answer, support_prob=None):
        from rellm.metrics.ranked_metrics import parse_numbered_list
        parsed = parse_numbered_list(answer or "")
        label = _parse_single_label(answer, support_prob, self.label_strategy)
        return {"labels_ranked": parsed, "label": label}

    def eng_preds(self, ex, a: str, model, support_prob=None, open_readout=None, readout_raw=None):
        a = ex.get("answer_override") or a
        ex['pred'] = {'answer': a}
        if support_prob is not None:
            ex['pred']['support_prob'] = support_prob
        if ex.get("support_prob_readout") is not None:
            ex['pred']['support_prob_readout'] = ex['support_prob_readout']
        if ex.get("readout_dbg") is not None:
            ex['pred']['readout_dbg'] = ex['readout_dbg']
        _carry_open_readout(ex, open_readout)
        _carry_readout_raw(ex, readout_raw)
        ex['pred'].update(self.parse_answer(a, support_prob))

    def run_generate(self, dataset, model, use_cache=True, max_new_tokens=None, compute_tf=None):
        """Single scored pass; enough room for a short answer / label list.

        Also records a teacher-forced argmax accuracy on the majority-vote label
        unless disabled. Locality uses ``compute_tf=False``.
        """
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
                    pred["tf_acc"] = model.teacher_forced_acc(ex["prompt"], target)

    def eval(self, tqadataset):
        from rellm.metrics.ranked_metrics import (
            build_labels_true, soft_acc, rank_acc, simple_acc, set_f1, ndcg_at_k,
            js_distance, ndcg_readout, tau_readout, counts_on_token_keys,
        )
        acc_sum = hacc_sum = racc_sum = sacc_sum = js_sum = 0.0
        acc1_sum = racc1_sum = f1_sum = ndcg_sum = 0.0
        tf_sum = 0.0
        js_n = 0
        ndcg_r_sum = ndcg_r_n = 0.0
        tau_r_sum = tau_r_n = 0.0
        js_g_sum = js_g_n = 0.0
        ndcg_g_sum = ndcg_g_n = 0.0
        tau_g_sum = tau_g_n = 0.0
        tf_n = 0
        n = 0
        retr = _RetrievalCounter()
        for ex in tqadataset.data:
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
