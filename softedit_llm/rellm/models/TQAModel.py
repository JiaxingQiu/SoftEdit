import torch
import logging
from contextlib import contextmanager
from typing import Dict

from .utils import (
    get_hf_model, get_tokenizer, get_preprocess, verify_inner_params,
    clean_answer, nll_to_probs, SYSTEM_PROMPT,
)

LOG = logging.getLogger(__name__)


class TQAModel(torch.nn.Module):
    """Textual Question Answering model wrapper — works with all decoder-only LLMs.

    Text-only analog of the VLM `VQAModel`: same public API (encode / generate /
    generate_scored / prepare_training_batch / teacher-forced + continuation
    checks) so metrics, editors and the soft objective port unchanged. Images,
    the vision processor, and the encoder-decoder paths are dropped; every
    backbone here (Qwen2.5-7B, Llama-3.1-8B, Mistral-7B) is decoder-only.
    """
    def __init__(self, config):
        super(TQAModel, self).__init__()
        self.config = config
        self.device = config.device
        self.temp = getattr(config.model, "temperature", 1.0)

        self.model = get_hf_model(config)
        self.model.eval()
        self.tokenizer = get_tokenizer(config)
        self.preprocess = get_preprocess(config)
        # Surface layer-count / tied-embedding issues at load time.
        self.arch_info = verify_inner_params(self.model, config)

    def forward(self, inputs):
        inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        output = self.model(**inputs)
        self.loss = getattr(output, "loss", None)
        return output.logits if hasattr(output, "logits") else output

    def encode(self, prompts, tokenize=False):
        prompts = [prompts] if isinstance(prompts, str) else prompts
        inputs = self.preprocess(prompts, self.tokenizer, tokenize=tokenize)
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        return inputs

    def question_lookup_pos(self, prompt, question):
        """0-based token index of the LAST token of ``question`` within the
        chat-templated ``prompt`` — AlphaEdit's fact-lookup ("key") position.

        The templated prompt appends a fixed instruction + assistant
        generation-prompt after the question, so the last *prompt* token is
        identical for every edit; keying there makes the null-space cache
        degenerate (see ``editors/alphaedit/alphaedit.md``). Instead we key on the
        last token of the question span, which varies per edit. Returns ``None``
        when the question can't be located or the tokenizer can't give offsets, so
        callers fall back to the last prompt token.

        Assumes a single unpadded sequence (AlphaEdit edits one sample at a time);
        the returned index is into the prompt tokens, which are the leading tokens
        of the concatenated prompt+answer sequence.
        """
        q = (question or "").strip()
        if not q:
            return None
        system_prompt = getattr(getattr(self, "config", None), "model", None)
        system_prompt = getattr(system_prompt, "system_prompt", None) or SYSTEM_PROMPT
        try:
            text = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            return None
        # rfind: the user-turn copy of the question is after the system prompt.
        qpos = text.rfind(q)
        if qpos < 0:
            return None
        qend = qpos + len(q)
        try:
            enc = self.tokenizer(text, add_special_tokens=False,
                                 return_offsets_mapping=True)
        except (TypeError, NotImplementedError):
            return None  # slow tokenizer without offset mapping
        offsets = enc.get("offset_mapping") if isinstance(enc, dict) else enc["offset_mapping"]
        if not offsets:
            return None
        # Last token whose char span starts inside the question span. Tokens that
        # start at/after qend belong to the trailing instruction/template, and
        # special tokens reported as (0, 0) are excluded by the lower bound.
        pos = None
        for i, (s, e) in enumerate(offsets):
            if qpos <= s < qend:
                pos = i
            elif s >= qend:
                break
        return pos

    def _apply_gen_defaults(self, kwargs):
        """Fill in shared generation defaults (temperature / greedy) in place."""
        kwargs.setdefault("temperature", self.temp)
        if kwargs["temperature"] <= 0:
            kwargs["do_sample"] = False
            kwargs["num_beams"] = 1
            kwargs.pop("temperature", None)
        return kwargs

    def generate(self, prompts, **kwargs):
        prompts = [prompts] if isinstance(prompts, str) else prompts
        inputs = self.encode(prompts, tokenize=False)
        self._apply_gen_defaults(kwargs)
        in_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **kwargs)
        gen = outputs[:, in_len:]
        texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        return [clean_answer(o, p) for (o, p) in zip(texts, prompts)]

    @torch.no_grad()
    def generate_scored(self, prompts, candidate_labels, open_labels=None, **kwargs):
        """Generate greedy answers AND first-token support probs in one forward.

        The distribution over the first generated position (generate's
        ``scores[0]``) is identical to a single forward's last-position logits,
        so the answer string and the label distribution come from the same pass.

        Args:
            candidate_labels: list[list[str]] — support labels per example.
            open_labels: optional list[list[str]] — human labels per example.
                When given, an open-vocabulary readout is also computed from the
                same scores (see ``open_readout_probs``).
        Returns:
            (answers, support_probs) — or
            (answers, support_probs, open_readouts, raw_readouts) when
            ``open_labels`` is provided. support_probs is a list[dict] of
            {label: prob} renormalized over each example's support.
        """
        prompts = [prompts] if isinstance(prompts, str) else prompts
        inputs = self.encode(prompts, tokenize=False)
        self._apply_gen_defaults(kwargs)
        kwargs["return_dict_in_generate"] = True
        kwargs["output_scores"] = True
        in_len = inputs["input_ids"].shape[1]
        outputs = self.model.generate(**inputs, **kwargs)
        gen = outputs.sequences[:, in_len:]
        outputs_text = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        answers = [clean_answer(o, p) for (o, p) in zip(outputs_text, prompts)]

        scores = getattr(outputs, "scores", None)
        support_probs = []
        for b, labels in enumerate(candidate_labels):
            if scores:
                support_probs.append(
                    self._first_token_probs(scores[0][b], labels, renormalize=True)
                )
            else:
                support_probs.append({})
        if open_labels is None:
            return answers, support_probs
        open_readouts = []
        raw_readouts = []
        for b, hlabels in enumerate(open_labels):
            if scores and hlabels:
                logits_b = scores[0][b]
                open_readouts.append(self.open_readout_probs(logits_b, hlabels))
                all_labels = list(dict.fromkeys(
                    list(candidate_labels[b]) + list(hlabels)))
                raw_readouts.append(self.build_readout_raw(logits_b, all_labels))
            else:
                open_readouts.append(None)
                raw_readouts.append(None)
        return answers, support_probs, open_readouts, raw_readouts

    def _first_token_probs(self, logits, candidate_labels, renormalize: bool = False):
        """Map a 1-D logits vector → {label: prob} via each label's first token."""
        probs = torch.softmax(logits.float(), dim=-1)
        out = {}
        for label in candidate_labels:
            token_ids = self.tokenizer.encode(label, add_special_tokens=False)
            out[label] = float(probs[token_ids[0]].item()) if token_ids else 0.0
        if renormalize:
            z = sum(out.values())
            if z > 0:
                out = {k: v / z for k, v in out.items()}
        return out

    def open_readout_probs(self, logits, human_labels):
        """Open-vocabulary readout over (model top-k ∪ human first tokens) + OTHER.

        The support is the model's OWN top-k first-token ids (k = number of
        distinct human-label first-token ids — the only external knowledge is
        how MANY labels the annotators gave, never which strings to probe)
        unioned with the human labels' first-token ids. Probabilities are the
        raw full-vocab softmax values; the residual mass goes into a reserved
        "__OTHER__" bucket instead of being renormalized away, so mass the model
        puts outside the support counts as disagreement in js_dist.

        Returns:
            (probs, label2key) — probs is {token_key: prob, "__OTHER__": residual}
            summing to 1; label2key maps each human label to its token key.
            (None, None) when no human label tokenizes.
        """
        probs = torch.softmax(logits.float(), dim=-1)

        label2id = {}
        for label in human_labels:
            token_ids = self.tokenizer.encode(str(label), add_special_tokens=False)
            if token_ids:
                label2id[str(label)] = int(token_ids[0])
        if not label2id:
            return None, None
        human_ids = set(label2id.values())
        k = len(human_ids)
        topk_ids = {int(t) for t in probs.topk(k).indices.tolist()}
        support_ids = human_ids | topk_ids

        # Key by the normalized decoded token string (readable dicts in preds);
        # distinct ids decoding to the same normalized string merge by summing.
        # "__OTHER__" is reserved: js_distance lowercases keys, so a bare "OTHER"
        # could collide with a genuine "other" answer.
        id2key = {}
        out = {}
        for tid in support_ids:
            key = self.tokenizer.decode([tid]).strip().lower() or f"tok_{tid}"
            id2key[tid] = key
            out[key] = out.get(key, 0.0) + float(probs[tid].item())
        out["__OTHER__"] = max(1.0 - sum(out.values()), 0.0)
        label2key = {label: id2key[tid] for label, tid in label2id.items()}
        return out, label2key

    def build_readout_raw(self, logits, labels, top_m=20):
        """Build a JSON-safe raw readout record from a 1-D logits vector.

        Sufficient to reconstruct any readout convention offline (open /
        restricted / different k / temperature rescaling) without the model.
        """
        logits = logits.float()
        logZ = float(torch.logsumexp(logits, dim=-1).item())
        topk = logits.topk(min(top_m, logits.shape[-1]))
        top_ids = topk.indices.tolist()
        top_logits = topk.values.tolist()
        top_tokens = [self.tokenizer.decode([tid]).strip().lower() or f"tok_{tid}"
                      for tid in top_ids]
        label_ids, label_logits, label_tokens = {}, {}, {}
        for label in labels:
            tids = self.tokenizer.encode(str(label), add_special_tokens=False)
            if tids:
                tid = tids[0]
                label_ids[label] = tid
                label_logits[label] = float(logits[tid].item())
                label_tokens[label] = self.tokenizer.decode([tid]).strip().lower() or f"tok_{tid}"
        return {
            "top_ids": top_ids, "top_logits": top_logits, "top_tokens": top_tokens,
            "label_ids": label_ids, "label_logits": label_logits, "label_tokens": label_tokens,
            "logZ": logZ,
        }

    @torch.no_grad()
    def get_loss_y(self, prompt, label, inputs=None, add_special_tokens=False):
        if inputs is None:
            inputs = self.encode(prompt, tokenize=False)

        label_ids = self.tokenizer(
            label, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids.to(self.device)

        # decoder-only: concat and mask
        input_ids = inputs["input_ids"]
        attn = inputs.get("attention_mask", None)

        full_ids = torch.cat([input_ids, label_ids], dim=1)
        if attn is not None:
            label_attn = torch.ones_like(label_ids, device=self.device)
            full_attn = torch.cat([attn, label_attn], dim=1)
        else:
            full_attn = None

        labels_mask = torch.full_like(full_ids, -100)
        prompt_len = input_ids.size(1)
        labels_mask[:, prompt_len:] = full_ids[:, prompt_len:]

        model_inputs = dict(inputs)
        model_inputs["input_ids"] = full_ids
        if full_attn is not None:
            model_inputs["attention_mask"] = full_attn
        model_inputs["labels"] = labels_mask

        out = self.model(**model_inputs)
        avg_nll = float(out.loss.item())
        num_tokens = full_ids.size(1) - prompt_len
        num_tokens = max(1, num_tokens)
        return avg_nll, avg_nll * num_tokens, num_tokens

    @torch.no_grad()
    def score_choices_single(self, pr, lbls, use_avg: bool = False, temperature: float = 1.0):
        inputs = self.encode(pr, tokenize=False)
        scores = {}
        for lbl in lbls:
            avg_nll, sum_nll, ntok = self.get_loss_y(pr, lbl, inputs, add_special_tokens=False)
            scores[lbl] = {"avg_nll": avg_nll, "sum_nll": sum_nll, "num_tokens": ntok}
        probs = nll_to_probs(scores, use_avg=use_avg, temperature=temperature)
        for lbl, p in probs.items():
            scores[lbl]["prob"] = float(p)
        return scores

    def get_loss(self, batch: Dict):
        """Return differentiable loss tensor for a loader batch (for finetuning).

        Requires batch with 'prompts' and 'golds' (list of dicts with 'label').
        """
        prompts = batch.get("prompts")
        prompt_inputs = self.encode(prompts, tokenize=False)

        golds = batch["golds"]
        gold_texts = [str(g.get("label_train", g["label"])) for g in golds]

        labels_ids, label_attn = self._tokenize_targets(gold_texts)
        if labels_ids.shape[1] == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        input_ids = prompt_inputs.get("input_ids")
        attn = prompt_inputs.get("attention_mask")

        # Decoder-only: concatenate prompt + labels; mask prompt tokens and answer padding
        full_ids = torch.cat([input_ids, labels_ids], dim=1)
        full_attn = torch.cat([attn, label_attn], dim=1) if attn is not None else None
        labels = torch.full_like(full_ids, -100)
        prompt_len = int(input_ids.shape[1])
        labels[:, prompt_len:] = labels_ids.masked_fill(label_attn == 0, -100)

        model_inputs = dict(prompt_inputs)
        model_inputs["input_ids"] = full_ids
        if full_attn is not None:
            model_inputs["attention_mask"] = full_attn
        out = self.model(**model_inputs, labels=labels)
        return out.loss

    def prepare_training_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Prepare a training batch into model inputs with labels (for editors).

        Like ``get_loss`` but returns the inputs dict (with a `-100`-masked
        `labels`) instead of computing loss, so editors call `model(**inputs)`
        and handle the loss themselves (via `soft_utils.compute_edit_loss`).
        """
        prompts = batch.get("prompts")
        prompt_inputs = self.encode(prompts, tokenize=False)

        golds = batch["golds"]
        gold_texts = [str(g.get("label_train", g["label"])) for g in golds]

        # Right-pad the (possibly multi-token) targets so the answer stays
        # contiguous after the prompt; keep the pad mask so pad is never supervised.
        labels_ids, label_attn = self._tokenize_targets(gold_texts)

        input_ids = prompt_inputs.get("input_ids")
        attn = prompt_inputs.get("attention_mask")

        full_ids = torch.cat([input_ids, labels_ids], dim=1)
        full_attn = torch.cat([attn, label_attn], dim=1) if attn is not None else None

        labels = torch.full_like(full_ids, -100)
        prompt_len = int(input_ids.shape[1])
        answer_labels = labels_ids.masked_fill(label_attn == 0, -100)
        labels[:, prompt_len:] = answer_labels

        model_inputs = dict(prompt_inputs)
        model_inputs["input_ids"] = full_ids
        if full_attn is not None:
            model_inputs["attention_mask"] = full_attn
        model_inputs["labels"] = labels
        return model_inputs

    def _tokenize_targets(self, gold_texts):
        """Tokenize supervision targets with a trailing EOS and right padding.

        Each target is encoded (add_special_tokens=False) and an EOS token
        appended so the supervised sequence is ``answer + <eos>``: teacher-forcing
        trains the model to STOP after the answer. Without it, editing only raises
        the answer tokens' probability and never P(eos|answer), so a hard edit
        collapses free generation into repeated tokens.

        Right padding keeps answer tokens contiguous after the prompt for
        decoder-only concatenation; the attention_mask masks pad positions out of
        both the loss (-100) and attention. Falls back to no EOS when undefined,
        and never double-appends.
        """
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos_id if eos_id is not None else 0

        seqs = []
        for text in gold_texts:
            ids = self.tokenizer.encode(str(text), add_special_tokens=False)
            if eos_id is not None and (not ids or ids[-1] != eos_id):
                ids = ids + [eos_id]
            seqs.append(ids)

        max_len = max((len(s) for s in seqs), default=0)
        input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            if s:
                input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
                attn[i, :len(s)] = 1
        return input_ids.to(self.device), attn.to(self.device)

    def _tf_token_match(self, pred_ids, gold_ids):
        """Position-wise match rate between predicted and gold token ids, compared
        on decoded surface form after casefold/strip (raw-id equality is too
        strict: models differ only in surface form / SentencePiece ▁-prefixes)."""
        pred_ids = pred_ids.reshape(-1).tolist()
        gold_ids = gold_ids.reshape(-1).tolist()
        if not gold_ids:
            return 0.0
        matches = 0
        for p, g in zip(pred_ids, gold_ids):
            p_txt = self.tokenizer.decode([int(p)]).strip().casefold()
            g_txt = self.tokenizer.decode([int(g)]).strip().casefold()
            if p_txt == g_txt:
                matches += 1
        return float(matches) / float(len(gold_ids))

    @contextmanager
    def _grace_query_at(self, key_id):
        """Temporarily point any GRACE adaptor's retrieval query at ``key_id``.

        GRACE is the only editor whose forward is position-sensitive (queries a
        single token position, default -1). During a teacher-forced concat forward
        that -1 falls on a target token instead of the keyed prompt-end, so the
        edit silently doesn't fire. Detected by class name to avoid importing the
        editor here; restores each adaptor's original ``key_id`` afterwards.
        """
        adaptors = [
            m for m in self.model.modules()
            if type(m).__name__ == "GRACEAdaptor" and hasattr(m, "key_id")
        ]
        saved = [m.key_id for m in adaptors]
        for m in adaptors:
            m.key_id = key_id
        try:
            yield
        finally:
            for m, k in zip(adaptors, saved):
                m.key_id = k

    @torch.no_grad()
    def teacher_forced_acc(self, prompt, target, add_special_tokens=False):
        """Teacher-forced token accuracy: fraction of `target` tokens that are the
        model's argmax prediction given `prompt` (prior-work "rewrite_acc" style).

        Measures whether an edit "took" at the logit level, independent of free
        generation / parsing. Scores a few case variants of the target and takes
        the best, crediting the edit whenever the model prefers the label in any
        casing (case can change tokenization outright)."""
        if not target:
            return 0.0
        inputs = self.encode(prompt, tokenize=False)
        variants = []
        for v in (target, target.lower(), target[:1].upper() + target[1:]):
            if v and v not in variants:
                variants.append(v)
        return max(self._tf_acc_one(inputs, v, add_special_tokens) for v in variants)

    @torch.no_grad()
    def _tf_acc_one(self, inputs, target, add_special_tokens=False):
        """Teacher-forced token match for one exact target string (prompt reused)."""
        label_ids = self.tokenizer(
            target, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids.to(self.device)
        if label_ids.numel() == 0:
            return 0.0

        # Decoder-only: concat prompt + target; logits at [prompt_len-1 : -1]
        # predict the target tokens.
        input_ids = inputs["input_ids"]
        attn = inputs.get("attention_mask", None)
        model_inputs = dict(inputs)
        model_inputs["input_ids"] = torch.cat([input_ids, label_ids], dim=1)
        if attn is not None:
            model_inputs["attention_mask"] = torch.cat([attn, torch.ones_like(label_ids)], dim=1)
        prompt_len = input_ids.size(1)
        # Pin any GRACE adaptor to the prompt-end token so tf_acc matches generate().
        with self._grace_query_at(prompt_len - 1):
            logits = self.model(**model_inputs).logits
        pred = logits[:, prompt_len - 1:-1, :].argmax(-1)
        return self._tf_token_match(pred, label_ids)

    @torch.no_grad()
    def continuation_passes(self, prompt, label):
        """Binary continuation check: would the model complete ``label`` correctly?

        Conditioned on the first token, checks whether the model's argmax at every
        subsequent position matches the gold continuation tokens. Single-token
        labels trivially pass. Uses surface-form normalization per token.
        """
        if not label:
            return False
        token_ids = self.tokenizer.encode(str(label), add_special_tokens=False)
        if len(token_ids) <= 1:
            return True

        inputs = self.encode(prompt, tokenize=False)
        label_tensor = torch.tensor([token_ids], device=self.device)

        input_ids = inputs["input_ids"]
        attn = inputs.get("attention_mask", None)
        model_inputs = dict(inputs)
        model_inputs["input_ids"] = torch.cat([input_ids, label_tensor], dim=1)
        if attn is not None:
            model_inputs["attention_mask"] = torch.cat(
                [attn, torch.ones_like(label_tensor)], dim=1)

        prompt_len = input_ids.size(1)
        with self._grace_query_at(prompt_len - 1):
            logits = self.model(**model_inputs).logits

        # logits[:, prompt_len : prompt_len+L-1] predict tokens 2:L of the label.
        pred_logits = logits[:, prompt_len:prompt_len + len(token_ids) - 1, :]
        preds = pred_logits.argmax(-1).reshape(-1).tolist()
        gold_continuation = token_ids[1:]

        for p, g in zip(preds, gold_continuation):
            p_txt = self.tokenizer.decode([int(p)]).strip().casefold()
            g_txt = self.tokenizer.decode([int(g)]).strip().casefold()
            if p_txt != g_txt:
                return False
        return True

    @torch.no_grad()
    def get_next_token_probs(self, prompt, candidate_labels):
        """First-token prob per candidate from a single forward pass (not renormalized)."""
        inputs = self.encode(prompt, tokenize=False)
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :][0]
        return self._first_token_probs(logits, candidate_labels, renormalize=False)
