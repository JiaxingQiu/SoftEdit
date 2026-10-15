import torch
import logging
import math
from contextlib import contextmanager
from PIL import Image
from .utils import *
from typing import Dict

LOG = logging.getLogger(__name__)


def resize_image(img, max_size, preserve_ratio=True):
    """Resize image to max_size. If preserve_ratio, keep aspect ratio with longest edge = max_size."""
    if preserve_ratio:
        w, h = img.size
        if max(w, h) <= max_size:
            return img
        scale = max_size / max(w, h)
        return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    else:
        sz = (max_size, max_size) if isinstance(max_size, int) else max_size
        return img.resize(sz, Image.LANCZOS)


class VQAModel(torch.nn.Module):
    """Vision Question Answering model wrapper - works with all VLMs"""
    def __init__(self, config):
        super(VQAModel, self).__init__()
        self.config = config
        self.device = config.device
        self.temp = getattr(config.model, "temperature", 1.0)
        self.image_size = getattr(config.model, "image_size", 336)# None) # none for no resizing

        self.model = get_hf_model(config)
        self.model.eval()
        self.processor = get_processor(config)
        self.tokenizer = get_tokenizer(config)
        self.preprocess = get_preprocess(config)
        
    
    def forward(self, inputs):
        inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        output = self.model(**inputs)
        self.loss = getattr(output, "loss", None)
        return output.logits if hasattr(output, "logits") else output

    def encode(self, images, prompts, tokenize=False):
        images = [images] if isinstance(images, (Image.Image, str)) else images
        images = [Image.open(img).convert("RGB") if isinstance(img, str) else img for img in images]
        prompts = [prompts] if isinstance(prompts, str) else prompts
        if self.image_size:
            images = [resize_image(img, self.image_size) for img in images]
        inputs = self.preprocess(images, prompts, self.processor, tokenize=tokenize)
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        return inputs
        

    def _apply_gen_defaults(self, kwargs):
        """Fill in shared generation defaults (temperature / greedy) in place."""
        kwargs.setdefault("temperature", self.temp)
        if kwargs["temperature"] <= 0:
            kwargs["do_sample"] = False
            kwargs["num_beams"] = 1
            kwargs.pop("temperature", None)
        return kwargs

    def generate(self, images, prompts, **kwargs):
        inputs = self.encode(images, prompts, tokenize=False)
        # Minimal deterministic defaults; can be overridden via kwargs
        self._apply_gen_defaults(kwargs)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **kwargs)
            outputs_text = self.processor.batch_decode(outputs, skip_special_tokens=True)
            answers = [clean_answer(o, i) for (o, i) in zip(outputs_text, prompts)]
        return answers

    @torch.no_grad()
    def generate_scored(self, images, prompts, candidate_labels, open_labels=None, **kwargs):
        """Generate greedy answers AND first-token support probs in one forward.

        The distribution over the first generated position (generate's
        ``scores[0]``) is identical to a single forward's last-position logits,
        so the answer string and the label distribution come from the same pass.

        Args:
            candidate_labels: list[list[str]] — support labels per example.
            open_labels: optional list[list[str]] — human labels per example.
                When given, an open-vocabulary readout is also computed from the
                same scores (see ``open_readout_probs``) and a third list is
                returned: entries are (probs, label2key) tuples or None.
        Returns:
            (answers, support_probs) — or (answers, support_probs, open_readouts)
            when ``open_labels`` is provided. support_probs is a list[dict] of
            {label: prob} renormalized over each example's support.
        """
        inputs = self.encode(images, prompts, tokenize=False)
        self._apply_gen_defaults(kwargs)
        kwargs["return_dict_in_generate"] = True
        kwargs["output_scores"] = True
        outputs = self.model.generate(**inputs, **kwargs)
        outputs_text = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)
        answers = [clean_answer(o, i) for (o, i) in zip(outputs_text, prompts)]

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
        See revlm/metrics/rank_metrics.md.

        Args:
            logits: 1-D logits vector at the answer (first generated) position.
            human_labels: the human answer labels for this example.
        Returns:
            (probs, label2key) — probs is {token_key: prob, "__OTHER__": residual}
            summing to 1; label2key maps each human label to its token key so
            the gold side can be aligned (js support / ndcg_readout relevance).
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
        # "__OTHER__" is reserved: js_distance lowercases keys, so a bare
        # "OTHER" would collide with a genuine "other" answer (VQA-v2 has them).
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

        The record is sufficient to reconstruct any readout convention offline
        (open / restricted / different k / temperature rescaling) without the
        model or tokenizer.

        Args:
            logits: 1-D logits at the answer (first generated token) position.
            labels: list of label strings whose first-token logits to capture.
            top_m: number of top tokens to store from the full vocabulary.
        Returns:
            dict with keys ``top_ids``, ``top_logits``, ``top_tokens``,
            ``label_ids``, ``label_logits``, ``label_tokens``, ``logZ``.
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
    def get_loss_y(self, image, prompt, label, inputs=None, add_special_tokens=False):
        # stay close to old behavior: no extra tokens by default
        if inputs is None:
            inputs = self.encode(image, prompt, tokenize=False)

        label_ids = self.tokenizer(
            label,
            return_tensors="pt",
            add_special_tokens=add_special_tokens
        ).input_ids.to(self.device)

        is_enc_dec = bool(getattr(self.model.config, "is_encoder_decoder", False))

        if is_enc_dec:
            out = self.model(**inputs, labels=label_ids)
            avg_nll = float(out.loss.item())
            num_tokens = label_ids.size(1)
            return avg_nll, avg_nll * num_tokens, num_tokens

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
    def score_choices_single(self, img, pr, lbls, use_avg: bool = False, temperature: float = 1.0):
        inputs = self.encode(img, pr, tokenize=False)
        scores = {}
        for lbl in lbls:
            avg_nll, sum_nll, ntok = self.get_loss_y(img, pr, lbl, inputs, add_special_tokens=False)
            scores[lbl] = {
                "avg_nll": avg_nll,
                "sum_nll": sum_nll,
                "num_tokens": ntok,
            }
        probs = nll_to_probs(scores, use_avg=use_avg, temperature=temperature)
        for lbl, p in probs.items():
            scores[lbl]["prob"] = float(p)
        return scores



    def get_loss(self, batch: Dict):
        """Return differentiable loss tensor for a loader batch (for finetuning).
        Requires batch with 'images', 'prompts', and 'golds' (list of dicts with 'label').
        
        Args:
            batch: Dict with keys:
                - 'images': List[PIL.Image] or PIL.Image
                - 'prompts': List[str] or str
                - 'golds': List[Dict] where each dict has 'label' key
                
        Returns:
            torch.Tensor: Loss value with gradients enabled
        """
        images = batch.get("images")
        prompts = batch.get("prompts")
        prompt_inputs = self.encode(images, prompts, tokenize=False)

        # Collect gold answer texts (strict requirement)
        golds = batch["golds"]  # expect list of dicts
        # Prefer task-specific training label if provided (e.g., MCI uses "(A) car")
        gold_texts = [str(g.get("label_train", g["label"])) for g in golds]

        labels_ids, label_attn = self._tokenize_targets(gold_texts)
        if labels_ids.shape[1] == 0:
            # No target tokens → zero loss
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        is_enc_dec = bool(getattr(getattr(self.model, "config", object()), "is_encoder_decoder", False))
        if is_enc_dec:
            out = self.model(**prompt_inputs, labels=labels_ids.masked_fill(label_attn == 0, -100))
            return out.loss

        input_ids = prompt_inputs.get("input_ids")
        attn = prompt_inputs.get("attention_mask")
        if input_ids is None:
            out = self.model(**prompt_inputs, labels=labels_ids.masked_fill(label_attn == 0, -100))
            return out.loss

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
        Returns the full input dict that can be passed to model(**inputs) or editor.edit().
        
        This is similar to get_loss() but returns the inputs dict instead of computing loss.
        Editors can then call model(**inputs) themselves and handle the loss computation.
        
        Args:
            batch: Dict with keys:
                - 'images': List[PIL.Image] or PIL.Image
                - 'prompts': List[str] or str
                - 'golds': List[Dict] where each dict has 'label' key
                
        Returns:
            Dict[str, torch.Tensor]: Model inputs including:
                - All processor outputs (pixel_values, pixel_mask, input_ids, attention_mask, etc.)
                - 'labels': torch.Tensor with -100 masking for prompt positions (decoder-only)
                           or raw label_ids (encoder-decoder)
        """
        images = batch.get("images")
        prompts = batch.get("prompts")
        prompt_inputs = self.encode(images, prompts, tokenize=False)

        # Collect gold answer texts
        golds = batch["golds"]
        # Prefer task-specific training label if provided (e.g., MCI uses "(A) car")
        gold_texts = [str(g.get("label_train", g["label"])) for g in golds]

        # Right-pad the (possibly multi-token) targets so the answer stays contiguous
        # after the prompt, and keep the pad mask so padding is never supervised.
        labels_ids, label_attn = self._tokenize_targets(gold_texts)

        is_enc_dec = bool(getattr(getattr(self.model, "config", object()), "is_encoder_decoder", False))

        if is_enc_dec:
            # Encoder-decoder: labels are separate decoder inputs (mask pad -> -100)
            model_inputs = dict(prompt_inputs)
            model_inputs["labels"] = labels_ids.masked_fill(label_attn == 0, -100)
            return model_inputs

        # Decoder-only: concatenate prompt + labels; mask prompt tokens
        input_ids = prompt_inputs.get("input_ids")
        attn = prompt_inputs.get("attention_mask")

        if input_ids is None:
            # Fallback if no input_ids in prompt_inputs
            model_inputs = dict(prompt_inputs)
            model_inputs["labels"] = labels_ids.masked_fill(label_attn == 0, -100)
            return model_inputs

        full_ids = torch.cat([input_ids, labels_ids], dim=1)
        full_attn = torch.cat([attn, label_attn], dim=1) if attn is not None else None
        
        # Create labels mask: -100 for prompt positions AND answer padding, actual
        # token ids for real answer positions (so multi-token targets are supervised
        # over their full length without ever training on pad tokens).
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

        Returns (input_ids, attention_mask). Each target is encoded
        (add_special_tokens=False) and an EOS token is appended so the supervised
        sequence is ``answer + <eos>``: teacher-forcing then trains the model to
        STOP after the answer. Without it, editing only ever raises the answer
        tokens' probability and never P(eos|answer), so a hard edit collapses free
        generation into repeated tokens (right first token, runaway continuation).

        Right padding keeps the answer tokens contiguous right after the prompt for
        decoder-only concatenation; the attention_mask lets callers mask pad
        positions out of both the loss (-100) and attention. Falls back to no EOS
        for tokenizers that don't define one, and never double-appends."""
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
        on decoded surface form after casefold/strip.

        Raw-id equality is too strict: models differ only in surface form (Gemma
        emits capitalized `'Happy'` id!=`'happy'`; SentencePiece uses a ▁-prefixed
        `' happy'`). Every other metric here normalizes case/whitespace, so TF must
        too, otherwise it reports 0 for a semantically correct rewrite."""
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

        GRACE is the only editor whose forward is position-sensitive (it queries a
        single token position, default -1). During a teacher-forced concat forward
        that -1 falls on a target token instead of the keyed prompt-end, so the
        edit silently doesn't fire. Detected by class name to avoid importing the
        editor here; restores each adaptor's original ``key_id`` afterwards. Yields
        immediately (and restores nothing) when no adaptor is present."""
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
    def teacher_forced_acc(self, image, prompt, target, add_special_tokens=False):
        """Teacher-forced token accuracy: fraction of `target` tokens that are the
        model's argmax prediction given `prompt` (prior-work "rewrite_acc" style).

        Measures whether an edit "took" at the logit level, independent of free
        generation / parsing. Targets are concatenated exactly as in editing
        (add_special_tokens=False), so it mirrors the supervised signal.

        Surface-form robustness: a backbone often answers the *same* label in a
        different case than the (lower-cased) gold — e.g. LLaVA emits "Anger"
        while gold is "anger". Case can change SentencePiece segmentation
        outright ("anger" -> 1 token, "Anger" -> ["An","ger"]), so a position-
        wise token match against the gold tokenization alone reads ~0 even
        though the edit clearly succeeded (acc / rank_acc / f1 stay ~1.0). This
        mainly bites in-context editors (ReasonEdit), which leave the backbone's
        surface form untouched; weight editors trained on the exact gold string
        already reproduce it verbatim. We therefore score a few case variants of
        the target and take the best, crediting the edit whenever the model
        prefers the label in *any* casing. `_tf_token_match` still casefolds each
        token so residual case differences never spuriously zero the metric."""
        if not target:
            return 0.0
        inputs = self.encode(image, prompt, tokenize=False)
        variants = []
        for v in (target, target.lower(), target[:1].upper() + target[1:]):
            if v and v not in variants:
                variants.append(v)
        # Encode once (vision tower is the cost); each variant only appends a
        # handful of target tokens for a cheap extra forward.
        return max(self._tf_acc_one(inputs, v, add_special_tokens) for v in variants)

    @torch.no_grad()
    def _tf_acc_one(self, inputs, target, add_special_tokens=False):
        """Teacher-forced token match for one exact target string.

        `inputs` is the already-encoded prompt (reused across case variants so
        the vision tower runs once). See `teacher_forced_acc` for the rationale."""
        label_ids = self.tokenizer(
            target, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids.to(self.device)
        if label_ids.numel() == 0:
            return 0.0

        is_enc_dec = bool(getattr(getattr(self.model, "config", object()), "is_encoder_decoder", False))
        if is_enc_dec:
            logits = self.model(**inputs, labels=label_ids).logits
            pred = logits.argmax(-1)
            return self._tf_token_match(pred, label_ids)

        # Decoder-only: concat prompt + target; the logits at positions
        # [prompt_len-1 : -1] predict the target tokens.
        input_ids = inputs["input_ids"]
        attn = inputs.get("attention_mask", None)
        model_inputs = dict(inputs)
        model_inputs["input_ids"] = torch.cat([input_ids, label_ids], dim=1)
        if attn is not None:
            model_inputs["attention_mask"] = torch.cat([attn, torch.ones_like(label_ids)], dim=1)
        # Gemma3 (and other multimodal) masks index by token_type_ids; extend with
        # zeros (text) for the appended target so its length matches input_ids.
        tti = inputs.get("token_type_ids", None)
        if tti is not None:
            zeros = torch.zeros(
                label_ids.shape[0], label_ids.shape[1], dtype=tti.dtype, device=tti.device
            )
            model_inputs["token_type_ids"] = torch.cat([tti, zeros], dim=1)
        prompt_len = input_ids.size(1)
        # GRACE routes on the query at a single token position (default key_id=-1).
        # Its key was captured at the PROMPT-END token, which is where generate()'s
        # prefill queries. But here we forward prompt+target concatenated, so -1
        # points at the last *target* token — a position GRACE never keyed, so the
        # edit wouldn't fire and tf_acc would read the un-edited model. Pin any
        # GRACE adaptor to the prompt-end token for this forward so tf_acc matches
        # what generate() sees. No-op for every other editor (no adaptor found).
        with self._grace_query_at(prompt_len - 1):
            logits = self.model(**model_inputs).logits
        pred = logits[:, prompt_len - 1:-1, :].argmax(-1)
        return self._tf_token_match(pred, label_ids)

    @torch.no_grad()
    def continuation_passes(self, image, prompt, label):
        """Binary continuation check: would the model complete ``label`` correctly?

        Conditioned on the first token, checks whether the model's argmax at
        every subsequent position matches the gold continuation tokens.
        Single-token labels trivially pass. Multi-token labels pass iff ALL
        continuation tokens (positions 2:L) are predicted correctly under
        teacher forcing.

        Uses surface-form normalization (casefold/strip) per token, consistent
        with ``_tf_token_match`` / ``teacher_forced_acc``.

        Returns True if the continuation passes, False otherwise.
        """
        if not label:
            return False
        token_ids = self.tokenizer.encode(str(label), add_special_tokens=False)
        if len(token_ids) <= 1:
            return True

        inputs = self.encode(image, prompt, tokenize=False)
        label_tensor = torch.tensor([token_ids], device=self.device)

        is_enc_dec = bool(getattr(
            getattr(self.model, "config", object()),
            "is_encoder_decoder", False))
        if is_enc_dec:
            logits = self.model(**inputs, labels=label_tensor).logits
            preds = logits.argmax(-1)
            for p, g in zip(preds.reshape(-1).tolist()[1:],
                            token_ids[1:]):
                p_txt = self.tokenizer.decode([int(p)]).strip().casefold()
                g_txt = self.tokenizer.decode([int(g)]).strip().casefold()
                if p_txt != g_txt:
                    return False
            return True

        input_ids = inputs["input_ids"]
        attn = inputs.get("attention_mask", None)
        model_inputs = dict(inputs)
        model_inputs["input_ids"] = torch.cat([input_ids, label_tensor], dim=1)
        if attn is not None:
            model_inputs["attention_mask"] = torch.cat(
                [attn, torch.ones_like(label_tensor)], dim=1)
        tti = inputs.get("token_type_ids", None)
        if tti is not None:
            zeros = torch.zeros(
                label_tensor.shape[0], label_tensor.shape[1],
                dtype=tti.dtype, device=tti.device)
            model_inputs["token_type_ids"] = torch.cat([tti, zeros], dim=1)

        prompt_len = input_ids.size(1)
        with self._grace_query_at(prompt_len - 1):
            logits = self.model(**model_inputs).logits

        # logits[:, prompt_len-1, :] predicts the 1st answer token (position 0)
        # logits[:, prompt_len,   :] predicts the 2nd answer token (position 1)  <- check starts here
        # ...
        # logits[:, prompt_len+L-2, :] predicts the L-th answer token
        pred_logits = logits[:, prompt_len:prompt_len + len(token_ids) - 1, :]
        preds = pred_logits.argmax(-1).reshape(-1).tolist()
        gold_continuation = token_ids[1:]  # tokens 2:L

        for p, g in zip(preds, gold_continuation):
            p_txt = self.tokenizer.decode([int(p)]).strip().casefold()
            g_txt = self.tokenizer.decode([int(g)]).strip().casefold()
            if p_txt != g_txt:
                return False
        return True

    @torch.no_grad()
    def get_next_token_probs(self, image, prompt, candidate_labels):
        """First-token prob per candidate from a single forward pass (not renormalized)."""
        inputs = self.encode(image, prompt, tokenize=False)
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :][0]  # [vocab_size]
        return self._first_token_probs(logits, candidate_labels, renormalize=False)
