"""Self-contained readout utilities for SoftEdit.

Computes answer-level probability readout and distributional metrics
(js_dist, ndcg_readout, tau_readout) independently of the eval pipeline
(no imports from editeval or taskengineer).

Four levels of use:

1. **From live logits** (during generation):
       ``first_token_probs``, ``open_readout``, ``build_readout_raw``

2. **Continuation gate** (validate multi-token answers):
       ``apply_continuation_gate`` — gates first-token probabilities by
       checking whether the model would complete each human answer correctly
       under teacher forcing.  Single-token answers pass trivially.

3. **From a saved readout_raw record** (offline reconstruction, no model):
       ``reconstruct_open_readout``

4. **Scoring** (given any readout distribution + human labels):
       ``score_readout``

Verification:
    ``verify_pred_scores`` — recompute readout scores from a pred dict and
    compare to pre-computed values; returns per-metric deltas.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from revlm.metrics.ranked_metrics import (
    build_labels_true,
    counts_on_token_keys,
    js_distance,
    ndcg_readout as _ndcg_readout,
    tau_readout as _tau_readout,
    norm,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ReadoutResult:
    """Complete readout output from a single example."""
    support_prob: Dict[str, float]
    support_prob_open: Optional[Dict[str, float]]
    label2key: Optional[Dict[str, str]]
    readout_raw: Optional[dict]
    scores: Dict[str, Optional[float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. Live-logit readout (mirrors VQAModel methods, standalone)
# ---------------------------------------------------------------------------

def first_token_probs(logits: torch.Tensor, candidate_labels: List[str],
                      tokenizer, *, renormalize: bool = False) -> Dict[str, float]:
    """Map a 1-D logits vector -> {label: prob} via each label's first token.

    Standalone equivalent of ``VQAModel._first_token_probs``.
    """
    probs = torch.softmax(logits.float(), dim=-1)
    out: Dict[str, float] = {}
    for label in candidate_labels:
        token_ids = tokenizer.encode(str(label), add_special_tokens=False)
        out[label] = float(probs[token_ids[0]].item()) if token_ids else 0.0
    if renormalize:
        z = sum(out.values())
        if z > 0:
            out = {k: v / z for k, v in out.items()}
    return out


def open_readout(logits: torch.Tensor, human_labels: List[str],
                 tokenizer) -> Tuple[Optional[Dict[str, float]],
                                     Optional[Dict[str, str]]]:
    """Open-vocabulary readout: (model top-k union human first tokens) + OTHER.

    Standalone equivalent of ``VQAModel.open_readout_probs``.

    Returns (probs_dict, label2key) or (None, None) when no label tokenizes.
    """
    probs = torch.softmax(logits.float(), dim=-1)

    label2id: Dict[str, int] = {}
    for label in human_labels:
        token_ids = tokenizer.encode(str(label), add_special_tokens=False)
        if token_ids:
            label2id[str(label)] = int(token_ids[0])
    if not label2id:
        return None, None

    human_ids = set(label2id.values())
    k = len(human_ids)
    topk_ids = {int(t) for t in probs.topk(k).indices.tolist()}
    support_ids = human_ids | topk_ids

    id2key: Dict[int, str] = {}
    out: Dict[str, float] = {}
    for tid in support_ids:
        key = tokenizer.decode([tid]).strip().lower() or f"tok_{tid}"
        id2key[tid] = key
        out[key] = out.get(key, 0.0) + float(probs[tid].item())
    out["__OTHER__"] = max(1.0 - sum(out.values()), 0.0)
    label2key = {label: id2key[tid] for label, tid in label2id.items()}
    return out, label2key


def build_readout_raw(logits: torch.Tensor, labels: List[str],
                      tokenizer, *, top_m: int = 20) -> dict:
    """Build a JSON-safe raw readout record from a 1-D logits vector.

    Standalone equivalent of ``VQAModel.build_readout_raw``.
    """
    logits = logits.float()
    logZ = float(torch.logsumexp(logits, dim=-1).item())
    topk = logits.topk(min(top_m, logits.shape[-1]))
    top_ids = topk.indices.tolist()
    top_logits = topk.values.tolist()
    top_tokens = [tokenizer.decode([tid]).strip().lower() or f"tok_{tid}"
                  for tid in top_ids]
    label_ids: Dict[str, int] = {}
    label_logits: Dict[str, float] = {}
    label_tokens: Dict[str, str] = {}
    for label in labels:
        tids = tokenizer.encode(str(label), add_special_tokens=False)
        if tids:
            tid = tids[0]
            label_ids[label] = tid
            label_logits[label] = float(logits[tid].item())
            label_tokens[label] = (tokenizer.decode([tid]).strip().lower()
                                   or f"tok_{tid}")
    return {
        "top_ids": top_ids, "top_logits": top_logits, "top_tokens": top_tokens,
        "label_ids": label_ids, "label_logits": label_logits,
        "label_tokens": label_tokens, "logZ": logZ,
    }


# ---------------------------------------------------------------------------
# 2. Continuation gate
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Output of ``apply_continuation_gate``."""
    support_prob_open: Dict[str, float]
    label2key: Dict[str, str]
    gate_log: List[dict]

    # Per-key mapping: which full answer (if any) claimed this first-token key.
    # None when the key was not matched or failed all continuations.
    key2answer: Dict[str, Optional[str]] = field(default_factory=dict)


def _is_multitoken(label: str, tokenizer) -> bool:
    tids = tokenizer.encode(str(label), add_special_tokens=False)
    return len(tids) > 1


def apply_continuation_gate(
    model,
    image,
    prompt: str,
    support_prob_open: Dict[str, float],
    human_labels: List[str],
    label2key: Dict[str, str],
    tokenizer=None,
) -> GateResult:
    """Apply the binary continuation gate to an open readout distribution.

    For each first-token key in ``support_prob_open`` that corresponds to a
    human answer's first token:

    - If the human answer is single-token: passes trivially (no check).
    - If multi-token: runs ``model.continuation_passes(image, prompt, label)``
      to verify the model would complete the full answer under teacher forcing.

    When multiple human answers share the same first-token key (collision),
    each is checked in turn.  **First-pass-wins**: the first answer whose
    continuation passes claims the probability; all others get 0.  If none
    pass, the probability is zeroed (mass flows to ``__OTHER__``).

    Args:
        model: VQAModel instance (needs ``.continuation_passes``).
        image: PIL image or path.
        prompt: the engineered prompt string.
        support_prob_open: the ungated open readout ``{key: prob}``.
        human_labels: list of human answer strings.
        label2key: ``{human_label: first_token_key}`` from ``open_readout``.
        tokenizer: optional; falls back to ``model.tokenizer``.
    Returns:
        GateResult with the gated distribution, label2key, and per-label log.
    """
    if tokenizer is None:
        tokenizer = model.tokenizer

    # Group human labels by their first-token key.
    key2labels: Dict[str, List[str]] = {}
    for lbl, key in label2key.items():
        key2labels.setdefault(key, []).append(lbl)

    gated = dict(support_prob_open)
    gate_log: List[dict] = []
    key2answer: Dict[str, Optional[str]] = {}

    for key, labels_for_key in key2labels.items():
        if key not in gated or key == "__OTHER__":
            continue

        # Check if ANY label for this key is multi-token.
        any_multi = any(_is_multitoken(lbl, tokenizer) for lbl in labels_for_key)
        if not any_multi:
            # All labels sharing this key are single-token: no gate needed.
            for lbl in labels_for_key:
                gate_log.append({"label": lbl, "key": key, "tokens": 1,
                                 "gate": "skip", "passed": True})
            if labels_for_key:
                key2answer[key] = labels_for_key[0]
            continue

        # At least one multi-token label: run continuation checks.
        winner = None
        for lbl in labels_for_key:
            n_tokens = len(tokenizer.encode(str(lbl), add_special_tokens=False))
            if n_tokens <= 1:
                # Single-token label in a collision group: passes trivially.
                entry = {"label": lbl, "key": key, "tokens": n_tokens,
                         "gate": "trivial", "passed": True}
                gate_log.append(entry)
                if winner is None:
                    winner = lbl
                continue

            passed = model.continuation_passes(image, prompt, lbl)
            entry = {"label": lbl, "key": key, "tokens": n_tokens,
                     "gate": "checked", "passed": passed}
            gate_log.append(entry)
            if passed and winner is None:
                winner = lbl

        if winner is None:
            # No label passed: zero this key's probability.
            gated[key] = 0.0
            key2answer[key] = None
        else:
            key2answer[key] = winner

    # Recompute __OTHER__ from the residual.
    non_other = sum(v for k, v in gated.items() if k != "__OTHER__")
    gated["__OTHER__"] = max(1.0 - non_other, 0.0)

    return GateResult(
        support_prob_open=gated,
        label2key=label2key,
        gate_log=gate_log,
        key2answer=key2answer,
    )


# ---------------------------------------------------------------------------
# 3. Offline reconstruction from readout_raw (no model / tokenizer)
# ---------------------------------------------------------------------------

def reconstruct_open_readout(
    raw: dict,
    human_labels: Optional[List[str]] = None,
    k: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Rebuild the open readout from a saved ``readout_raw`` record.

    Args:
        raw: readout_raw dict (see ``build_readout_raw`` / rank_metrics.md).
        human_labels: if given, the human labels whose first-token keys are
            unioned with the model top-k.  Falls back to ``raw["label_ids"]``
            keys.
        k: number of model top-k tokens.  Defaults to the number of distinct
            human-label token ids.
    Returns:
        (probs_dict, label2key) — same format as ``open_readout``.
    """
    logZ = raw["logZ"]
    top_ids = raw["top_ids"]
    top_logits = raw["top_logits"]
    top_tokens = raw["top_tokens"]
    label_ids = raw.get("label_ids", {})
    label_logits = raw.get("label_logits", {})
    label_tokens = raw.get("label_tokens", {})

    if human_labels is None:
        human_labels = list(label_ids.keys())

    human_id_set = set()
    label2id: Dict[str, int] = {}
    for lbl in human_labels:
        tid = label_ids.get(lbl)
        if tid is not None:
            human_id_set.add(tid)
            label2id[lbl] = tid

    if k is None:
        k = max(len(human_id_set), 1)
    model_top_ids = set(top_ids[:k])
    support_ids = human_id_set | model_top_ids

    id2logit: Dict[int, float] = {}
    id2token: Dict[int, str] = {}
    for tid, lg, tok in zip(top_ids, top_logits, top_tokens):
        id2logit[tid] = lg
        id2token[tid] = tok
    for lbl in human_labels:
        tid = label_ids.get(lbl)
        lg = label_logits.get(lbl)
        tok = label_tokens.get(lbl)
        if tid is not None and lg is not None:
            id2logit.setdefault(tid, lg)
            id2token.setdefault(tid, tok or f"tok_{tid}")

    out: Dict[str, float] = {}
    id2key: Dict[int, str] = {}
    for tid in support_ids:
        key = id2token.get(tid, f"tok_{tid}")
        id2key[tid] = key
        prob = math.exp(id2logit.get(tid, -100.0) - logZ)
        out[key] = out.get(key, 0.0) + prob
    out["__OTHER__"] = max(1.0 - sum(out.values()), 0.0)

    label2key = {lbl: id2key[tid] for lbl, tid in label2id.items()}
    return out, label2key


# ---------------------------------------------------------------------------
# 3. Scoring (pure functions, given any readout distribution + human labels)
# ---------------------------------------------------------------------------

def score_readout(
    support_prob_open: Optional[Dict[str, float]],
    label2key: Optional[Dict[str, str]],
    answer_counts: Dict[str, float],
    labels_true=None,
) -> Dict[str, Optional[float]]:
    """Compute js_dist, ndcg_readout, tau_readout from a readout distribution.

    Exactly replicates the scoring logic inside
    ``QATaskEngineer.eval`` / ``PostEditQATaskEngineer.eval``, extracted so
    SoftEdit (and tests) can call it without the full eval pipeline.

    Args:
        support_prob_open: {token_key: prob, "__OTHER__": residual}.
        label2key: {human_label: token_key} mapping.
        answer_counts: {label: count} from the human annotations.
        labels_true: pre-built LabelsTrue, or auto-built from answer_counts.
    Returns:
        {"js_dist": float|None, "ndcg_readout": float|None,
         "tau_readout": float|None}
    """
    if labels_true is None:
        labels_true = build_labels_true(answer_counts) if answer_counts else []

    scores: Dict[str, Optional[float]] = {
        "js_dist": None,
        "ndcg_readout": None,
        "tau_readout": None,
    }

    if not support_prob_open:
        return scores

    if answer_counts:
        target = counts_on_token_keys(answer_counts, label2key)
        scores["js_dist"] = js_distance(target, support_prob_open)

    if labels_true:
        scores["ndcg_readout"] = _ndcg_readout(
            support_prob_open, labels_true, label2key)
        scores["tau_readout"] = _tau_readout(
            support_prob_open, labels_true, label2key)

    return scores


# ---------------------------------------------------------------------------
# 4. Full pipeline helper (model + inputs → ReadoutResult)
# ---------------------------------------------------------------------------

def compute_readout(
    model,
    image,
    prompt: str,
    candidate_labels: List[str],
    human_labels: List[str],
    answer_counts: Dict[str, float],
    *,
    continuation_gate: bool = False,
) -> ReadoutResult:
    """One forward pass -> readout distributions + scores.

    ``model`` is expected to be a ``VQAModel`` (or any object with
    ``.encode``, ``.model``, ``.tokenizer``).

    When ``continuation_gate=True``, the gated readout and gated scores are
    also computed and stored in ``result.scores_gated`` / ``result.gate``.
    """
    inputs = model.encode(image, prompt, tokenize=False)
    with torch.no_grad():
        outputs = model.model(**inputs)
        logits_1d = (outputs.logits if hasattr(outputs, "logits")
                     else outputs)[:, -1, :][0]

    sp = first_token_probs(logits_1d, candidate_labels, model.tokenizer,
                           renormalize=True)
    spo, l2k = open_readout(logits_1d, human_labels, model.tokenizer)

    all_labels = list(dict.fromkeys(list(candidate_labels) + list(human_labels)))
    raw = build_readout_raw(logits_1d, all_labels, model.tokenizer)

    scores = score_readout(spo, l2k, answer_counts)

    result = ReadoutResult(
        support_prob=sp,
        support_prob_open=spo,
        label2key=l2k,
        readout_raw=raw,
        scores=scores,
    )

    if continuation_gate and spo is not None and l2k is not None:
        gate = apply_continuation_gate(
            model, image, prompt, spo, human_labels, l2k)
        result.scores_gated = score_readout(
            gate.support_prob_open, gate.label2key, answer_counts)
        result.gate = gate

    return result


# ---------------------------------------------------------------------------
# 5. Verification: compare against eval-pipeline results
# ---------------------------------------------------------------------------

def verify_pred_scores(
    pred: dict,
    answer_counts: Dict[str, float],
    *,
    atol: float = 1e-6,
) -> Dict[str, Optional[float]]:
    """Recompute readout scores from a pred dict and return per-metric deltas.

    Useful for verifying that the standalone scoring exactly matches what the
    eval pipeline stored.  A delta of 0.0 means perfect agreement.

    Args:
        pred: a single example's ``pred`` dict (as saved in prediction dumps).
        answer_counts: the example's human vote counts.
        atol: absolute tolerance below which a delta is considered zero.
    Returns:
        {"js_dist": delta|None, "ndcg_readout": delta|None,
         "tau_readout": delta|None, "match": bool}
        where delta = |recomputed - stored| (None if either side is None).
    """
    spo = pred.get("support_prob_open")
    l2k = pred.get("open_label2key")

    recomputed = score_readout(spo, l2k, answer_counts)

    deltas: Dict[str, Optional[float]] = {}
    all_match = True
    for key in ("js_dist", "ndcg_readout", "tau_readout"):
        stored = pred.get(key)
        new = recomputed.get(key)
        if stored is None and new is None:
            deltas[key] = None
        elif stored is None or new is None:
            deltas[key] = float("inf")
            all_match = False
        else:
            d = abs(float(new) - float(stored))
            deltas[key] = d
            if d > atol:
                all_match = False
    deltas["match"] = all_match
    return deltas


def verify_from_raw(
    pred: dict,
    answer_counts: Dict[str, float],
    human_labels: Optional[List[str]] = None,
    *,
    atol: float = 1e-6,
) -> Dict[str, Optional[float]]:
    """Reconstruct the open readout from ``readout_raw`` and verify scores.

    Deeper check than ``verify_pred_scores``: recomputes the readout from raw
    logits (no model needed) and compares to the stored open readout.
    """
    raw = pred.get("readout_raw")
    if raw is None:
        return {"js_dist": None, "ndcg_readout": None, "tau_readout": None,
                "match": True, "note": "no readout_raw (editor override)"}

    spo_recon, l2k_recon = reconstruct_open_readout(raw, human_labels)
    scores_recon = score_readout(spo_recon, l2k_recon, answer_counts)

    spo_stored = pred.get("support_prob_open", {})
    scores_stored = score_readout(spo_stored, pred.get("open_label2key"),
                                  answer_counts)

    deltas: Dict[str, Optional[float]] = {}
    all_match = True
    for key in ("js_dist", "ndcg_readout", "tau_readout"):
        s = scores_stored.get(key)
        r = scores_recon.get(key)
        if s is None and r is None:
            deltas[key] = None
        elif s is None or r is None:
            deltas[key] = float("inf")
            all_match = False
        else:
            d = abs(float(r) - float(s))
            deltas[key] = d
            if d > atol:
                all_match = False
    deltas["match"] = all_match
    return deltas
