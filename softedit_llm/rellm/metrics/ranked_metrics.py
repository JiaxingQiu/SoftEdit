"""Metrics for ranked multilabel prediction with tie-aware ground truth.

Ground truth format (LabelsTrue):
    A list of tiers, each tier a list of (label, optional_score) tuples.
    Labels within the same tier are tied (unordered).
    Tiers are ordered from most relevant (index 0) to least.

    Example with scores (annotator counts):
        [[("umbrella", 38)], [("sunscreen", 36)], [("sun hat", 14)]]

    Example with ties:
        [[("bed", 45)], [("shower", 2), ("tv", 2)]]

    Example without scores (ranking only):
        [[("umbrella", None)], [("sunscreen", None), ("sun hat", None)]]

NOTE (LLM port): identical to the rankedit VLM `ranked_metrics.py` EXCEPT the
free-text label parser, which is generalized from *first word* to *first
phrase segment* so ProtoQA's multi-word answers ("sun hat", "crying baby") are
matched as whole labels. The scoring functions and the first-token distribution
metrics are byte-for-byte the VLM procedure.
"""
import math
import re
from typing import Dict, List, Optional, Tuple

LabelsTrue = List[List[Tuple[str, Optional[float]]]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm(s: str) -> str:
    return str(s).strip().lower()


_LABEL_SEP_RE = re.compile(r'\s*(?:\bor\b|\band\b|/|,|;|&)\s*', re.IGNORECASE)


def labels_match(a: str, b: str) -> bool:
    """Connector-aware match after normalization.

    ProtoQA gold labels often use "/" as a synonym separator (e.g.
    "talking/arguing", "alarm clock/alarm"). The prediction parser
    (`parse_single_answer`) splits on connectors and keeps only the first
    segment, so a prediction of "talking" must match the gold "talking/arguing".
    We succeed when the normalized prediction equals any connector-split segment
    of the gold label, or vice versa.
    """
    na, nb = norm(a), norm(b)
    if na == nb:
        return True
    for seg in _LABEL_SEP_RE.split(nb):
        seg = seg.strip()
        if seg and na == seg:
            return True
    for seg in _LABEL_SEP_RE.split(na):
        seg = seg.strip()
        if seg and nb == seg:
            return True
    return False


def has_scores(labels_true: LabelsTrue) -> bool:
    """True if every label in labels_true has a non-None score."""
    for tier in labels_true:
        for _, score in tier:
            if score is None:
                return False
    return True


def flatten_labels(labels_true: LabelsTrue) -> List[str]:
    """Flatten tiers into a single ordered list of normalized labels."""
    return [norm(label) for tier in labels_true for label, _ in tier]


def build_labels_true(answer_counts: Dict[str, int]) -> LabelsTrue:
    """Build LabelsTrue from answer_counts dict (scenario 2: scores available).

    Groups labels by count (descending), each group is a tier.
    Only includes labels with count > 0.
    """
    positive = {label: count for label, count in answer_counts.items() if count > 0}
    if not positive:
        return []
    by_count: Dict[int, List[str]] = {}
    for label, count in positive.items():
        by_count.setdefault(count, []).append(label)
    tiers = []
    for count in sorted(by_count.keys(), reverse=True):
        tier = [(label, float(count)) for label in sorted(by_count[count])]
        tiers.append(tier)
    return tiers


def align_probs(dist: Optional[Dict[str, float]], labels: List[str]) -> List[float]:
    """Project a {label: value} dict onto an ordered `labels` list, renormalized.

    Missing labels get 0. Used by the per-sample debug log so gold / read / pred
    distributions are all shown over the SAME column order (labels_true).
    Returns [] if `dist` is empty/degenerate.
    """
    if not dist or not labels:
        return []
    d = {norm(k): max(float(v), 0.0) for k, v in dist.items()}
    vals = [d.get(norm(l), 0.0) for l in labels]
    s = sum(vals)
    return [v / s for v in vals] if s > 0 else vals


# ---------------------------------------------------------------------------
# Single-QA label parsing (the ONE strategy switch)
# ---------------------------------------------------------------------------
# The single-QA predicted label is a *derived* field. Predicted JSONs always
# persist the raw model `answer` (and, when scored, `support_prob`), so the label
# can be (re)parsed from the raw output under any strategy at any time. This is
# the only place the strategy is interpreted; the edit/locality split and every
# accuracy metric simply read the already-parsed `pred["label"]` (text-to-text).
#
#   strategy="text": normalized free-text answer.
#   strategy="prob": argmax over the support distribution (a guaranteed-valid
#                    support label); falls back to text when support_prob absent.
LABEL_STRATEGIES = ("text", "prob")


def parse_qa_label(answer: str, support_prob: Optional[Dict[str, float]] = None,
                   strategy: str = "text") -> str:
    """Resolve a single-QA generation into one normalized label string.

    Single source of truth for the single-QA label, so ad-hoc callers (notebooks via
    ``qa_pred_label``) never diverge from the pipeline:
      - ``strategy="text"``: delegates to ``parse_single_answer`` — the exact parse the
        pipeline stores in ``pred["label"]`` and uses for the edit/locality split.
      - ``strategy="prob"``: argmax over the support distribution (guaranteed-valid).
    """
    if strategy == "prob" and support_prob:
        return norm(max(support_prob.items(), key=lambda kv: kv[1])[0])
    return parse_single_answer(answer or "")


def qa_pred_label(ex: dict, strategy: str = "text") -> str:
    """Convenience reader: single-QA label from an example's stored raw prediction.

    The pipeline reads `pred["label"]` directly; this helper is for ad-hoc use
    (e.g. notebooks) where only the raw record is available.
    """
    pred = ex.get("pred") or ex.get("pred_qa") or {}
    return parse_qa_label(pred.get("answer", pred.get("label", "")), pred.get("support_prob"), strategy)


# ---------------------------------------------------------------------------
# Distributional metric (predicted support distribution vs human votes)
# ---------------------------------------------------------------------------

def counts_on_token_keys(answer_counts: Dict[str, float],
                         label2key: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    """Project human vote counts onto open-readout token keys.

    The open readout dict is keyed by decoded FIRST-TOKEN strings (e.g. Qwen:
    "happiness" -> "h", "surprise" -> "sur"), so the human side must be mapped
    into the same key space before js_distance — otherwise the supports look
    disjoint and JS saturates even for a perfect model. Labels sharing a first
    token merge by summing counts (see rank_metrics.md). Identity fallback for
    labels missing from ``label2key`` (e.g. untokenizable).
    """
    if not label2key:
        return answer_counts
    out: Dict[str, float] = {}
    for lbl, c in (answer_counts or {}).items():
        key = label2key.get(lbl) or label2key.get(norm(lbl)) or lbl
        out[key] = out.get(key, 0.0) + float(c)
    return out


def js_distance(answer_counts: Dict[str, float], pred_probs: Dict[str, float]) -> float:
    """Jensen-Shannon distance between the human vote distribution and the
    model's predicted support distribution.

    Args:
        answer_counts: {label: vote_count}  (human target; normalized here)
        pred_probs:    {label: prob}         (model support_prob; normalized here)

    Both are aligned over the union of labels. Uses JS divergence with log base
    2, so the divergence is in [0, 1] and its square root (the JS *distance*, a
    true metric) is also in [0, 1]. 0 = identical, 1 = disjoint support.
    Returns 1.0 if either distribution is empty/degenerate.
    """
    if not answer_counts or not pred_probs:
        return 1.0
    p = {norm(k): float(v) for k, v in answer_counts.items()}
    q = {norm(k): float(v) for k, v in pred_probs.items()}
    labels = set(p) | set(q)
    P = [max(p.get(l, 0.0), 0.0) for l in labels]
    Q = [max(q.get(l, 0.0), 0.0) for l in labels]
    sp, sq = sum(P), sum(Q)
    if sp <= 0 or sq <= 0:
        return 1.0
    P = [x / sp for x in P]
    Q = [x / sq for x in Q]
    jsd = 0.0
    for pi, qi in zip(P, Q):
        mi = 0.5 * (pi + qi)
        if pi > 0:
            jsd += 0.5 * pi * math.log2(pi / mi)
        if qi > 0:
            jsd += 0.5 * qi * math.log2(qi / mi)
    jsd = min(max(jsd, 0.0), 1.0)
    return math.sqrt(jsd)


# ---------------------------------------------------------------------------
# Single-label accuracy metrics
# ---------------------------------------------------------------------------

def soft_acc(pred: str, labels_true: LabelsTrue) -> float:
    """VQA-style soft accuracy using annotator scores.

    Formula:  min(count(pred) / k, 1.0)
    where     k = ceil(0.3 * N),  N = sum of all scores.

    Requires scores in labels_true.
    Returns 0.0 if pred not found or no scores available.
    """
    if not pred or not labels_true:
        return 0.0
    pred_n = norm(pred)
    total = 0.0
    pred_score = 0.0
    for tier in labels_true:
        for label, score in tier:
            if score is None:
                return 0.0
            total += score
            if labels_match(pred_n, label):
                pred_score = score
    if total == 0:
        return 0.0
    k = max(math.ceil(0.3 * total), 1)
    return min(pred_score / k, 1.0)


def rank_acc(pred: str, labels_true: LabelsTrue) -> float:
    """Tier-position accuracy (no scores needed).

    Returns (n_tiers - tier_index) / n_tiers if pred found, else 0.
    Top tier → 1.0, second tier → (n-1)/n, ..., not found → 0.0.
    """
    if not pred or not labels_true:
        return 0.0
    pred_n = norm(pred)
    n_tiers = len(labels_true)
    for tier_idx, tier in enumerate(labels_true):
        for label, _ in tier:
            if labels_match(pred_n, label):
                return (n_tiers - tier_idx) / n_tiers
    return 0.0


def simple_acc(pred: str, labels_true: LabelsTrue) -> float:
    """Binary hit accuracy: 1.0 if pred matches ANY acceptable label, else 0.0.

    The most lenient accuracy -- ignores tier/rank position and annotator counts.
    Equivalent to (rank_acc > 0). On single-label samples it coincides with
    soft_acc / rank_acc; on multilabel samples it is >= both.
    """
    if not pred or not labels_true:
        return 0.0
    pred_n = norm(pred)
    for tier in labels_true:
        for label, _ in tier:
            if labels_match(pred_n, label):
                return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Ranked-list metrics
# ---------------------------------------------------------------------------

def acc_at_1(pred_ranked: List[str], labels_true: LabelsTrue) -> float:
    """Accuracy of the top-1 prediction.

    Uses soft_acc if scores available, else rank_acc.
    """
    if not pred_ranked or not labels_true:
        return 0.0
    if has_scores(labels_true):
        return soft_acc(pred_ranked[0], labels_true)
    return rank_acc(pred_ranked[0], labels_true)


def set_f1(pred_ranked: List[str], labels_true: LabelsTrue) -> float:
    """F1 treating prediction and ground truth as unordered sets."""
    if not labels_true:
        return 1.0 if not pred_ranked else 0.0
    if not pred_ranked:
        return 0.0
    gold_set = set(flatten_labels(labels_true))
    pred_set = {norm(x) for x in pred_ranked}
    tp = sum(1 for p in pred_set if any(labels_match(p, g) for g in gold_set))
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    return 2 * precision * recall / (precision + recall)


GAIN_MODES = ("linear", "log", "binary")


def _tier_relevance(n_tiers: int, tier_idx: int, gain: str = "linear") -> float:
    """Relevance grade for a tier, controlled by ``gain``.

    ``linear``  (default): n_tiers - tier_idx  (original behaviour).
    ``log``:     1 + log2(n_tiers - tier_idx)   (compressed; still ordered).
    ``binary``:  1.0 for any gold label         (ignores tier ordering).
    """
    if gain == "binary":
        return 1.0
    raw = n_tiers - tier_idx  # always >= 1
    if gain == "log":
        return 1.0 + math.log2(raw)
    return float(raw)


def _dcg(relevances: List[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(pred_ranked: List[str], labels_true: LabelsTrue,
              k: Optional[int] = None, gain: str = "log") -> float:
    """nDCG@K with tie-aware relevance.

    Labels in the same tier receive the same relevance value.
    K defaults to total number of gold labels.

    ``gain`` controls how tier position maps to relevance (see ``_tier_relevance``):
      - ``"linear"`` (default): n_tiers − tier_idx.
      - ``"log"``:   1 + log2(n_tiers − tier_idx)  — softens first-label dominance.
      - ``"binary"``: 1.0 for every gold label — ignores tier ordering entirely.
    """
    if not labels_true:
        return 1.0 if not pred_ranked else 0.0
    if not pred_ranked:
        return 0.0
    n_tiers = len(labels_true)
    n_gold = sum(len(tier) for tier in labels_true)
    if k is None:
        k = n_gold

    gold_relevance: Dict[str, float] = {}
    for tier_idx, tier in enumerate(labels_true):
        rel = _tier_relevance(n_tiers, tier_idx, gain)
        for label, _ in tier:
            gold_relevance[norm(label)] = rel

    def _get_relevance(p: str) -> float:
        for g, rel in gold_relevance.items():
            if labels_match(p, g):
                return rel
        return 0.0

    pred_rels = [_get_relevance(p) for p in pred_ranked[:k]]
    ideal_rels = sorted(gold_relevance.values(), reverse=True)[:k]

    dcg = _dcg(pred_rels)
    idcg = _dcg(ideal_rels)
    return dcg / idcg if idcg > 0 else 0.0


def ndcg_readout(pred_probs_open: Dict[str, float], labels_true: LabelsTrue,
                 label2key: Optional[Dict[str, str]] = None,
                 k: Optional[int] = None,
                 gain: str = "log") -> float:
    """nDCG of the model's own open-readout ranking vs the human ranked labels.

    The predicted list is the open-vocabulary readout support (model top-k ∪
    human first tokens; see rank_metrics.md) ordered by probability, with the
    reserved "__OTHER__" bucket dropped. Probabilities are used ONLY for ordering —
    no distribution scores enter the metric. The gold tiers are converted into
    the same first-token key space via ``label2key`` (identity when absent);
    gold labels colliding on one key keep the higher (better) tier. Scoring is
    delegated to ``ndcg_at_k`` so tie handling and tier relevance stay
    identical to the text-generation metric.

    ``gain`` is forwarded to ``ndcg_at_k``; use ``"log"`` or ``"binary"`` to
    reduce first-label dominance (see ``_tier_relevance``).
    """
    if not labels_true:
        return 1.0 if not pred_probs_open else 0.0
    if not pred_probs_open:
        return 0.0

    l2k = {norm(l): norm(v) for l, v in (label2key or {}).items()}
    seen = set()
    key_tiers: LabelsTrue = []
    for tier in labels_true:
        kt = []
        for label, score in tier:
            key = l2k.get(norm(label), norm(label))
            if key in seen:
                continue  # first-token collision: keep the higher tier
            seen.add(key)
            kt.append((key, score))
        if kt:
            key_tiers.append(kt)
    if not key_tiers:
        return 0.0

    ranked = [key for key, _ in sorted(
        ((norm(kk), float(v)) for kk, v in pred_probs_open.items() if kk != "__OTHER__"),
        key=lambda kv: kv[1], reverse=True)]
    return ndcg_at_k(ranked, key_tiers, k=k, gain=gain)


# ---------------------------------------------------------------------------
# Kendall's tau (pairwise rank concordance)
# ---------------------------------------------------------------------------

TAU_MODES = ("union", "gold")


def _tiers_to_rank(tiers: LabelsTrue) -> Dict[str, int]:
    """Map every label in a tier-list to its tier index (lower = better)."""
    out: Dict[str, int] = {}
    for tier_idx, tier in enumerate(tiers):
        for label, _ in tier:
            key = norm(label)
            if key not in out:
                out[key] = tier_idx
    return out


def _flatten_tiers(tiers: LabelsTrue) -> List[str]:
    """Flatten tiers into a single list preserving tier order."""
    return [norm(label) for tier in tiers for label, _ in tier]


def _probs_to_tiers(probs: Dict[str, float]) -> LabelsTrue:
    """Group {label: prob} into descending tiers of equal probability."""
    by_prob: Dict[float, List[str]] = {}
    for key, prob in probs.items():
        p = float(prob)
        by_prob.setdefault(p, []).append(norm(key))
    return [[(label, prob) for label in sorted(labels)]
            for prob, labels in sorted(by_prob.items(), reverse=True)]


def kendall_tau(pred_tiers: LabelsTrue, gold_tiers: LabelsTrue,
                mode: str = "union") -> float:
    """Pairwise rank concordance between two tier-lists.

    Both ``pred_tiers`` and ``gold_tiers`` use the same LabelsTrue format:
    a list of tiers (best first), each tier a list of ``(label, score)``
    tuples. Labels within a tier are tied (unordered).

    For every pair of items the concordance rule is:

      - **Both sides tie** → concordant.
      - **One side ties, the other is strict** → discordant (the tying side
        failed to distinguish).
      - **Both sides strict, same direction** → concordant.
      - **Both sides strict, opposite direction** → discordant.

    Returns the fraction of concordant pairs, in [0, 1].

    ``mode`` controls which item pairs are compared:

      ``"union"``:  Items = gold labels ∪ predicted labels.  Missing items get
                    a worst-rank tier on the side where they are absent.
                    Penalises both missing gold labels and non-gold predictions
                    in top positions.

      ``"gold"``:   Items = gold labels only.  Gold labels missing from the
                    predicted list get worst predicted tier.  Non-gold predicted
                    labels are ignored.
    """
    if not gold_tiers:
        return 1.0 if not pred_tiers else 0.0
    if not pred_tiers:
        return 0.0

    gold_rank = _tiers_to_rank(gold_tiers)
    pred_rank = _tiers_to_rank(pred_tiers)

    gold_flat = _flatten_tiers(gold_tiers)
    pred_flat = _flatten_tiers(pred_tiers)

    if mode == "union":
        items = list(dict.fromkeys(gold_flat + pred_flat))
    else:
        items = list(dict.fromkeys(gold_flat))

    worst_gold = len(gold_tiers)
    worst_pred = len(pred_tiers)

    concordant = 0
    total = 0
    for i in range(len(items)):
        gi = gold_rank.get(items[i], worst_gold)
        pi = pred_rank.get(items[i], worst_pred)
        for j in range(i + 1, len(items)):
            gj = gold_rank.get(items[j], worst_gold)
            pj = pred_rank.get(items[j], worst_pred)
            g_cmp = (gi > gj) - (gi < gj)   # -1 / 0 / +1
            p_cmp = (pi > pj) - (pi < pj)
            if g_cmp == 0 and p_cmp == 0:
                concordant += 1              # both tied
            elif g_cmp == 0 or p_cmp == 0:
                pass                         # one ties, other strict → discordant
            elif g_cmp == p_cmp:
                concordant += 1              # same strict direction
            total += 1

    return concordant / total if total > 0 else 1.0


def tau_readout(pred_probs_open: Dict[str, float], labels_true: LabelsTrue,
                label2key: Optional[Dict[str, str]] = None,
                mode: str = "union") -> float:
    """Kendall's tau of the model's open-readout ranking vs human ranked labels.

    Analogous to ``ndcg_readout`` but uses pairwise concordance instead of DCG.
    All label pairs are weighted equally — no label or position is privileged.

    The predicted probabilities are grouped into tiers of equal probability so
    that predicted ties are represented faithfully (same format as gold tiers).
    ``__OTHER__`` is dropped. Gold tiers are converted to first-token key space
    via ``label2key``. Scoring is delegated to ``kendall_tau``.

    ``mode`` is forwarded to ``kendall_tau``; ``"union"`` (default) penalises
    non-gold tokens occupying top positions (analogous to JS penalising mass
    on non-gold tokens).
    """
    if not labels_true:
        return 1.0 if not pred_probs_open else 0.0
    if not pred_probs_open:
        return 0.0

    l2k = {norm(l): norm(v) for l, v in (label2key or {}).items()}
    seen = set()
    key_tiers: LabelsTrue = []
    for tier in labels_true:
        kt = []
        for label, score in tier:
            key = l2k.get(norm(label), norm(label))
            if key in seen:
                continue
            seen.add(key)
            kt.append((key, score))
        if kt:
            key_tiers.append(kt)
    if not key_tiers:
        return 0.0

    filtered = {norm(k): float(v) for k, v in pred_probs_open.items()
                if k != "__OTHER__"}
    pred_tiers = _probs_to_tiers(filtered)
    return kendall_tau(pred_tiers, key_tiers, mode=mode)


# ---------------------------------------------------------------------------
# Prediction parsing
# ---------------------------------------------------------------------------
# LLM port: the two functions below are the ONLY deviation from the VLM
# ``ranked_metrics.py``. The pipeline is identical (strip list markers /
# markdown / parentheticals / trailing punctuation, split on hedging
# connectors); we only widen the returned label *unit* from the first word to
# the first cleaned phrase segment, and drop the single-word filter, so
# ProtoQA's multi-word answers survive. Output is unchanged on single-word
# answers, so this is backward-compatible with the VLM datasets.

def _dedupe_ordered(items: List[str]) -> List[str]:
    """Remove duplicates, preserving first-occurrence order.

    LLM port: unlike the VLM version, multi-word items are KEPT (ProtoQA answers
    like "sun hat" are valid labels); only exact-duplicate normalized items drop.
    """
    seen = set()
    result = []
    for item in items:
        key = norm(item)
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _clean_label(s: str) -> str:
    """Clean a single parsed label: take only the first line, strip markdown/commentary."""
    s = s.split('\n')[0]
    s = re.sub(r'\*\*(.*?)\*\*', r'\1', s)
    s = re.sub(r'\*\*(.*)$', r'\1', s)
    s = re.sub(r'\(.*?\)', '', s)
    s = s.rstrip('.,;:!? ')
    return norm(s)


# Connectors an LLM uses to hedge between labels in a single free answer
# ("umbrella or sunscreen", "shade/hat", "bed, shower"). Split on these so a
# hedged answer yields its first choice instead of being discarded.
_CONNECTOR_RE = re.compile(r'\s*(?:\bor\b|\band\b|/|,|;|&)\s*', re.IGNORECASE)


def parse_single_answer(answer: str) -> str:
    """Resolve a free single-answer generation into one normalized label PHRASE.

    LLM port: same pipeline as the VLM ``parse_single_answer`` (strip list
    markers / markdown / parentheticals / trailing punctuation, split on hedging
    connectors ``or``/``and``/``/``/``,``/``;``/``&``), but returns the FIRST
    cleaned SEGMENT as a whole phrase instead of only its first word — so
    multi-word ProtoQA answers ("sun hat", "crying baby") are matched intact.
    Examples: "Sun hat or sunscreen." -> "sun hat", "shade/umbrella" -> "shade",
    "A crying baby." -> "a crying baby", "" -> "". Identical to the VLM behaviour
    on single-word answers.
    """
    if not answer:
        return ""
    text = str(answer)
    # If it's a numbered list, keep the first item's text.
    m = re.search(r'(?:^|\n)\s*1[\.\)]', text)
    if m:
        first = re.match(r'\s*1[\.\)]\s*(.+)', text[m.start():], re.DOTALL)
        if first:
            text = first.group(1)
    label = _clean_label(text)  # first line, markdown/parenthetical/punct stripped, lowercased
    if not label:
        return ""
    segment = _CONNECTOR_RE.split(label, maxsplit=1)[0].strip()
    return segment


def parse_numbered_list(text: str) -> List[str]:
    """Parse a numbered-list response into an ordered list of unique labels.

    Handles formats like:
        "1. umbrella 2. sunscreen 3. sun hat"
        "1. umbrella\\n2. sunscreen\\n3. sun hat"
        "1) umbrella 2) sunscreen"
        "Here's a list:\\n\\n1. umbrella\\n2. sunscreen"  (preamble stripped)
        "umbrella, sunscreen, sun hat"  (fallback: comma-separated)

    Duplicates are removed, keeping first appearance only. LLM port: multi-word
    items are kept (see ``_dedupe_ordered``).
    """
    m = re.search(r'(?:^|\n)\s*1[\.\)]', text)
    if m:
        text = text[m.start():]

    items: List[str] = []
    raw = re.findall(r'\d+[\.\)]\s*(.+?)(?=\s*\d+[\.\)]|$)', text, re.DOTALL)
    if raw:
        items = [_clean_label(item) for item in raw if item.strip()]
    elif ',' in text:
        items = [_clean_label(item) for item in text.split(',') if item.strip()]
    elif '\n' in text:
        items = [_clean_label(line) for line in text.strip().split('\n') if line.strip()]
    else:
        # Bare single answer (no list/comma/newline). Use _clean_label so trailing
        # punctuation ("Anger.") and markdown are stripped, same as the other paths.
        cleaned = _clean_label(text)
        items = [cleaned] if cleaned else []
    return _dedupe_ordered(items)
