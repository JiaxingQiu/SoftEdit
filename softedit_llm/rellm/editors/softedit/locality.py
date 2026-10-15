"""Locality anchors: text-only "declare-then-ask" supervision points.

For each of the K labels we build ``locality_n`` anchors. Each anchor is a prompt

    "{declarative} {question} {instruction}"

paired with a one-hot target on its label. The edit should still answer that
re-description correctly **without** disturbing unrelated inputs — this is the
locality term of the objective.

Text-only rewrite of the VLM ``locality.py``: the image, grey-image blend and
VLM captioning are dropped (no visual context in the LLM setting). Declaratives:
  - "template" : fixed "The answer to 'Q' is <label>." (always contains the label)
  - "llm"      : the backbone LLM naturalizes the template into a fluent sentence
  - "both"     : template + llm  (2x anchors per label)
  - "random"   : ablation control (generic sentences, no edit-specific content)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import torch


RANDOM_DECLARATIVES: List[str] = [
    "Answer the following question.",
    "This is a question answering task.",
    "Read the question carefully before answering.",
    "Consider the question and respond concisely.",
    "Use common sense to determine your response.",
    "Think about what most people would say.",
    "The question has a short factual answer.",
    "Provide the most likely answer.",
    "Give a brief, direct response.",
    "Focus on the key information in the question.",
    "The answer should be a word or short phrase.",
    "Respond with the single most plausible answer.",
    "Consider the everyday meaning of the question.",
    "Answer as a typical person would.",
    "The question concerns a common scenario.",
    "Reflect on the most natural response.",
    "State the most common answer to the question.",
    "The question is about ordinary knowledge.",
    "Give the answer that comes to mind first.",
    "Keep the response short and to the point.",
    "The question expects a general answer.",
    "Answer based on what is usually true.",
    "Provide a concise factual response.",
    "Consider the most frequent response people give.",
    "The question has a widely agreed-upon answer.",
    "Respond with a plausible everyday answer.",
    "Think about the typical situation described.",
    "The answer is something most people would name.",
    "Give the answer that best fits the question.",
    "Respond briefly with the most likely option.",
]


@dataclass
class LocalityAnchor:
    """One locality supervision point (text-only)."""

    prompt: str           # full prompt including the answer instruction
    target_label: str
    target_q: np.ndarray  # [K] one-hot target on ``target_label``


# --------------------------------------------------------------------------- #
# text cleanup                                                                  #
# --------------------------------------------------------------------------- #

def _clean_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?:\s*\n", "", text, count=1)
    text = re.sub(r'^["\']|["\']$', "", text.strip())
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_QUOTES = r'["\'\u201c\u201d\u2018\u2019]'


def _first_sentence(text: str) -> str:
    """Keep only the first sentence; strip LLM chattiness and quotes."""
    text = re.sub(rf'^{_QUOTES}|{_QUOTES}$', "", text.strip())
    m = re.search(rf'(?<=[a-z0-9])\.(?:\s|{_QUOTES}|$)', text, re.IGNORECASE)
    if m:
        text = text[:m.start() + 1]
    text = re.sub(rf'^{_QUOTES}|{_QUOTES}$', "", text.strip())
    return text


DECLARATIVE_PROMPT: str = (
    'Convert the following to a single natural declarative sentence. '
    'Return a single sentence. '
    '"The answer to \'{q}\' is {label}."'
)


@torch.no_grad()
def _llm_generate(wrapper, prompt: str, max_new_tokens: int = 100,
                  temperature: float = 1.0) -> str:
    """One sampled generation from the backbone LLM → cleaned text."""
    out = wrapper.generate([prompt], max_new_tokens=max_new_tokens,
                           do_sample=True, temperature=temperature, use_cache=True)
    text = out[0] if isinstance(out, (list, tuple)) else str(out)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _clean_text(text)


def _declarative_template(question: str, label: str) -> str:
    """Fixed template — always contains the correct label."""
    q = question.strip().rstrip("?").strip()
    return f"The answer to '{q}' is {label.strip()}."


def _declarative_llm(question: str, label: str, wrapper, max_retries: int = 3) -> str:
    """Use the backbone LLM to naturalize the declarative.

    Rejects outputs that are too close to the template ("answer" present) or that
    diverge completely (target label missing).
    """
    q = question.strip().rstrip("?").strip()
    label_lower = label.strip().lower()
    prompt = DECLARATIVE_PROMPT.format(q=q, label=label.strip())
    for _ in range(max_retries):
        text = _first_sentence(_llm_generate(wrapper, prompt, max_new_tokens=100,
                                             temperature=1.0))
        low = text.lower()
        if label_lower not in low:
            continue
        if "answer" in low:
            continue
        if len(text.split()) <= 3:
            continue
        return text
    return _declarative_template(question, label)


def _compose_prompt(declarative: str, question: str, answer_instruction: str) -> str:
    parts = [declarative.strip(), question.strip()]
    if answer_instruction:
        parts.append(answer_instruction.strip())
    return " ".join(parts)


def _one_hot(label: str, labels: List[str]) -> np.ndarray:
    q = np.zeros(len(labels), dtype=np.float64)
    q[labels.index(label)] = 1.0
    return q


# --------------------------------------------------------------------------- #
# builder                                                                      #
# --------------------------------------------------------------------------- #

def build_locality_anchors(
    model,
    wrapper,
    tokenizer,
    question: str,
    labels: List[str],
    answer_instruction: str = "",
    *,
    n: int = 1,
    declarative: str = "both",
    verbose: bool = True,
) -> List[LocalityAnchor]:
    """Build locality anchors (one-hot target per label).

    Yields ``n * K`` anchors for "template", "llm", or "random" mode, and
    ``2 * n * K`` for "both" (one template + one llm anchor per (label, i)).

    Args:
        model / wrapper / tokenizer: LLM handles (``wrapper.generate`` for "llm").
        question: bare edit question (no answer instruction).
        labels: K label strings.
        n: anchors per label (per declarative style).
        declarative: "template" | "llm" | "both" | "random".
    """
    if declarative not in ("template", "llm", "both", "random"):
        raise ValueError(f"declarative must be 'template', 'llm', 'both', or 'random', "
                         f"got {declarative!r}")

    anchors: List[LocalityAnchor] = []

    if declarative == "random":
        rng = np.random.RandomState(hash(question) & 0xFFFFFFFF)
        for li, label in enumerate(labels):
            target_q = _one_hot(label, labels)
            pool = list(RANDOM_DECLARATIVES)
            rng.shuffle(pool)
            for i in range(n):
                decl = pool[(li * n + i) % len(pool)]
                prompt = _compose_prompt(decl, question, answer_instruction)
                anchors.append(LocalityAnchor(
                    prompt=prompt, target_label=label, target_q=target_q))
                if verbose:
                    print(f"  [loc {i + 1}/{n}] {label}: [random] decl=\"{decl}\"",
                          flush=True)
        return anchors

    decl_modes = {
        "template": ["template"],
        "llm": ["llm"],
        "both": ["template", "llm"],
    }[declarative]

    for li, label in enumerate(labels):
        target_q = _one_hot(label, labels)
        for i in range(n):
            for dm in decl_modes:
                if dm == "llm":
                    decl = _declarative_llm(question, label, wrapper)
                else:
                    decl = _declarative_template(question, label)
                prompt = _compose_prompt(decl, question, answer_instruction)
                anchors.append(LocalityAnchor(
                    prompt=prompt, target_label=label, target_q=target_q))
                if verbose:
                    mode_tag = f"[{dm}] " if declarative == "both" else ""
                    print(f"  [loc {i + 1}/{n}] {label}: {mode_tag}decl=\"{decl}\"",
                          flush=True)
    return anchors
