"""Text-generality (t-gen): rephrase ProtoQA questions with GPT-4o (synchronous).

Meaning- and answer-preserving rephrasings of the original question — the gold
answer distribution is unchanged, so a t-gen variant reuses the same
`answer_counts`. Faithful port of the VLM `prep_generality/generality/tgen.py`
(text-only; no image).
"""
from __future__ import annotations

import re
import time
from typing import List

from openai import AzureOpenAI

from data_process.tokens import azure_key, azure_endpoint

AZURE_ENDPOINT = azure_endpoint.split("/openai/")[0].rstrip("/") + "/"
API_VERSION = "2025-01-01-preview"
DEFAULT_DEPLOYMENT = "gpt-4o"          # sync-capable deployment
BATCH_DEPLOYMENT = "gpt-4o-batch"      # for the optional batch backend

SYSTEM_PROMPT = (
    "You rephrase questions. Preserve the exact meaning so the correct answer is "
    "unchanged. Do not add, remove, or alter any facts. Output only the rephrased "
    "questions, one per line, as a numbered list, with no extra commentary."
)

_NUM_RE = re.compile(r"^\s*\d+[\.\)]\s*")


def make_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=azure_key,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
    )


def build_user_prompt(question: str, k: int) -> str:
    return (
        f"Rephrase the following question in {k} different ways while keeping the "
        f"meaning and the correct answer identical.\n\nQuestion: {question}"
    )


def parse_variants(text: str, question: str, k: int) -> List[str]:
    """Parse a numbered list into variants, dropping empty / identical lines."""
    seen = {question.strip().lower()}
    out: List[str] = []
    for line in (text or "").splitlines():
        cand = _NUM_RE.sub("", line.strip()).strip()
        low = cand.lower()
        if cand and low not in seen:
            seen.add(low)
            out.append(cand)
    return out[:k]


def rephrase_question(
    client: AzureOpenAI,
    question: str,
    *,
    k: int = 4,
    deployment: str = DEFAULT_DEPLOYMENT,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> List[str]:
    """Return up to k meaning-preserving rephrasings. Returns [] on repeated failure."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, k)},
    ]
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=messages,
                temperature=temperature,
                max_tokens=256,
            )
            return parse_variants(resp.choices[0].message.content, question, k)
        except Exception as e:  # noqa: BLE001 - transient API/network errors
            if attempt == max_retries - 1:
                print(f"[tgen] failed after {max_retries} tries for {question!r}: {e}")
                return []
            time.sleep(2.0 * (attempt + 1))
    return []


def build_request_body(question: str, k: int, *, deployment: str = BATCH_DEPLOYMENT,
                       temperature: float = 0.7) -> dict:
    """Chat body for the optional Azure batch backend (one request per question)."""
    return {
        "model": deployment,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, k)},
        ],
        "temperature": temperature,
        "max_tokens": 256,
    }


__all__ = [
    "AZURE_ENDPOINT", "API_VERSION", "DEFAULT_DEPLOYMENT", "BATCH_DEPLOYMENT",
    "SYSTEM_PROMPT", "make_client", "build_user_prompt", "parse_variants",
    "rephrase_question", "build_request_body",
]
