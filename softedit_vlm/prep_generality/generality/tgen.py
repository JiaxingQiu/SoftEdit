"""Text-generality (t-gen): rephrase questions with gpt-4o-mini (synchronous).

Meaning- and answer-preserving rephrasings of the original question. The image
is unchanged; only the question text varies.

Uses the Azure OpenAI chat completions endpoint (same resource as the soft-gen
GPT-4o validator). The deployment name is configurable (default gpt-4o-mini).
"""

from __future__ import annotations

import re
import time
from typing import List

from openai import AzureOpenAI

from data_process.tokens import azure_key, azure_endpoint

# Derive the base resource endpoint (strip the "/openai/..." path if present).
AZURE_ENDPOINT = azure_endpoint.split("/openai/")[0].rstrip("/") + "/"
API_VERSION = "2025-01-01-preview"
# Sync-capable deployment on this Azure resource. "gpt-4o-batch" is batch-only;
# no "gpt-4o-mini" deployment exists. Override with --deployment if that changes.
DEFAULT_DEPLOYMENT = "gpt-4o"

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


def _parse_variants(text: str, question: str, k: int) -> List[str]:
    """Parse a numbered list into variants, dropping empty / identical lines.

    No min-length filter: many source questions are legitimately short
    (e.g. "What color?"), so short rephrasings are valid.
    """
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
    k: int = 2,
    deployment: str = DEFAULT_DEPLOYMENT,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> List[str]:
    """Return up to k meaning-preserving rephrasings of ``question``.

    Returns [] on repeated failure (caller decides how to handle).
    """
    user = (
        f"Rephrase the following question in {k} different ways while keeping the "
        f"meaning and the correct answer identical.\n\nQuestion: {question}"
    )
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=256,
            )
            return _parse_variants(resp.choices[0].message.content, question, k)
        except Exception as e:  # noqa: BLE001 - transient API/network errors
            if attempt == max_retries - 1:
                print(f"[tgen] failed after {max_retries} tries for {question!r}: {e}")
                return []
            time.sleep(2.0 * (attempt + 1))
    return []


__all__ = [
    "AZURE_ENDPOINT",
    "API_VERSION",
    "DEFAULT_DEPLOYMENT",
    "make_client",
    "rephrase_question",
]
