"""Soft-generality (soft-gen) context generation with GPT-4o.

For a ProtoQA question Q with human answers {A_1, A_2, A_3}, and a chosen TARGET
answer A_i, we ask GPT-4o for a brief natural **premise** to prepend to Q such
that, given the premise, A_i becomes clearly the single most likely answer —
more likely than the other listed answers. The soft-gen eval sample is then:

    prompt = compose_question(context, Q) + answer instruction
    gold   = one-hot on A_i   (answer_counts = {A_i: 1})

and `sloc` is the distribution-alignment metric on these one-hot samples.

This module is the **single source of truth** for the prompt. The prompt-eng
notebook (`notebooks/prelim/examine_softgen_prompt.ipynb`) and the batch driver
both import `SYSTEM_PROMPT` / `build_user_prompt` from here, so iterating in the
notebook and firing the batch never diverge.

Sync client mirrors `prep_generality/generality/tgen.py`.
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
DEFAULT_DEPLOYMENT = "gpt-4o"          # sync-capable deployment
BATCH_DEPLOYMENT = "gpt-4o-batch"      # batch-only deployment (see gen_support pattern)


# ---------------------------------------------------------------------------
# The prompt (edit here — or override SYSTEM_PROMPT / build_user_prompt in the
# notebook while iterating, then paste the final version back into this file).
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You write a brief, natural PREMISE (1 sentence) to place BEFORE a question so that, "
    "given the premise, one specific TARGET answer becomes clearly the single most likely answer "
    "to that question — more likely than the other plausible answers listed.\n\n"
    "Rules:\n"
    "- Do NOT mention, name, spell, or obviously synonymize the target answer, and never write "
    "'the answer is'. The premise should make the target the natural conclusion, not state it.\n"
    "- Do NOT change, restate, or answer the question. The premise is only added context that "
    "comes before the original question verbatim.\n"
    "- Make it realistic and specific: it should steer a reasonable person toward the target and "
    "away from each of the other listed answers, without being a trivial giveaway.\n"
    "- Keep it concise, self-contained, and neutral in tone.\n"
    "Output ONLY the premise text — no quotes, no labels, no preamble."
)


def build_user_prompt(question: str, target: str, others: List[str]) -> str:
    """User message: the question, the target to elevate, and the answers to suppress."""
    others_str = ", ".join(str(o) for o in others) if others else "(none)"
    return (
        f"Question: {question}\n"
        f"Target answer to make most likely: {target}\n"
        f"Other plausible answers to make less likely: {others_str}\n\n"
        f"Write the premise."
    )


def compose_question(context: str, question: str) -> str:
    """Combine a soft-gen premise with the original question for eval prompting.

    The task engineer appends the answer instruction; this only prepends context.
    """
    context = (context or "").strip()
    return f"{context} {question}".strip() if context else str(question).strip()


def parse_context(content: str) -> str:
    """Clean a GPT premise reply: drop a leading 'Premise:'/'Context:' label and quotes."""
    if not content:
        return ""
    t = content.strip()
    t = re.sub(r"^\s*(premise|context)\s*[:\-]\s*", "", t, flags=re.IGNORECASE)
    t = t.strip().strip('"').strip("'").strip()
    return t


# ---------------------------------------------------------------------------
# Sync client (prototyping / prompt-eng notebook)
# ---------------------------------------------------------------------------

def make_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=azure_key,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
    )


def generate_context(
    client: AzureOpenAI,
    question: str,
    target: str,
    others: List[str],
    *,
    deployment: str = DEFAULT_DEPLOYMENT,
    temperature: float = 0.7,
    max_tokens: int = 220,
    max_retries: int = 3,
) -> str:
    """Return one premise for (question, target) that suppresses `others`.

    Returns "" on repeated failure (caller decides how to handle).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, target, others)},
    ]
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return parse_context(resp.choices[0].message.content)
        except Exception as e:  # noqa: BLE001 - transient API/network errors
            if attempt == max_retries - 1:
                print(f"[softgen] failed after {max_retries} tries for "
                      f"{question!r}/{target!r}: {e}")
                return ""
            time.sleep(2.0 * (attempt + 1))
    return ""


def build_request_body(question: str, target: str, others: List[str],
                       *, deployment: str = BATCH_DEPLOYMENT,
                       temperature: float = 0.7, max_tokens: int = 220) -> dict:
    """Chat-completions body for the Azure batch JSONL (one request per target)."""
    return {
        "model": deployment,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, target, others)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


__all__ = [
    "AZURE_ENDPOINT", "API_VERSION", "DEFAULT_DEPLOYMENT", "BATCH_DEPLOYMENT",
    "SYSTEM_PROMPT", "build_user_prompt", "compose_question", "parse_context",
    "make_client", "generate_context", "build_request_body",
]
