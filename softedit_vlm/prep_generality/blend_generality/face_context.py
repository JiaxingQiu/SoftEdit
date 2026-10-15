"""One-sentence, emotion-invariant face description via GPT-4o vision (Azure).

Used by FERPlus blend generation: the description grounds the T2I anchor in the
original face's identity/framing (age, gender, pose, lighting, ...) so FLUX makes
a plausible FER-distribution face instead of a generic stock portrait. It must
NOT mention the expression — that is owned by the ``{label}`` clause in the prompt
(see ``prompts.build_t2i_prompt``). Kept short so the context does not overwhelm
image generation.
"""

from __future__ import annotations

import base64
import time
from functools import lru_cache
from pathlib import Path

from openai import AzureOpenAI

from data_process.tokens import azure_key, azure_endpoint

AZURE_ENDPOINT = azure_endpoint.split("/openai/")[0].rstrip("/") + "/"
API_VERSION = "2025-01-01-preview"
DEPLOYMENT = "gpt-4o"

SYSTEM = (
    "Describe the person's demographics (age, gender, skin tone), facing direction, looking direction, and other basic information, in a few words. "
    "Ignore facial expression. "
    "Reply with a short sentence."
)


def _make_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=azure_key,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
    )


_client: AzureOpenAI | None = None


def get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = _make_client()
    return _client


def _encode_image(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


@lru_cache(maxsize=None)
def describe_face(
    image_path: str | Path,
    *,
    max_retries: int = 3,
) -> str:
    """Return a 1-sentence, expression-free description of the face at ``image_path``.

    Cached per path: the same original face is reused across blend rounds. Returns
    an empty string on failure (caller falls back to the no-context template).
    """
    client = get_client()
    b64 = _encode_image(image_path)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=DEPLOYMENT,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "low",
                        }},
                        {"type": "text", "text": "Describe this face in one sentence."},
                    ]},
                ],
                temperature=0.0,
                max_tokens=60,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[face_context] failed after {max_retries} tries: {e}")
                return ""
            time.sleep(2.0 * (attempt + 1))
    return ""
