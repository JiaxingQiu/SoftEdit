"""1-sentence scene caption via GPT-4o vision (Azure).

Used by VQAv2 blend generation: the caption provides scene context to the T2I
prompt so FLUX generates images grounded in the original scene rather than
arbitrary compositions.
"""

from __future__ import annotations

import base64
import time
from functools import lru_cache
from pathlib import Path

from openai import AzureOpenAI
from PIL import Image

from data_process.tokens import azure_key, azure_endpoint

AZURE_ENDPOINT = azure_endpoint.split("/openai/")[0].rstrip("/") + "/"
API_VERSION = "2025-01-01-preview"
DEPLOYMENT = "gpt-4o"

SYSTEM = (
    "You describe images in exactly one short sentence (max 15 words). "
    "Focus on the main subject, scene layout, and setting. "
    "Do NOT mention text, labels, or answer any question about the image."
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


def caption_image(
    image_path: str | Path,
    *,
    max_retries: int = 3,
) -> str:
    """Return a 1-sentence scene description of the image at ``image_path``."""
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
                        {"type": "text", "text": "Describe this image in one sentence."},
                    ]},
                ],
                temperature=0.0,
                max_tokens=60,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[caption] failed after {max_retries} tries: {e}")
                return ""
            time.sleep(2.0 * (attempt + 1))
    return ""
