"""Locality anchors: text-only "describe-then-declare" supervision points.

For each of the K labels we build ``locality_n`` anchors. Each anchor is a prompt

    "{caption}. {declarative}. {question} {instruction}"

paired with a (usually grey) image and a one-hot target on its label. The edit
should still answer correctly on these re-descriptions **without** disturbing
unrelated inputs — this is the locality term of the objective.

Components (independent, composed only here):
  - caption:     VLM description of the anchor image (optional, ``caption="on"``).
  - declarative: "template" (fixed), "llm" (VLM naturalization),
                 "both" (template + llm → 2× anchors), or "random" (ablation:
                 generic VQA sentences with no edit-specific context).
  - image:       blend of anchor and grey (``blend`` knob; 0 = grey, 1 = anchor).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

GREY_VALUE: int = 128

CAPTION_PROMPTS: List[str] = [
    "Describe this image. Answer in one to two sentences.",
    "What objects do you see in this image? Answer in one to two sentences.",
    "Describe the colors and textures in this image. Answer in one to two sentences.",
    "What is happening in this scene? Answer in one to two sentences.",
    "Describe the background of this image. Answer in one to two sentences.",
    "What details stand out in this image? Answer in one to two sentences.",
]

RANDOM_DECLARATIVES: List[str] = [
    "Answer the question based on the image.",
    "This is a visual question answering task.",
    "Look at the image carefully before answering.",
    "Consider what you see in the photograph.",
    "The image contains relevant visual information.",
    "Pay attention to the details in the scene.",
    "Use the visual context to determine your response.",
    "Examine the objects and their arrangement.",
    "The visual content provides clues for the correct response.",
    "Think about what is depicted in this picture.",
    "Observe the scene and identify the key elements.",
    "The photograph shows a real-world scenario.",
    "Focus on the most prominent features of the image.",
    "Consider the spatial relationships between objects.",
    "The image was taken in a natural setting.",
    "Notice the colors and shapes in the scene.",
    "The visual information is sufficient to determine an outcome.",
    "Study the composition of the image.",
    "The scene depicts everyday objects and activities.",
    "Identify the main subject of the photograph.",
    "The image shows a common visual scenario.",
    "Look at the foreground and background elements.",
    "Consider the lighting and perspective of the image.",
    "The photograph captures a specific moment.",
    "Analyze the visual elements to form your judgment.",
    "The image provides context for understanding the situation.",
    "Notice the textures and materials visible in the scene.",
    "The visual content relates to a factual observation.",
    "Examine the image for relevant details.",
    "The scene contains multiple objects of interest.",
    "Consider the overall theme of the image.",
    "The photograph was taken from a particular viewpoint.",
    "Look for patterns and regularities in the scene.",
    "The image depicts a recognizable environment.",
    "Focus on what is happening in the image.",
    "The visual information can help disambiguate the situation.",
    "Study the relationships between the elements shown.",
    "The image contains both foreground and background detail.",
    "Consider how the objects interact with each other.",
    "The photograph shows a scene from daily life.",
    "Examine the proportions and scale of the objects.",
    "The visual content is informative and descriptive.",
    "Notice what is present and what is absent.",
    "The image captures a specific visual arrangement.",
    "Consider the context provided by the surroundings.",
    "The scene includes distinguishable features.",
    "Look at the image as a whole before focusing on parts.",
    "The photograph contains meaningful visual content.",
    "Analyze the scene systematically.",
    "The image provides a clear view of the subject matter.",
]


@dataclass
class LocalityAnchor:
    """One locality supervision point."""

    image: Any            # PIL Image (grey by default) or None (text-only)
    prompt: str           # full prompt including the answer instruction
    target_label: str
    target_q: np.ndarray  # [K] one-hot target on ``target_label``


# --------------------------------------------------------------------------- #
# image                                                                        #
# --------------------------------------------------------------------------- #

def blend_image(
    anchor: Any,
    w: float,
    size: Optional[Tuple[int, int]] = None,
) -> Optional[Image.Image]:
    """``w * anchor + (1 - w) * grey``. w<0 → None (text-only), 0 → grey, 1 → anchor."""
    if w < 0:
        return None
    img = anchor
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    if size is None:
        size = img.size
    if w <= 0:
        return Image.new("RGB", size, color=(GREY_VALUE, GREY_VALUE, GREY_VALUE))
    img = img.resize(size, Image.LANCZOS)
    if w >= 1:
        return img
    arr = np.array(img).astype(np.float32)
    blended = w * arr + (1 - w) * GREY_VALUE
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
# text                                                                         #
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


@torch.no_grad()
def _vlm_caption(model, wrapper, tokenizer, image, prompt: str,
                 max_new_tokens: int = 200, temperature: float = 0.9) -> str:
    """One sampled VLM caption of ``image`` → cleaned text."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    inputs = wrapper.encode([image], [prompt], tokenize=False)
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=True, temperature=temperature, use_cache=True,
    )
    prompt_len = inputs["input_ids"].shape[1]
    text = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True).strip()
    del output, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _clean_text(text)


DECLARATIVE_PROMPT: str = (
    'Convert the following to a single natural declarative sentence. '
    'Return a single sentence. '
    '"The answer to \'{q}\' is {label}."'
)

_GREY_IMG_CACHE = None


def _grey_img(size=(224, 224)):
    """Tiny grey image for text-only VLM calls (cached)."""
    global _GREY_IMG_CACHE
    if _GREY_IMG_CACHE is None:
        _GREY_IMG_CACHE = Image.new("RGB", size, (GREY_VALUE, GREY_VALUE, GREY_VALUE))
    return _GREY_IMG_CACHE


def _declarative_template(question: str, label: str) -> str:
    """Fixed template — always contains the correct label."""
    q = question.strip().rstrip("?").strip()
    return f"The answer to '{q}' is {label.strip()}."


def _declarative_vlm(question: str, label: str,
                     model, wrapper, tokenizer,
                     max_retries: int = 3) -> str:
    """Use the VLM itself (with grey image) to naturalize the declarative.

    Rejects outputs that are too close to the template ("answer to" present)
    or that diverge completely (target label missing).
    """
    q = question.strip().rstrip("?").strip()
    label_lower = label.strip().lower()
    prompt = DECLARATIVE_PROMPT.format(q=q, label=label.strip())
    grey = _grey_img()
    for _ in range(max_retries):
        text = _vlm_caption(model, wrapper, tokenizer, grey, prompt,
                            max_new_tokens=100, temperature=1.0)
        text = _first_sentence(text)
        low = text.lower()
        if label_lower not in low:
            continue
        if "answer" in low:
            continue
        if len(text.split()) <= 3:
            continue
        return text
    return _declarative_template(question, label)


def _compose_prompt(caption: Optional[str], declarative: str,
                    question: str, answer_instruction: str) -> str:
    parts = []
    if caption:
        parts.append(caption.strip())
    parts.append(declarative.strip())
    parts.append(question.strip())
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
    image: Any,
    question: str,
    labels: List[str],
    answer_instruction: str = "",
    *,
    n: int = 1,
    blend: float = 0.0,
    caption: str = "on",
    declarative: str = "template",
    gen_image_size: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
) -> List[LocalityAnchor]:
    """Build locality anchors (one-hot target per label).

    Yields ``n * K`` anchors for "template", "llm", or "random" mode, and
    ``2 * n * K`` for "both" (one template + one llm anchor per (label, i)).

    "random" is an ablation control: each label gets a generic VQA sentence
    (no edit-specific context) to show locality needs meaningful content.

    Args:
        model / wrapper / tokenizer: VLM handles (captioning only).
        image: anchor image (PIL or path) — used for captioning + blending.
        question: bare edit question (no answer instruction).
        labels: K label strings.
        n: anchors per label (per declarative style).
        blend: image blend weight (0 = grey, 1 = anchor, <0 = text-only).
        caption: "on" (VLM caption of the anchor image) or "off".
        declarative: "template" | "llm" | "both" | "random".
    """
    if declarative not in ("template", "llm", "both", "random"):
        raise ValueError(f"declarative must be 'template', 'llm', 'both', or 'random', "
                         f"got {declarative!r}")
    use_caption = caption == "on"
    gen_img = blend_image(image, blend, size=gen_image_size)

    anchors: List[LocalityAnchor] = []

    if declarative == "random":
        rng = np.random.RandomState(hash(question) & 0xFFFFFFFF)
        for li, label in enumerate(labels):
            target_q = _one_hot(label, labels)
            pool = list(RANDOM_DECLARATIVES)
            rng.shuffle(pool)
            for i in range(n):
                decl = pool[(li * n + i) % len(pool)]
                prompt = _compose_prompt(None, decl, question, answer_instruction)
                anchors.append(LocalityAnchor(
                    image=gen_img, prompt=prompt,
                    target_label=label, target_q=target_q,
                ))
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
            cap = None
            if use_caption:
                pool_idx = (li * n + i) % len(CAPTION_PROMPTS)
                cap = _vlm_caption(model, wrapper, tokenizer, image,
                                   CAPTION_PROMPTS[pool_idx])
            for dm in decl_modes:
                if dm == "llm":
                    decl = _declarative_vlm(question, label,
                                            model, wrapper, tokenizer)
                else:
                    decl = _declarative_template(question, label)
                prompt = _compose_prompt(cap, decl, question, answer_instruction)
                anchors.append(LocalityAnchor(
                    image=gen_img, prompt=prompt,
                    target_label=label, target_q=target_q,
                ))
                if verbose:
                    tag = f'cap="{cap[:50]}..." ' if cap else ""
                    mode_tag = f"[{dm}] " if declarative == "both" else ""
                    print(f"  [loc {i + 1}/{n}] {label}: {mode_tag}{tag}decl=\"{decl}\"",
                          flush=True)
    return anchors
