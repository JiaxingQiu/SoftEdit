"""Auto-tune probes for SoftEdit ``reject_threshold_pct``.

``Augmenter`` is the caller. Each probe jointly perturbs image and question.

Image (in order):
  crop 0.75–0.95, gaussian noise, aspect stretch, then ReasonEdit mosaic
  (``pad_with_mosaic``: Picsum 4×4 + 20% noise tiles, area_pct=0.9, canvas
  kept larger), then rotate ±10° + color jitter.

  Stretch/noise magnitudes match i-gen bin 3 (scale 0.75). Stretch is
  squeeze-only; empty bands are filled from the Picsum cache.

Text: chained Qwen2.5-1.5B rephrase (same recipe as the shared utils Augmenter).
"""
import random
import re

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_JITTER = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
_MOSAIC_AREA_PCT = 0.9
_MOSAIC_N_TILES = 4
_IGEN_SCALE = 0.75          # i-gen default bin 3
_NOISE = (6.0, 14.0)        # gaussian sigma at scale 1.0
_STRETCH = 0.25             # max per-axis squeeze at scale 1.0
_ROT = 10.0                 # ReasonEdit ``RandomRotation(10)``


class Augmenter:
    """Joint image+text augmenter used by auto-threshold calibration."""

    _llm_name = "Qwen/Qwen2.5-1.5B-Instruct"
    _instance = None

    def __init__(self):
        self._llm = None
        self._llm_tok = None

    @classmethod
    def shared(cls) -> "Augmenter":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def perturb(self, image, question: str, seed: int, kind: str = "both"):
        """One generality probe. ``kind`` is ``both`` (default), ``image``, or ``text``."""
        q = question or ""
        img = image
        if kind in ("both", "image") and image is not None:
            try:
                img = self.image(image, seed)
            except Exception:
                img = image
        if kind in ("both", "text"):
            try:
                q = self.question(q, int(seed) + 7919)
            except Exception as exc:
                print(f"[Augmenter] question perturb failed (seed={seed}): {exc}", flush=True)
        return img, q

    def image(self, img, seed: int) -> Image.Image:
        rng = random.Random(int(seed))
        _set_seed(int(seed))
        im = self._as_rgb(img)
        w, h = im.size
        frac = rng.uniform(0.75, 0.95)
        cw, ch = max(1, int(w * frac)), max(1, int(h * frac))
        left = rng.randint(0, max(0, w - cw))
        top = rng.randint(0, max(0, h - ch))
        im = im.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BICUBIC)

        fill_bg = self._fill_bg(w, h)
        im = self._gaussian_noise(im, rng)
        im = self._stretch(im, rng, fill_bg)
        try:
            from revlm.editors.utils import pad_with_mosaic
            im = pad_with_mosaic(im, area_pct=_MOSAIC_AREA_PCT, n_tiles=_MOSAIC_N_TILES)
        except Exception:
            pass
        im = T.functional.rotate(im, rng.uniform(-_ROT, _ROT))
        im = _JITTER(im)
        return im

    def question(self, q, seed: int = 0) -> str:
        _set_seed(int(seed))
        return self.rephrase(q or "", mode="rephrase") or (q or "")

    @staticmethod
    def _as_rgb(img) -> Image.Image:
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        if hasattr(img, "convert"):
            return img.convert("RGB")
        raise TypeError(f"unsupported image type: {type(img)}")

    @staticmethod
    def _fill_bg(w: int, h: int):
        try:
            from revlm.editors.utils import get_bg_cache
            return get_bg_cache().random_crop((w, h))
        except Exception:
            return None

    @staticmethod
    def _gaussian_noise(im: Image.Image, rng: random.Random,
                        scale: float = _IGEN_SCALE) -> Image.Image:
        sigma = rng.uniform(*_NOISE) * scale
        arr = np.asarray(im).astype(np.float32)
        noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    @staticmethod
    def _stretch(im: Image.Image, rng: random.Random, fill_bg,
                 scale: float = _IGEN_SCALE) -> Image.Image:
        w, h = im.size
        fx = 1.0 - rng.uniform(0.0, _STRETCH) * scale
        fy = 1.0 - rng.uniform(0.0, _STRETCH) * scale
        nw, nh = max(1, int(round(w * fx))), max(1, int(round(h * fy)))
        resized = im.resize((nw, nh), Image.LANCZOS)
        if fill_bg is not None:
            canvas = fill_bg.convert("RGB").resize((w, h), Image.LANCZOS)
        else:
            canvas = Image.new("RGB", (w, h), (0, 0, 0))
        canvas.paste(resized, ((w - nw) // 2, (h - nh) // 2))
        return canvas

    def _get_llm(self):
        if self._llm is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[Augmenter] Loading {self._llm_name}...")
            self._llm_tok = AutoTokenizer.from_pretrained(self._llm_name)
            self._llm = AutoModelForCausalLM.from_pretrained(
                self._llm_name, torch_dtype=torch.float16, device_map="auto"
            )
            self._llm.eval()
        return self._llm, self._llm_tok

    @staticmethod
    def _is_english(text: str) -> bool:
        return bool(re.fullmatch(r"[a-zA-Z0-9\s.,!?;:'\"\-()]+", text)) if text else False

    def rephrase(self, text, mode: str = "rephrase"):
        if not text:
            return text
        model, tok = self._get_llm()

        if mode == "question":
            instruction = ("Ask a general question that can be answered by this "
                           "sentence. Only output the question, nothing else.")
            temp, max_new_tokens, n_chain = 0.1, 32, 1
        elif mode == "normalize":
            instruction = ("Simplify this text slightly, keeping the same meaning. "
                           "Output only the simplified text.")
            temp, max_new_tokens, n_chain = 0.1, 64, 1
        else:
            instruction = ("Rephrase this sentence while keeping the same meaning. "
                           "Only output the rephrased sentence, nothing else.")
            temp, max_new_tokens, n_chain = 1.5, 64, 3

        for _attempt in range(3):
            candidates = []
            for _c in range(n_chain):
                prev = candidates[-1] if candidates else text
                messages = [{"role": "user", "content": f"{instruction}\n\n{prev}"}]
                prompt = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = tok(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs, max_new_tokens=max_new_tokens, temperature=temp,
                        do_sample=True, pad_token_id=tok.eos_token_id,
                    )
                out = tok.decode(
                    out_ids[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True).strip()
                if not out or out == text or len(out) < 5 or not self._is_english(out):
                    continue
                if mode == "question" and not out.endswith("?"):
                    continue
                candidates.append(out)
            if candidates:
                break

        return random.choice(candidates) if candidates else text


def perturb_pair(image, question: str, seed: int, kind: str = "both"):
    """Thin wrapper around ``Augmenter.shared().perturb``."""
    return Augmenter.shared().perturb(image, question, seed, kind=kind)
