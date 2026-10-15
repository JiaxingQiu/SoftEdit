"""Auto-tune probes for SoftEdit ``reject_threshold_pct``.

``Augmenter`` is the caller. Text-only: chained Qwen2.5-1.5B rephrase
(n_chain=3, English filter, pick one).
"""
import random
import re

import numpy as np
import torch


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Augmenter:
    """Text augmenter used by auto-threshold calibration."""

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

    def perturb(self, question: str, seed: int) -> str:
        try:
            return self.question(question or "", seed)
        except Exception as exc:
            print(f"[Augmenter] question perturb failed (seed={seed}): {exc}", flush=True)
            return question or ""

    def question(self, q, seed: int = 0) -> str:
        _set_seed(int(seed))
        return self.rephrase(q or "", mode="rephrase") or (q or "")

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


def perturb_question(question: str, seed: int) -> str:
    """Thin wrapper around ``Augmenter.shared().perturb``."""
    return Augmenter.shared().perturb(question, seed)
