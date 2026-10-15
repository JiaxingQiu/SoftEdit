"""Yes/No paraphrase judge (Qwen2.5-1.5B, two-token softmax).

Self-contained locality filter for retrieval: given the query question and the
retrieved edit's question, returns P(Yes) that they ask about the same thing.
"""

import torch
from typing import List, Tuple


class ParaphraseJudge:
    MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    THRESH = 0.5
    PROMPT = (
        'Is "{q1}" a paraphrased question asking about the same thing as "{q2}"? '
        "Answer Yes or No."
    )

    def __init__(self):
        self._model = None
        self._tok = None
        self._yes_ids: List[int] = []
        self._no_ids: List[int] = []

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[ParaphraseJudge] loading {self.MODEL} ...", flush=True)
        self._tok = AutoTokenizer.from_pretrained(self.MODEL)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.MODEL, torch_dtype=torch.float16, device_map="auto")
        self._model.eval()
        self._yes_ids = self._ids(["Yes", "yes", " Yes", " yes"])
        self._no_ids = self._ids(["No", "no", " No", " no"])

    def _ids(self, variants):
        out = []
        for v in variants:
            enc = self._tok.encode(v, add_special_tokens=False)
            if enc:
                out.append(enc[0])
        return sorted(set(out))

    @torch.no_grad()
    def score(self, q1: str, q2: str) -> float:
        """P(Yes) that q1 asks about the same thing as q2."""
        self._load()
        messages = [{"role": "user", "content": self.PROMPT.format(q1=q1, q2=q2)}]
        prompt = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        logits = self._model(**inputs).logits[0, -1].float()
        yes = torch.logsumexp(logits[self._yes_ids], dim=0)
        no = torch.logsumexp(logits[self._no_ids], dim=0)
        return float(torch.softmax(torch.stack([no, yes]), dim=0)[1])

    def accept(self, q1: str, q2: str) -> Tuple[bool, float]:
        p = self.score(q1, q2)
        return p >= self.THRESH, p
