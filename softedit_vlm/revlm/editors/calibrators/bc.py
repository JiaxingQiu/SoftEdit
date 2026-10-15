"""Batch Calibration (Zhou et al., ICLR 2024), soft-label BCL adaptation.

The paper has no code release. This follows the printed formulas. On a batch
of inputs the contextual bias is the mean class score,

    p_hat(y) = (1/N) sum_i p(y | x_i),

and the labeled variant (BCL) subtracts a scaled copy of that bias,

    p_BCL(y | x) = p(y | x) - gamma * p_hat(y).

gamma = 0 leaves the model unchanged. gamma = 1 is unlabeled BC. The paper
picks gamma by a uniform grid on [-5, 5]; here the grid minimizes KL to the
human vote distribution instead of maximizing accuracy.

Adaptations for this benchmark, where each question has its own answer set:

* p_hat is the mean first-answer-token softmax over the shared vocabulary,
  estimated once on D_edit and then frozen. Eval batches have size 1, so
  re-estimating the bias there would cancel the correction.
* The same frozen bias is subtracted at every generated position. The paper
  only scores one label vector per input.
* Generation softmaxes the LM-head output, so the hook returns the log of the
  shifted scores after they are shifted to be non-negative and normalized.
  The greedy token matches argmax(p - gamma p_hat). Align, SGen, and SLoc
  read that normalized distribution.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from revlm.editors.soft_utils.targets import SoftTarget

# Paper grid: uniform samples of gamma on [-5, 5]. 21 points include 0 and 1.
_GAMMA_MIN = -5.0
_GAMMA_MAX = 5.0
_GAMMA_STEPS = 21
_CHUNK = 32


def _resolve_lm_head(model) -> nn.Linear:
    """Locate the lm_head across the VLMs this repo runs."""
    for path in ("lm_head", "language_model.lm_head"):
        mod = model
        for attr in path.split("."):
            mod = getattr(mod, attr, None)
            if mod is None:
                break
        if isinstance(mod, nn.Linear):
            return mod
    raise ValueError("model has no nn.Linear lm_head")


def _shifted_log_prob(probs: torch.Tensor, mu: torch.Tensor, gamma: float) -> torch.Tensor:
    """Log of the normalized BCL scores. argmax matches argmax(p - gamma mu)."""
    shifted = probs - float(gamma) * mu
    shifted = shifted - shifted.amin(dim=-1, keepdim=True)
    shifted = shifted / shifted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return shifted.clamp_min(1e-12).log()


class _BiasHook:
    """Subtract gamma * p_hat from the next-token distribution.

    gamma == 0, or a missing bias, returns the LM-head output unchanged.
    """

    def __init__(self):
        self.gamma = 0.0
        self.mu: Optional[torch.Tensor] = None
        self._handle = None

    def __call__(self, module, inputs, output):
        if self.mu is None or abs(self.gamma) < 1e-8:
            return output
        mu = self.mu.to(device=output.device, dtype=torch.float32)
        probs = torch.softmax(output.float(), dim=-1)
        corrected = _shifted_log_prob(probs, mu, self.gamma)
        return corrected.to(dtype=output.dtype)

    def install(self, module: nn.Module):
        self.remove()
        self._handle = module.register_forward_hook(self)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class BCSoft(nn.Module):
    """Batch BCL calibrator. Same editor interface as ft_retrain."""

    log_name = "bc_soft"
    gamma_min = _GAMMA_MIN
    gamma_max = _GAMMA_MAX
    gamma_steps = _GAMMA_STEPS

    def __init__(self, config, model):
        super().__init__()
        self.model = model.model if hasattr(model, "model") else model
        self.tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None
        self.device = config.device
        self.head = _resolve_lm_head(self.model)
        self._hook = _BiasHook()

        for p in self.model.parameters():
            p.requires_grad = False

        self.soft = True
        self.soft_mode = "loss"
        self.rank_temp = 1.0
        self._soft_target: Optional[SoftTarget] = None
        self._soft_cont = None

        first = next(self.model.parameters(), None)
        dtype = getattr(first, "dtype", torch.bfloat16)
        self.autocast_dtype = torch.float16 if dtype == torch.float16 else torch.bfloat16

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    @staticmethod
    def _prompt_end_pos(labels: torch.Tensor) -> int:
        """Logit row that predicts the first answer token. Matches soft_utils.kl."""
        row = labels[0] if labels.dim() > 1 else labels
        nz = (row != -100).nonzero(as_tuple=False)
        if nz.numel() == 0:
            return int(row.shape[0]) - 1
        return max(int(nz[0].item()) - 1, 0)

    def _to_device(self, batch):
        return {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()}

    @torch.no_grad()
    def _collect(self, history):
        """Unscaled prompt-end logits and the human target for every edit."""
        self.model.eval()
        self._hook.remove()
        rows = []
        targets: List[SoftTarget] = []
        for batch in history:
            target = batch.get("_soft_target")
            labels = batch.get("labels")
            if target is None or labels is None:
                continue
            inputs = self._to_device(
                {k: v for k, v in batch.items() if k not in ("_soft_target", "_soft_cont")})
            with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
                outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            pos = self._prompt_end_pos(inputs["labels"])
            rows.append(logits[0, pos, :].float().cpu())
            targets.append(target)
        if not rows:
            return None, None
        return torch.stack(rows, dim=0), targets

    def _mean_prob(self, logits: torch.Tensor) -> torch.Tensor:
        """p_hat: mean softmax over the edit batch. Stays on CPU."""
        total = torch.zeros(logits.size(-1), dtype=torch.float32)
        n = logits.size(0)
        for start in range(0, n, _CHUNK):
            chunk = logits[start:start + _CHUNK].to(self.device)
            total += torch.softmax(chunk, dim=-1).sum(dim=0).cpu()
        return total / max(n, 1)

    def _mean_kl(self, logits, targets, mu, gamma: float) -> float:
        n = len(targets)
        total = 0.0
        for start in range(0, n, _CHUNK):
            chunk = logits[start:start + _CHUNK].to(self.device)
            if abs(gamma) < 1e-8:
                logp = torch.log_softmax(chunk, dim=-1)
            else:
                probs = torch.softmax(chunk, dim=-1)
                logp = _shifted_log_prob(probs, mu.to(chunk.device), gamma)
            for i, tgt in enumerate(targets[start:start + _CHUNK]):
                q = tgt.q.to(logp.device)
                ids = tgt.first_ids.to(logp.device)
                lp = logp[i].index_select(0, ids)
                total += float((q * (torch.log(q.clamp_min(1e-12)) - lp)).sum().clamp_min(0.0))
        return total / max(n, 1)

    def _fit(self, logits, targets):
        """Grid-search gamma. Returns (gamma, p_hat)."""
        mu = self._mean_prob(logits)
        grid = torch.linspace(self.gamma_min, self.gamma_max, self.gamma_steps)
        best_gamma, best_kl = 0.0, float("inf")
        for gamma in grid.tolist():
            kl = self._mean_kl(logits, targets, mu, gamma)
            mark = ""
            if abs(gamma) < 1e-8:
                mark = " (uncalibrated)"
            elif abs(gamma - 1.0) < 1e-8:
                mark = " (BC)"
            print(f"[{self.log_name}] gamma={gamma:+.2f} KL={kl:.4f}{mark}", flush=True)
            if kl < best_kl:
                best_gamma, best_kl = float(gamma), kl
        print(f"[{self.log_name}] fitted gamma={best_gamma:+.2f} (KL={best_kl:.4f})",
              flush=True)
        return best_gamma, mu

    def edit(self, config, tokens, batch_history=None):
        """Estimate p_hat on D_edit, pick gamma, then hook the LM head."""
        history = list(batch_history or []) + [tokens]
        logits, targets = self._collect(history)
        if logits is None:
            print(f"[{self.log_name}] no soft targets, keeping gamma=0", flush=True)
            self._hook.gamma = 0.0
            self._hook.mu = None
        else:
            print(f"[{self.log_name}] fitting BCL on {len(targets)} edits", flush=True)
            self._hook.gamma, self._hook.mu = self._fit(logits, targets)
        self._hook.install(self.head)
        self.model.eval()
        return self.model

    def save_resume_state(self):
        mu = None if self._hook.mu is None else self._hook.mu.detach().cpu()
        return {"gamma": self._hook.gamma, "mu": mu}

    def load_resume_state(self, state):
        if not state:
            return
        self._hook.gamma = float(state.get("gamma", 0.0))
        mu = state.get("mu")
        self._hook.mu = None if mu is None else mu.float().cpu()
        self._hook.install(self.head)
        print(f"[{self.log_name}] resumed gamma={self._hook.gamma:+.2f}", flush=True)
