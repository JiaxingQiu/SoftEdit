"""Adaptive Temperature Scaling (Xie et al., EMNLP 2024).

Official code: github.com/johnathan-xie/adaptive-temperature-scaling
Their released config trains a transformer decoder layer on hidden states
with ``num_train_epochs=2, lr=5e-5, selective_smoothing``.

This baseline uses the same architecture (one decoder layer + linear → 1)
and the same learning rate, but trains for up to 30 epochs and keeps the
best head. It stops early when the human-vote KL does not improve for 5
epochs. The loss is that KL instead of their correctness-dependent
objective. The backbone is frozen and the head dimensions are read from
the LLM at init time.

    T(h) = clip(exp(head(h)), 0.1, 10)
    z'   = z / T(h)

A positive per-position T does not change the greedy token.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rellm.editors.soft_utils.targets import SoftTarget

_LR = 5e-5
_EPOCHS = 30
_PATIENCE = 5
_CHUNK = 32
_T_MIN = 0.1
_T_MAX = 10.0


def _resolve_lm_head(model) -> nn.Linear:
    for path in ("lm_head", "language_model.lm_head"):
        mod = model
        for attr in path.split("."):
            mod = getattr(mod, attr, None)
            if mod is None:
                break
        if isinstance(mod, nn.Linear):
            return mod
    raise ValueError("model has no nn.Linear lm_head")


# ------------------------------------------------------------------
# Transformer calibration head (matches their TransformerHead)
# ------------------------------------------------------------------

class _RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        # Compute in fp32, then return the incoming dtype. Casting back to
        # ``x.dtype`` after ``x.float()`` leaves the residual in fp32, and the
        # next bf16 linear layer rejects it.
        dtype = x.dtype
        x = x.float()
        w = self.weight.float()
        y = w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.to(dtype)


class _MLP(nn.Module):
    def __init__(self, d, intermediate):
        super().__init__()
        self.gate = nn.Linear(d, intermediate, bias=False)
        self.up   = nn.Linear(d, intermediate, bias=False)
        self.down = nn.Linear(intermediate, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _SelfAttn(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.n_kv = n_kv_heads
        self.head_dim = d // n_heads
        self.kv_groups = n_heads // n_kv_heads
        self.q = nn.Linear(d, n_heads * self.head_dim, bias=False)
        self.k = nn.Linear(d, n_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(d, n_kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(n_heads * self.head_dim, d, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        hd = self.head_dim
        qr = self.q(x).view(B, L, self.n_heads, hd).transpose(1, 2)
        kr = self.k(x).view(B, L, self.n_kv, hd).transpose(1, 2)
        vr = self.v(x).view(B, L, self.n_kv, hd).transpose(1, 2)
        if self.kv_groups > 1:
            kr = kr.repeat_interleave(self.kv_groups, dim=1)
            vr = vr.repeat_interleave(self.kv_groups, dim=1)
        a = F.scaled_dot_product_attention(qr, kr, vr, is_causal=True)
        return self.o(a.transpose(1, 2).reshape(B, L, -1))


class _DecoderLayer(nn.Module):
    def __init__(self, d, intermediate, n_heads, n_kv_heads):
        super().__init__()
        self.attn = _SelfAttn(d, n_heads, n_kv_heads)
        self.mlp = _MLP(d, intermediate)
        self.norm1 = _RMSNorm(d)
        self.norm2 = _RMSNorm(d)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _TransformerTemperature(nn.Module):
    """One decoder layer → linear → 1 scalar per position."""

    def __init__(self, d, intermediate, n_heads, n_kv_heads):
        super().__init__()
        self.layer = _DecoderLayer(d, intermediate, n_heads, n_kv_heads)
        self.proj = nn.Linear(d, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def temperature(self, hidden: torch.Tensor) -> torch.Tensor:
        # Generation feeds fp32 hidden states into a bf16 head. Match the head.
        weight = self.proj.weight
        h = hidden.to(device=weight.device, dtype=weight.dtype)
        out = self.layer(h.unsqueeze(0) if h.ndim == 2 else h)
        raw = self.proj(out).squeeze(-1).squeeze(0)
        return raw.float().exp().clamp(min=_T_MIN, max=_T_MAX)

    def scale_logits(self, hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        temp = self.temperature(hidden)
        while temp.ndim < logits.ndim:
            temp = temp.unsqueeze(-1)
        return logits / temp.to(dtype=logits.dtype)


# ------------------------------------------------------------------
# Hook
# ------------------------------------------------------------------

class _ATSHook:
    def __init__(self, head):
        self.head = head
        self._handle = None

    def __call__(self, module, inputs, output):
        hidden = inputs[0] if isinstance(inputs, tuple) else inputs
        return self.head.scale_logits(hidden, output)

    def install(self, module: nn.Module):
        self.remove()
        self.head.to(device=module.weight.device, dtype=module.weight.dtype)
        self._handle = module.register_forward_hook(self)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ------------------------------------------------------------------
# Resolve transformer dimensions from the LLM config
# ------------------------------------------------------------------

def _head_dims(model):
    """Read hidden_size, intermediate_size, n_heads, n_kv_heads from the LLM."""
    cfg = getattr(model, "config", None)
    lang = getattr(model, "language_model", None)
    if lang is not None:
        cfg = getattr(lang, "config", cfg)
    text_cfg = getattr(cfg, "text_config", cfg)

    d = getattr(text_cfg, "hidden_size", None)
    inter = getattr(text_cfg, "intermediate_size", None)
    n_heads = getattr(text_cfg, "num_attention_heads", None)
    n_kv = getattr(text_cfg, "num_key_value_heads", n_heads)

    if d is None:
        raise ValueError("cannot read hidden_size from model config")
    if inter is None:
        inter = 4 * d
    if n_heads is None:
        n_heads = max(d // 128, 1)
    return int(d), int(inter), int(n_heads), int(n_kv)


# ------------------------------------------------------------------
# Editor
# ------------------------------------------------------------------

class ATSSoft(nn.Module):
    """Batch adaptive temperature scaling. Same editor interface as ft_retrain."""

    log_name = "ats_soft"

    def __init__(self, config, model):
        super().__init__()
        self.model = model.model if hasattr(model, "model") else model
        self.tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None
        self.device = config.device
        self.lm_head = _resolve_lm_head(self.model)

        d, inter, n_heads, n_kv = _head_dims(self.model)
        self.temp_head = _TransformerTemperature(d, inter, n_heads, n_kv)
        self._hook = _ATSHook(self.temp_head)

        editor = getattr(config, "editor", config)
        self.lr = float(getattr(editor, "edit_lr", _LR))
        self.epochs = int(getattr(editor, "n_iter", _EPOCHS))

        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.temp_head.parameters():
            p.requires_grad = True

        self.soft = True
        self.soft_mode = "loss"
        self.rank_temp = 1.0
        self._soft_target: Optional[SoftTarget] = None
        self._soft_cont = None

        first = next(self.model.parameters(), None)
        dtype = getattr(first, "dtype", torch.bfloat16)
        self.autocast_dtype = torch.float16 if dtype == torch.float16 else torch.bfloat16

        n_params = sum(p.numel() for p in self.temp_head.parameters())
        print(f"[{self.log_name}] head: d={d} inter={inter} heads={n_heads} "
              f"kv={n_kv} params={n_params:,}", flush=True)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    @staticmethod
    def _prompt_end_pos(labels: torch.Tensor) -> int:
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
        """Prompt-end hidden states, logits, and human targets."""
        self.model.eval()
        self._hook.remove()
        hidden_rows, logit_rows = [], []
        targets: List[SoftTarget] = []
        captured = {}

        def _grab(_module, inputs, _output):
            h = inputs[0] if isinstance(inputs, tuple) else inputs
            captured["h"] = h.detach()

        handle = self.lm_head.register_forward_hook(_grab)
        try:
            for batch in history:
                target = batch.get("_soft_target")
                labels = batch.get("labels")
                if target is None or labels is None:
                    continue
                inputs = self._to_device(
                    {k: v for k, v in batch.items()
                     if k not in ("_soft_target", "_soft_cont")})
                with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
                    outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                pos = self._prompt_end_pos(inputs["labels"])
                hidden_rows.append(captured["h"][0, pos, :].float().cpu())
                logit_rows.append(logits[0, pos, :].float().cpu())
                targets.append(target)
        finally:
            handle.remove()
        if not logit_rows:
            return None, None, None
        return torch.stack(hidden_rows), torch.stack(logit_rows), targets

    def _kl(self, logp, targets) -> torch.Tensor:
        total = logp.new_zeros(())
        for i, tgt in enumerate(targets):
            q = tgt.q.to(logp.device)
            ids = tgt.first_ids.to(logp.device)
            lp = logp[i].index_select(0, ids)
            total = total + (q * (torch.log(q.clamp_min(1e-12)) - lp)).sum().clamp_min(0.0)
        return total

    def _fit(self, hidden, logits, targets):
        self.temp_head.to(self.device)
        self.temp_head.train()
        opt = torch.optim.AdamW(self.temp_head.parameters(), lr=self.lr)
        n = len(targets)
        total_steps = max(1, self.epochs * ((n + _CHUNK - 1) // _CHUNK))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
        best_loss = float("inf")
        best_state = None
        wait = 0
        for epoch in range(self.epochs):
            total = 0.0
            for start in range(0, n, _CHUNK):
                sl = slice(start, min(start + _CHUNK, n))
                batch_targets = targets[sl]
                opt.zero_grad(set_to_none=True)
                h = hidden[sl].to(self.device)
                z = logits[sl].to(self.device)
                temp = self.temp_head.temperature(h).unsqueeze(-1)
                logp = torch.log_softmax(z / temp, dim=-1)
                loss = self._kl(logp, batch_targets) / len(batch_targets)
                loss.backward()
                opt.step()
                sched.step()
                total += float(loss.detach()) * len(batch_targets)
            mean = total / n
            print(f"[{self.log_name}] epoch {epoch + 1}/{self.epochs} "
                  f"KL={mean:.4f}", flush=True)
            if mean < best_loss:
                best_loss = mean
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.temp_head.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= _PATIENCE:
                    print(f"[{self.log_name}] KL flat for {_PATIENCE} epochs, "
                          f"stopping", flush=True)
                    break
        if best_state is not None:
            self.temp_head.load_state_dict(best_state)
            print(f"[{self.log_name}] restored best KL={best_loss:.4f}", flush=True)
        self.temp_head.eval()

    def edit(self, config, tokens, batch_history=None):
        history = list(batch_history or []) + [tokens]
        hidden, logits, targets = self._collect(history)
        if hidden is None:
            print(f"[{self.log_name}] no soft targets, keeping T=1", flush=True)
        else:
            print(f"[{self.log_name}] fitting on {len(targets)} edits", flush=True)
            self._fit(hidden, logits, targets)
        self._hook.install(self.lm_head)
        self.model.eval()
        return self.model

    def save_resume_state(self):
        return {"head": {k: v.detach().cpu()
                         for k, v in self.temp_head.state_dict().items()}}

    def load_resume_state(self, state):
        if not state or "head" not in state:
            return
        self.temp_head.load_state_dict(state["head"])
        self.temp_head.eval()
        self._hook.install(self.lm_head)
        print(f"[{self.log_name}] resumed transformer head", flush=True)
