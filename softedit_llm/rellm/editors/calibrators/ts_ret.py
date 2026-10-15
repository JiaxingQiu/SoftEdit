"""Per-edit temperature scaling with nearest-key retrieval.

Global temperature scaling fits one T on the whole edit set. That average is a
poor fit when edits target different vocabulary slices. This editor fits one
temperature per edit and stores it with the prompt-end hidden state of the
last language layer (the LM-head input):

    T_e* = argmin_{T>0} KL( q_e || softmax(z_e / T) )

At generation, the prefill hidden state retrieves the closest stored key by
minimum L2 distance and that edit's T scales every logit of the sample. Decode
steps reuse the prefill temperature: with a KV cache they only see the new
token, and without one the last position is a generated token.

Temperatures do not depend on later edits, so the sequential loop appends one
pair at a time and checkpoints are a cache. Positive T does not change the
greedy token.
"""

from typing import Optional

import torch
import torch.nn as nn
from rellm.editors.soft_utils.targets import SoftTarget

_LR = 1e-2
_STEPS = 200
_PATIENCE = 20
_T_MIN = 0.1
_T_MAX = 10.0


def _resolve_lm_head(model) -> nn.Linear:
    """Locate the lm_head on the decoder-only LLMs this repo runs."""
    for path in ("lm_head", "language_model.lm_head"):
        mod = model
        for attr in path.split("."):
            mod = getattr(mod, attr, None)
            if mod is None:
                break
        if isinstance(mod, nn.Linear):
            return mod
    raise ValueError("model has no nn.Linear lm_head")


def _wrap_reset(fn, hook):
    """Clear the latched temperature at the start of each generation call."""
    if fn is None or getattr(fn, "_ts_ret_wrapped", False):
        return fn

    def wrapped(*args, **kwargs):
        hook.reset_latch()
        return fn(*args, **kwargs)

    wrapped._ts_ret_wrapped = True
    return wrapped


def _kl_one(logits: torch.Tensor, target: SoftTarget, T: torch.Tensor) -> torch.Tensor:
    """KL(q || softmax(z / T)) on one prompt-end logit row."""
    logp = torch.log_softmax(logits / T, dim=-1)
    q = target.q.to(device=logp.device, dtype=logp.dtype)
    ids = target.first_ids.to(logp.device)
    lp = logp.index_select(0, ids)
    return (q * (torch.log(q.clamp_min(1e-12)) - lp)).sum().clamp_min(0.0)


class _RetrievalTemperatureHook:
    """Divide LM-head logits by the temperature of the nearest stored edit."""

    def __init__(self):
        self.keys: Optional[torch.Tensor] = None
        self.temps: Optional[torch.Tensor] = None
        self._handle = None
        self._latched_T: Optional[torch.Tensor] = None
        self._prompt_len: Optional[int] = None
        self._last_len: Optional[int] = None

    def reset_latch(self):
        self._latched_T = None
        self._prompt_len = None
        self._last_len = None

    def _is_new_prefill(self, seq_len: int, batch: int) -> bool:
        if self._prompt_len is None or self._latched_T is None:
            return True
        if self._latched_T.shape[0] != batch:
            return True
        if seq_len == 1:
            return False
        if self._last_len == 1:
            return True
        if seq_len < self._prompt_len:
            return True
        if self._last_len is not None and seq_len > self._last_len + 1:
            return True
        return False

    def _lookup(self, rows: torch.Tensor) -> torch.Tensor:
        """Top-1 minimum-L2 neighbor. ``rows`` is [B, D]. Returns T [B]."""
        keys = self.keys.to(device=rows.device, dtype=torch.float32)
        idx = torch.cdist(rows.float(), keys, p=2).argmin(dim=-1)
        temps = self.temps.to(device=rows.device, dtype=torch.float32)
        return temps.index_select(0, idx)

    def __call__(self, module, inputs, output):
        if self.keys is None or self.keys.numel() == 0 or self.temps is None:
            return output
        hidden = inputs[0] if isinstance(inputs, tuple) else inputs
        if not isinstance(hidden, torch.Tensor):
            return output
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        if hidden.ndim != 3:
            return output
        batch, seq_len, _ = hidden.shape
        if self._is_new_prefill(seq_len, batch):
            rows = hidden[:, -1, :]
            self._latched_T = self._lookup(rows)
            self._prompt_len = seq_len
        elif self._latched_T is None:
            return output
        self._last_len = seq_len
        scale = self._latched_T.to(dtype=output.dtype).clamp_min(1e-6)
        while scale.ndim < output.ndim:
            scale = scale.unsqueeze(-1)
        return output / scale

    def install(self, module: nn.Module):
        self.remove()
        self._handle = module.register_forward_hook(self)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class TSRetSoft(nn.Module):
    """One temperature per edit, retrieved by prompt-end hidden state."""

    log_name = "ts_ret_soft"

    def __init__(self, config, model):
        super().__init__()
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None
        self.device = config.device
        self.head = _resolve_lm_head(self.model)
        self._hook = _RetrievalTemperatureHook()

        for p in self.model.parameters():
            p.requires_grad = False

        self.soft = True
        self.soft_mode = "loss"
        self.rank_temp = 1.0
        self._soft_target: Optional[SoftTarget] = None
        self._soft_cont = None

        editor = getattr(config, "editor", config)
        self.n_iter = int(getattr(editor, "n_iter", _STEPS) or _STEPS)
        self.edit_lr = float(getattr(editor, "edit_lr", _LR) or _LR)
        self.early_stop_patience = int(getattr(editor, "early_stop_patience", _PATIENCE) or _PATIENCE)

        first = next(self.model.parameters(), None)
        dtype = getattr(first, "dtype", torch.bfloat16)
        self.autocast_dtype = torch.float16 if dtype == torch.float16 else torch.bfloat16
        self._patch_generate()

    def _patch_generate(self):
        """Reset the latch on the wrapper that eval actually calls."""
        if self.wrapper is None:
            return
        for name in ("generate", "generate_scored"):
            fn = getattr(self.wrapper, name, None)
            wrapped = _wrap_reset(fn, self._hook)
            if wrapped is not None and wrapped is not fn:
                setattr(self.wrapper, name, wrapped)

    def generate(self, *args, **kwargs):
        self._hook.reset_latch()
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
    def _forward_one(self, tokens):
        """Unscaled prompt-end hidden state and logits for the current edit."""
        self.model.eval()
        self._hook.remove()
        captured = {}

        def _grab(_module, inputs, _output):
            hidden = inputs[0] if isinstance(inputs, tuple) else inputs
            captured["h"] = hidden.detach()

        handle = self.head.register_forward_hook(_grab)
        try:
            inputs = self._to_device(
                {k: v for k, v in tokens.items() if k not in ("_soft_target", "_soft_cont")})
            with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
                outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            pos = self._prompt_end_pos(inputs["labels"])
            hidden = captured["h"]
            if hidden.ndim == 2:
                hidden = hidden.unsqueeze(0)
            return hidden[0, pos, :].float().cpu(), logits[0, pos, :].float().cpu()
        finally:
            handle.remove()

    def _fit_one(self, logits: torch.Tensor, target: SoftTarget):
        """Adam on log T for a single edit. Returns (T, KL)."""
        logits = logits.to(self.device)
        log_T = nn.Parameter(torch.zeros((), device=logits.device))
        opt = torch.optim.Adam([log_T], lr=self.edit_lr)
        best_loss, best_T, wait = float("inf"), 1.0, 0
        for _step in range(self.n_iter):
            opt.zero_grad(set_to_none=True)
            T = log_T.exp().clamp(min=_T_MIN, max=_T_MAX)
            loss = _kl_one(logits, target, T)
            loss.backward()
            opt.step()
            value = float(loss.detach())
            T_value = float(T.detach())
            if value < best_loss - 1e-6:
                best_loss, best_T, wait = value, T_value, 0
            else:
                wait += 1
                if wait >= self.early_stop_patience:
                    break
        return best_T, best_loss

    def _append(self, hidden: torch.Tensor, temperature: float):
        row = hidden.detach().float().cpu().view(1, -1)
        temp = torch.tensor([float(temperature)], dtype=torch.float32)
        if self._hook.keys is None or self._hook.keys.numel() == 0:
            self._hook.keys = row
            self._hook.temps = temp
        else:
            self._hook.keys = torch.cat([self._hook.keys, row], dim=0)
            self._hook.temps = torch.cat([self._hook.temps, temp], dim=0)

    def _self_check(self, hidden: torch.Tensor):
        """Top-1 of this edit against the codebook. Its own key should win."""
        keys = self._hook.keys
        if keys is None or keys.shape[0] == 0:
            return 0, 0.0
        dist = torch.cdist(hidden.float().view(1, -1), keys.float(), p=2).view(-1)
        top = int(dist.argmin().item())
        return top, float(dist[top].item())

    def edit(self, config, tokens, batch_history=None):
        """Fit T for this edit only and append it to the codebook."""
        del config, batch_history
        target = self._soft_target
        if target is None and isinstance(tokens, dict):
            target = tokens.get("_soft_target")
        labels = tokens.get("labels") if isinstance(tokens, dict) else None
        if target is None or labels is None:
            print(f"[{self.log_name}] no soft target, codebook unchanged "
                  f"(n={0 if self._hook.temps is None else int(self._hook.temps.shape[0])})",
                  flush=True)
        else:
            hidden, logits = self._forward_one(tokens)
            temperature, kl = self._fit_one(logits, target)
            self._append(hidden, temperature)
            top, dist = self._self_check(hidden)
            n = int(self._hook.temps.shape[0])
            own = top == n - 1
            print(f"[{self.log_name}] edit {n} T={temperature:.4f} KL={kl:.4f} "
                  f"self_top={top} l2={dist:.4f} own={own}", flush=True)
        self._hook.install(self.head)
        self.model.eval()
        return self.model

    def save_resume_state(self):
        keys = None if self._hook.keys is None else self._hook.keys.detach().cpu()
        temps = None if self._hook.temps is None else self._hook.temps.detach().cpu()
        return {"keys": keys, "temps": temps}

    def load_resume_state(self, state):
        if not state:
            return
        keys = state.get("keys")
        temps = state.get("temps")
        self._hook.keys = None if keys is None else keys.float().cpu()
        self._hook.temps = None if temps is None else temps.float().cpu()
        self._hook.reset_latch()
        self._hook.install(self.head)
        n = 0 if self._hook.temps is None else int(self._hook.temps.shape[0])
        print(f"[{self.log_name}] resumed codebook n={n}", flush=True)
