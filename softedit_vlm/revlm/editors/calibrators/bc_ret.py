"""Per-edit first-token distribution with nearest-key retrieval.

For each edit, store the human first-token distribution q_e with the
prompt-end hidden state of the last language layer (the LM-head input).
At generation the prefill hidden state retrieves the closest key by minimum
L2 distance (top-1) and that edit's q replaces the prompt-end logit row:

    z'[prompt end, first_ids] = log q
    z'[prompt end, elsewhere] = large negative

Later tokens keep the frozen model's logits. Decode steps (KV cache length 1)
are left alone: the first token was already taken from the prefill row.
A full re-forward still writes the pinned prompt-end position, not the last
generated token.

Entries do not depend on later edits, so the sequential loop appends one
pair at a time and checkpoints are a cache.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from revlm.editors.soft_utils.targets import SoftTarget

# Below every log q we will write. Representable in fp16 and bf16.
_LOGIT_FLOOR = -1e4


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


def _wrap_reset(fn, hook):
    """Clear the latched entry at the start of each generation call."""
    if fn is None or getattr(fn, "_bc_ret_wrapped", False):
        return fn

    def wrapped(*args, **kwargs):
        hook.reset_latch()
        return fn(*args, **kwargs)

    wrapped._bc_ret_wrapped = True
    return wrapped


def _pack_target(target: SoftTarget) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unique first-token ids and a probability vector that sums to 1."""
    ids = target.first_ids.detach().long().cpu().view(-1)
    q = target.q.detach().float().cpu().view(-1)
    if ids.numel() == 0 or q.numel() == 0 or ids.numel() != q.numel():
        raise ValueError("soft target has no aligned first-token distribution")
    uniq, inv = torch.unique(ids, return_inverse=True)
    packed = torch.zeros(uniq.shape[0], dtype=torch.float32)
    packed.scatter_add_(0, inv, q.clamp_min(0))
    packed = packed / packed.sum().clamp_min(1e-12)
    return uniq, packed


class _RetrievalDistributionHook:
    """Replace the prompt-end logit row with the nearest edit's q."""

    def __init__(self):
        self.keys: Optional[torch.Tensor] = None
        self.ids: List[torch.Tensor] = []
        self.probs: List[torch.Tensor] = []
        self._handle = None
        self._latched_idx: Optional[torch.Tensor] = None
        self._prompt_len: Optional[int] = None
        self._last_len: Optional[int] = None

    def reset_latch(self):
        self._latched_idx = None
        self._prompt_len = None
        self._last_len = None

    def __len__(self) -> int:
        return len(self.ids)

    def _is_new_prefill(self, seq_len: int, batch: int) -> bool:
        if self._prompt_len is None or self._latched_idx is None:
            return True
        if self._latched_idx.shape[0] != batch:
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
        """Top-1 minimum-L2 neighbor. ``rows`` is [B, D]. Returns indices [B]."""
        keys = self.keys.to(device=rows.device, dtype=torch.float32)
        return torch.cdist(rows.float(), keys, p=2).argmin(dim=-1)

    def _write(self, output: torch.Tensor, pos: int, index: torch.Tensor) -> torch.Tensor:
        """Overwrite one sequence position with log q. Other positions stay."""
        out = output.clone()
        floor = out.new_full((out.shape[-1],), _LOGIT_FLOOR)
        for b, j in enumerate(index.tolist()):
            ids = self.ids[j].to(device=out.device)
            q = self.probs[j].to(device=out.device, dtype=torch.float32)
            q = q / q.sum().clamp_min(1e-12)
            row = floor.clone()
            row[ids] = q.clamp_min(1e-12).log().to(dtype=out.dtype)
            out[b, pos, :] = row
        return out

    def __call__(self, module, inputs, output):
        if self.keys is None or self.keys.numel() == 0 or not self.ids:
            return output
        if not isinstance(output, torch.Tensor):
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
            self._latched_idx = self._lookup(hidden[:, -1, :])
            self._prompt_len = seq_len
            pos = seq_len - 1
        elif seq_len == 1:
            # KV-cache decode. The first token already came from the prefill row.
            self._last_len = seq_len
            return output
        else:
            pos = self._prompt_len - 1
        self._last_len = seq_len
        return self._write(output, pos, self._latched_idx)

    def install(self, module: nn.Module):
        self.remove()
        self._handle = module.register_forward_hook(self)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class BCRetSoft(nn.Module):
    """One first-token distribution per edit, retrieved by prompt-end hidden state."""

    log_name = "bc_ret_soft"

    def __init__(self, config, model):
        super().__init__()
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None
        self.device = config.device
        self.head = _resolve_lm_head(self.model)
        self._hook = _RetrievalDistributionHook()

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
    def _forward_one(self, tokens) -> torch.Tensor:
        """Unscaled prompt-end hidden state for the current edit."""
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
                self.model(**inputs)
            pos = self._prompt_end_pos(inputs["labels"])
            hidden = captured["h"]
            if hidden.ndim == 2:
                hidden = hidden.unsqueeze(0)
            return hidden[0, pos, :].float().cpu()
        finally:
            handle.remove()

    def _append(self, hidden: torch.Tensor, ids: torch.Tensor, probs: torch.Tensor):
        row = hidden.detach().float().cpu().view(1, -1)
        if self._hook.keys is None or self._hook.keys.numel() == 0:
            self._hook.keys = row
        else:
            self._hook.keys = torch.cat([self._hook.keys, row], dim=0)
        self._hook.ids.append(ids.detach().long().cpu())
        self._hook.probs.append(probs.detach().float().cpu())

    def _self_check(self, hidden: torch.Tensor):
        """Top-1 of this edit against the codebook. Its own key should win."""
        keys = self._hook.keys
        if keys is None or keys.shape[0] == 0:
            return 0, 0.0
        dist = torch.cdist(hidden.float().view(1, -1), keys.float(), p=2).view(-1)
        top = int(dist.argmin().item())
        return top, float(dist[top].item())

    def edit(self, config, tokens, batch_history=None):
        """Store this edit's first-token distribution. No parameter fit."""
        del config, batch_history
        target = self._soft_target
        if target is None and isinstance(tokens, dict):
            target = tokens.get("_soft_target")
        labels = tokens.get("labels") if isinstance(tokens, dict) else None
        if target is None or labels is None:
            print(f"[{self.log_name}] no soft target, codebook unchanged "
                  f"(n={len(self._hook)})", flush=True)
        else:
            hidden = self._forward_one(tokens)
            ids, probs = _pack_target(target)
            self._append(hidden, ids, probs)
            top, dist = self._self_check(hidden)
            n = len(self._hook)
            own = top == n - 1
            print(f"[{self.log_name}] edit {n} K={int(ids.numel())} "
                  f"self_top={top} l2={dist:.4f} own={own}", flush=True)
        self._hook.install(self.head)
        self.model.eval()
        return self.model

    def save_resume_state(self):
        keys = None if self._hook.keys is None else self._hook.keys.detach().cpu()
        return {
            "keys": keys,
            "ids": [t.detach().cpu() for t in self._hook.ids],
            "probs": [t.detach().cpu() for t in self._hook.probs],
        }

    def load_resume_state(self, state):
        if not state:
            return
        keys = state.get("keys")
        ids = state.get("ids") or []
        probs = state.get("probs") or []
        self._hook.keys = None if keys is None else keys.float().cpu()
        self._hook.ids = [t.long().cpu() for t in ids]
        self._hook.probs = [t.float().cpu() for t in probs]
        self._hook.reset_latch()
        self._hook.install(self.head)
        print(f"[{self.log_name}] resumed codebook n={len(self._hook)}", flush=True)
