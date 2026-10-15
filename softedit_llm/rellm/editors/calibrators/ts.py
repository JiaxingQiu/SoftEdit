"""Temperature Scaling (Guo et al., ICML 2017), soft-label adaptation.

Fits one global temperature on the whole edit set:

    T* = argmin_{T>0} mean_e KL( q_e || softmax(z_e / T) )

q_e is the human vote distribution (the same soft target the ``*_soft`` editors
use). z_e is the unscaled prompt-end logit row. A forward hook then divides
every LM-head logit by T, so reliability, generality, and locality generation
all see the calibrated model. Model weights are not changed.

Hyperparameters follow the editor convention: defaults live here, and
``config.editor`` can override them. There is no calibrator YAML.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from rellm.editors.soft_utils.targets import SoftTarget


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


class _TemperatureHook:
    """Divide an nn.Linear output by T. T=1 leaves logits unchanged."""

    def __init__(self, T: float = 1.0):
        self.T = max(float(T), 1e-6)
        self._handle = None

    def __call__(self, module, inputs, output):
        return output / self.T

    def install(self, module: nn.Module):
        self.remove()
        self._handle = module.register_forward_hook(self)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class TSSoft(nn.Module):
    """Batch temperature scaling. Same editor interface as ft_retrain."""

    log_name = "ts_soft"

    def __init__(self, config, model):
        super().__init__()
        self.model = model.model if hasattr(model, "model") else model
        self.tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None
        self.device = config.device
        self.head = _resolve_lm_head(self.model)
        self._hook = _TemperatureHook(1.0)

        for p in self.model.parameters():
            p.requires_grad = False

        # Pipeline builds the human distribution q and stores it on each edit.
        self.soft = True
        self.soft_mode = "loss"
        self.rank_temp = 1.0
        self._soft_target: Optional[SoftTarget] = None
        self._soft_cont = None

        editor = getattr(config, "editor", config)
        self.n_iter = int(getattr(editor, "n_iter", 1000))
        self.edit_lr = float(getattr(editor, "edit_lr", 1e-2))
        self.early_stop_patience = int(getattr(editor, "early_stop_patience", 20))

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

    def _kl_chunk(self, logits, targets, T):
        logp = torch.log_softmax(logits / T, dim=-1)
        total = logits.new_zeros(())
        for i, tgt in enumerate(targets):
            q = tgt.q.to(logp.device)
            ids = tgt.first_ids.to(logp.device)
            lp = logp[i].index_select(0, ids)
            total = total + (q * (torch.log(q.clamp_min(1e-12)) - lp)).sum().clamp_min(0.0)
        return total

    def _fit(self, logits, targets):
        """Adam on log T. Chunked so the vocab-sized graph is one chunk at a time."""
        log_T = nn.Parameter(torch.zeros((), device=logits.device))
        opt = torch.optim.Adam([log_T], lr=self.edit_lr)
        best_loss, best_T, wait = float("inf"), 1.0, 0
        n = len(targets)
        chunk = 32

        for step in range(self.n_iter):
            opt.zero_grad(set_to_none=True)
            total = 0.0
            T_value = 1.0
            for start in range(0, n, chunk):
                sl = slice(start, start + chunk)
                # Fresh T each chunk: one shared T would free its graph on the
                # first backward and the next chunk would fail.
                T = log_T.exp().clamp_min(1e-4)
                loss = self._kl_chunk(logits[sl], targets[sl], T)
                (loss / n).backward()
                total += float(loss.detach())
                T_value = float(T.detach())
            mean = total / n
            if mean < best_loss:
                best_loss, best_T, wait = mean, T_value, 0
            else:
                wait += 1
            opt.step()
            if (step + 1) % 20 == 0 or step == 0 or wait >= self.early_stop_patience:
                print(f"[{self.log_name}] iter {step + 1}/{self.n_iter} "
                      f"T={T_value:.4f} KL={mean:.4f}", flush=True)
            if wait >= self.early_stop_patience:
                break

        print(f"[{self.log_name}] fitted T={best_T:.4f} (KL={best_loss:.4f})", flush=True)
        return best_T

    def edit(self, config, tokens, batch_history=None):
        """Fit T on every accumulated edit, then hook the LM head."""
        history = list(batch_history or []) + [tokens]
        logits, targets = self._collect(history)
        if logits is None:
            print(f"[{self.log_name}] no soft targets, keeping T=1", flush=True)
            self._hook.T = 1.0
        else:
            print(f"[{self.log_name}] fitting T on {len(targets)} edits", flush=True)
            self._hook.T = self._fit(logits.to(self.device), targets)
        self._hook.install(self.head)
        self.model.eval()
        return self.model

    def save_resume_state(self):
        return {"T": self._hook.T}

    def load_resume_state(self, state):
        if not state:
            return
        self._hook.T = float(state.get("T", 1.0))
        self._hook.install(self.head)
        print(f"[{self.log_name}] resumed T={self._hook.T:.4f}", flush=True)
