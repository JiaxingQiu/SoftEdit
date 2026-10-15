import torch
from .utils import brackets_to_periods, parent_module
from .soft import compute_edit_loss


class Finetune(torch.nn.Module):
    """
    Fine-tuning editor - directly finetunes chosen weights given new inputs.
    """
    log_name = "ft"  # overridden by the factory (e.g. "ft_soft" for the soft variant)

    def __init__(self, config, model):
        super(Finetune, self).__init__()
        self.model = model.model if hasattr(model, 'model') else model
        self.tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        
        # Keep original pname for logging / compatibility
        self.pnames = [brackets_to_periods(config.inner_params[0])]
        self.device = config.device
        self.edit_lr = float(getattr(config.editor, "edit_lr", 5e-4))  # final chosen default
        
        # AMP configuration: use GradScaler only for FP16, BF16 does not need/allow it
        first_param = next(self.model.parameters(), None)
        model_dtype = getattr(first_param, 'dtype', torch.bfloat16)
        self.autocast_dtype = torch.float16 if model_dtype == torch.float16 else torch.bfloat16
        self.scaler = torch.amp.GradScaler('cuda') if self.autocast_dtype == torch.float16 else None

        # Resolve inner_params[0] to a module (so we can finetune weight + bias together)
        layer_spec = config.inner_params[0]
        suffixes = [".weight", ".bias"]
        layer = layer_spec.rsplit(".", 1)[0] if any(layer_spec.endswith(s) for s in suffixes) else layer_spec
        self.layer_path = brackets_to_periods(layer)

        edit_module = parent_module(self.model, self.layer_path)
        layer_name = layer.rsplit(".", 1)[-1]
        self.layer_module = getattr(edit_module, layer_name)
        
        # Reduce memory and ensure gradients flow with checkpointing
        # Disable KV cache for training (if model supports it)
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        # BLIP (InstructBLIP) is an encoder-decoder model that also needs enable_input_require_grads()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        # Enable gradient checkpointing if available (memory saving)
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        elif hasattr(self.model, 'enable_gradient_checkpointing'):
            self.model.enable_gradient_checkpointing()
        
        # Freeze all parameters except those in the target module (weight + bias together)
        train_params = set(self.layer_module.parameters())
        for p in self.model.parameters():
            p.requires_grad = p in train_params
        if train_params:
            print(f"Finetuning module {layer}")

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def edit(self, config, tokens, batch_history):
        self.model.train()

        params = list(self.layer_module.parameters())
        weight_decay = float(getattr(config.editor, "weight_decay", 0.0))
        opt = torch.optim.Adam(params, lr=self.edit_lr, weight_decay=weight_decay)
        n_iter = int(getattr(config.editor, "n_iter", 500))
        early_stop_patience = int(getattr(config.editor, "early_stop_patience", 50))
        # Loss floor: stop once fit "well enough"; driving loss->0 collapses generation
        # into repeated tokens. Soft (KL/continuation) uses its own floor.
        loss_floor = float(getattr(config.editor, "loss_floor", 1e-1))
        if getattr(self, "soft_mode", "target") in ("loss", "rank_loss"):
            loss_floor = float(getattr(config.editor, "soft_loss_floor", 1e-1))
        # Clamp each edited weight to w0 ± eps (L-inf ball) after every step; None disables.
        norm_constraint = getattr(config.editor, "norm_constraint", 5e-4)
        norm_constraint = float(norm_constraint) if norm_constraint not in (None, "", "null") else None

        # Cache weights at the start of THIS edit call as the reference for the clamp.
        w0 = {p: p.detach().clone() for p in params} if norm_constraint is not None else None

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_iter))
        self.losses = []

        best_loss = float('inf')
        patience_counter = 0

        for i in range(n_iter):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                outputs = self.model(**tokens)
                # CE on the target (base / soft="target") or JS on the label
                # distribution (soft="loss"/"rank_loss"); dispatched by mode.
                loss = compute_edit_loss(self, outputs, tokens)

            loss_value = loss.detach().cpu().item()
            self.losses.append(loss_value)

            # Loss floor reached: target fit well enough, stop before over-fitting.
            if loss_value < loss_floor:
                print(f"[{self.log_name}] iter {i+1}/{n_iter} - loss: {loss_value:.4f} (< floor {loss_floor:g}, stop)")
                break

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(opt)
                self.scaler.update()
            else:
                loss.backward()
                opt.step()

            scheduler.step()

            # L-inf norm constraint around the pre-edit weights.
            if norm_constraint is not None:
                with torch.no_grad():
                    for p in params:
                        p[...] = torch.clamp(p, min=w0[p] - norm_constraint, max=w0[p] + norm_constraint)

            # Secondary safety: stop if loss stops improving.
            if loss_value < best_loss:
                best_loss = loss_value
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break

            # Print loss every 10 iterations or on first/last iteration
            if (i + 1) % 10 == 0 or i == 0 or i == n_iter - 1:
                print(f"[{self.log_name}] iter {i+1}/{n_iter} - loss: {loss_value:.4f}")

        self.loss = loss if 'loss' in locals() else None
        return self.model

    def save_resume_state(self):
        """FT mutates only `self.layer_module` (weight+bias) in place; persist it."""
        return {"layer": {k: v.detach().cpu() for k, v in self.layer_module.state_dict().items()}}

    def load_resume_state(self, state):
        """Copy the edited weights back (load_state_dict handles device/dtype)."""
        self.layer_module.load_state_dict(state["layer"])


class FinetuneSoft(Finetune):
    """Multilabel FT: teacher-forces the ranked multi-label string.

    The supervision target swap is performed generically in the edit pipeline
    (edit_utils) whenever ``editor.soft`` is True; this subclass only flips the
    flag so the rest of FT behaves identically.
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        self.soft = True
        self.soft_mode = getattr(config.editor, "soft", "loss")
        self.rank_temp = float(getattr(config.editor, "rank_temp", 1.0))

