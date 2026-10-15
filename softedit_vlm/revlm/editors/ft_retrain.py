import os
import torch
from .utils import brackets_to_periods, parent_module
from .soft import compute_edit_loss


class Finetune_retrain(torch.nn.Module):
    """
    Fine-tuning editor that trains on ALL accumulated edits.
    
    For each new edit, continues from current weights and trains on all
    accumulated edits (including the new one). Incremental fine-tuning.
    """
    log_name = "ft_retrain"

    def __init__(self, config, model):
        torch.nn.Module.__init__(self)
        self.model = model.model if hasattr(model, 'model') else model
        self.tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        
        self.pnames = [brackets_to_periods(config.inner_params[0])]
        self.device = config.device
        editor_config = getattr(config, 'editor', config)
        self.edit_lr = float(getattr(editor_config, 'edit_lr', 1e-2))
        
        # AMP configuration
        first_param = next(self.model.parameters(), None)
        model_dtype = getattr(first_param, 'dtype', torch.bfloat16)
        self.autocast_dtype = torch.float16 if model_dtype == torch.float16 else torch.bfloat16
        self.scaler = torch.amp.GradScaler('cuda') if self.autocast_dtype == torch.float16 else None

        # Resolve inner_params[0] to a module (finetune weight + bias together)
        layer_spec = config.inner_params[0]
        suffixes = [".weight", ".bias"]
        layer = layer_spec.rsplit(".", 1)[0] if any(layer_spec.endswith(s) for s in suffixes) else layer_spec
        self.layer_path = brackets_to_periods(layer)

        edit_module = parent_module(self.model, self.layer_path)
        layer_name = layer.rsplit(".", 1)[-1]
        self.layer_module = getattr(edit_module, layer_name)
        
        # Disable KV cache for training
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        # Enable gradient checkpointing
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        elif hasattr(self.model, 'enable_gradient_checkpointing'):
            self.model.enable_gradient_checkpointing()
        
        # Freeze all parameters except target module
        train_params = set(self.layer_module.parameters())
        for p in self.model.parameters():
            p.requires_grad = p in train_params
        if train_params:
            print(f"Finetuning module {layer} (incremental retrain on all edits)", flush=True)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def _move_to_device(self, batch_tokens):
        """Move a CPU-resident history item to the training device."""
        return {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch_tokens.items()}

    def _training_ckpt_dir(self, config):
        edit_dir = getattr(config, "edit_dir", None)
        if not edit_dir:
            return None
        return os.path.join(edit_dir, "_resume")

    def _save_training_ckpt(self, config, epoch, opt, scheduler, best_loss,
                            patience_counter, losses):
        ckpt_dir = self._training_ckpt_dir(config)
        if ckpt_dir is None:
            return
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"train_ckpt_{epoch+1}.pt")
        tmp = path + ".tmp"
        torch.save({
            "epoch": epoch,
            "layer": {k: v.detach().cpu() for k, v in self.layer_module.state_dict().items()},
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
            "patience_counter": patience_counter,
            "losses": losses,
        }, tmp)
        os.replace(tmp, path)
        print(f"[{self.log_name}] training ckpt saved (epoch {epoch+1})", flush=True)

    def _load_training_ckpt(self, config):
        """Load the latest training checkpoint (highest epoch number)."""
        ckpt_dir = self._training_ckpt_dir(config)
        if ckpt_dir is None or not os.path.isdir(ckpt_dir):
            return None
        ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("train_ckpt_") and f.endswith(".pt")]
        if not ckpts:
            return None
        # overwrite=True forces a fresh retrain: wipe stale ckpts so a changed
        # n_iter (or edit set) can't resume past the new budget and skip training.
        if getattr(config, "overwrite", False):
            for f in ckpts:
                os.remove(os.path.join(ckpt_dir, f))
            return None
        latest = max(ckpts, key=lambda f: int(f.split("_")[-1].split(".")[0]))
        path = os.path.join(ckpt_dir, latest)
        return torch.load(path, map_location="cpu", weights_only=False)

    def edit(self, config, tokens, batch_history=None):
        """
        Edit by training on ALL accumulated edits (continues from current weights).
        
        tokens: current edit batch (not yet in batch_history)
        batch_history: list of previous edit batches (may be empty or None)
        """
        if batch_history is None:
            batch_history = []
        
        all_history = batch_history + [tokens]
        
        self.model.train()
        
        params = list(self.layer_module.parameters())
        opt = torch.optim.Adam(params, lr=self.edit_lr)
        editor_config = getattr(config, 'editor', config)
        n_iter = int(getattr(editor_config, 'n_iter', 20))
        early_stop_patience = int(getattr(editor_config, 'early_stop_patience', 10))
        loss_floor = float(getattr(editor_config, "loss_floor", 1e-1))
        ckpt_every = int(getattr(editor_config, "ckpt_every", 1))
        
        retrain_batch_size = int(getattr(editor_config, "retrain_batch_size", 10))
        retrain_batch_size = max(1, retrain_batch_size)
        n_groups = (len(all_history) + retrain_batch_size - 1) // retrain_batch_size
        min_lr = float(getattr(editor_config, "min_lr", 1e-5))
        total_steps = n_groups * n_iter
        print(
            f"[{self.log_name}] Retraining on {len(all_history)} edits (including current) "
            f"| retrain_batch_size={retrain_batch_size} | groups={n_groups} "
            f"| n_iter={n_iter} | lr={self.edit_lr:.2e}->{min_lr:.2e} | patience={early_stop_patience}",
            flush=True
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, total_steps), eta_min=min_lr)
        
        self.losses = []
        best_loss = float("inf")
        patience_counter = 0
        start_epoch = 0

        # Resume from a mid-training checkpoint if available
        ckpt = self._load_training_ckpt(config)
        if ckpt is not None:
            start_epoch = ckpt["epoch"] + 1
            self.layer_module.load_state_dict(ckpt["layer"])
            opt.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            best_loss = ckpt["best_loss"]
            patience_counter = ckpt["patience_counter"]
            self.losses = ckpt["losses"]
            print(f"[{self.log_name}] resumed training from epoch {start_epoch} "
                  f"(best_loss={best_loss:.4f})", flush=True)

        for epoch in range(start_epoch, n_iter):
            epoch_loss_sum = 0.0

            for group_idx in range(n_groups):
                start = group_idx * retrain_batch_size
                end = min(len(all_history), start + retrain_batch_size)
                group = all_history[start:end]
                group_size = len(group)

                opt.zero_grad(set_to_none=True)
                group_loss_sum = 0.0

                for batch_tokens in group:
                    batch_tokens = self._move_to_device(batch_tokens)
                    self._soft_target = batch_tokens.get("_soft_target", None)
                    # This edit's continuation ids (tiny CPU tensors from history);
                    # continuation_nll moves them to device and reconstructs the
                    # prompt+y_k forward from model_inputs, so no shuttling needed.
                    self._soft_cont = batch_tokens.get("_soft_cont", None)
                    model_inputs = {k: v for k, v in batch_tokens.items()
                                    if k not in ("_soft_target", "_soft_cont")}
                    with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
                        outputs = self.model(**model_inputs)
                        loss = compute_edit_loss(self, outputs, model_inputs)

                    group_loss_sum += loss.detach().cpu().item()

                    scaled_loss = loss / float(max(1, group_size))
                    if self.scaler is not None:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                if self.scaler is not None:
                    self.scaler.step(opt)
                    self.scaler.update()
                else:
                    opt.step()

                scheduler.step()

                avg_group_loss = group_loss_sum / float(max(1, group_size))
                epoch_loss_sum += avg_group_loss

            avg_epoch_loss = epoch_loss_sum / float(max(1, n_groups))
            self.losses.append(avg_epoch_loss)

            if avg_epoch_loss < loss_floor:
                print(
                    f"[{self.log_name}] epoch {epoch+1}/{n_iter} - avg_loss: {avg_epoch_loss:.4f} "
                    f"(< floor {loss_floor:g}, stop)", flush=True
                )
                break

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"[{self.log_name}] epoch {epoch+1}/{n_iter} - early stop (patience)", flush=True)
                    break

            current_lr = opt.param_groups[0]["lr"]
            print(
                f"[{self.log_name}] epoch {epoch+1}/{n_iter} - avg_loss: {avg_epoch_loss:.4f} - lr: {current_lr:.2e}",
                flush=True
            )

            # Periodic training checkpoint
            if ckpt_every > 0 and (epoch + 1) % ckpt_every == 0:
                self._save_training_ckpt(config, epoch, opt, scheduler, best_loss,
                                         patience_counter, self.losses)
        
        self.loss = avg_epoch_loss if "avg_epoch_loss" in locals() else (loss if "loss" in locals() else None)
        return self.model

    def save_resume_state(self):
        """Persist the edited layer weights for sequential resume / weight caching."""
        return {"layer": {k: v.detach().cpu() for k, v in self.layer_module.state_dict().items()}}

    def load_resume_state(self, state):
        """Restore edited weights."""
        self.layer_module.load_state_dict(state["layer"])


class FinetuneRetrainSoft(Finetune_retrain):
    """Multilabel batch FT: trains on the ranked multi-label distribution.

    Inherits the full-batch retrain loop from Finetune_retrain; this subclass
    only flips the soft flags so compute_edit_loss dispatches to KL/JS instead
    of CE.  Mirrors FinetuneSoft's relationship to Finetune.
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        self.soft = True
        self.soft_mode = getattr(config.editor, "soft", "loss")
        self.rank_temp = float(getattr(config.editor, "rank_temp", 1.0))
