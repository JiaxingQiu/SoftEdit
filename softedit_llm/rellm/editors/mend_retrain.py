import os
import torch
import torch.nn.functional as F
from .mend import MEND, GradientTransform, get_shape
from .soft import compute_edit_loss
from .utils import get_inner_params, brackets_to_periods, hook_model


class MEND_retrain(MEND):
    """
    MEND editor that trains on ALL accumulated edits.

    Similar to ft_retrain: called once after all edits are collected,
    trains hypernetwork on all edits iteratively (multiple epochs).
    """
    log_name = "mend_retrain"

    def __init__(self, config, model, tokenizer, device, mend=None):
        super().__init__(config, model, tokenizer, device, mend)

        editor_config = getattr(config, 'editor', config)
        self.retrain_batch_size = int(getattr(editor_config, "retrain_batch_size", 10))
        self.retrain_batch_size = max(1, self.retrain_batch_size)

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
        pd = dict(self.model.named_parameters())
        names = list(self.pnames) + list(self.bias_map.values())
        weights = {n: pd[n].detach().cpu() for n in names if n in pd}
        path = os.path.join(ckpt_dir, f"train_ckpt_{epoch+1}.pt")
        tmp = path + ".tmp"
        torch.save({
            "epoch": epoch,
            "weights": weights,
            "hypernet": {k: v.detach().cpu() for k, v in self.mend.state_dict().items()},
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
        Train hypernetwork on ALL accumulated edits.

        tokens: current edit batch (not yet in batch_history)
        batch_history: list of previous edit batches (may be empty or None)
        """
        if batch_history is None:
            batch_history = []

        all_history = batch_history + [tokens]

        editor_config = getattr(config, 'editor', config)
        edit_lr = float(getattr(editor_config, 'edit_lr', 1e-2))
        min_lr = float(getattr(editor_config, 'min_lr', 1e-5))
        n_iter = int(getattr(editor_config, 'n_iter', 20))
        early_stop_patience = int(getattr(editor_config, 'early_stop_patience', 5))
        loss_floor = float(getattr(editor_config, "loss_floor", 1e-1))
        # Scale the applied rank-1 update, exactly as sequential MEND. Unscaled
        # add_() is a full-gradient step from a near-random hypernet -> it scrambles
        # the layer (model_loss > log|V|) and only "recovers" after thousands of
        # steps. Scaling keeps every step small so the loss descends instead.
        lr_scale = float(getattr(editor_config, 'lr_scale', 0.1))
        ckpt_every = int(getattr(editor_config, "ckpt_every", 1))

        n_groups = (len(all_history) + self.retrain_batch_size - 1) // self.retrain_batch_size
        print(
            f"[{self.log_name}] Training on {len(all_history)} edits "
            f"| batch_size={self.retrain_batch_size} | groups={n_groups} "
            f"| n_iter={n_iter} | lr={edit_lr:.2e}->{min_lr:.2e} "
            f"| lr_scale={lr_scale:g} | patience={early_stop_patience}",
            flush=True
        )

        opt = torch.optim.Adam(self.outer_parameters(), lr=edit_lr)

        total_steps = n_groups * n_iter
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps), eta_min=min_lr)

        self.losses = []
        best_loss = float('inf')
        patience_counter = 0
        start_epoch = 0
        param_dict = dict(self.model.named_parameters())
        # Keep the best (lowest model_loss) weights so a late diverging epoch can't
        # return a scrambled layer; restored after the loop. Snapshotted on CPU.
        edited_names = list(self.pnames) + list(self.bias_map.values())
        best_state = None

        # Resume from a mid-training checkpoint if available
        ckpt = self._load_training_ckpt(config)
        if ckpt is not None:
            start_epoch = ckpt["epoch"] + 1
            with torch.no_grad():
                for n, w in ckpt["weights"].items():
                    if n in param_dict:
                        param_dict[n].copy_(w.to(param_dict[n].device, param_dict[n].dtype))
            self.mend.load_state_dict(ckpt["hypernet"])
            opt.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            best_loss = ckpt["best_loss"]
            patience_counter = ckpt["patience_counter"]
            self.losses = ckpt["losses"]
            print(f"[{self.log_name}] resumed training from epoch {start_epoch} "
                  f"(best_loss={best_loss:.4f})", flush=True)

        for epoch in range(start_epoch, n_iter):
            epoch_loss_sum = 0.0
            epoch_hypernet_loss = 0.0

            for group_idx in range(n_groups):
                start = group_idx * self.retrain_batch_size
                end = min(len(all_history), start + self.retrain_batch_size)
                group = all_history[start:end]
                group_size = len(group)

                opt.zero_grad(set_to_none=True)
                group_loss_sum = 0.0
                group_hypernet_loss = 0.0

                for batch_tokens in group:
                    batch_tokens = self._move_to_device(batch_tokens)
                    self._soft_target = batch_tokens.get("_soft_target", None)
                    # This edit's continuation ids (tiny CPU tensors from history);
                    # continuation_nll moves them to device and reconstructs the
                    # prompt+y_k forward from model_inputs, so no shuttling needed.
                    self._soft_cont = batch_tokens.get("_soft_cont", None)
                    model_inputs = {k: v for k, v in batch_tokens.items()
                                    if k not in ("_soft_target", "_soft_cont")}

                    outputs = self.model(**model_inputs)
                    base_loss = compute_edit_loss(self, outputs, model_inputs)

                    base_loss.backward()

                    transformed_factors = {}
                    target_grads = {}
                    for n, p in get_inner_params(self.model.named_parameters(), self.pnames):
                        x = self._select_token(p.__x__)
                        delta = self._select_token(p.__delta__)
                        transformed_factors[n] = self.mend[n.replace(".", "#")](x, delta)
                        if p.grad is not None:
                            target_grads[n] = p.grad.detach().clone()

                    updates = {}
                    for n, (x_t, delta_t) in transformed_factors.items():
                        updates[n] = torch.matmul(delta_t.view(-1, 1), x_t.view(1, -1))

                    hypernet_loss = torch.tensor(0.0, device=self.device)
                    for n in updates:
                        upd = updates[n].T if self._transpose else updates[n]
                        upd = upd.to(target_grads[n].dtype)
                        target = -target_grads[n]
                        cos_sim = F.cosine_similarity(upd.view(1, -1), target.view(1, -1))
                        hypernet_loss = hypernet_loss - cos_sim
                        mag_loss = (upd.norm() - target.norm()).abs() * 0.1
                        hypernet_loss = hypernet_loss + mag_loss

                    scaled_loss = hypernet_loss / float(max(1, group_size))
                    scaled_loss.backward()

                    group_loss_sum += base_loss.detach().cpu().item()
                    group_hypernet_loss += hypernet_loss.detach().cpu().item()

                    with torch.no_grad():
                        for n in updates:
                            upd = updates[n].T if self._transpose else updates[n]
                            param_dict[n].add_((lr_scale * upd).to(param_dict[n].dtype))
                            if n in self.bias_map:
                                _, delta_t = transformed_factors[n]
                                bias_name = self.bias_map[n]
                                b_upd = delta_t.mean(dim=0) if delta_t.dim() == 2 else delta_t
                                param_dict[bias_name].add_((lr_scale * b_upd).to(param_dict[bias_name].dtype))

                    self.model.zero_grad()

                torch.nn.utils.clip_grad_norm_(self.outer_parameters(), max_norm=1.0)
                opt.step()
                scheduler.step()

                avg_group_loss = group_loss_sum / float(max(1, group_size))
                epoch_loss_sum += avg_group_loss
                epoch_hypernet_loss += group_hypernet_loss / float(max(1, group_size))

            avg_epoch_loss = epoch_loss_sum / float(max(1, n_groups))
            avg_hypernet_loss = epoch_hypernet_loss / float(max(1, n_groups))
            self.losses.append(avg_epoch_loss)

            # Track/keep the best model_loss (the CE we actually care about), not the
            # hypernet cosine-alignment proxy.
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
                best_state = {n: param_dict[n].detach().cpu().clone()
                              for n in edited_names if n in param_dict}
            else:
                patience_counter += 1

            if avg_epoch_loss < loss_floor:
                print(
                    f"[{self.log_name}] epoch {epoch+1}/{n_iter} - model_loss: {avg_epoch_loss:.4f} "
                    f"(< floor {loss_floor:g}, stop)", flush=True
                )
                break

            if patience_counter >= early_stop_patience:
                print(f"[{self.log_name}] epoch {epoch+1}/{n_iter} - early stop (model_loss not improving)", flush=True)
                break

            lr_current = opt.param_groups[0]['lr']
            print(f"[{self.log_name}] epoch {epoch+1}/{n_iter} - model_loss: {avg_epoch_loss:.4f} - hypernet_loss: {avg_hypernet_loss:.4f} - lr: {lr_current:.2e}", flush=True)

            if epoch % 10 == 0:
                torch.cuda.empty_cache()

            # Periodic training checkpoint
            if ckpt_every > 0 and (epoch + 1) % ckpt_every == 0:
                self._save_training_ckpt(config, epoch, opt, scheduler, best_loss,
                                         patience_counter, self.losses)

        # Restore the best-loss weights (a diverging tail epoch shouldn't ship).
        if best_state is not None:
            with torch.no_grad():
                for n, w in best_state.items():
                    if n in param_dict:
                        param_dict[n].copy_(w.to(param_dict[n].device, param_dict[n].dtype))

        final_model_loss = best_loss if best_loss != float('inf') else float('nan')
        print(f"[{self.log_name}] Training complete. Best model_loss: {final_model_loss:.4f}", flush=True)

        self.loss = final_model_loss
        return self.model


class MENDRetrainSoft(MEND_retrain):
    """Multilabel batch MEND: trains the hypernetwork against the ranked
    multi-label distribution.

    Inherits the full-batch retrain loop from MEND_retrain; this subclass
    only flips the soft flags so compute_edit_loss dispatches to KL/JS instead
    of CE.  Mirrors MENDSoft's relationship to MEND.
    """

    def __init__(self, config, model, tokenizer, device, mend=None):
        super().__init__(config, model, tokenizer, device, mend=mend)
        self.soft = True
        self.soft_mode = getattr(config.editor, "soft", "loss")
        self.rank_temp = float(getattr(config.editor, "rank_temp", 1.0))
