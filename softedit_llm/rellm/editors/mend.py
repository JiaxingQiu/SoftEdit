import torch
from .utils import get_inner_params, brackets_to_periods, hook_model
from .soft import compute_edit_loss
import transformers
import torch.nn.functional as F


class GradientTransform(torch.nn.Module):
    """Transforms gradients for MEND (same backbone as GRACE, lightly adapted)."""

    def __init__(self, x_dim, delta_dim):
        super(GradientTransform, self).__init__()
        self.mlp1 = torch.nn.Linear(x_dim, x_dim)
        self.mlp2 = torch.nn.Linear(delta_dim, delta_dim)

    def forward(self, x, delta):
        # If we've got grads for each token, just grab the last representation
        if len(x.shape) == 3:
            x = x[:, -1, :]
            delta = delta[:, -1, :]

        # Ensure dtypes match Linear weights (handles bf16 / fp16 backbones)
        x = x.to(self.mlp1.weight.dtype)
        delta = delta.to(self.mlp2.weight.dtype)

        return self.mlp1(x), self.mlp2(delta)


def get_shape(p, model):
    """Get shape for gradient transform (GRACE-style logic)."""
    if isinstance(model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
        return p.shape
    # Generic linear layer: weight [out, in] → (in, out)
    return (p.shape[1], p.shape[0])


class MEND(torch.nn.Module):
    """MEND: Model Editing Networks using Gradient Decomposition (GRACE version online editing)."""

    log_name = "mend"  # overridden by the factory (e.g. "mend_soft")

    def __init__(self, config, model, tokenizer, device, mend=None):
        super().__init__()

        # Unwrap the model wrapper (e.g., TQAModel) to get underlying HF model
        if mend is None:
            self.model = model.model if hasattr(model, "model") else model
        else:
            self.model = model

        self.tokenizer = tokenizer
        self.device = device
        self.config = config

        # Memory-focused tweaks that do NOT change the editing algorithm:
        # - disable KV cache during editing (top-level + any nested LM)
        # - enable gradient checkpointing when available
        # - enable input gradients if the model supports it
        def _enable_memory_savers(mod):
            if mod is None:
                return
            if hasattr(mod, "config") and hasattr(mod.config, "use_cache"):
                mod.config.use_cache = False
            if hasattr(mod, "enable_input_require_grads"):
                try:
                    mod.enable_input_require_grads()
                except Exception:
                    pass
            if hasattr(mod, "gradient_checkpointing_enable"):
                try:
                    mod.gradient_checkpointing_enable()
                except Exception:
                    pass
            elif hasattr(mod, "enable_gradient_checkpointing"):
                try:
                    mod.enable_gradient_checkpointing()
                except Exception:
                    pass

        core_model = self.model
        _enable_memory_savers(core_model)
        for child_name in ("language_model", "vision_model", "qformer"):
            _enable_memory_savers(getattr(core_model, child_name, None))

        # Use rellm NestedConfig: inner_params is already flattened
        params_dict = dict(self.model.named_parameters())
        self.bias_map = {}
        self.pnames = []
        for inner in config.inner_params:
            pname = brackets_to_periods(inner)
            self.pnames.append(pname)
            if pname.endswith(".weight"):
                bias_name = pname[:-7] + ".bias"
                if bias_name in params_dict:
                    self.bias_map[pname] = bias_name

        # Freeze the backbone except the edited inner params (same as FT).
        # Otherwise base_loss.backward() materializes grads for the full model
        # and Adam cannot allocate hypernet state on a single GPU.
        train_params = {params_dict[n] for n in self.pnames if n in params_dict}
        train_params.update(params_dict[n] for n in self.bias_map.values() if n in params_dict)
        for p in self.model.parameters():
            p.requires_grad = p in train_params

        # Install hooks that populate p.weight.__x__ and p.weight.__delta__
        hook_model(self.model, self.pnames)

        # GPT-2 uses transposed convention; others use standard [out, in]
        if not isinstance(
            self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel
        ):
            transpose = False
        else:
            transpose = True
        self._transpose = transpose

        model_dtype = next(self.model.parameters()).dtype

        # Build (or re-use) the GradientTransform hypernets
        if mend is None:
            self.mend = torch.nn.ModuleDict({})
            for n, p in get_inner_params(self.model.named_parameters(), self.pnames):
                shape = get_shape(p, self.model)
                # Generic case: x_dim=in, delta_dim=out. Match backbone
                # dtype so the square Linear(out, out) is not a fp32 0.5GB slab.
                self.mend[n.replace(".", "#")] = GradientTransform(
                    shape[0], shape[1]
                ).to(device=device, dtype=model_dtype)
        else:
            self.mend = mend

        self.loss = None
        self.losses = []
        self._key_idx = -1

    def outer_parameters(self):
        return list(self.mend.parameters())

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def get_model_loss(self, model, logits, batch):
        # Optional hook for custom loss (kept from GRACE code)
        if hasattr(model, "get_loss"):
            return model.get_loss(logits, batch)
        if hasattr(model, "model") and hasattr(model.model, "get_loss"):
            return model.model.get_loss(logits, batch)
        return None

    def edit(self, config, tokens, batch_history):
        """
        Memory-efficient MEND training for large models.

        Uses gradient matching: train hypernetwork to produce updates that
        align with the direction that reduces loss on the edit example.
        """
        del batch_history  # not used in this simple variant

        editor_config = getattr(config, 'editor', config)
        opt = torch.optim.Adam(self.outer_parameters(), lr=float(getattr(editor_config, 'edit_lr', 2e-3)))
        n_iter = int(getattr(editor_config, 'n_iter', 500))
        early_stop_patience = int(getattr(editor_config, 'early_stop_patience', 50))
        # Scale each applied rank-1 update (unscaled add_() is a full grad step -> weights explode).
        lr_scale = float(getattr(editor_config, 'lr_scale', 0.03))
        # Loss floor: stop once fit "well enough"; driving loss->0 collapses generation
        # into repeated tokens. Soft (KL/continuation) uses its own floor.
        loss_floor = float(getattr(editor_config, 'loss_floor', 1e-1))
        if getattr(self, "soft_mode", "target") in ("loss", "rank_loss"):
            loss_floor = float(getattr(editor_config, 'soft_loss_floor', 1e-1))
        # L-inf clamp on cumulative drift of edited weights from pre-edit values; None disables.
        norm_constraint = getattr(editor_config, 'norm_constraint', 1e-3)
        norm_constraint = float(norm_constraint) if norm_constraint not in (None, "", "null") else None

        self.losses = []
        self._key_idx = self._compute_key_idx(tokens)

        best_loss = float('inf')
        patience_counter = 0

        # Cache pre-edit weights (and biases we touch) as the clamp reference.
        w0 = None
        if norm_constraint is not None:
            pd0 = dict(self.model.named_parameters())
            names0 = list(self.pnames) + list(self.bias_map.values())
            w0 = {n: pd0[n].detach().clone() for n in names0 if n in pd0}

        for i in range(n_iter):
            opt.zero_grad()

            # Step 1: Forward + backward to populate hooks and get base gradients.
            # base_loss is CE (base / soft="target") or JS on the label
            # distribution (soft="loss"/"rank_loss"); the hypernetwork then learns
            # updates that reduce whichever objective is active.
            outputs = self.model(**tokens)
            base_loss = compute_edit_loss(self, outputs, tokens)

            # Loss floor: the current (already-updated) weights fit the target well
            # enough -> stop before accumulating further destructive updates.
            base_loss_value = base_loss.detach().cpu().item()
            if base_loss_value < loss_floor:
                print(f"[{self.log_name}] iter {i+1}/{n_iter} - loss: {base_loss_value:.4f} (< floor {loss_floor:g}, stop)")
                break

            base_loss.backward()  # Populates __x__, __delta__ via hooks

            # Step 2: Transform gradients with hypernetwork (KEEPS GRADIENT FLOW!)
            transformed_factors = {}
            target_grads = {}  # Store the actual gradients as targets
            for n, p in get_inner_params(self.model.named_parameters(), self.pnames):
                x = self._select_token(p.__x__)
                delta = self._select_token(p.__delta__)
                transformed_factors[n] = self.mend[n.replace(".", "#")](x, delta)
                # Target: the actual gradient scaled for this loss
                if p.grad is not None:
                    target_grads[n] = p.grad.detach().clone()

            # Drop backbone activations / grads before Adam allocates state.
            del outputs, base_loss
            self.model.zero_grad(set_to_none=True)

            # Step 3: Build updates from hypernetwork output
            updates = {}
            for n, (x_t, delta_t) in transformed_factors.items():
                updates[n] = torch.matmul(delta_t.view(-1, 1), x_t.view(1, -1))

            # Step 4: Compute hypernetwork loss - match transformed update to target direction
            # This trains the hypernetwork to produce updates aligned with loss reduction
            hypernet_loss = torch.tensor(0.0, device=self.device)
            for n in updates:
                upd = updates[n].T if self._transpose else updates[n]
                upd = upd.to(target_grads[n].dtype)
                # Negative cosine similarity: we want update to align with negative gradient
                # (gradient points uphill, we want to go downhill)
                target = -target_grads[n]  # Negative gradient = direction of steepest descent
                cos_sim = F.cosine_similarity(upd.view(1, -1), target.view(1, -1))
                hypernet_loss = hypernet_loss - cos_sim  # Minimize negative cosine = maximize alignment

                # Also add magnitude matching term
                mag_loss = (upd.norm() - target.norm()).abs() * 0.1
                hypernet_loss = hypernet_loss + mag_loss

            # Step 5: Backprop to train hypernetwork
            hypernet_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.outer_parameters(), max_norm=1.0)
            torch.cuda.empty_cache()
            opt.step()

            # Step 6: Apply updates to model weights (no grad needed).
            # Cache the touched params first so a non-improving step can be reverted
            # (greedy accept/reject) -> the online hypernet's noisy early steps and
            # near-minimum overshoot can't push the edit loss back up.
            self.model.zero_grad()
            param_dict = dict(self.model.named_parameters())
            touched = list(updates) + [self.bias_map[n] for n in updates if n in self.bias_map]
            pre_step = {n: param_dict[n].detach().clone() for n in touched}
            with torch.no_grad():
                for n in updates:
                    upd = updates[n].T if self._transpose else updates[n]
                    param_dict[n].add_((lr_scale * upd).to(param_dict[n].dtype))
                    # Handle bias if present
                    if n in self.bias_map:
                        bias_name = self.bias_map[n]
                        _, delta_t = transformed_factors[n]
                        b_upd = delta_t.mean(dim=0) if delta_t.dim() == 2 else delta_t
                        param_dict[bias_name].add_((lr_scale * b_upd).to(param_dict[bias_name].dtype))
                # L-inf clamp: keep cumulative drift within w0 +/- norm_constraint.
                if norm_constraint is not None:
                    for n, w in w0.items():
                        param_dict[n][...] = torch.clamp(
                            param_dict[n], min=w - norm_constraint, max=w + norm_constraint)

            # Step 7: Evaluate post-edit loss for logging/early stopping
            with torch.no_grad():
                outputs = self.model(**tokens)
                post_loss = compute_edit_loss(self, outputs, tokens)

            post_value = post_loss.cpu().item() if post_loss is not None else base_loss_value
            # Greedy accept/reject: revert the weight step if it did not reduce the loss.
            # (The hypernet still took its opt.step() above, so it keeps improving and a
            # later step can be accepted -> reliable, monotonic edit-loss descent.)
            if post_value > base_loss_value:
                with torch.no_grad():
                    for n in touched:
                        param_dict[n].copy_(pre_step[n])
                loss_value = base_loss_value
            else:
                loss_value = post_value
            self.losses.append(loss_value)
            self.loss = post_loss.detach() if (post_loss is not None and post_value <= base_loss_value) else None

            # Clear CUDA cache periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()

            # Early stopping
            if loss_value < best_loss:
                best_loss = loss_value
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break

            # Print loss
            if (i + 1) % 10 == 0 or i == 0 or i == n_iter - 1:
                print(f"[{self.log_name}] iter {i+1}/{n_iter} - loss: {loss_value:.4f}")

        return self.model

    def save_resume_state(self):
        """MEND mutates the inner-param weights (+ optional biases) in place and
        trains the hypernetwork `self.mend` across edits; persist both."""
        pd = dict(self.model.named_parameters())
        names = list(self.pnames) + list(self.bias_map.values())
        weights = {n: pd[n].detach().cpu() for n in names if n in pd}
        return {
            "weights": weights,
            "hypernet": {k: v.detach().cpu() for k, v in self.mend.state_dict().items()},
        }

    def load_resume_state(self, state):
        pd = dict(self.model.named_parameters())
        with torch.no_grad():
            for n, w in state["weights"].items():
                if n in pd:
                    pd[n].copy_(w.to(pd[n].device, pd[n].dtype))
        self.mend.load_state_dict(state["hypernet"])

    def _select_token(self, tensor):
        if tensor.dim() == 3:
            idx = self._key_idx if self._key_idx is not None and self._key_idx >= 0 else tensor.shape[1] - 1
            idx = min(idx, tensor.shape[1] - 1)
            tensor = tensor[:, idx, :]
        return tensor

    def _compute_key_idx(self, tokens):
        labels = tokens.get("labels")
        if labels is None:
            return -1
        if labels.dim() == 1:
            non_masked = (labels != -100)
            if non_masked.any():
                return non_masked.nonzero().max().item()
            return labels.numel() - 1
        non_masked = (labels != -100)
        if non_masked.any():
            return non_masked.sum(dim=1).max().item() - 1
        return labels.shape[1] - 1


class MENDSoft(MEND):
    """Multilabel MEND: the hypernetwork is trained against the ranked
    multi-label string. The supervision target swap happens generically in
    edit_utils when ``editor.soft`` is True; this subclass only flips the flag.
    """

    def __init__(self, config, model, tokenizer, device, mend=None):
        super().__init__(config, model, tokenizer, device, mend=mend)
        self.soft = True
        self.soft_mode = getattr(config.editor, "soft", "loss")
        self.rank_temp = float(getattr(config.editor, "rank_temp", 1.0))
