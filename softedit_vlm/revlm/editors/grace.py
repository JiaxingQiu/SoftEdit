import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from .utils import parent_module, brackets_to_periods
from .soft import compute_edit_loss
import transformers


def perturb_values(chosen_value, num_pert, device):
    """Add noise to values for adversarial training (matching original GRACE)."""
    noise = torch.normal(0, 1, chosen_value.shape, device=device)
    noise[0] = noise[0] * 0
    noise.requires_grad = True
    return chosen_value + noise


def _mmd(query, key):
    """Maximum Mean Discrepancy distance (fp32: cdist has no bf16 CUDA kernel)."""
    key, query = key.float(), query.float()
    kdist = torch.exp(-torch.cdist(key, key)).mean(-1).mean(-1)
    qdist = torch.exp(-torch.cdist(query, query)).mean(-1).mean(-1)
    kqdist = torch.exp(-torch.cdist(query, key)).mean(-1).mean(-1)
    return kdist + qdist - kqdist


def _cos(query, key, eps=1e-8):
    """Cosine distance"""
    if len(key.shape) < 2:
        key = key.view(1, -1)
    query_n, key_n = query.norm(dim=1)[:, None], key.norm(dim=1)[:, None]
    query_norm = query / torch.clamp(query_n, min=eps)
    key_norm = key / torch.clamp(key_n, min=eps)
    sim_mt = torch.mm(key_norm, query_norm.T)
    return 1-sim_mt


def _euc(query, key):
    """Euclidean distance (fp32: cdist has no bf16 CUDA kernel)."""
    if len(key.shape) < 2:
        key = key.view(1, -1)
    return torch.cdist(key.float(), query.float(), p=2)


def pairwise_dist(query, keys, dist_fn):
    """Compute distance from query to all keys"""
    dists = []
    if dist_fn == "mmd":
        d_fn = _mmd
    elif dist_fn == "cos":
        d_fn = _cos
    elif dist_fn == "euc":
        d_fn = _euc
    else:
        raise ValueError(f"Distance name {dist_fn} does not exist")

    for i in range(len(keys)):
        dists.append(d_fn(query, keys[i]).view(-1, 1))
    return torch.stack(dists).view(-1, len(query))


class GRACE(torch.nn.Module):
    """GRACE: General Retrieval Adaptors for Continual Editing"""
    log_name = "grace"  # overridden by the factory (e.g. "grace_soft")

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.log_dict = {}
        self.model = model.model if hasattr(model, 'model') else model
        self.tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        
        layer = config.inner_params[0]
        self.device = config.device

        suffixes = [".weight", ".bias"]
        self.layer = layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer
        
        for n, p in self.model.named_parameters():
            p.requires_grad = False
        
        if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True

        edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        layer_name = self.layer.rsplit(".", 1)[-1]
        original_layer = getattr(edit_module, layer_name)
        # Per-dataset initial ε radius (optimal GRACE differs by dataset: identical-
        # question sets need a tight radius so distinct edits get distinct keys).
        dataset_name = getattr(getattr(config, "experiment", None), "dataset_name", None)
        init_eps = self._resolve_eps(config.editor, dataset_name)
        print(f"[GRACE] init_epsilon={init_eps} (dataset={dataset_name})", flush=True)
        # Wrap the original layer with GRACEAdaptor using editor-specific config
        wrapped_layer = GRACEAdaptor(config.editor, original_layer, transpose=transpose,
                                     init_epsilon=init_eps).to(self.device)
        setattr(edit_module, layer_name, wrapped_layer)
        self.target_layer = wrapped_layer

    @staticmethod
    def _resolve_eps(editor_cfg, dataset_name):
        """Initial ε radius, per-dataset if `eps_by_dataset` maps this dataset,
        else the global `eps`. Handles both dict and namespace config forms."""
        default = float(getattr(editor_cfg, "eps", 10.0))  # euc radius (mmGRACE calibration)
        m = getattr(editor_cfg, "eps_by_dataset", None)  # optional per-dataset override map
        if not m or not dataset_name:
            return default
        val = m.get(dataset_name, None) if isinstance(m, dict) else getattr(m, dataset_name, None)
        return float(val) if val is not None else default
        
    def __call__(self, **kwargs):
        if self.config.task == "hallucination":
            key_id = (kwargs.get("labels", torch.tensor([])) == -100).sum() - 1
            setattr(self.target_layer, "key_id", key_id)
        return self.model(**kwargs)
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
        
    def edit(self, config, tokens, batch_history):
        # Capture the codebook key at the PROMPT-END position (the last prompt
        # token = the last -100 in labels), NOT the default last token.
        #
        # During training the sequence is prompt+answer, so key_id=-1 would key on
        # the last *answer* token. But reliability here is scored with generative
        # generate(): the prefill query is the last *prompt* token (key_id=-1 over
        # a prompt-only sequence). Those positions differ, so a last-answer-token
        # key never matches the query -> GRACE routes on ~0 examples and
        # reliability collapses to 0. Keying on the last prompt token aligns the
        # train/inference positions (that activation is answer-independent by
        # causal masking), so the edit actually fires at generation time.
        # (Score-based tasks like MC also remain correct: prompt+choice ends at
        # this same prompt-end token for the matching choice.)
        #
        # NOTE: assumes prompt tokens are contiguous (no interior right-padding),
        # i.e. run sequential editing with batch_size=1 (one edit at a time).
        labels = tokens.get("labels", None)
        if labels is not None:
            prompt_len = int((labels == -100).sum(dim=-1).min().item())
            setattr(self.target_layer, "key_id", max(0, prompt_len - 1))

        setattr(self.target_layer, "training", True)
        setattr(self.target_layer, "edit_label", tokens.get("labels", None))
                
        self.losses = []
        n_iter = int(getattr(config.editor, "n_iter", 1000))
        edit_lr = float(getattr(config.editor, "edit_lr", 2e-2))
        early_stop_patience = int(getattr(config.editor, "early_stop_patience", 150))
        # Target loss for "memorized" convergence: stop as soon as the edit is
        # fully learned (loss ~ 0) instead of burning the full n_iter budget.
        loss_eps = float(getattr(config.editor, "loss_eps", 1e-2))
        # In the distribution-aware modes the value is trained to match q (a
        # divergence ~0 at a perfect match); allow a separately-tunable eps.
        if getattr(self, "soft_mode", "target") in ("loss", "rank_loss"):
            loss_eps = float(getattr(config.editor, "soft_loss_eps", 1e-2))
        # Keep the LR from annealing all the way to 0: a cosine decay to eta_min=0
        # freezes the value before it reaches ~0 loss. A high floor keeps lr large
        # in the tail so the value can keep inflating the answer-token logits
        # (softmax saturation) all the way toward loss ~ 0.
        eta_min = edit_lr * float(getattr(config.editor, "eta_min_frac", 0.5))
        # Improvement is judged on a *relative* drop so the slow tail (e.g. loss
        # 0.015 -> 0.014, sub-1e-6/step but still real) keeps resetting patience
        # instead of tripping the plateau guard while loss is still descending.
        plateau_rel_tol = float(getattr(config.editor, "plateau_rel_tol", 1e-3))
        
        best_loss = float('inf')
        patience_counter = 0
        
        for i in range(n_iter):
            setattr(self.target_layer, "iter", i)
            outputs = self.model(**tokens)
            
            # Create optimizer after first forward pass (so it includes self.values Parameter)
            if i == 0:
                optimizer = torch.optim.Adam(self.model.parameters(), edit_lr)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, n_iter), eta_min=eta_min)
            
            # Optimize the stored value vector with CE (base / soft="target") or
            # JS on the label distribution (soft="loss"/"rank_loss").
            loss = compute_edit_loss(self, outputs, tokens)
            
            loss_value = loss.detach().cpu().item()
            self.losses.append(loss_value)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            # Converged: the sample is memorized (loss ~ 0) → done. This fires at
            # iter 0 too when the edit key already falls inside an existing
            # codebook entry (same label) that yields the right answer — common on
            # datasets with an identical question (e.g. FERPlus), where the edit is
            # already covered and needs no optimization.
            if loss_value <= loss_eps:
                print(f"[{self.log_name}] converged at iter {i+1}/{n_iter} - loss: {loss_value:.4f}")
                break
            
            # Plateau guard: only fires on a genuine stall. A >=plateau_rel_tol
            # relative drop counts as progress and resets patience, so a slow but
            # real tail descent keeps training toward loss_eps instead of being
            # cut early; only a truly flat loss trips the guard.
            if loss_value < best_loss * (1.0 - plateau_rel_tol):
                patience_counter = 0
            else:
                patience_counter += 1
                if i >= 100 and patience_counter >= early_stop_patience:
                    break
            if loss_value < best_loss:
                best_loss = loss_value
            
            # Print loss every 200 iterations or on first/last iteration
            if (i + 1) % 200 == 0 or i == 0 or i == n_iter - 1:
                print(f"[{self.log_name}] iter {i+1}/{n_iter} - loss: {loss_value:.4f}")
        
        self.loss = loss if 'loss' in locals() else None
        setattr(self.target_layer, "training", False)
        # Reset the iteration counter so a stale `iter == 0` (left over when this
        # edit converged at iter 1) can never re-open the codebook-mutation branch
        # during the subsequent read-only evaluation forwards.
        setattr(self.target_layer, "iter", -1)
        # Restore the default query position so generate()'s prefill queries the
        # last prompt token (key_id=-1 over the prompt-only sequence) — the same
        # position the key was captured from above.
        setattr(self.target_layer, "key_id", -1)
        
        # Log info
        if hasattr(self.target_layer, "chosen_key"):
            self.log_dict["chosen_key"] = getattr(self.target_layer, "chosen_key")
        if hasattr(self.target_layer, "keys"):
            self.log_dict["nkeys"] = len(getattr(self.target_layer, "keys"))

    def save_resume_state(self):
        """GRACE freezes the base model; all edit state is the adaptor codebook
        (keys/values/epsilons/key_labels), created lazily on the first edit."""
        tl = self.target_layer
        if not hasattr(tl, "keys"):
            return {"has_codebook": False}
        return {
            "has_codebook": True,
            "keys": tl.keys.detach().cpu(),
            "values": tl.values.detach().cpu(),
            "epsilons": tl.epsilons.detach().cpu(),
            "key_labels": [kl.detach().cpu() if isinstance(kl, torch.Tensor) else kl
                           for kl in tl.key_labels],
        }

    def load_resume_state(self, state):
        if not state or not state.get("has_codebook"):
            return
        tl = self.target_layer
        dev = getattr(tl, "device", self.device)
        tl.keys = state["keys"].to(dev)
        tl.values = torch.nn.Parameter(state["values"].to(dev), requires_grad=True)
        tl.epsilons = state["epsilons"].to(dev)
        tl.key_labels = [kl.to(dev) if isinstance(kl, torch.Tensor) else kl
                         for kl in state["key_labels"]]
        print(f"[GRACE] resume: restored {len(tl.keys)} codebook keys", flush=True)

    def rekey(self, model_wrapper, edit_ds, n_samples=None):
        """Rewrite codebook keys for the current GPU (resume-only).

        Keys are bf16 activations and can drift across GPU architectures.
        Values / epsilons / labels are left untouched. Matching is nearest-
        neighbor against fresh activations of already-edited samples — no
        UID tracking and no change to edit/NEW/SPLIT/MERGE behavior.
        """
        tl = self.target_layer
        if not hasattr(tl, "keys") or len(tl.keys) == 0:
            print("[GRACE] rekey: no codebook keys", flush=True)
            return

        from PIL import Image

        samples = list(getattr(edit_ds, "data", []) or [])
        if n_samples is not None:
            samples = samples[: max(0, int(n_samples))]
        if not samples:
            print("[GRACE] rekey: no edit samples to forward", flush=True)
            return

        raw_model = self.model
        was_training = raw_model.training
        raw_model.eval()
        tl._rekey_mode = True
        new_acts = []
        t0 = time.time()

        with torch.no_grad():
            for sample in samples:
                img_path = sample.get("image", "")
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception as exc:
                    print(f"[GRACE] rekey: skip bad image ({exc})", flush=True)
                    continue
                if "prompt" not in sample or "gold" not in sample:
                    print("[GRACE] rekey: skip sample missing prompt/gold", flush=True)
                    continue
                tokens = model_wrapper.prepare_training_batch({
                    "images": [img],
                    "prompts": [sample["prompt"]],
                    "golds": [sample["gold"]],
                    "idxs": [0],
                })
                labels = tokens.get("labels", None)
                if labels is not None:
                    prompt_len = int((labels == -100).sum(dim=-1).min().item())
                    tl.key_id = max(0, prompt_len - 1)
                else:
                    tl.key_id = -1
                raw_model(**tokens)
                new_acts.append(tl._rekey_activation.detach().reshape(-1))

        tl._rekey_mode = False
        tl.key_id = -1
        if was_training:
            raw_model.train()

        if not new_acts:
            print("[GRACE] rekey: no activations collected", flush=True)
            return

        bank = torch.stack(new_acts, dim=0).to(device=tl.keys.device)  # [N, H]
        dists = torch.cdist(tl.keys.float(), bank.float(), p=2)       # [K, N]
        nearest = dists.argmin(dim=1)
        mean_d = float(dists.gather(1, nearest.view(-1, 1)).mean().item())
        tl.keys = bank[nearest].to(dtype=tl.keys.dtype)
        print(f"[GRACE] rekey: rewrote {len(tl.keys)} keys from {len(bank)} samples "
              f"in {time.time() - t0:.1f}s (mean NN dist={mean_d:.4f})", flush=True)

    def reset_counters(self):
        """Reset edit application counters."""
        self.target_layer._edit_applied_count = 0
        self.target_layer._total_forward_count = 0

    def print_stats(self):
        """Print edit application stats (call after running inference)."""
        applied = self.target_layer._edit_applied_count
        total = self.target_layer._total_forward_count
        print(f"[GRACE] applied weight update to {applied}/{total} examples", flush=True)


class GRACESoft(GRACE):
    """Multilabel GRACE: the stored value is trained toward the ranked
    multi-label string. The supervision target swap happens generically in
    edit_utils when ``editor.soft`` is True; this subclass only flips the flag.
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        self.soft = True
        self.soft_mode = getattr(config.editor, "soft", "loss")
        self.rank_temp = float(getattr(config.editor, "rank_temp", 1.0))

    @contextmanager
    def cont_guard(self):
        """Mark the adaptor so the soft continuation forwards skip codebook
        mutation (add/split/merge) while leaving retrieval/routing untouched.
        `compute_edit_loss` wraps the continuation forwards in this."""
        tl = self.target_layer
        prev = getattr(tl, "_cont_mode", False)
        tl._cont_mode = True
        try:
            yield
        finally:
            tl._cont_mode = prev


class GRACEAdaptor(torch.nn.Module):
    def __init__(self, config, layer, transpose, init_epsilon=None):
        super().__init__()
        self.layer = layer
        # init_epsilon is resolved per-dataset by GRACE.__init__; fall back to global eps.
        self.init_epsilon = float(init_epsilon) if init_epsilon is not None else getattr(config, 'eps', 10.0)
        self.dist_fn = getattr(config, 'dist_fn', 'euc')  # mmGRACE uses euclidean
        self.replacement = getattr(config, 'replacement', 'replace_all')
        self.val_init = getattr(config, 'val_init', 'warm')
        self.val_train = getattr(config, 'val_train', 'sgd')
        self.eps_expand = getattr(config, 'eps_expand', 'coverage')
        self.device = layer.weight.device
        self.config = config
        self.num_pert = getattr(config, 'num_pert', 10)
        self.pool_method = getattr(config, 'pool_method', 'last')  # "mean" or "last"
        self.key_id = -1
    
        if transpose:
            self.key_shape = layer.weight.shape[1]
            self.value_shape = layer.weight.shape[0]
        else:
            self.key_shape = layer.weight.shape[0]
            self.value_shape = layer.weight.shape[1]
        self.training = False
        
        # Tracking counters for edit application
        self._edit_applied_count = 0
        self._total_forward_count = 0

    def add_key(self, new_key, new_value):
        keys = torch.vstack([self.keys, new_key.detach()])
        values = torch.nn.Parameter(torch.vstack([self.values, new_value]), requires_grad=True)
        new_epsilon = torch.tensor(self.init_epsilon, device=self.device).view(1)
        epsilons = torch.vstack([self.epsilons, new_epsilon])
        key_labels = self.key_labels + [self.edit_label]
        return keys, values, epsilons, key_labels

    def init_key_value(self, query, value):
        key = query.detach()
        epsilon = torch.tensor(self.init_epsilon, device=self.device, requires_grad=False).view(1)
        key_label = [self.edit_label]
        return key, value, epsilon, key_label

    def label_match(self, edit_label, key_label):
        if isinstance(edit_label, torch.Tensor) and isinstance(key_label, torch.Tensor):
            return edit_label.float().mean() == key_label.float().mean()
        return edit_label == key_label

    def split_epsilons_in_half(self, nearest_key, smallest_distance):
        # Ensure dtype matches existing epsilons (e.g., bfloat16 vs float32)
        sd = smallest_distance.to(self.epsilons.dtype)
        half = (sd / 2) - self.epsilons.new_tensor(1e-5)
        self.epsilons[nearest_key] = half
        self.epsilons[-1] = sd / 2

    def _pool_activation(self, act, safe_key_id):
        """Extract key/query from activation based on pool_method."""
        if act.dim() == 3:
            if self.pool_method == "mean":
                return act.mean(dim=1)  # [B, H]
            else:  # "last" - use position
                return act[:, safe_key_id, :]
        else:
            return act.mean(dim=0, keepdim=True)
    
    def forward(self, *args):
        layer_out = self.layer(*args)
        
        # Clamp key_id to valid bounds for current sequence length
        if args[0].dim() == 3:
            seq_len = args[0].shape[1]
            safe_key_id = min(self.key_id, seq_len - 1) if self.key_id >= 0 else max(-seq_len, self.key_id)
        else:
            safe_key_id = self.key_id

        # Resume-only capture path: pool activation and skip codebook logic.
        if getattr(self, "_rekey_mode", False):
            self._rekey_activation = self._pool_activation(args[0], safe_key_id)
            return layer_out
        
        # If we have never initialized keys:
        if not hasattr(self, "keys"):
            # No keys and not in training mode → behave as identity (no GRACE effect before editing)
            if not self.training:
                return layer_out

            # Initialize on first forward pass during training (GRACE.edit sets edit_label & key_id)
            init_query = self._pool_activation(args[0], safe_key_id)
            init_value_out = self._pool_activation(layer_out, safe_key_id)

            key, value, epsilon, key_label = self.init_key_value(init_query, init_value_out)
            self.keys = key
            self.values = torch.nn.Parameter(value, requires_grad=True)
            self.epsilons = epsilon
            self.key_labels = key_label

        # Compute query for retrieval (handles both 2D and 3D activations)
        query = self._pool_activation(args[0], safe_key_id)
        
        # --- compute distance from query to all keys and find the closest key ---
        # cdist has no bf16 CUDA kernel, so compute Euclidean distance in fp32
        # (keys/query are bf16 for Gemma/Qwen); the fp32 result compares fine below.
        if self.dist_fn == "euc":
            dists = torch.cdist(self.keys.float(), query.float(), p=2).view(-1, query.shape[0])
        elif self.dist_fn == "cos":
            dists = 1 - F.cosine_similarity(self.keys, query, dim=1).view(-1, query.shape[0])
        else:
            # fall back to Euclidean if unsupported
            dists = torch.cdist(self.keys.float(), query.float(), p=2).view(-1, query.shape[0])

        smallest_distance, nearest_key = dists.min(0)
        
        # --- optionally update codebook (only on iter == 0 during editing) ---
        # MUST be gated on self.training: `iter` is a leftover attribute from the
        # previous edit's loop and is left at 0 whenever an edit converges at iter 1
        # (i == 0, common for a MERGE already at loss ~0). Without the training
        # guard, every inference/eval forward would then re-enter this branch and
        # add a NEW key for each far eval query, exploding the codebook and routing
        # edited queries to junk keys (accuracy collapse).
        # `_cont_mode` (set by GRACESoft.cont_guard around the soft continuation
        # forwards) skips codebook mutation only: the main forward already
        # created/selected this edit's key on the majority label, so the extra
        # continuation forwards must not add/split/merge. Routing (query, distance,
        # eps, replace_all) below runs unchanged, so the just-created key still
        # fires at ~0 distance and the value is supervised on the continuation.
        if self.training and getattr(self, "iter", 0) == 0 and not getattr(self, "_cont_mode", False):
            _min_d = float(smallest_distance.min().item())
            if smallest_distance > (self.init_epsilon + self.epsilons[nearest_key]):
                # No close key → make a new key
                value_out = self._pool_activation(layer_out, safe_key_id)
                self.keys, self.values, self.epsilons, self.key_labels = self.add_key(query, value_out)
                self.chosen_key = len(self.keys) - 1
                if self.training:
                    print(f"[grace] key: NEW (min_dist={_min_d:.4f} > eps={self.init_epsilon}) "
                          f"-> {len(self.keys)} keys", flush=True)
            else:
                # Handle conflicts with nearest key
                if not self.label_match(self.edit_label, self.key_labels[nearest_key]):
                    value_out = self._pool_activation(layer_out, safe_key_id)
                    self.keys, self.values, self.epsilons, self.key_labels = self.add_key(query, value_out)
                    self.split_epsilons_in_half(nearest_key, smallest_distance)
                    self.chosen_key = len(self.keys) - 1
                    if self.training:
                        print(f"[grace] key: SPLIT (min_dist={_min_d:.4f} <= eps, label != nearest) "
                              f"-> {len(self.keys)} keys, eps halved to {float(self.epsilons[-1].item()):.4f}", flush=True)
                else:
                    # Same label: possibly expand epsilon
                    if smallest_distance > self.epsilons[nearest_key]:
                        # Cast distance to match epsilons dtype (e.g., bfloat16 for Qwen3)
                        sd = smallest_distance.to(self.epsilons.dtype)
                        if self.eps_expand == "coverage":
                            self.epsilons[nearest_key] = sd
                        elif self.eps_expand == "moving_average":
                            a = 0.5
                            self.keys[nearest_key] = a * self.keys[nearest_key] + (1 - a) * query
                            self.epsilons[nearest_key] = sd
                    self.chosen_key = nearest_key
                    if self.training:
                        print(f"[grace] key: MERGE (min_dist={_min_d:.4f} <= eps, label == nearest) "
                              f"-> reuse key {int(nearest_key)}/{len(self.keys)}", flush=True)
        else:
            # No codebook change; just use nearest key
            self.chosen_key = nearest_key
        
        smallest_dist = smallest_distance.view(-1, 1)
        chosen_value = self.values[self.chosen_key]
        eps = self.epsilons[self.chosen_key].view(-1, 1)

        # Track edit application (only during inference)
        if not self.training:
            self._total_forward_count += 1
            if (smallest_dist <= eps).any():
                self._edit_applied_count += 1

        # Optional adversarial value training (matches original GRACE config)
        if (self.val_train == "adv") and self.training:
            chosen_value = perturb_values(chosen_value, self.num_pert, self.device)

        # --- apply replacement policy ---
        if len(layer_out.shape) == 3:
            token_to_edit = safe_key_id
            # Ensure chosen_value is [1, H] for broadcasting
            if chosen_value.dim() == 1:
                chosen_value_b = chosen_value.unsqueeze(0)  # [1, H]
            else:
                chosen_value_b = chosen_value  # assume [1, H] or [B, H]

            if self.replacement == "replace_all":
                # Broadcast chosen_value across all time steps
                value_full = chosen_value_b.view(1, 1, -1).expand(layer_out.shape[0], layer_out.shape[1], -1)
                layer_out = torch.where(
                    (smallest_dist <= eps).view(-1, 1, 1),
                    value_full,
                    layer_out,
                )
            elif self.replacement == "replace_last":
                layer_out[:, token_to_edit, :] = torch.where(
                    (smallest_dist <= eps),
                    chosen_value_b,
                    layer_out[:, token_to_edit, :],
                )
            elif self.replacement == "replace_prompt":
                if token_to_edit > 0:
                    value_prompt = chosen_value_b.view(1, 1, -1).expand(layer_out.shape[0], token_to_edit, -1)
                    layer_out[:, :token_to_edit, :] = torch.where(
                        (smallest_dist <= eps).view(-1, 1, 1),
                        value_prompt,
                        layer_out[:, :token_to_edit, :],
                    )
        else:
            # 2D activations (e.g., vision patches) – apply a simplified replacement
            token_to_edit = min(safe_key_id if safe_key_id >= 0 else layer_out.shape[0] - 1,
                                layer_out.shape[0] - 1)
            if self.replacement == "replace_all":
                layer_out = torch.where(
                    (smallest_dist <= eps),
                    chosen_value,
                    layer_out,
                )
            elif self.replacement == "replace_last":
                layer_out[token_to_edit] = torch.where(
                    (smallest_dist <= eps).view(-1),
                    chosen_value.squeeze(0),
                    layer_out[token_to_edit],
                )
            elif self.replacement == "replace_prompt" and token_to_edit > 0:
                layer_out[:token_to_edit] = torch.where(
                    (smallest_dist <= eps),
                    chosen_value,
                    layer_out[:token_to_edit],
                )

        return layer_out

