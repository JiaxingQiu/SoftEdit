"""ReasonEdit: answer-only multimodal in-context knowledge editing.

A "multimodal IKE": each edit stores one key/value in a codebook.
- key   = dual_sbert embedding of <image, question>
          = concat(VLM vision activation, lang_scaler * SBERT(question))
- value = "New Fact: The answer to 'Q' is <majority-vote label>."

At eval time the nearest fact(s) are retrieved (hubness-corrected, with a
rejection gate for locality) and prepended to the prompt.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

from .utils import brackets_to_periods, parent_module, Augmenter


class ReasonEdit(nn.Module):
    """Answer-only multimodal-IKE codebook for VLM editing.

    Codebook entry: [key_emb, value, radius]
    - key_emb: dual_sbert embedding of <image, question>
    - value: "New Fact: The answer to 'Q' is <label>."
    - radius: d(key, aug(key)); feeds the rejection threshold and key merging

    All knobs live in __init__ defaults below; the editor yaml only selects
    the editor (_name), it does not override behavior.

    One key per edit. At query time the nearest fact(s) are retrieved
    (hubness-corrected, with a rejection gate for locality) and prepended
    to the prompt via `apply_to_dataset`.
    """

    # ==================== 1. CORE ====================

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        cfg = getattr(config, "editor", config)

        # Model References
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.device = getattr(config, "device", torch.device("cpu"))
        self.seed = getattr(cfg, "seed", None)

        # Core Retrieval
        self.cap_keys = int(getattr(cfg, "cap_keys", 1))  # top-k facts to retrieve/prepend
        self.hubness_keys = getattr(cfg, "hubness_keys", False)
        self.hubness_eps = float(getattr(cfg, "hubness_eps", 1e-6))
        self.hubness_knn = int(getattr(cfg, "hubness_knn", 3))
        self.reject_threshold_pct = float(getattr(cfg, "reject_threshold_pct", 0.1))  # 0=disabled; percentile of pairwise key distances + intra-edit aug distances (see _compute_reject_threshold)
        
        # Embedding Config
        self.mode = getattr(cfg, "mode", "dual_sbert")              # "vision", "language", "language_last", "dual_sbert"
        self.pool_method = getattr(cfg, "pool_method", "mean")      # "mean" or "last"
        self.distance = getattr(cfg, "distance", "l2")              # "l2" or "cosine"
        self.lang_scaler = float(getattr(config.model, "lang_scaler_sbert", 30.0))
        
        # Radius Estimation
        self.radius_area_pct = float(getattr(cfg, "radius_area_pct", 0.9))  # mosaic area for augmentation
        self.n_radius_samples = int(getattr(cfg, "n_radius_samples", 0))
        self.radius_percentile = float(getattr(cfg, "radius_percentile", 50))
        self.radius_scaler = float(getattr(cfg, "radius_scaler", 1.0))
        
        # Key Merging (separate from radius)
        self.merge_keys = getattr(cfg, "merge_keys", False)  # IKE-style default: keep each edit's fact separate
        self.merge_ioa_threshold = float(getattr(cfg, "merge_ioa_threshold", 0.9))
        self.merge_dist_pct = float(getattr(cfg, "merge_dist_pct", 0.1))

        # Internal State
        self._added_uids = set()
        self._edit_count = 0
        self._act = None
        self._sbert = None
        self._last_retr = {"d1": None, "uids": []}
        self._hubness_logged = False
        
        # Codebook Storage
        self.codebook = []
        self.key_embs = None   # [N, hidden]
        self.key_radii = None  # [N]
        self.key_sigmas = None  # [N] - local neighborhood scale for hubness correction
        self.key_dist_threshold = None  # rejection threshold for far queries
        
        # Logging
        self.last_retrieval_log = None
        self.plot_codebook_pct_threshold = getattr(cfg, "plot_codebook_pct_threshold", 85)

        # Setup Hooks & Tools
        dataset_name = getattr(getattr(config, "experiment", None), "dataset_name", None)
        self.augmenter = Augmenter(self.wrapper, seed=self.seed, mosaic_prob=1.0, dataset_name=dataset_name)
        
        # VLM activation hooks
        model_cfg = getattr(config, "model", config)
        inner_params_vision = getattr(model_cfg, "inner_params_vision", [])
        inner_params_lang = getattr(model_cfg, "inner_params_lang", [])
        inner_params = getattr(model_cfg, "inner_params", [])
        
        # Validate required params based on mode
        if self.mode == "vision" and not inner_params_vision:
            raise ValueError("mode='vision' requires config.model.inner_params_vision")
        elif self.mode == "language" and not inner_params_lang:
            raise ValueError("mode='language' requires config.model.inner_params_lang")
        elif self.mode == "language_last" and not inner_params:
            raise ValueError("mode='language_last' requires config.model.inner_params")
        elif self.mode == "dual_sbert" and not inner_params_vision:
            raise ValueError("mode='dual_sbert' requires config.model.inner_params_vision")
        
        def _setup_hook(param_name):
            name = param_name.rsplit(".", 1)[0] if param_name.endswith((".weight", ".bias")) else param_name
            mod = parent_module(self.model, brackets_to_periods(name))
            layer = getattr(mod, name.rsplit(".", 1)[-1])
            return layer.register_forward_hook(
                lambda m, i, o: setattr(self, "_act", i[0].detach() if isinstance(i[0], torch.Tensor) else None)
            )
        
        # Setup single hook based on mode
        if self.mode == "vision":
            self._hook = _setup_hook(inner_params_vision[0])
        elif self.mode == "language":
            self._hook = _setup_hook(inner_params_lang[0])
        elif self.mode == "language_last":
            self._hook = _setup_hook(inner_params[0])
        elif self.mode == "dual_sbert":
            self._hook = _setup_hook(inner_params_vision[0])
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    def generate(self, *a, **kw):
        return (self.model if hasattr(self.model, "generate") else self.wrapper).generate(*a, **kw)

    # ==================== 2. ENCODING ====================

    def _pool_act(self, act, batch_size):
        """Pool activation to [B, hidden] shape."""
        if act is None:
            raise RuntimeError("Hook failed to capture activation")
        act = act.to(self.device, torch.float32)
        if act.dim() == 3:
            return act[:, -1, :] if self.pool_method == "last" else act.mean(dim=1)
        elif act.dim() == 2:
            if act.shape[0] == batch_size:
                return act
            elif act.shape[0] % batch_size == 0:
                patches = act.shape[0] // batch_size
                if self.pool_method == "last":
                    return act.view(batch_size, patches, -1)[:, -1, :]
                return act.view(batch_size, patches, -1).mean(dim=1)
            else:
                if self.pool_method == "last":
                    return act[-1:].expand(batch_size, -1)
                return act.mean(dim=0, keepdim=True).expand(batch_size, -1)
        raise RuntimeError(f"Expected 2D or 3D activation, got {act.shape}")

    def _get_sbert(self):
        """Lazy load sentence-transformers model."""
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")
            self._sbert.to(self.device)
        return self._sbert

    def _encode_sbert(self, texts: List[str]) -> torch.Tensor:
        """Get sentence embeddings from SBERT. Returns [B, 768]."""
        sbert = self._get_sbert()
        emb = sbert.encode(texts, convert_to_tensor=True)
        return emb.to(self.device, torch.float32)

    @torch.no_grad()
    def _encode_vlm(self, images: List, texts: List[str]) -> torch.Tensor:
        """Get VLM embedding for <image, text> pairs.
        
        Returns: [B, hidden] or [B, hidden+lang_dim] tensor (for dual_sbert)
        """
        self.model.eval()
        batch_size = len(images) if isinstance(images, list) else 1
        
        # Forward pass to capture activation
        self._act = None
        inputs = self.wrapper.encode(images, texts, tokenize=False)
        self.model(**inputs)
        emb = self._pool_act(self._act, batch_size)
        self._act = None
        
        # For dual_sbert, concat with SBERT embedding
        if self.mode == "dual_sbert":
            lang_emb = self._encode_sbert(texts) * self.lang_scaler
            emb = torch.cat([emb, lang_emb], dim=-1)
        
        return emb

    # ==================== 3. KEY MANAGEMENT ====================

    @torch.no_grad()
    def _estimate_radius(self, key_emb: torch.Tensor, img, text: str) -> float:
        """Estimate radius using augmented samples. n_radius_samples=0 disables
        augmentation (radius 0 -> no intra samples in reject threshold, no merging)."""
        if self.n_radius_samples <= 0:
            return 0.0
        aug_dists = []
        for _ in range(self.n_radius_samples):
            aug_img = self.augmenter.image(img, area_pct=self.radius_area_pct)
            aug_text = self.augmenter.question(text) if text else ""
            aug_emb = self._encode_vlm([aug_img], [aug_text])
            aug_dists.append(float(torch.norm(aug_emb.cpu() - key_emb.cpu())))
        return float(np.percentile(aug_dists, self.radius_percentile)) * self.radius_scaler

    def _circle_intersection(self, d: float, r1: float, r2: float) -> float:
        """Compute intersection area of two circles."""
        if d >= r1 + r2:
            return 0.0
        if d <= abs(r1 - r2):
            return np.pi * min(r1, r2) ** 2
        part1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
        part2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
        part3 = 0.5 * np.sqrt((r1+r2-d) * (d+r1-r2) * (d-r1+r2) * (d+r1+r2))
        return part1 + part2 - part3

    def _circle_ioa_pair(self, d: float, r1: float, r2: float) -> Tuple[float, float]:
        """Compute IoA pair: (intersection/area1, intersection/area2)."""
        intersection = self._circle_intersection(d, r1, r2)
        area1, area2 = np.pi * r1 ** 2, np.pi * r2 ** 2
        return (intersection / area1 if area1 > 0 else 0.0,
                intersection / area2 if area2 > 0 else 0.0)

    def _try_merge_key(self, emb: torch.Tensor, radius: float, value: str) -> Tuple[bool, float]:
        """Try to merge new key with existing keys. Returns (merged, final_radius)."""
        if self.key_embs is None or len(self.codebook) == 0:
            return False, radius
        
        emb_cpu = emb.cpu().squeeze(0) if emb.dim() > 1 else emb.cpu()
        dists = torch.norm(self.key_embs - emb_cpu, dim=1).numpy()
        radii = self.key_radii.numpy()
        
        overlapping = np.where(dists < radius + radii)[0]
        if len(overlapping) == 0:
            return False, radius
        
        for idx in overlapping:
            d = dists[idx]
            # Distance-based merge
            if d < self.merge_dist_pct * radius and d < self.merge_dist_pct * radii[idx]:
                self._do_merge(idx, d, radius, value)
                return True, radius
            # IoA-based merge
            ioa_new, ioa_old = self._circle_ioa_pair(d, radius, radii[idx])
            if ioa_new > self.merge_ioa_threshold and ioa_old > self.merge_ioa_threshold:
                self._do_merge(idx, d, radius, value)
                return True, radius
        
        return False, radius

    def _do_merge(self, idx: int, dist: float, radius: float, value: str):
        """Perform merge of new key into existing key at idx."""
        self.key_radii[idx] = max(float(self.key_radii[idx]), dist + radius)
        existing_value = self.codebook[idx].get("value", "")
        if value and value != existing_value and value not in existing_value:
            self.codebook[idx]["value"] = f"{existing_value} {value}".strip()
        self.codebook[idx]["is_merged"] = True
        self.codebook[idx]["merge_count"] = self.codebook[idx].get("merge_count", 1) + 1

    @torch.no_grad()
    def _add_edit(self, img, question: str, answer: str, uid=None):
        """Add one multimodal-IKE key for an edit.

        key   = dual_sbert embedding of <image, question>
        value = "New Fact: The answer to 'Q' is <answer>."
        (Answer-only: no rationale / patch keys.)
        """
        answer_value = f"New Fact: The answer to '{question}' is {answer}." if answer else ""
        edit_idx = self._edit_count
        self._edit_count += 1

        emb = self._encode_vlm([img], [question])
        if self.distance == "cosine":
            emb = F.normalize(emb, dim=-1)
        # Radius = d(key, aug(key)): one intra-edit distance sample, used by the
        # rejection threshold population (and by key merging).
        radius = self._estimate_radius(emb[0:1], img, question)
        emb = emb.cpu()

        entry = {
            "value": answer_value, "edit_idx": edit_idx, "key_text": question,
            "uid": str(uid) if uid is not None else None,
        }

        merged = False
        if self.merge_keys:
            merged, radius = self._try_merge_key(emb, radius, answer_value)
        if not merged:
            entry["original_radius"] = radius
            self.codebook.append(entry)
            if self.key_embs is None:
                self.key_embs = emb
                self.key_radii = torch.tensor([radius])
            else:
                self.key_embs = torch.cat([self.key_embs, emb], dim=0)
                self.key_radii = torch.cat([self.key_radii, torch.tensor([radius])])

        # Invalidate caches (recomputed lazily at retrieval)
        self.key_sigmas = None
        self.key_dist_threshold = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _compute_key_sigmas(self, chunk_size: int = 1000):
        """Compute σ_k = median distance to hubness_knn nearest neighbor keys (for hubness correction)."""
        if not self.hubness_keys or self.key_embs is None or len(self.key_embs) < 2:
            self.key_sigmas = None
            return
        N = len(self.key_embs)
        k = min(self.hubness_knn, N - 1)
        sigmas = torch.zeros(N)
        embs = self.key_embs.float()
        for i in range(0, N, chunk_size):
            chunk = embs[i:i+chunk_size]
            dists = torch.cdist(chunk, embs, p=2)  # [chunk, N]
            dists[:, i:i+len(chunk)].fill_diagonal_(float('inf'))  # exclude self
            topk_dists, _ = dists.topk(k, dim=1, largest=False)
            sigmas[i:i+len(chunk)] = topk_dists.median(dim=1).values
        self.key_sigmas = sigmas
        print(f"[Hubness] computed sigma for {N} keys (k={k}, median={sigmas.median():.2f})")

    def _refresh_retrieval_stats(self):
        """Recompute whatever `_add_edit` invalidated. Sigmas first: the
        threshold is a percentile of the distances retrieval actually uses."""
        if self.hubness_keys and self.key_sigmas is None:
            self._compute_key_sigmas()
        if self.key_dist_threshold is None:
            self._compute_reject_threshold()

    def _compute_reject_threshold(self, max_sample: int = 2000):
        """Compute rejection threshold as percentile of pairwise key distances
        plus intra-edit aug distances (original_radius = d(key, aug(key))).

        The aug distances re-inject the intra-edit mode into the otherwise
        purely inter-edit distance distribution (one key per edit), so a low
        percentile lands in the intra/inter gap. NOTE: the intra share of the
        population shrinks as ~2/N, so the effective meaning of a fixed pct
        drifts toward the inter (chance) scale as the codebook grows.
        """
        if self.reject_threshold_pct <= 0 or self.key_embs is None or len(self.key_embs) < 2:
            self.key_dist_threshold = None
            return
        N = len(self.key_embs)
        idx = torch.randperm(N)[:min(max_sample, N)]
        sample = self.key_embs[idx].float()
        dists = torch.cdist(sample, sample, p=2)
        # Apply hubness scaling to match retrieval distances
        if self.hubness_keys and self.key_sigmas is not None:
            dists = dists / (self.key_sigmas[idx] + self.hubness_eps)  # scale by target key's sigma
        inter = dists[torch.triu(torch.ones_like(dists), diagonal=1) == 1]  # upper triangle

        # Intra-edit samples: original_radius is the pre-merge aug distance,
        # so merged keys' inflated key_radii don't leak into the population.
        intra = torch.tensor([float(e.get("original_radius", 0.0)) for e in self.codebook])
        if self.hubness_keys and self.key_sigmas is not None:
            intra = intra / (self.key_sigmas + self.hubness_eps)
        intra = intra[intra > 0]

        population = torch.cat([intra, inter])
        self.key_dist_threshold = float(np.percentile(population.numpy(), self.reject_threshold_pct))
        print(f"[Reject] threshold={self.key_dist_threshold:.2f} (p{self.reject_threshold_pct:g} of "
              f"{len(intra)} intra + {len(inter)} inter, hubness={self.hubness_keys})")

    # ==================== 4. RETRIEVAL ====================

    def _compute_distances(self, q_embs: torch.Tensor, key_indices: List[int]) -> torch.Tensor:
        """Compute [n_queries, n_keys] distance matrix, optionally hubness-scaled."""
        key_embs = self.key_embs[torch.tensor(key_indices)]
        if self.distance == "cosine":
            dists = 1 - (q_embs @ key_embs.t())
        else:
            dists = torch.cdist(q_embs.float(), key_embs.float(), p=2)
        self._refresh_retrieval_stats()
        if self.hubness_keys and self.key_sigmas is not None:
            sigmas = self.key_sigmas[torch.tensor(key_indices)]
            dists = dists / (sigmas + self.hubness_eps)
            if not getattr(self, '_hubness_logged', False):
                print(f"[Hubness correction] enabled, scaling distances by σ_k")
                self._hubness_logged = True
        return dists

    @torch.no_grad()
    def _retrieve(self, image, question: str) -> List[str]:
        """Retrieve values for a query <image, question>.

        Also records `_last_retr` = {d1, uids} (min scaled distance + uids of the
        retrieved entries) for the retrieval diagnostics stamped by
        apply_to_dataset (retr_rate / retr_acc, same as SoftEdit).
        """
        self._last_retr = {"d1": None, "uids": []}
        if self.key_embs is None or len(self.codebook) == 0:
            return []
        
        query_patches = [image]  # Use whole image for query
        return self._retrieve_from_keys(query_patches, question)

    @torch.no_grad()
    def _retrieve_from_keys(self, query_patches: List, question: str, 
                            key_indices: List[int] = None) -> List[str]:
        """Retrieve top cap_keys facts."""
        # Build query embeddings
        q_embs = self._encode_vlm(query_patches, [question] * len(query_patches))
        if self.distance == "cosine":
            q_embs = F.normalize(q_embs, dim=-1)
        q_embs = q_embs.cpu()
        
        if key_indices is None:
            key_indices = list(range(len(self.codebook)))
        if not key_indices:
            return []
        
        # Compute distances [n_patches, n_keys]
        dist_matrix = self._compute_distances(q_embs, key_indices)
        
        # Sort by min distance, take top cap_keys
        min_dists = dist_matrix.min(dim=0).values
        self._last_retr["d1"] = float(min_dists.min())
        
        # Rejection gate: query too far from all keys
        if self.key_dist_threshold is not None and min_dists.min() > self.key_dist_threshold:
            return []
        
        top_local = min_dists.argsort()[:self.cap_keys].tolist()
        
        # Collect unique values
        retrieved = set()
        for local_i in top_local:
            entry = self.codebook[key_indices[local_i]]
            if entry["value"]:
                retrieved.add(entry["value"])
                self._last_retr["uids"].append(entry.get("uid"))
        return list(retrieved)

    @torch.no_grad()
    def _get_matched_indices(self, image, question: str, apply_cap_k: bool = True) -> set:
        """Get matched key indices for plotting."""
        if self.key_embs is None or len(self.codebook) == 0:
            return set()
        
        query_patches = [image]  # Use whole image for query
        
        q_embs = self._encode_vlm(query_patches, [question] * len(query_patches)).cpu()
        if self.distance == "cosine":
            q_embs = F.normalize(q_embs, dim=-1)
        
        key_indices = list(range(len(self.codebook)))
        dist_matrix = self._compute_distances(q_embs, key_indices)
        min_dists = dist_matrix.min(dim=0).values
        
        # Rejection gate: query too far from all keys
        if self.key_dist_threshold is not None and min_dists.min() > self.key_dist_threshold:
            return set()
        
        top_k = min_dists.argsort()
        if apply_cap_k:
            top_k = top_k[:self.cap_keys]
        return {key_indices[i.item()] for i in top_k}

    # ==================== 5. DATASET & IO ====================

    def apply_to_dataset(self, dataset):
        """Apply retrieved facts to dataset prompts."""
        applied = 0
        log = []
        data = getattr(dataset, "data", [])
        
        for ex in data:
            prompt_orig = ex.get("prompt_orig") or ex.get("prompt", "")
            q, img = ex.get("question", ""), ex.get("image")
            if not prompt_orig or img is None:
                continue
            
            facts = self._retrieve(img, q) if q else []

            # Retrieval diagnostics (same schema as SoftEdit v2): aggregated by
            # the task engineer's _RetrievalCounter into retr_rate / retr_acc.
            # hit = ANY retrieved entry is the example's own edit (ReasonEdit
            # can inject multiple facts).
            dbg = {"src": "fact" if facts else None}
            if self._last_retr["d1"] is not None:
                dbg["d1"] = round(self._last_retr["d1"], 4)
            if facts:
                uids = [u for u in self._last_retr["uids"] if u is not None]
                dbg["entry_uid"] = uids[0] if len(uids) == 1 else uids
                dbg["hit"] = str(ex.get("uid")) in {str(u) for u in uids}
            ex["readout_dbg"] = dbg

            if facts:
                ex["prompt_orig"] = prompt_orig
                ex["prompt"] = f"{' '.join(facts)} {prompt_orig}"
                applied += 1
            else:
                ex["prompt"] = prompt_orig
            
            log.append({"uid": ex.get("uid"), "n_facts": len(facts), "facts": facts})
        
        self.last_retrieval_log = log
        print(f"[ReasonEdit] applied facts to {applied}/{len(data)} examples", flush=True)
        if getattr(getattr(dataset, "config", None), "debug", False):
            for i, ex in enumerate(data[:3]):
                print(f"  [{i}] q='{ex.get('question','')}' answer='{ex.get('answer','')}' "
                      f"prompt='{ex.get('prompt','')}'", flush=True)

    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None):
        """Add edits to the codebook (answer-only multimodal IKE).

        One key per edit: <image, question> -> "The answer to 'Q' is <label>",
        where the answer is the majority-vote label in gold.label.
        """
        if edit_ds is None:
            return self.model

        n_before = len(self.codebook)
        n_added = 0
        for ex in getattr(edit_ds, "data", []):
            uid = ex.get("uid") or (ex.get("image"), ex.get("question"))
            if uid in self._added_uids:
                continue
            img = ex.get("image")
            answer = self._edit_answer(ex)
            if img is None or not answer:
                continue
            self._add_edit(img, ex.get("question", ""), answer, uid=uid)
            self._added_uids.add(uid)
            n_added += 1
            print(f"\r[ReasonEdit] edit {n_added}...", end="", flush=True)

        n_after = len(self.codebook)
        mem_mb = self.key_embs.numel() * 2 / 1024 / 1024 if self.key_embs is not None else 0
        print(f"\n[ReasonEdit] +{n_added} edits (mode={self.mode}, merge={self.merge_keys}), "
              f"{n_before}->{n_after} keys, {mem_mb:.1f} MB", flush=True)

        # NOTE: apply_to_dataset is called externally in edit_utils.py before
        # evaluation to avoid O(N²) hubness recomputation after each edit batch.
        return self.model

    def _edit_answer(self, ex) -> str:
        """Stored value answer for one example (majority-vote label).
        Overridden by ReasonEditSoft to store the full ranked multi-label list."""
        return (ex.get("gold") or {}).get("label") or ex.get("answer") or ex.get("target") or ""

    def save_index(self, path):
        """Save codebook to disk."""
        torch.save({
            "codebook": self.codebook,
            "key_embs": self.key_embs,
            "key_radii": self.key_radii,
            "key_sigmas": self.key_sigmas,
            "key_dist_threshold": self.key_dist_threshold,
            "_edit_count": self._edit_count,
        }, path)
        print(f"[ReasonEdit] saved {len(self.codebook)} keys to {path}", flush=True)
    
    def load_index(self, path):
        """Load codebook from disk."""
        data = torch.load(path, map_location=self.device)
        self.codebook = data["codebook"]
        self.key_embs = data["key_embs"].to(self.device)
        self._edit_count = data.get("_edit_count", len(set(e.get("edit_idx", 0) for e in self.codebook)))
        self.key_radii = data["key_radii"].to(self.device)
        self.key_sigmas = data.get("key_sigmas")
        if self.key_sigmas is not None:
            self.key_sigmas = self.key_sigmas.to(self.device)
        self.key_dist_threshold = data.get("key_dist_threshold")
        print(f"[ReasonEdit] loaded {len(self.codebook)} keys from {path}", flush=True)

    def save_resume_state(self) -> Dict:
        """Serializable codebook state for resume (see revlm/run/resume.py).

        Tensors stay on CPU (that is where `_add_edit` keeps them and where
        retrieval computes distances), so no device juggling on reload.
        """
        return {
            "codebook": self.codebook,
            "key_embs": self.key_embs,
            "key_radii": self.key_radii,
            "key_sigmas": self.key_sigmas,
            "key_dist_threshold": self.key_dist_threshold,
            "_edit_count": self._edit_count,
            "_added_uids": self._added_uids,  # needed for incremental dedup after resume
        }

    def load_resume_state(self, state: Dict) -> None:
        """Restore codebook state produced by `save_resume_state`."""
        self.codebook = state["codebook"]
        self.key_embs = state.get("key_embs")
        self.key_radii = state.get("key_radii")
        self.key_sigmas = state.get("key_sigmas")
        self.key_dist_threshold = state.get("key_dist_threshold")
        self._edit_count = state.get("_edit_count", len(self.codebook))
        self._added_uids = set(state.get("_added_uids") or set())
        print(f"[ReasonEdit] resume: restored {len(self.codebook)} keys "
              f"(_edit_count={self._edit_count}, uids={len(self._added_uids)})", flush=True)

    def get_stats(self) -> Dict:
        """Return statistics about stored keys."""
        n_merged = sum(1 for e in self.codebook if e.get("is_merged", False))
        stats = {
            "num_keys": len(self.codebook),
            "num_merged_keys": n_merged,
            "num_edits": len(self._added_uids),
            "merge_keys": self.merge_keys,
            "cap_keys": self.cap_keys,
            "emb_size_mb": self.key_embs.numel() * 2 / 1024 / 1024 if self.key_embs is not None else 0,
        }
        if self.key_radii is not None:
            stats["avg_radius"] = float(self.key_radii.mean())
            stats["min_radius"] = float(self.key_radii.min())
            stats["max_radius"] = float(self.key_radii.max())
        return stats

    # ==================== 6. VISUALIZATION ====================

    @torch.no_grad()
    def plot_codebook(self, max_edits=20, query_img=None, query_text=None, 
                      apply_cap_k=True, figsize=None, dpi=150):
        """Plot force-directed network of keys."""
        import matplotlib.pyplot as plt
        import networkx as nx
        
        if self.key_embs is None or len(self.codebook) == 0:
            print("[ReasonEdit] No keys to plot")
            return
        
        # Sample edits
        all_edits = sorted(set(e.get("edit_idx", 0) for e in self.codebook))
        if figsize is None:
            figsize = (4, 3) if len(all_edits) < 10 else (6, 4)
        if len(all_edits) > max_edits:
            import random
            selected_edits = set(random.sample(all_edits, max_edits))
        else:
            selected_edits = set(all_edits)
        
        indices = [i for i, e in enumerate(self.codebook) if e.get("edit_idx", 0) in selected_edits]
        idx_to_local = {g: l for l, g in enumerate(indices)}
        embs = self.key_embs[indices].float().cpu().numpy()
        n_keys = len(indices)
        
        # Get matched indices
        matched_global, q_emb = set(), None
        if query_img is not None and query_text is not None:
            matched_global = self._get_matched_indices(query_img, query_text, apply_cap_k=apply_cap_k)
            query_patches = [query_img]  # Use whole image for query
            q_embs = self._encode_vlm(query_patches, [query_text] * len(query_patches)).cpu()
            if self.distance == "cosine":
                q_embs = F.normalize(q_embs, dim=-1)
            q_emb = q_embs[0].numpy()
        matched_local = {idx_to_local[g] for g in matched_global if g in idx_to_local}
        
        # Build graph
        dists = np.linalg.norm(embs[:, None] - embs[None, :], axis=-1)
        sims = 1 / (1 + dists)
        
        G = nx.Graph()
        G.add_nodes_from(range(n_keys))
        thresh = np.percentile(sims[np.triu_indices(n_keys, k=1)], self.plot_codebook_pct_threshold) if n_keys > 1 else 0
        for i in range(n_keys):
            for j in range(i + 1, n_keys):
                if sims[i, j] > thresh:
                    G.add_edge(i, j, weight=sims[i, j])
        
        q_node = None
        if q_emb is not None:
            q_dists = np.linalg.norm(embs - q_emb, axis=-1)
            q_sims = 1 / (1 + q_dists)
            q_node = n_keys
            G.add_node(q_node)
            for i in range(n_keys):
                if q_sims[i] > thresh:
                    G.add_edge(q_node, i, weight=q_sims[i])
        
        pos = nx.spring_layout(G, weight='weight', seed=42, k=2/np.sqrt(len(G.nodes())))
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        nx.draw_networkx_edges(G, pos, alpha=0.08, width=0.1, ax=ax)
        
        edit_list = sorted(selected_edits)
        edit_to_color = {e: i for i, e in enumerate(edit_list)}
        cmap = plt.cm.get_cmap('tab20', max(len(edit_list), 1))
        
        # Node types: (name, size, marker, filter_fn)
        node_types = [
            ("original", 60, 'o', lambda i: not self.codebook[indices[i]].get("is_merged", False)),
            ("merged", 50, 's', lambda i: self.codebook[indices[i]].get("is_merged", False)),
        ]
        for node_type, size, marker, filter_fn in node_types:
            nodelist = [i for i in range(n_keys) if filter_fn(i)]
            if not nodelist:
                continue
            colors = [cmap(edit_to_color[self.codebook[indices[i]].get("edit_idx", 0)]) for i in nodelist]
            edgecolors = ['black' if i in matched_local else 'none' for i in nodelist]
            linewidths = [1.5 if i in matched_local else 0 for i in nodelist]
            nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=colors, node_shape=marker,
                                   node_size=size, alpha=0.8, ax=ax, edgecolors=edgecolors, linewidths=linewidths)
        
        if q_node is not None:
            ax.scatter(pos[q_node][0], pos[q_node][1], c='black', s=80, marker='*', zorder=10)
        
        # Legend - only show items that exist in the plot
        ax.scatter([], [], c='gray', s=40, marker='o', label='original')
        if self.merge_keys:
            ax.scatter([], [], c='gray', s=18, marker='s', label='merged')
        if q_node is not None:
            ax.scatter([], [], c='black', s=40, marker='*', label='query')
        if matched_local:
            ax.scatter([], [], c='gray', s=30, marker='o', edgecolors='black', linewidths=1.5, label='retrieved')
        # ax.legend(loc='lower left', fontsize=6, frameon=False, handletextpad=0.1)
        
        if self.mode == "dual_sbert":
            title = "Balanced Embedding"
        elif self.mode in ["language", "language_last"]:
            title = "Language-biased Embedding"
        else:
            title = "Vision-biased Embedding"

        ax.set_title(title, fontsize=10)
        # ax.set_title(f'Codebook ({len(edit_list)} edits, {n_keys} keys, {len(matched_local)} retrieved)', fontsize=8)
        ax.axis('off')
        ax.margins(0.05)
        plt.tight_layout(pad=0.3)
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)
        plt.show()


class ReasonEditSoft(ReasonEdit):
    """Multilabel ReasonEdit: store the full ranked multi-label list as the value.

    Identical to ReasonEdit except each codebook entry's value carries the
    numbered ranked-label string ("1. l1\\n2. l2\\n...") instead of the single
    majority-vote label. Falls back to the majority-vote label when no ranking
    is available.
    """

    def _edit_answer(self, ex) -> str:
        from .soft import ranked_label_str
        gold = ex.get("gold") or {}
        return ranked_label_str(gold) or gold.get("label") or ex.get("answer") or ""
