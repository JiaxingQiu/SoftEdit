"""IKE: Simple In-Context Knowledge Editing

Pattern (same as ReasonEdit):
1. edit() → incrementally add entries to corpus
2. apply_to_dataset() → retrieve from corpus, prepend to prompts
"""

import torch
from sentence_transformers import SentenceTransformer
from typing import List, Any, Optional


class IKE(torch.nn.Module):

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.wrapper = model if hasattr(model, "model") else None
        self.model = model.model if hasattr(model, "model") else model
        self.device = config.device

        editor_cfg = getattr(config, "editor", config)
        self.k: int = int(getattr(editor_cfg, "k", 1))  # top-k facts to retrieve/prepend
        self.sentence_model = SentenceTransformer(
            getattr(editor_cfg, "sentence_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        ).to(self.device)

        # Corpus storage
        self.corpus_sentences: List[str] = []
        self.corpus_embeddings: Optional[torch.Tensor] = None
        self._added_uids: set = set()

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    def generate(self, *a, **kw):
        return (self.model if hasattr(self.model, "generate") else self.wrapper).generate(*a, **kw)

    # -------------------------------------------------------------------------
    # Core: Add & Retrieve
    # -------------------------------------------------------------------------

    @staticmethod
    def _key_text(ex) -> str:
        """Text the retrieval key/fact is built from: the raw question (no
        instruction), used identically on the edit and query sides."""
        return ex.get("question") or ex.get("prompt_orig") or ex.get("prompt", "")

    @staticmethod
    def _retrieval_key(text: str) -> str:
        """Deterministic retrieval key: lowercase + collapse whitespace.

        Used identically for stored edits and queries so the key and the query live
        in the same text space (prompt only, no target), which makes top-k retrieval
        symmetric. (Deliberately not the LLM `normalize` augment, which is stochastic
        and would desync the corpus/query keys.)
        """
        return " ".join((text or "").strip().lower().split())

    @torch.no_grad()
    def _add_edit(self, key_text: str, target: str, uid: Any) -> bool:
        """Add one edit to corpus. Returns True if added, False if skipped.

        The retrieval key is built from ``key_text`` (the raw question); the
        stored/prepended fact reads as ``"{key_text} {target}"``.
        """
        if uid in self._added_uids:
            return False

        # Retrieval key = normalized question only (symmetric with the query side);
        # the stored/prepended fact keeps the readable question + target.
        key = self._retrieval_key(key_text)
        sentence = f"{key_text} {target}"

        # Encode the key (not the fact) so queries (prompt-only) match the same space.
        # Store the RAW (un-normalized) embedding; retrieval is nearest-neighbor by L2.
        emb = self.sentence_model.encode(key, convert_to_tensor=True, show_progress_bar=False)
        emb = emb.unsqueeze(0).to(self.device)

        # Append
        self.corpus_sentences.append(sentence)
        if self.corpus_embeddings is None:
            self.corpus_embeddings = emb
        else:
            self.corpus_embeddings = torch.cat([self.corpus_embeddings, emb], dim=0)

        self._added_uids.add(uid)
        return True

    # -------------------------------------------------------------------------
    # API: edit() and apply_to_dataset()
    # -------------------------------------------------------------------------

    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None):
        """Add new edits to corpus."""
        if edit_ds is None:
            return self.model

        data = getattr(edit_ds, "data", [])
        n_added = 0
        for ex in data:
            uid = ex.get("uid") or (ex.get("image"), ex.get("question"))
            # Key/fact text = the raw question (symmetric with the query side).
            key_text = self._key_text(ex)
            target = self._edit_target(ex)
            if key_text and target and self._add_edit(key_text, target, uid):
                n_added += 1

        print(f"[{type(self).__name__}] +{n_added} edits, corpus={len(self.corpus_sentences)}", flush=True)
        return self.model

    def _edit_target(self, ex) -> str:
        """Stored-fact target for one example (majority-vote label).
        Overridden by IKESoft to store the full ranked multi-label list."""
        return ex.get("gold", {}).get("label", "")

    @torch.no_grad()
    def apply_to_dataset(self, dataset, train_ds=None, inplace=True):
        """Retrieve and prepend to prompts (batch optimized)."""
        data = getattr(dataset, "data", [])
        if not data or not self.corpus_sentences:
            print(f"[IKE] applied to 0/{len(data)} examples (k={self.k})", flush=True)
            return [], dataset

        # Prepend target is the eval prompt (bare question); the retrieval key is
        # built the same way as the edit side (symmetric retrieval space).
        prompts = [ex.get("prompt_orig") or ex.get("prompt", "") for ex in data]
        key_texts = [self._key_text(ex) for ex in data]
        valid_mask = [bool(p) for p in prompts]

        # Batch encode all queries at once
        valid_keys_raw = [k for k, v in zip(key_texts, valid_mask) if v]
        if not valid_keys_raw:
            print(f"[IKE] applied to 0/{len(data)} examples (k={self.k})", flush=True)
            return [], dataset

        # Build queries in the same text space as stored keys (symmetric retrieval).
        valid_keys = [self._retrieval_key(k) for k in valid_keys_raw]
        q_embs = self.sentence_model.encode(valid_keys, convert_to_tensor=True, batch_size=64, show_progress_bar=False)
        q_embs = q_embs.to(self.device)  # RAW (un-normalized)

        # Nearest-neighbor by L2 over raw embeddings: the k smallest distances per query.
        dists = torch.cdist(q_embs.float(), self.corpus_embeddings.float(), p=2)  # [Q, C]
        kk = min(self.k, dists.shape[1])
        nn_idx = torch.topk(dists, kk, dim=1, largest=False).indices.tolist()  # [Q][kk]

        # Apply results
        applied = 0
        hit_idx = 0
        for ex, prompt, valid in zip(data, prompts, valid_mask):
            if not valid:
                continue
            facts = [self.corpus_sentences[i] for i in nn_idx[hit_idx]]
            hit_idx += 1
            if facts:
                facts_str = "New Fact: " + "\n".join(facts) + "\n"
                ex["prompt_orig"] = prompt
                ex["prompt"] = facts_str + prompt
                applied += 1

        print(f"[IKE] applied to {applied}/{len(data)} examples (k={self.k})", flush=True)
        return [], dataset


class IKESoft(IKE):
    """Multilabel IKE: store the full ranked multi-label list as the fact target.

    Identical to IKE except the stored/prepended fact carries the numbered
    ranked-label string ("1. l1\\n2. l2\\n...") instead of the single
    majority-vote label.
    """

    def _edit_target(self, ex) -> str:
        from .soft import ranked_label_str
        return ranked_label_str(ex.get("gold", {}))
