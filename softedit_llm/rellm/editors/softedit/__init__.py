"""SoftEdit (LLM): retrieval codebook of small lm_head weight updates.

Text-only adaptation of the VLM SoftEdit. Each edit trains a tiny lm_head weight
update ΔW so the first-token distribution matches the human answer proportions,
regularized by a **generality** (graph-Laplacian Jacobian coupling) and a
**locality** (declare-then-ask anchors) term. ΔW is stored in an SBERT-keyed
retrieval codebook and spliced into the lm_head before inference for any query
that retrieves it (restored afterwards). Fully self-contained.

Layout:
  - ``softedit.py``   : the editor (SBERT retrieval + codebook + apply/patch)
  - ``train.py``      : capture supervision + train ΔW / ΔW_cont + apply/restore
  - ``generality.py`` : affinity graph + graph_gen term (verbatim port)
  - ``locality.py``   : text-only locality anchors (template / llm declaratives)
  - ``judge.py``      : paraphrase judge for the retrieval locality gate (verbatim)
"""
from .softedit import SoftEdit

__all__ = ["SoftEdit"]
