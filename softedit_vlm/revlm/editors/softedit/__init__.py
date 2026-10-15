"""SoftEdit: retrieval codebook of small lm_head weight updates.

Each edit trains a tiny lm_head weight update ΔW so the first-token distribution
matches the human answer proportions, regularized by two terms — **generality**
(graph-Laplacian Jacobian coupling, ``generality.py``) and **locality**
(describe-then-declare anchors, ``locality.py``). ΔW is stored in a retrieval
codebook and applied to the lm_head before inference for any query that
retrieves it. Fully self-contained (no cross-editor dependencies).

Layout:
  - ``softedit.py``   : the editor (retrieval + codebook + apply/patch)
  - ``train.py``      : capture supervision + train ΔW / ΔW_cont + apply/restore
  - ``generality.py`` : affinity graph + graph_gen term
  - ``locality.py``   : locality anchors (VLM caption + template declarative)
  - ``judge.py``      : paraphrase judge for the retrieval locality gate
"""
from .softedit import SoftEdit
from .softedit_task import SoftEditTask

__all__ = ["SoftEdit", "SoftEditTask"]
