"""AlphaEdit hyperparameters (LLM / rankedit_llm).

Ported from the official ``AlphaEdit/AlphaEdit_hparams.py`` (ICLR 2025) and pared
down to the fields our wired implementation actually consumes. Presets are seeded
from the paper's ``hparams/AlphaEdit/Llama3-8B.json`` and given mid-depth analogs
for the other two backbones (see ``layers`` notes below).

Deviations from upstream (documented in ``alphaedit.md``):
  * ``fact_token="last"`` (last prompt token) instead of ``subject_last`` — ProtoQA
    has no subject span.
  * ``v_loss_layer`` / ``kl_factor`` / ``mom2_update_weight`` / ``mom2_adjustment``
    / ``layer_selection`` are dropped: our ``compute_z`` runs a real chat-templated
    ``model(**tokens)`` forward with a delta hook and reads the model's own final
    logits (hard: CE; soft: the Stage-6 objective), so the manual ln_f@lm_head
    projection + unrelated-prompt KL regularizer are not reconstructed. The delta
    L2-ball clamp (``clamp_norm_factor``) and delta weight decay (``v_weight_decay``)
    are kept.
"""
from dataclasses import dataclass, field, replace
from typing import List


@dataclass
class AlphaEditHyperParams:
    # Which decoder blocks' MLP down_proj to edit (a contiguous early-middle range).
    layers: List[int]

    # Module path templates (HF Llama/Qwen/Phi share this naming).
    rewrite_module_tmp: str = "model.layers.{}.mlp.down_proj"
    layer_module_tmp: str = "model.layers.{}"
    ln_f_module: str = "model.norm"
    lm_head_module: str = "lm_head"

    # Fact-lookup token. Upstream uses "subject_last"; ProtoQA has no subject span.
    fact_token: str = "last"

    # z* (right-vector) optimization.
    v_num_grad_steps: int = 25
    v_lr: float = 1e-1
    v_weight_decay: float = 0.5
    clamp_norm_factor: float = 0.75
    z_verbose: bool = True  # one-line-per-edit z* convergence log

    # Covariance (wikipedia mom2 of the down_proj fan-in) + null-space threshold.
    # Sample count keys the cache (`_n100000_`); it does NOT change P's shape
    # (always [n_layers, d_in, d_in], d_in = down_proj fan-in).
    mom2_dataset: str = "wikipedia"
    mom2_n_samples: int = 100000  # paper-faithful; 10k caches still coexist
    mom2_dtype: str = "float32"
    nullspace_threshold: float = 2e-2

    # Ridge term in the projected MEMIT solve.
    L2: float = 10.0

    # Batched sequential editing: edits are buffered and solved in chunks of
    # ``num_edits`` (upstream default 100), giving rank-N keys per solve and a
    # rank-N ``cache_c`` fold — the regime the L2/clamp were tuned for. ``1``
    # reproduces the old one-edit-per-solve behavior.
    num_edits: int = 100
    # Context-template averaging of keys (upstream averages over ~11 contexts):
    # each edit's key is the mean down_proj input over ``n_contexts`` neutral
    # wrappers of the question. ``1`` = identity context only (no averaging).
    n_contexts: int = 5


# Per-backbone presets. Layer ranges:
#   * Llama-3.1-8B (32 blocks): [4,5,6,7,8] verbatim from the paper.
#   * Mistral-7B   (32 blocks): same depth as Llama -> [4,5,6,7,8].
#   * Qwen2.5-7B   (28 blocks): mid-depth analog of 4..8/32 -> [3,4,5,6,7].
_PRESETS = {
    "llama": AlphaEditHyperParams(layers=[4, 5, 6, 7, 8]),
    "qwen": AlphaEditHyperParams(layers=[3, 4, 5, 6, 7]),
    "mistral": AlphaEditHyperParams(layers=[4, 5, 6, 7, 8]),
}


def _backbone_key(model_name: str) -> str:
    n = (model_name or "").lower()
    if "mistral" in n:
        return "mistral"
    if "llama" in n:
        return "llama"
    if "qwen" in n:
        return "qwen"
    raise ValueError(
        f"AlphaEdit: no layer preset for model {model_name!r}. "
        "Add one to rellm/editors/alphaedit/hparams.py::_PRESETS."
    )


# Fields overridable via `config.editor.<field>` (CLI / editor preset yaml).
_OVERRIDABLE = (
    "layers", "rewrite_module_tmp", "layer_module_tmp", "ln_f_module",
    "lm_head_module", "fact_token", "v_num_grad_steps", "v_lr", "v_weight_decay",
    "clamp_norm_factor", "z_verbose", "mom2_dataset", "mom2_n_samples", "mom2_dtype",
    "nullspace_threshold", "L2", "num_edits", "n_contexts",
)


def resolve_hparams(config) -> AlphaEditHyperParams:
    """Backbone preset with any ``config.editor.<field>`` overrides applied."""
    base = _PRESETS[_backbone_key(getattr(config.model, "name", ""))]
    editor_cfg = getattr(config, "editor", None)
    overrides = {}
    for name in _OVERRIDABLE:
        val = getattr(editor_cfg, name, None) if editor_cfg is not None else None
        if val is not None:
            overrides[name] = list(val) if name == "layers" else val
    return replace(base, **overrides)
