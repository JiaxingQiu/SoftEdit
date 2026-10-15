from .ft import Finetune, FinetuneSoft
from .ft_retrain import Finetune_retrain, FinetuneRetrainSoft
from .mend import MEND, MENDSoft
from .mend_retrain import MEND_retrain, MENDRetrainSoft
from .grace import GRACE, GRACESoft
from .ike import IKE, IKESoft
from .alphaedit import AlphaEdit, AlphaEditSoft
from .softedit import SoftEdit
from .calibrators import TSSoft, BCSoft, ATSSoft, TSRetSoft, BCRetSoft

SUPPORTED_EDITORS = (
    "ft", "ft_soft", "ft_retrain", "ft_retrain_soft",
    "mend", "mend_soft", "mend_retrain", "mend_retrain_soft",
    "grace", "grace_soft",
    "ike", "ike_soft",
    "alphaedit", "alphaedit_soft",
    "softedit", "softedit_major",
    "ts_soft", "bc_soft", "ats_soft", "ts_ret_soft", "bc_ret_soft",
)


def get_editor(config, model):
    """Factory function to get editor based on config.

    `*_soft` variants are the multilabel counterparts of each base editor: they
    supervise on the full ranked multi-label list instead of the single
    majority-vote label. Weight editors do this via a generic `editor.soft`
    hook in edit_utils; retrieval editors (ike) override their stored value
    directly.

    AlphaEdit (Stage 11) is a closed-form null-space editor: sequential by default,
    its own down_proj layer range + wikipedia covariance.

    SoftEdit (Stage 12) is a retrieval codebook of tiny lm_head weight updates:
    SBERT-keyed retrieval + a per-sample ΔW patch spliced into the lm_head at
    generation time. Inherently multilabel (no `_soft` variant).
    """
    device = config.device
    editor_name = getattr(
        config.editor, "_name", config.editor if hasattr(config, "editor") else None
    )

    if editor_name == "ft":
        editor = Finetune(config, model)
    elif editor_name == "ft_soft":
        editor = FinetuneSoft(config, model)
    elif editor_name == "ft_retrain":
        editor = Finetune_retrain(config, model)
    elif editor_name == "ft_retrain_soft":
        editor = FinetuneRetrainSoft(config, model)
    elif editor_name == "mend":
        tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        editor = MEND(config, model, tokenizer, device)
    elif editor_name == "mend_soft":
        tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        editor = MENDSoft(config, model, tokenizer, device)
    elif editor_name == "mend_retrain":
        tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        editor = MEND_retrain(config, model, tokenizer, device)
    elif editor_name == "mend_retrain_soft":
        tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        editor = MENDRetrainSoft(config, model, tokenizer, device)
    elif editor_name == "grace":
        editor = GRACE(config, model)
    elif editor_name == "grace_soft":
        editor = GRACESoft(config, model)
    elif editor_name == "ike":
        editor = IKE(config, model)
    elif editor_name == "ike_soft":
        editor = IKESoft(config, model)
    elif editor_name == "alphaedit":
        editor = AlphaEdit(config, model)
    elif editor_name == "alphaedit_soft":
        editor = AlphaEditSoft(config, model)
    elif editor_name in ("softedit", "softedit_major"):
        editor = SoftEdit(config, model)
    elif editor_name == "ts_soft":
        editor = TSSoft(config, model)
    elif editor_name == "bc_soft":
        editor = BCSoft(config, model)
    elif editor_name == "ats_soft":
        editor = ATSSoft(config, model)
    elif editor_name == "ts_ret_soft":
        editor = TSRetSoft(config, model)
    elif editor_name == "bc_ret_soft":
        editor = BCRetSoft(config, model)
    else:
        raise ValueError(
            f"Unknown editor: {editor_name!r}. Choose from {list(SUPPORTED_EDITORS)} or 'baseline'."
        )

    # Single source of truth for log prefixes: soft variants report "{editor}_soft"
    # so their per-iteration training logs are distinguishable from the base editor.
    editor.log_name = editor_name

    return editor
