from .ft import Finetune, FinetuneSoft
from .ft_retrain import Finetune_retrain, FinetuneRetrainSoft
from .mend import MEND, MENDSoft
from .mend_retrain import MEND_retrain, MENDRetrainSoft
from .grace import GRACE, GRACESoft
from .balancedit import BalancEdit, BalancEditSoft
from .ike import IKE, IKESoft
from .reasonedit import ReasonEdit, ReasonEditSoft
from .softedit import SoftEdit, SoftEditTask
from .gt import GTEdit, GTEditSoft
from .calibrators import TSSoft, BCSoft, ATSSoft, TSRetSoft, BCRetSoft


def get_editor(config, model):
    """Factory function to get editor based on config.

    `*_soft` variants are the multilabel counterparts of each base editor: they
    supervise on the full ranked multi-label list instead of the single
    majority-vote label. Weight editors do this via a generic `editor.soft`
    hook in edit_utils; retrieval editors override their stored value directly.
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
    elif editor_name == "balancedit":
        editor = BalancEdit(config, model)
    elif editor_name == "balancedit_soft":
        editor = BalancEditSoft(config, model)
    elif editor_name == "ike":
        editor = IKE(config, model)
    elif editor_name == "ike_soft":
        editor = IKESoft(config, model)
    elif editor_name == "reasonedit":
        editor = ReasonEdit(config, model)
    elif editor_name == "reasonedit_soft":
        editor = ReasonEditSoft(config, model)
    elif editor_name in ("softedit", "softedit_major"):
        editor = SoftEdit(config, model)
    elif editor_name == "softedit_task":
        editor = SoftEditTask(config, model)
    elif editor_name == "gt":
        editor = GTEdit(config, model)
    elif editor_name == "gt_soft":
        editor = GTEditSoft(config, model)
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
        raise ValueError(f"Unknown editor: {editor_name}")

    # Single source of truth for log prefixes: soft variants report "{editor}_soft"
    # so their per-iteration training logs are distinguishable from the base editor.
    editor.log_name = editor_name

    return editor
