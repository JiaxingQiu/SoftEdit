import os
import logging
import math
from typing import Dict

import torch
from torch import nn
import transformers

LOG = logging.getLogger(__name__)


def cache_dir():
    """HuggingFace model/tokenizer cache dir for rankedit_llm.

    Mirrors the rankedit convention (`./ckpts`), which resolves to
    `rankedit_llm/ckpts/` because the run entry points chdir to the project root.
    This is where downloaded backbones are stored/read from — isolated from the
    VLM `rankedit/ckpts/`. NOT where finetuned checkpoints are saved (that's
    `config.ckpt_dir`).
    """
    path = os.path.join(".", "ckpts")
    os.makedirs(path, exist_ok=True)
    return path


def get_tokenizer(config):
    """Load the tokenizer for a decoder-only LLM.

    Left padding so batched generation / first-token scoring stay aligned (the
    last real token — and generate's first-step scores — line up across the
    batch). Falls back to EOS as the pad token when the tokenizer defines none
    (Llama-3.1 has no pad token). HPC download-timeout handling mirrors rankedit.
    """
    model_name = config.model.name
    ckpt_cache = cache_dir()
    tok = _from_pretrained_with_timeout(
        transformers.AutoTokenizer, model_name, cache_dir=ckpt_cache,
        trust_remote_code=True, use_fast=True,
    )
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _from_pretrained_with_timeout(loader, model_name, **kwargs):
    """`loader.from_pretrained` trying local cache first, then a timed download."""
    try:
        return loader.from_pretrained(model_name, local_files_only=True, **kwargs)
    except Exception:
        original_timeout = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT")
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
        try:
            return loader.from_pretrained(model_name, local_files_only=False, **kwargs)
        finally:
            if original_timeout is not None:
                os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = original_timeout
            elif "HF_HUB_DOWNLOAD_TIMEOUT" in os.environ:
                del os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]


def get_hf_model(config):
    """Load a decoder-only causal LM (Qwen2.5-7B / Llama-3.1-8B / Mistral-7B).

    Uniform `AutoModelForCausalLM` path (no per-family classes needed): bf16 on
    GPU, sdpa attention, single-device `device_map` from `config.device`.
    """
    model_path = getattr(config.model, "pt", None) or config.model.name
    device = getattr(config, "device", "auto")
    device_map = str(device) if device != "auto" else "auto"
    load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": device_map,
        "attn_implementation": "sdpa",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else None,
    }
    if not getattr(config.model, "pt", None):
        load_kwargs["cache_dir"] = cache_dir()
    model = _from_pretrained_with_timeout(
        transformers.AutoModelForCausalLM, model_path,
        **{k: v for k, v in load_kwargs.items() if v is not None},
    )
    dropout = getattr(config, "dropout", None)
    if dropout is not None:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = dropout
            elif hasattr(m, "dropout") and isinstance(m.dropout, float):
                m.dropout = dropout
            elif hasattr(m, "activation_dropout") and isinstance(m.activation_dropout, float):
                m.activation_dropout = dropout
    return model


# Uniform system message across all backbones. Passing our own system message
# overrides each model's chat_template default (Qwen injects a persona,
# Llama/Phi inject their own) — see WORKBOOK "system prompt" finding — so the
# system context is identical across models rather than an uncontrolled confound.
SYSTEM_PROMPT = "You are a helpful assistant to answer question in a single word or phrase."


def get_preprocess(config):
    """Return callable(prompts, tokenizer) -> dict of tensors on CPU.

    Instruct models: wrap each prompt in the model's chat template (with our
    fixed system message + a trailing generation prompt), then tokenize
    (`add_special_tokens=False` since the template already emits BOS/role
    markers). The wrapper moves tensors to the device. Override the system
    message per-run via `config.model.system_prompt`.
    """
    system_prompt = getattr(getattr(config, "model", None), "system_prompt", None) or SYSTEM_PROMPT

    def _causal_lm(prompts, tokenizer, tokenize=False):
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in prompts
        ]
        return tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False,
        )

    return _causal_lm


def verify_inner_params(model, config):
    """Sanity-check the configured `inner_params` against the loaded model.

    Confirms each configured module path resolves to a real parameter, and warns
    when the LM head is tied to the input embeddings (editing `lm_head.weight`
    then also moves `embed_tokens`). Non-fatal: logs
    warnings so a misconfigured layer index surfaces early at load time.
    """
    named = dict(model.named_parameters())
    n_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    for name in (getattr(config.model, "inner_params", None) or []):
        if name not in named:
            LOG.warning("[verify_inner_params] '%s' not found in model params "
                        "(num_hidden_layers=%s). Check the layer index/path.",
                        name, n_layers)
    tied = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))
    if tied:
        LOG.warning("[verify_inner_params] tie_word_embeddings=True: editing "
                    "'lm_head.weight' also moves input embeddings (embed_tokens).")
    return {"num_hidden_layers": n_layers, "tie_word_embeddings": tied}


def clean_answer(o, i):
    s = o or ""
    if "ASSISTANT:" in s:
        s = s.split("ASSISTANT:")[-1].strip()
    elif "model\n" in s:
        s = s.split("model\n")[-1].strip()
    elif "assistant" in s:
        s = s.split("assistant")[-1].replace("\n", "").replace(":", "").strip()
    elif "Answer:" in s:
        s = s.split("Answer:")[-1].strip()
    elif "ANSWER:" in s:
        s = s.split("ANSWER:")[-1].strip()
    elif "answer:" in s:
        s = s.split("answer:")[-1].strip()
    else:
        s = s.replace(i, "").strip()
    return s


# "subtract max" softmax — numerically stable.
def nll_to_probs(label_losses: Dict[str, Dict[str, float]], use_avg: bool = False, temperature: float = 1.0) -> Dict[str, float]:
    scores = {}
    temp = max(1e-8, float(temperature))
    for lbl, d in label_losses.items():
        nll = float(d['avg_nll'] if use_avg else d['sum_nll'])
        scores[lbl] = -(nll) / temp
    m = max(scores.values()) if scores else 0.0
    exps = {lbl: math.exp(s - m) for lbl, s in scores.items()}
    Z = sum(exps.values()) or 1.0
    return {lbl: v / Z for lbl, v in exps.items()}


def load_inner_params_from_ckpt(model, ckpt_dir, layer_idx=0):
    """Load finetuned inner parameter weights into a TQAModel wrapper."""
    from rellm.editors.utils import validate_and_correct_param_name

    ckpt_base = ckpt_dir
    model_name = model.config.model.name
    layer_dir = os.path.join(ckpt_base, model_name, f"layer_{layer_idx}")
    layer_path = os.path.join(layer_dir, "layer.pt")
    layer_name_path = os.path.join(layer_dir, "layer_name.pt")

    inner_param_name = torch.load(layer_name_path)
    resolved_name = validate_and_correct_param_name(model.model, inner_param_name)

    param_tensor = torch.load(layer_path)
    target_param = dict(model.model.named_parameters())[resolved_name]
    target_param.data.copy_(param_tensor.to(target_param.data.device))
    print(f"Loaded finetuned layer '{inner_param_name}' from {layer_dir}")
    return resolved_name


def save_inner_params_to_ckpt(model, config, layer_idx=0):
    """Save finetuned inner parameter weights from a TQAModel wrapper."""
    ckpt_dir = config.ckpt_dir
    inner_param = config.model.inner_params[layer_idx]
    param_tensor = dict(model.model.named_parameters())[inner_param]
    model_name = model.config.model.name

    save_root = os.path.join(ckpt_dir, model_name)
    layer_dir = os.path.join(save_root, f"layer_{layer_idx}")
    layer_path = os.path.join(layer_dir, "layer.pt")
    layer_name_path = os.path.join(layer_dir, "layer_name.pt")

    os.makedirs(layer_dir, exist_ok=True)
    torch.save(param_tensor.detach().cpu().clone(), layer_path)
    torch.save(inner_param, layer_name_path)
    print(f"Saved finetuned layer '{inner_param}' to {layer_dir}")
