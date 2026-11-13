"""Shared editor helpers (text-only).

Ported from the VLM repo's ``revlm/editors/utils.py`` with the image-specific
machinery removed (mosaic backgrounds, the `Augmenter` online rephraser, and its
torchvision/PIL deps). The Stage-7 editors (ft/mend/grace/ike and the retrain
variants) only need the parameter-name and MEND hook helpers below.

``ranked_label_str`` (the soft target-string builder) lives alongside the
distribution-based soft supervision in ``rellm/editors/soft.py``.
"""
import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    """Set random seed for reproducibility (random, numpy, torch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_inner_params(named_parameters, inner_names):
    """Get parameters by name"""
    param_dict = dict(named_parameters)
    return [(n, param_dict[n]) for n in inner_names if n in param_dict]


def param_subset(named_parameters, inner_names):
    """Get subset of parameters"""
    param_dict = dict(named_parameters)
    return [param_dict[n] for n in inner_names if n in param_dict]


def parent_module(model, pname):
    """Get parent module for a parameter name"""
    components = pname.split('.')
    parent = model
    for component in components[:-1]:
        if hasattr(parent, component):
            parent = getattr(parent, component)
        elif component.isdigit():
            parent = parent[int(component)]
        else:
            raise RuntimeError(f"Couldn't find child module {component}")
    if not hasattr(parent, components[-1]):
        raise RuntimeError(f"Couldn't find child module {components[-1]}")
    return parent


def brackets_to_periods(name):
    """Convert brackets to periods in parameter names"""
    return name.replace("[", ".").replace("]", "")


def linear_backward_hook(mod, grad_in, grad_out):
    """Hook for capturing gradients in MEND"""
    if not hasattr(mod, "weight"):
        return
    if hasattr(mod.weight, "__x__"):
        assert len(grad_out) == 1
        mod.weight.__delta__ = grad_out[0].detach()


def linear_forward_hook(mod, activations, output):
    """Hook for capturing activations in MEND"""
    assert len(activations) == 1
    mod.weight.__x__ = activations[0].detach()


def hook_model(model, pnames):
    """Add forward and backward hooks to model for MEND"""
    handles = []
    for pname in pnames:
        parent = parent_module(model, pname)
        handles.append(parent.register_forward_hook(linear_forward_hook))
        handles.append(parent.register_full_backward_hook(linear_backward_hook))
    model.handles = handles


def explore_layers(model, top_k=10):
    """Find suggested layer candidates for finetuning.

    This function searches for layers that are commonly used for finetuning,
    such as language model heads, attention projections, and MLP layers.

    Args:
        model: PyTorch model to explore
        top_k: Maximum number of suggestions to return

    Returns:
        list: List of parameter names that match finetuning keywords
    """
    all_names = [n for n, p in model.named_parameters()]
    keywords = ['lm_head', 'embed_out', 'output', 'classifier', 'head',
                'self_attn.q_proj', 'self_attn.v_proj', 'self_attn.k_proj',
                'mlp.c_fc', 'mlp.c_proj', 'gate_proj', 'up_proj', 'down_proj']

    suggestions = []
    for name in all_names:
        for kw in keywords:
            if kw in name.lower():
                suggestions.append(name)
                break

    return suggestions[:top_k] if suggestions else all_names[:1]


def validate_and_correct_param_name(model, param_name, logger=None):
    """Validate parameter name exists, try to correct if not found.

    This is useful for handling variations in parameter names across different
    model versions or HuggingFace implementations.

    Args:
        model: PyTorch model to search
        param_name: Parameter name to validate (may include brackets)
        logger: Optional logger for warnings/info (if None, uses print)

    Returns:
        str: Valid parameter name (original if exists, corrected if found, fallback otherwise)
    """
    param_name = brackets_to_periods(param_name)
    model_params = dict(model.named_parameters())

    if param_name in model_params:
        return param_name

    # Exact-suffix match: handles wrapper/prefix differences (e.g. a live
    # PeftModel prefixes every param with 'base_model.model.') while PRESERVING
    # the layer index and container (transformer vs global_transformer).
    suffix_matches = [n for n in model_params if n == param_name or n.endswith("." + param_name)]
    if suffix_matches:
        corrected = sorted(suffix_matches, key=len)[0]
        msg = f"  Suffix-matched layer: {corrected}"
        logger.info(msg) if logger else print(msg)
        return corrected

    # PeftModel suffix match: LoRA-targeted params are renamed from
    # e.g. ``gate_proj.weight`` → ``gate_proj.base_layer.weight``.
    # Strip ``.base_layer`` from candidate names before comparing suffixes
    # so the original yaml name still matches the correct layer index.
    peft_matches = []
    for n in model_params:
        n_stripped = n.replace(".base_layer.", ".")
        if n_stripped == param_name or n_stripped.endswith("." + param_name):
            peft_matches.append(n)
    if peft_matches:
        corrected = sorted(peft_matches, key=len)[0]
        msg = f"  Suffix-matched layer (PeftModel .base_layer): {corrected}"
        logger.info(msg) if logger else print(msg)
        return corrected

    # Index-aware fuzzy matching: preserves ALL numeric parts (layer indices)
    # from the original param_name so we don't accidentally pick layer 0
    # when layer 30 was requested.
    log_msg = f"Parameter '{param_name}' not found in model. Attempting to find match..."
    if logger:
        logger.warning(log_msg)
    else:
        print(f"WARNING: {log_msg}")

    layer_parts = param_name.split('.')

    # Extract numeric indices and their preceding container name so we can
    # require e.g. "layers" + "30" to appear adjacently in the candidate.
    index_pairs = []
    for i, part in enumerate(layer_parts):
        if part.isdigit() and i > 0:
            index_pairs.append((layer_parts[i - 1], part))

    key_parts = [p for p in layer_parts if p and not p.isdigit()]

    if len(key_parts) >= 3:
        matches = [n for n in model_params.keys()
                   if all(part.lower() in n.lower() for part in key_parts[-3:])]
        # Narrow to candidates containing every (container.index) pair so
        # that e.g. "layers.30" is required, not just "layers".
        if matches and index_pairs:
            narrowed = [n for n in matches
                        if all(f"{container}.{idx}" in n for container, idx in index_pairs)]
            if narrowed:
                matches = narrowed
        if matches:
            corrected = matches[0]
            log_msg = f"  Found corrected layer: {corrected}"
            if logger:
                logger.info(log_msg)
            else:
                print(log_msg)
            return corrected

    fallback = list(model_params.keys())[0]
    log_msg = f"  No match found. Using fallback: {fallback}"
    if logger:
        logger.warning(log_msg)
    else:
        print(f"WARNING: {log_msg}")
    return fallback
