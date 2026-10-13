"""LoRA application helpers for VLMs."""

from __future__ import annotations

from typing import List, Optional

import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model


DEFAULT_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def text_model_target_names(model, leaf_names: List[str]) -> List[str]:
    """Prefer LM (text_model) Linear leaves so vision towers are not adapted."""
    leaves = set(leaf_names)
    names: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.split(".")[-1] not in leaves:
            continue
        # SmolVLM / Idefics: language stack lives under text_model
        if "text_model" in name or "language_model" in name or "model.layers" in name:
            names.append(name)
    if names:
        return names
    # Fallback: leaf-name matching (may include vision)
    return list(leaf_names)


def apply_lora(
    model,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    use_rslora: bool = False,
):
    """Wrap a causal / VLM model with LoRA adapters on LM attention + MLP."""
    leaves = target_modules or DEFAULT_TARGETS
    filtered = text_model_target_names(model, leaves)
    print(f"[LoRA] targeting {len(filtered)} modules (text-first)")

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=filtered,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
        use_rslora=use_rslora,
    )
    peft_model = get_peft_model(model, cfg)
    peft_model.print_trainable_parameters()
    return peft_model
