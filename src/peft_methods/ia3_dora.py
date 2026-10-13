"""IA3 and DoRA PEFT wrappers."""

from __future__ import annotations

from typing import List, Optional

from peft import IA3Config, LoraConfig, TaskType, get_peft_model

from .lora import DEFAULT_TARGETS, text_model_target_names


DEFAULT_IA3_TARGETS = ["k_proj", "v_proj", "down_proj"]


def apply_ia3(
    model,
    target_modules: Optional[List[str]] = None,
    feedforward_modules: Optional[List[str]] = None,
):
    """Apply IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)."""
    leaves = target_modules or DEFAULT_IA3_TARGETS
    targets = text_model_target_names(model, leaves)
    ff_leaves = set(feedforward_modules or ["down_proj"])
    ff = [t for t in targets if t.split(".")[-1] in ff_leaves] or list(ff_leaves)
    print(f"[IA3] targeting {len(targets)} modules (text-first)")
    cfg = IA3Config(
        task_type=TaskType.CAUSAL_LM,
        target_modules=targets,
        feedforward_modules=ff,
    )
    peft_model = get_peft_model(model, cfg)
    peft_model.print_trainable_parameters()
    return peft_model


def apply_dora(
    model,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
):
    """Apply DoRA (Weight-Decomposed Low-Rank Adaptation)."""
    leaves = target_modules or DEFAULT_TARGETS
    filtered = text_model_target_names(model, leaves)
    print(f"[DoRA] targeting {len(filtered)} modules (text-first)")
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=filtered,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_dora=True,
    )
    peft_model = get_peft_model(model, cfg)
    peft_model.print_trainable_parameters()
    return peft_model
