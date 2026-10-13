"""CPU-safe post-training quantization (Windows-friendly).

Uses pure PyTorch weight-only absmax INT8/INT4. Dequantizes in forward.
Does not require MSVC / quanto_cpp (which fails on this laptop).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantConfig:
    bits: int = 8  # 8 or 4
    exclude_patterns: Optional[list] = None


class WeightOnlyLinear(nn.Module):
    """Linear layer with packed int weights + dequant on the fly (CPU-safe)."""

    def __init__(self, linear: nn.Linear, bits: int = 8):
        super().__init__()
        assert bits in (4, 8)
        self.bits = bits
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        w = linear.weight.detach().float().cpu()
        max_q = (2 ** (bits - 1)) - 1
        absmax = w.abs().amax(dim=1).clamp(min=1e-8)
        scale = absmax / max_q
        q = torch.round(w / scale.unsqueeze(1)).clamp(-max_q, max_q).to(torch.int8)
        self.register_buffer("qweight", q)
        self.register_buffer("scale", scale)
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().float().cpu())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.qweight.float() * self.scale.unsqueeze(1)
        return F.linear(x.float(), w, self.bias)

    @property
    def weight(self):
        return self.qweight.float() * self.scale.unsqueeze(1)


def _should_skip(name: str, exclude: list) -> bool:
    n = name.lower()
    return any(ex.lower() in n for ex in exclude)


def quantize_model(model, bits: int = 8, weights=None):
    """Quantize Linear layers to INT8/INT4 weight-only (pure PyTorch)."""
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")

    exclude = ["lm_head", "embed_tokens", "visual", "vision_tower", "vision_model", "embed"]
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if _should_skip(name, exclude):
            continue
        if module.in_features * module.out_features < 1024:
            continue
        parent_name = ".".join(name.split(".")[:-1])
        leaf = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, WeightOnlyLinear(module, bits=bits))
        replaced += 1
    print(f"[quant-pytorch] replaced {replaced} Linear layers with INT{bits} weight-only")
    model.eval()
    return model


def model_disk_size_mb(model) -> float:
    """Estimate parameter / buffer storage size in MB."""
    total_bytes = 0.0
    counted = set()
    for mod in model.modules():
        if isinstance(mod, WeightOnlyLinear):
            # INT4 packed conceptually as 0.5 byte/elem; we store int8 for simplicity
            bpe = 1.0 if mod.bits == 8 else 0.5
            total_bytes += mod.qweight.numel() * bpe
            total_bytes += mod.scale.numel() * 4
            if mod.bias is not None:
                total_bytes += mod.bias.numel() * 4
            counted.add(id(mod.qweight))
            counted.add(id(mod.scale))
            if mod.bias is not None:
                counted.add(id(mod.bias))

    for p in list(model.parameters()) + list(model.buffers()):
        if id(p) in counted:
            continue
        if p.dtype in (torch.float32, torch.float):
            total_bytes += p.numel() * 4
        elif p.dtype in (torch.float16, torch.bfloat16):
            total_bytes += p.numel() * 2
        elif p.dtype == torch.int8:
            total_bytes += p.numel()
        else:
            total_bytes += p.numel() * 4
    return total_bytes / (1024**2)
