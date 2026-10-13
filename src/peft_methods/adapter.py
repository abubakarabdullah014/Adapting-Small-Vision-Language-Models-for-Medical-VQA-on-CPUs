"""Houlsby-style bottleneck adapters for Transformer blocks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


BOTTLENECK_META = "bottleneck_adapter.json"
BOTTLENECK_WEIGHTS = "adapter_weights.pt"


@dataclass
class BottleneckAdapterConfig:
    bottleneck_dim: int = 64
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    init_scale: float = 1e-3


class BottleneckAdapter(nn.Module):
    """Down-project -> nonlinearity -> up-project with residual."""

    def __init__(self, hidden_size: int, bottleneck_dim: int = 64, dropout: float = 0.05, init_scale: float = 1e-3):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_dim, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.down.weight, std=init_scale)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.up(self.act(self.down(x))))


class AdapterWrappedLinear(nn.Module):
    """Wraps an nn.Linear and applies a bottleneck adapter after it."""

    def __init__(self, linear: nn.Linear, adapter: BottleneckAdapter):
        super().__init__()
        self.linear = linear
        self.adapter = adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.linear(x))

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


def _freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _is_lm_path(name: str) -> bool:
    return "text_model" in name or "language_model" in name or "model.layers" in name


def apply_bottleneck_adapters(
    model: nn.Module,
    config: Optional[BottleneckAdapterConfig] = None,
) -> nn.Module:
    """
    Insert Houlsby-style bottleneck adapters after selected Linear modules.

    Freezes the base model; only adapter parameters train.
    Prefers language-model paths so vision towers are left alone.
    """
    cfg = config or BottleneckAdapterConfig()
    targets = set(cfg.target_modules or ["o_proj", "down_proj"])
    _freeze_all(model)

    candidates = []
    for name, module in list(model.named_modules()):
        leaf = name.split(".")[-1]
        if leaf not in targets or not isinstance(module, nn.Linear):
            continue
        candidates.append((name, module))

    lm_only = [(n, m) for n, m in candidates if _is_lm_path(n)]
    to_wrap = lm_only or candidates

    replaced = 0
    for name, module in to_wrap:
        leaf = name.split(".")[-1]
        parent_name = ".".join(name.split(".")[:-1])
        parent = model.get_submodule(parent_name) if parent_name else model
        # Skip if already wrapped
        if isinstance(getattr(parent, leaf, None), AdapterWrappedLinear):
            continue
        hidden = module.out_features
        adapter = BottleneckAdapter(
            hidden_size=hidden,
            bottleneck_dim=cfg.bottleneck_dim,
            dropout=cfg.dropout,
            init_scale=cfg.init_scale,
        )
        wrapped = AdapterWrappedLinear(module, adapter)
        setattr(parent, leaf, wrapped)
        replaced += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[Adapter] Replaced {replaced} modules | "
        f"trainable={trainable:,} ({100 * trainable / max(total, 1):.4f}% of {total:,})"
    )
    return model


def save_bottleneck_adapters(
    model: nn.Module,
    path: Union[str, Path],
    config: Optional[BottleneckAdapterConfig] = None,
) -> Path:
    """Save only adapter weights + a small JSON marker (not full HF model)."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if ".adapter." in k}
    if not state:
        raise RuntimeError("No bottleneck adapter weights found to save.")
    torch.save(state, out / BOTTLENECK_WEIGHTS)
    meta: Dict[str, Any] = {
        "type": "bottleneck",
        "num_tensors": len(state),
        "config": asdict(config or BottleneckAdapterConfig()),
    }
    (out / BOTTLENECK_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[Adapter] saved {len(state)} tensors -> {out}")
    return out


def load_bottleneck_adapters(
    model: nn.Module,
    path: Union[str, Path],
    config: Optional[BottleneckAdapterConfig] = None,
) -> nn.Module:
    """Re-wrap model with bottleneck adapters and load saved weights."""
    path = Path(path)
    meta_path = path / BOTTLENECK_META
    weights_path = path / BOTTLENECK_WEIGHTS
    if not weights_path.exists():
        # allow path pointing at run dir
        if (path / "adapter" / BOTTLENECK_WEIGHTS).exists():
            path = path / "adapter"
            meta_path = path / BOTTLENECK_META
            weights_path = path / BOTTLENECK_WEIGHTS
    if not weights_path.exists():
        raise FileNotFoundError(f"No {BOTTLENECK_WEIGHTS} under {path}")

    cfg = config
    if cfg is None and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        c = meta.get("config") or {}
        cfg = BottleneckAdapterConfig(
            bottleneck_dim=int(c.get("bottleneck_dim", 64)),
            dropout=float(c.get("dropout", 0.05)),
            target_modules=c.get("target_modules"),
            init_scale=float(c.get("init_scale", 1e-3)),
        )
    cfg = cfg or BottleneckAdapterConfig()
    model = apply_bottleneck_adapters(model, cfg)
    state = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"[Adapter] loaded {len(state)} tensors from {weights_path} "
        f"(missing={len(missing)} unexpected={len(unexpected)})"
    )
    return model


def is_bottleneck_adapter_dir(path: Union[str, Path]) -> bool:
    p = Path(path)
    return (p / BOTTLENECK_META).exists() or (p / BOTTLENECK_WEIGHTS).exists() or (
        (p / "adapter" / BOTTLENECK_META).exists() or (p / "adapter" / BOTTLENECK_WEIGHTS).exists()
    )
