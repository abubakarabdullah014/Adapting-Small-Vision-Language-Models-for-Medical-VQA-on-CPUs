"""CPU-optimized VLM registry for small medical VQA experiments (16GB laptop)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class ModelSpec:
    name: str
    hf_id: str
    approx_params_b: float
    family: str  # smolvlm | qwen2vl | qwen25vl | internvl
    default_max_pixels: int = 128 * 28 * 28  # lower visual tokens for CPU RAM
    trust_remote_code: bool = True
    cpu_priority: str = "primary"  # primary | optional


# Primary stack: tiny SmolVLMs that fit a 16GB CPU laptop.
# Optional larger models remain registered but are gated by the pilot.
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "smolvlm_256m": ModelSpec(
        name="smolvlm_256m",
        hf_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        approx_params_b=0.256,
        family="smolvlm",
        cpu_priority="primary",
    ),
    "smolvlm_500m": ModelSpec(
        name="smolvlm_500m",
        hf_id="HuggingFaceTB/SmolVLM-500M-Instruct",
        approx_params_b=0.5,
        family="smolvlm",
        cpu_priority="primary",
    ),
    "smolvlm_2b": ModelSpec(
        name="smolvlm_2b",
        hf_id="HuggingFaceTB/SmolVLM-Instruct",
        approx_params_b=2.2,
        family="smolvlm",
        cpu_priority="optional",  # already downloaded; eval-only unless pilot says OK
    ),
}


def list_models() -> Dict[str, ModelSpec]:
    return dict(MODEL_REGISTRY)


def primary_models() -> list:
    return [k for k, v in MODEL_REGISTRY.items() if v.cpu_priority == "primary"]


def configure_cpu(num_threads: int = 8) -> None:
    """Pin PyTorch CPU threads for reproducible latency."""
    torch.set_num_threads(num_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "float32").lower()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def load_model_and_processor(
    model_key: str,
    dtype: str = "float32",
    device: str = "cpu",
    cache_dir: Optional[str] = "data_cache",
    max_pixels: Optional[int] = None,
    low_cpu_mem_usage: bool = True,
) -> Tuple[Any, Any, ModelSpec]:
    """
    Load a registered VLM and its processor on CPU.

    Returns (model, processor, spec).
    """
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_key}'. Choose from {list(MODEL_REGISTRY)}")

    configure_cpu()
    spec = MODEL_REGISTRY[model_key]
    torch_dtype = _dtype_from_str(dtype)
    pixels = max_pixels if max_pixels is not None else spec.default_max_pixels

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        spec.hf_id,
        trust_remote_code=spec.trust_remote_code,
        cache_dir=cache_dir,
    )

    model = _load_family_model(spec, torch_dtype, cache_dir, low_cpu_mem_usage)
    model.to(device)
    model.eval()

    model._benchmark_spec = spec
    model._benchmark_max_pixels = pixels
    if hasattr(processor, "image_processor"):
        for attr, val in (
            ("max_pixels", pixels),
            ("size", {"longest_edge": 512}),
        ):
            if hasattr(processor.image_processor, attr):
                try:
                    setattr(processor.image_processor, attr, val)
                except Exception:
                    pass

    return model, processor, spec


def _load_family_model(
    spec: ModelSpec,
    torch_dtype: torch.dtype,
    cache_dir: Optional[str],
    low_cpu_mem_usage: bool,
):
    common = dict(
        trust_remote_code=spec.trust_remote_code,
        cache_dir=cache_dir,
        low_cpu_mem_usage=low_cpu_mem_usage,
        torch_dtype=torch_dtype,
    )

    if spec.family == "qwen2vl":
        from transformers import Qwen2VLForConditionalGeneration

        return Qwen2VLForConditionalGeneration.from_pretrained(spec.hf_id, **common)

    if spec.family == "qwen25vl":
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration

            return Qwen2_5_VLForConditionalGeneration.from_pretrained(spec.hf_id, **common)
        except Exception:
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq.from_pretrained(spec.hf_id, **common)

    if spec.family == "smolvlm":
        try:
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq.from_pretrained(spec.hf_id, **common)
        except Exception:
            from transformers import Idefics3ForConditionalGeneration

            return Idefics3ForConditionalGeneration.from_pretrained(spec.hf_id, **common)

    if spec.family == "internvl":
        from transformers import AutoModel

        return AutoModel.from_pretrained(spec.hf_id, **common)

    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(spec.hf_id, **common)


def count_parameters(model) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def peak_rss_gb() -> float:
    """Current process RSS in GB."""
    import psutil

    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
