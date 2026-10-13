"""CPU training loop for PEFT methods on medical VQA."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running as script from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.medvqa import MedVQADataset, load_medvqa
from src.models.registry import configure_cpu, count_parameters, load_model_and_processor, peak_rss_gb
from src.peft_methods import apply_bottleneck_adapters, apply_dora, apply_ia3, apply_lora
from src.peft_methods.adapter import BottleneckAdapterConfig
from src.profiling import PeakRSSTracker, energy_tracker


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_peft(model, method: str, cfg: Dict[str, Any]):
    method = method.lower()
    if method in {"none", "zeroshot", "base"}:
        for p in model.parameters():
            p.requires_grad = False
        return model
    if method == "lora":
        return apply_lora(
            model,
            r=int(cfg.get("lora_r", 8)),
            alpha=int(cfg.get("lora_alpha", 16)),
            dropout=float(cfg.get("lora_dropout", 0.05)),
        )
    if method == "dora":
        return apply_dora(
            model,
            r=int(cfg.get("lora_r", 8)),
            alpha=int(cfg.get("lora_alpha", 16)),
            dropout=float(cfg.get("lora_dropout", 0.05)),
        )
    if method == "ia3":
        return apply_ia3(model)
    if method in {"adapter", "bottleneck"}:
        return apply_bottleneck_adapters(
            model,
            BottleneckAdapterConfig(
                bottleneck_dim=int(cfg.get("adapter_dim", 64)),
                dropout=float(cfg.get("adapter_dropout", 0.05)),
            ),
        )
    raise ValueError(f"Unknown PEFT method: {method}")


def collate_keep_list(batch):
    """Keep list-of-dicts for processor-side batching in the training step."""
    return batch


def prepare_batch(processor, batch, device="cpu", max_length=512, family: str = "generic"):
    """Build model inputs; supervise ONLY answer tokens (critical for VLM PEFT)."""
    from src.data.medvqa import build_prompt

    images = [b["image"] for b in batch]
    full_texts = []
    prompt_texts = []

    use_chat = hasattr(processor, "apply_chat_template") and family in {"smolvlm", "qwen2vl", "qwen25vl"}
    for b in batch:
        user_body = b.get("prompt") or build_prompt(
            b["question"], b.get("answer_type", "open"), family=family
        )
        user_body = user_body.replace("<image>", "").strip()
        if use_chat:
            user_msgs = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": user_body}],
                }
            ]
            prompt_txt = processor.apply_chat_template(user_msgs, add_generation_prompt=True)
            # SmolVLM/Qwen chat templates ignore plain-string content; must use
            # [{"type":"text","text": ...}] or the answer is dropped and only EOS is supervised.
            full_msgs = user_msgs + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(b["answer"])}],
                }
            ]
            full_txt = processor.apply_chat_template(full_msgs, add_generation_prompt=False)
        else:
            prompt_txt = user_body if "<image>" in user_body or family not in {"smolvlm"} else f"<image>\n{user_body}"
            if family in {"smolvlm", "idefics", "idefics3"} and "<image>" not in prompt_txt:
                prompt_txt = f"<image>\n{prompt_txt}"
            full_txt = prompt_txt + " " + b["answer"]
        prompt_texts.append(prompt_txt)
        full_texts.append(full_txt)

    def _encode(texts):
        try:
            return processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
        except TypeError:
            return processor(text=texts[0], images=images[0], return_tensors="pt")

    enc_full = _encode(full_texts)
    enc_prompt = _encode(prompt_texts)
    enc = {k: v.to(device) for k, v in enc_full.items()}

    if "input_ids" in enc:
        labels = enc["input_ids"].clone()
        if "attention_mask" in enc:
            labels = labels.masked_fill(enc["attention_mask"] == 0, -100)
        # Mask the prompt prefix (keep answer / assistant tokens only)
        prompt_lens = enc_prompt["attention_mask"].sum(dim=1).tolist()
        for i, plen in enumerate(prompt_lens):
            plen = int(plen)
            # leave at least 1 supervised token
            plen = min(plen, labels.shape[1] - 1)
            labels[i, :plen] = -100
        enc["labels"] = labels
    return enc


def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    configure_cpu(int(cfg.get("num_threads", 8)))
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    out_dir = Path(cfg.get("output_dir", "checkpoints")) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.dump(cfg), encoding="utf-8")

    model, processor, spec = load_model_and_processor(
        cfg["model"],
        dtype=cfg.get("dtype", "float32"),
        device=cfg.get("device", "cpu"),
        cache_dir=cfg.get("cache_dir", "data_cache"),
        max_pixels=cfg.get("max_pixels"),
    )

    model = apply_peft(model, cfg.get("method", "lora"), cfg)
    model.train()

    train_samples = load_medvqa(
        cfg["dataset"],
        split=cfg.get("train_split", "train"),
        cache_dir=cfg.get("cache_dir", "data_cache"),
        max_samples=cfg.get("max_train_samples"),
    )
    ds = MedVQADataset(train_samples, processor=None, for_training=True)
    loader = DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_keep_list,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters — check PEFT method.")
    optim = torch.optim.AdamW(params, lr=float(cfg.get("lr", 2e-4)), weight_decay=0.01)

    max_steps = int(cfg.get("max_steps", 200))
    grad_accum = int(cfg.get("grad_accum", 4))
    log_every = int(cfg.get("log_every", 10))
    save_every = int(cfg.get("save_every", 100))
    max_length = int(cfg.get("max_length", 512))

    # Resume
    start_step = 0
    ckpt_path = out_dir / "last.pt"
    if cfg.get("resume", True) and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model"], strict=False)
        optim.load_state_dict(state["optim"])
        start_step = int(state.get("step", 0))
        print(f"[train] Resumed from step {start_step}")

    tracker = PeakRSSTracker()
    history = []
    step = start_step
    t0 = time.perf_counter()
    running_loss = 0.0
    optim.zero_grad(set_to_none=True)

    with energy_tracker(project_name=cfg["run_name"], output_dir=str(out_dir / "energy")) as energy:
        pbar = tqdm(total=max_steps, initial=start_step, desc=cfg["run_name"])
        while step < max_steps:
            for batch in loader:
                if step >= max_steps:
                    break
                try:
                    inputs = prepare_batch(
                        processor,
                        batch,
                        device=cfg.get("device", "cpu"),
                        max_length=max_length,
                        family=spec.family,
                    )
                    use_amp = cfg.get("dtype") in {"bf16", "bfloat16"}
                    if use_amp and hasattr(torch, "autocast"):
                        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                            out = model(**inputs)
                            loss = out.loss / grad_accum
                    else:
                        out = model(**inputs)
                        loss = out.loss / grad_accum
                    loss.backward()
                    running_loss += float(loss.item()) * grad_accum
                    if (step + 1) % grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                        optim.step()
                        optim.zero_grad(set_to_none=True)
                    step += 1
                    pbar.update(1)
                    tracker.tick()
                    if step % log_every == 0:
                        avg = running_loss / log_every
                        running_loss = 0.0
                        rec = {
                            "step": step,
                            "loss": avg,
                            "rss_gb": tracker.peak,
                            "elapsed_s": time.perf_counter() - t0,
                        }
                        history.append(rec)
                        print(rec)
                    if step % save_every == 0 or step >= max_steps:
                        torch.save(
                            {"model": model.state_dict(), "optim": optim.state_dict(), "step": step},
                            ckpt_path,
                        )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "not enough memory" in str(e).lower():
                        print(f"[train] OOM at step {step}: {e}")
                        torch.save({"error": "oom", "step": step}, out_dir / "oom.json")
                        raise
                    raise
        pbar.close()

    # Save adapters (PEFT vs custom bottleneck)
    method = str(cfg.get("method", "")).lower()
    merged = False
    if method in {"adapter", "bottleneck"}:
        from src.peft_methods.adapter import save_bottleneck_adapters

        save_bottleneck_adapters(
            model,
            out_dir / "adapter",
            BottleneckAdapterConfig(
                bottleneck_dim=int(cfg.get("adapter_dim", 64)),
                dropout=float(cfg.get("adapter_dropout", 0.05)),
            ),
        )
    else:
        try:
            model.save_pretrained(out_dir / "adapter")
        except Exception:
            torch.save(model.state_dict(), out_dir / "model_state.pt")
        try:
            from peft import PeftModel

            if isinstance(model, PeftModel) or hasattr(model, "merge_and_unload"):
                merged_model = model.merge_and_unload()
                merged_model.save_pretrained(out_dir / "merged")
                processor.save_pretrained(out_dir / "merged")
                merged = True
        except Exception as e:
            print(f"[train] merge skipped: {e}")

    param_counts = count_parameters(model)
    result = {
        "run_name": cfg["run_name"],
        "model": cfg["model"],
        "method": cfg.get("method"),
        "dataset": cfg["dataset"],
        "seed": seed,
        "steps": step,
        "wall_clock_s": time.perf_counter() - t0,
        "peak_rss_gb": tracker.peak,
        "mean_rss_gb": float(sum(tracker.samples) / max(len(tracker.samples), 1)),
        "param_counts": param_counts,
        "trainable_pct": 100.0 * param_counts["trainable"] / max(param_counts["total"], 1),
        "energy_kwh": energy.get("energy_kwh"),
        "merged": merged,
        "output_dir": str(out_dir),
        "history": history,
        "spec": {"hf_id": spec.hf_id, "approx_params_b": spec.approx_params_b},
    }
    with open(out_dir / "train_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    # Also append to global results log
    results_dir = Path(cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "train_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if "run_name" not in cfg:
        cfg["run_name"] = (
            f"{cfg['model']}_{cfg.get('method','lora')}_{cfg['dataset']}_s{cfg.get('seed',0)}"
        )
    train(cfg)


if __name__ == "__main__":
    main()
