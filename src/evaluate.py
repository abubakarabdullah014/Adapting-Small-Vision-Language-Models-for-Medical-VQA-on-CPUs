"""Evaluation + profiling for medical VQA VLMs (CPU)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.medvqa import build_prompt, load_medvqa
from src.metrics import bootstrap_ci, per_sample_scores, score_predictions
from src.models.registry import configure_cpu, count_parameters, load_model_and_processor, peak_rss_gb
from src.profiling import LatencyStats, PeakRSSTracker, energy_tracker, measure_generation_latency, summarize_latencies
from src.quant.quanto_quant import model_disk_size_mb, quantize_model


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def maybe_load_adapter(model, adapter_path: Optional[str]):
    if not adapter_path:
        return model
    from peft import PeftModel

    from src.peft_methods.adapter import is_bottleneck_adapter_dir, load_bottleneck_adapters

    path = Path(adapter_path).expanduser().resolve()
    # Custom Houlsby bottleneck (not PEFT)
    if is_bottleneck_adapter_dir(path):
        model = load_bottleneck_adapters(model, path)
        print(f"[eval] loaded bottleneck adapter from {path}")
        return model

    if (path / "adapter" / "adapter_config.json").exists():
        path = path / "adapter"
    elif not (path / "adapter_config.json").exists():
        raise FileNotFoundError(f"No adapter_config.json / bottleneck adapter under {adapter_path}")
    # Absolute POSIX path avoids HF treating Windows paths as repo ids
    model = PeftModel.from_pretrained(model, path.as_posix())
    try:
        model = model.merge_and_unload()
        print(f"[eval] merged adapter from {path}")
    except Exception as e:
        print(f"[eval] merge skipped ({e}); using PeftModel as-is")
    return model


def _prompt_family(model) -> str:
    spec = getattr(model, "_benchmark_spec", None)
    return getattr(spec, "family", "generic") if spec is not None else "generic"


def generate_answer(model, processor, image, question: str, answer_type: str, max_new_tokens: int = 32) -> str:
    family = _prompt_family(model)
    prompt = build_prompt(question, answer_type, family=family)

    # Prefer chat template when available (SmolVLM / Qwen2-VL)
    inputs = None
    if hasattr(processor, "apply_chat_template"):
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt.replace("<image>", "").strip()},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt")
        except Exception:
            inputs = None
    if inputs is None:
        try:
            inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        except TypeError:
            inputs = processor(text=prompt, images=image, return_tensors="pt")

    inputs = {k: v.cpu() if hasattr(v, "cpu") else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # Decode only new tokens when possible
    if hasattr(out, "shape") and input_len > 0 and out.shape[-1] >= input_len:
        gen = out[:, input_len:]
    else:
        gen = out
    try:
        text = processor.batch_decode(gen, skip_special_tokens=True)[0]
    except Exception:
        text = processor.tokenizer.decode(gen[0], skip_special_tokens=True)
    # Strip prompt echo if present
    if "Answer:" in text:
        text = text.split("Answer:")[-1]
    return text.strip(), int(gen.shape[-1]) if hasattr(gen, "shape") else 1


def evaluate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    configure_cpu(int(cfg.get("num_threads", 8)))
    run_name = cfg.get(
        "run_name",
        f"eval_{cfg['model']}_{cfg.get('method','zeroshot')}_{cfg.get('quant','fp')}_{cfg['dataset']}",
    )
    out_dir = Path(cfg.get("output_dir", "results")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor, spec = load_model_and_processor(
        cfg["model"],
        dtype=cfg.get("dtype", "float32"),
        device=cfg.get("device", "cpu"),
        cache_dir=cfg.get("cache_dir", "data_cache"),
        max_pixels=cfg.get("max_pixels"),
    )
    model = maybe_load_adapter(model, cfg.get("adapter_path"))

    quant = cfg.get("quant") or cfg.get("precision") or "fp32"
    if quant in {"int8", "8", 8}:
        model = quantize_model(model, bits=8)
        quant_label = "int8"
    elif quant in {"int4", "4", 4}:
        model = quantize_model(model, bits=4)
        quant_label = "int4"
    else:
        quant_label = "fp32"

    model.eval()
    samples = load_medvqa(
        cfg["dataset"],
        split=cfg.get("eval_split", "test"),
        cache_dir=cfg.get("cache_dir", "data_cache"),
        max_samples=cfg.get("max_eval_samples"),
    )

    preds: List[str] = []
    golds: List[str] = []
    types: List[str] = []
    latencies: List[LatencyStats] = []
    rss = PeakRSSTracker()
    max_new = int(cfg.get("max_new_tokens", 32))

    t0 = time.perf_counter()
    with energy_tracker(project_name=run_name, output_dir=str(out_dir / "energy")) as energy:
        for s in tqdm(samples, desc=run_name):
            try:
                t_a = time.perf_counter()
                text, ntok = generate_answer(model, processor, s.image, s.question, s.answer_type, max_new)
                t_b = time.perf_counter()
                e2e = t_b - t_a
                latencies.append(
                    LatencyStats(
                        ttft_s=e2e,
                        tokens_per_s=(ntok / e2e) if e2e > 0 else 0.0,
                        e2e_s=e2e,
                        n_tokens=ntok,
                    )
                )
            except Exception as e:
                text = ""
                print(f"[eval] generation failed for {s.sample_id}: {e}")
                latencies.append(LatencyStats(ttft_s=0, tokens_per_s=0, e2e_s=0, n_tokens=0))

            preds.append(text)
            golds.append(s.answer)
            types.append(s.answer_type)
            rss.tick()

    metrics = score_predictions(preds, golds, types)
    scores = per_sample_scores(preds, golds, types)
    mean, lo, hi = bootstrap_ci(scores, n_boot=int(cfg.get("n_boot", 1000)), seed=int(cfg.get("seed", 0)))
    lat_summary = summarize_latencies(latencies)
    lat_summary.peak_rss_gb = rss.peak
    lat_summary.mean_rss_gb = float(sum(rss.samples) / max(len(rss.samples), 1))
    lat_summary.energy_kwh = energy.get("energy_kwh")
    lat_summary.duration_s = time.perf_counter() - t0

    # Save predictions
    pred_rows = [
        {
            "sample_id": s.sample_id,
            "question": s.question,
            "gold": g,
            "pred": p,
            "answer_type": t,
        }
        for s, p, g, t in zip(samples, preds, golds, types)
    ]
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for row in pred_rows:
            f.write(json.dumps(row) + "\n")

    result = {
        "run_name": run_name,
        "model": cfg["model"],
        "method": cfg.get("method", "zeroshot"),
        "quant": quant_label,
        "dataset": cfg["dataset"],
        "seed": cfg.get("seed", 0),
        "n_samples": len(samples),
        "metrics": metrics,
        "overall_mean": mean,
        "overall_ci95": [lo, hi],
        "latency": {
            "p50_e2e_s": lat_summary.p50_e2e_s,
            "p95_e2e_s": lat_summary.p95_e2e_s,
            "mean_ttft_s": lat_summary.mean_ttft_s,
            "mean_tok_s": lat_summary.mean_tok_s,
        },
        "peak_rss_gb": lat_summary.peak_rss_gb,
        "mean_rss_gb": lat_summary.mean_rss_gb,
        "disk_size_mb": model_disk_size_mb(model),
        "param_counts": count_parameters(model),
        "energy_kwh": lat_summary.energy_kwh,
        "wall_clock_s": lat_summary.duration_s,
        "spec": {"hf_id": spec.hf_id, "approx_params_b": spec.approx_params_b},
        "output_dir": str(out_dir),
    }
    with open(out_dir / "eval_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    results_dir = Path(cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "eval_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    n_empty = sum(1 for p in preds if not (p or "").strip())
    result["n_empty_preds"] = n_empty
    # Only hard-fail if generations clearly crashed (near-zero latency + mostly empty)
    mean_e2e = float((result.get("latency") or {}).get("mean_ttft_s") or 0.0)
    if len(preds) > 0 and n_empty / len(preds) > 0.8 and mean_e2e < 0.05:
        result["status"] = "failed_generations"
        with open(out_dir / "eval_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(json.dumps({k: result[k] for k in ("run_name", "metrics", "latency", "peak_rss_gb")}, indent=2))
        raise RuntimeError(
            f"Too many empty predictions ({n_empty}/{len(preds)}) with near-zero latency."
        )

    print(json.dumps({k: result[k] for k in ("run_name", "metrics", "latency", "peak_rss_gb")}, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
