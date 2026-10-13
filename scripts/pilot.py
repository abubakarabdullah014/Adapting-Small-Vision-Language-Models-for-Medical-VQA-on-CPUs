"""Pilot: measure step time and peak RAM for each model (CPU)."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.registry import MODEL_REGISTRY, configure_cpu, load_model_and_processor, peak_rss_gb
from src.peft_methods import apply_lora
from src.profiling import PeakRSSTracker


def pilot_model(model_key: str, steps: int = 3, dtype: str = "float32") -> dict:
    configure_cpu(8)
    rss = PeakRSSTracker()
    result = {
        "model": model_key,
        "status": "ok",
        "hf_id": MODEL_REGISTRY[model_key].hf_id,
        "approx_params_b": MODEL_REGISTRY[model_key].approx_params_b,
    }
    t_load0 = time.perf_counter()
    try:
        model, processor, spec = load_model_and_processor(model_key, dtype=dtype, device="cpu")
        rss.tick()
        result["load_s"] = time.perf_counter() - t_load0
        result["rss_after_load_gb"] = rss.peak

        model = apply_lora(model, r=8, alpha=16)
        model.train()
        rss.tick()
        result["rss_after_lora_gb"] = peak_rss_gb()

        # Load a few samples BEFORE heavy training to avoid RAM spikes
        train_samples = None
        test_samples = None
        try:
            from src.data.medvqa import build_prompt, load_medvqa
            from src.train import prepare_batch

            train_samples = load_medvqa("vqa_rad", split="train", max_samples=2)
            test_samples = load_medvqa("vqa_rad", split="test", max_samples=1)
            batch = [
                {
                    "image": s.image,
                    "question": s.question,
                    "answer_type": s.answer_type,
                    "prompt": build_prompt(s.question, s.answer_type, family=spec.family),
                    "answer": s.answer,
                }
                for s in train_samples
            ]
            optim = torch_optim(model)
            step_times = []
            for i in range(steps):
                t0 = time.perf_counter()
                inputs = prepare_batch(
                    processor,
                    batch[:1],
                    device="cpu",
                    max_length=256,
                    family=spec.family,
                )
                out = model(**inputs)
                loss = out.loss
                loss.backward()
                optim.step()
                optim.zero_grad(set_to_none=True)
                step_times.append(time.perf_counter() - t0)
                rss.tick()
                print(f"  step {i+1}/{steps}: {step_times[-1]:.2f}s | RSS={rss.peak:.2f}GB")
            result["step_times_s"] = step_times
            result["mean_step_s"] = sum(step_times) / len(step_times)
            result["est_200_steps_h"] = result["mean_step_s"] * 200 / 3600
        except Exception as e:
            result["train_pilot_error"] = str(e)
            result["train_pilot_trace"] = traceback.format_exc()

        # Inference latency probe
        try:
            from src.evaluate import generate_answer

            model.eval()
            if test_samples is None:
                from src.data.medvqa import load_medvqa

                test_samples = load_medvqa("vqa_rad", split="test", max_samples=1)
            s = test_samples[0]
            t0 = time.perf_counter()
            text, ntok = generate_answer(model, processor, s.image, s.question, s.answer_type, max_new_tokens=16)
            result["infer_s"] = time.perf_counter() - t0
            result["infer_tokens"] = ntok
            result["infer_preview"] = text[:120]
            rss.tick()
        except Exception as e:
            result["infer_error"] = str(e)

        result["peak_rss_gb"] = rss.peak
        # Feasibility gate for training
        result["train_feasible"] = rss.peak < 14.0 and result.get("mean_step_s", 999) < 180
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["trace"] = traceback.format_exc()
        result["peak_rss_gb"] = peak_rss_gb()
        result["train_feasible"] = False
    return result


def torch_optim(model):
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=2e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", default="results/pilot_report.json")
    args = ap.parse_args()

    Path("results").mkdir(parents=True, exist_ok=True)
    reports = []
    for m in args.models:
        print("=" * 60)
        print("PILOT", m)
        print("=" * 60)
        r = pilot_model(m, steps=args.steps)
        reports.append(r)
        print(json.dumps({k: r[k] for k in r if k != "trace" and k != "train_pilot_trace"}, indent=2))
        # Free memory between models
        import gc
        import torch

        gc.collect()
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print("Wrote", out)

    # Summary markdown
    md = ["# Pilot report\n", "| Model | Status | Peak RSS (GB) | Mean step (s) | Est. 200 steps (h) | Train feasible |",
          "|---|---|---:|---:|---:|:---:|"]
    for r in reports:
        md.append(
            f"| {r['model']} | {r['status']} | {r.get('peak_rss_gb', float('nan')):.2f} | "
            f"{r.get('mean_step_s', float('nan')):.1f} | {r.get('est_200_steps_h', float('nan')):.2f} | "
            f"{r.get('train_feasible', False)} |"
        )
    Path("results/pilot_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
