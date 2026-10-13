"""
Smoke test: validates metrics, tables, figures, and logging without downloading VLMs.
Uses synthetic predictions so the analysis pipeline is exercisable offline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import bootstrap_ci, mcnemar_closed, score_predictions


def main():
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    models = ["internvl25_1b", "qwen2_vl_2b", "smolvlm_2b", "qwen25_vl_3b"]
    methods = ["zeroshot", "lora", "adapter", "ia3", "dora"]
    quants = ["fp32", "int8", "int4"]

    # Synthetic closed/open labels
    n = 80
    golds = ["yes" if i % 2 == 0 else "left lower lobe infiltrate" for i in range(n)]
    types = ["closed" if i % 2 == 0 else "open" for i in range(n)]

    eval_path = results / "eval_runs_smoke.jsonl"
    # Smoke data is separate from real eval_runs.jsonl (integrity).
    with open(eval_path, "w", encoding="utf-8") as f:
        for model in models:
            for method in methods:
                for quant in quants:
                    # Simulate: LoRA > adapter > dora > ia3 > zeroshot; int4 slightly hurts; larger models better
                    base = 0.35 + 0.05 * models.index(model)
                    boost = {
                        "zeroshot": 0.0,
                        "ia3": 0.06,
                        "adapter": 0.09,
                        "dora": 0.10,
                        "lora": 0.12,
                    }[method]
                    qpen = {"fp32": 0.0, "int8": -0.01, "int4": -0.04}[quant]
                    p = np.clip(base + boost + qpen, 0.05, 0.95)
                    preds = []
                    for i in range(n):
                        if types[i] == "closed":
                            preds.append(golds[i] if rng.random() < p else ("no" if golds[i] == "yes" else "yes"))
                        else:
                            # open: sometimes copy gold, sometimes truncate
                            preds.append(golds[i] if rng.random() < p else golds[i].split()[0])
                    metrics = score_predictions(preds, golds, types)
                    scores = np.array(
                        [1.0 if (types[i] == "closed" and preds[i] == golds[i]) else metrics["open_f1"] for i in range(n)]
                    )
                    # simpler overall vector
                    from src.metrics import per_sample_scores

                    sc = per_sample_scores(preds, golds, types)
                    mean, lo, hi = bootstrap_ci(sc, n_boot=200, seed=0)
                    # Latency grows with model size; int4 may be slower on CPU
                    size = models.index(model) + 1
                    p50 = 2.5 * size * (1.15 if quant == "int4" else 0.85 if quant == "int8" else 1.0)
                    row = {
                        "run_name": f"smoke_{model}_{method}_{quant}_vqa_rad",
                        "model": model,
                        "method": method,
                        "quant": quant,
                        "dataset": "vqa_rad",
                        "seed": 0,
                        "n_samples": n,
                        "metrics": metrics,
                        "overall_mean": mean,
                        "overall_ci95": [lo, hi],
                        "latency": {
                            "p50_e2e_s": p50,
                            "p95_e2e_s": p50 * 1.4,
                            "mean_ttft_s": p50,
                            "mean_tok_s": 16 / p50,
                        },
                        "peak_rss_gb": 3.5 + 2.2 * size * (0.55 if quant == "int4" else 0.7 if quant == "int8" else 1.0),
                        "mean_rss_gb": 3.0 + 2.0 * size * (0.55 if quant == "int4" else 0.7 if quant == "int8" else 1.0),
                        "disk_size_mb": 1000 * size * (0.25 if quant == "int4" else 0.5 if quant == "int8" else 1.0),
                        "param_counts": {"total": int(1e9 * size), "trainable": 0},
                        "energy_kwh": 0.01 * size,
                        "wall_clock_s": p50 * n,
                        "smoke": True,
                        "note": "Synthetic smoke-test numbers for pipeline validation only; replace with logged real runs.",
                    }
                    f.write(json.dumps(row) + "\n")

    # McNemar demo
    preds_a = ["yes"] * 40 + ["no"] * 40
    preds_b = ["yes"] * 50 + ["no"] * 30
    golds_c = ["yes"] * 60 + ["no"] * 20
    types_c = ["closed"] * 80
    print("McNemar demo:", mcnemar_closed(preds_a, preds_b, golds_c, types_c))

    # Train smoke rows (separate file)
    with open(results / "train_runs_smoke.jsonl", "w", encoding="utf-8") as f:
        for model in models[:3]:
            for method in ["lora", "adapter", "ia3", "dora"]:
                f.write(
                    json.dumps(
                        {
                            "run_name": f"smoke_train_{model}_{method}",
                            "model": model,
                            "method": method,
                            "dataset": "vqa_rad",
                            "seed": 0,
                            "steps": 200,
                            "wall_clock_s": 3600 * (models.index(model) + 1),
                            "peak_rss_gb": 8 + models.index(model),
                            "trainable_pct": {"lora": 0.8, "adapter": 0.5, "ia3": 0.05, "dora": 0.85}[method],
                            "energy_kwh": 0.05 * (models.index(model) + 1),
                            "smoke": True,
                        }
                    )
                    + "\n"
                )

    print("Smoke JSONL written. Generating tables/figures...")
    from analysis.make_tables import main as tables_main
    from analysis.make_figures import main as figures_main

    tables_main()
    figures_main()
    print("Done.")


if __name__ == "__main__":
    main()
