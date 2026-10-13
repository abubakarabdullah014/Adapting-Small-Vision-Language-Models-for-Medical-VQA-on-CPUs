"""Generate YAML configs for the CPU-friendly small-VLM experiment grid."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"
CFG.mkdir(parents=True, exist_ok=True)

# Tiny models that fit a 16GB CPU laptop
MODELS = ["smolvlm_256m", "smolvlm_500m"]
TRAINABLE = ["smolvlm_256m", "smolvlm_500m"]
# Optional larger model (eval-only by default)
OPTIONAL_EVAL = ["smolvlm_2b"]
METHODS = ["lora", "lora_r16", "adapter", "ia3", "dora"]
DATASETS_EVAL = ["vqa_rad", "slake", "pathvqa"]
QUANTS = ["fp32", "int8", "int4"]
SEEDS = [0, 1, 2]


def write(name: str, cfg: dict) -> None:
    path = CFG / f"{name}.yaml"
    path.write_text(yaml.dump(cfg, sort_keys=False), encoding="utf-8")
    print("wrote", path.name)


def base(**kwargs):
    d = {
        "device": "cpu",
        "dtype": "float32",
        "num_threads": 8,
        "cache_dir": "data_cache",
        "results_dir": "results",
        "output_dir": "checkpoints",
        "max_pixels": 100352,  # 128*28*28 — lower visual tokens for RAM
        "max_length": 384,
        "max_new_tokens": 24,
        "resume": True,
    }
    d.update(kwargs)
    return d


def main():
    # Clear outdated large-model configs so the queue stays CPU-safe
    for old in CFG.glob("*.yaml"):
        if any(x in old.name for x in ("internvl", "qwen2_vl", "qwen25_vl")):
            old.unlink()
            print("removed", old.name)

    for m in MODELS:
        write(
            f"pilot_{m}",
            base(
                run_name=f"pilot_{m}",
                model=m,
                method="lora",
                dataset="vqa_rad",
                seed=0,
                max_train_samples=8,
                max_steps=5,
                batch_size=1,
                grad_accum=1,
                lr=2e-4,
                lora_r=8,
                lora_alpha=16,
                log_every=1,
                save_every=5,
                output_dir="checkpoints/pilot",
            ),
        )

    for m in MODELS + OPTIONAL_EVAL:
        for ds in DATASETS_EVAL:
            for q in QUANTS:
                name = f"eval_{m}_zeroshot_{q}_{ds}"
                write(
                    name,
                    base(
                        run_name=name,
                        model=m,
                        method="zeroshot",
                        quant=q,
                        dataset=ds,
                        eval_split="test",
                        max_eval_samples=50 if ds != "pathvqa" else 50,
                        seed=0,
                        output_dir="results",
                    ),
                )

    for m in TRAINABLE:
        for method in METHODS:
            for seed in SEEDS:
                lora_r = 16 if method == "lora_r16" else 8
                method_name = "lora" if method == "lora_r16" else method
                name = f"train_{m}_{method}_vqarad_s{seed}"
                write(
                    name,
                    base(
                        run_name=name,
                        model=m,
                        method=method_name,
                        dataset="vqa_rad",
                        train_split="train",
                        seed=seed,
                        lora_r=lora_r,
                        lora_alpha=32 if lora_r == 16 else 16,
                        adapter_dim=64,
                        max_train_samples=400,  # keep CPU training tractable
                        max_steps=150,
                        batch_size=1,
                        grad_accum=4,
                        lr=2e-4,
                        log_every=10,
                        save_every=50,
                    ),
                )
                for q in QUANTS:
                    ename = f"eval_{m}_{method}_{q}_vqarad_s{seed}"
                    write(
                        ename,
                        base(
                            run_name=ename,
                            model=m,
                            method=method_name,
                            quant=q,
                            dataset="vqa_rad",
                            eval_split="test",
                            adapter_path=f"checkpoints/{name}",
                            max_eval_samples=50,
                            seed=seed,
                            output_dir="results",
                        ),
                    )

    for m in TRAINABLE:
        for ds in ["slake", "pathvqa"]:
            name = f"eval_{m}_lora_fp32_{ds}_transfer"
            write(
                name,
                base(
                    run_name=name,
                    model=m,
                    method="lora",
                    quant="fp32",
                    dataset=ds,
                    eval_split="test",
                    adapter_path=f"checkpoints/train_{m}_lora_vqarad_s0",
                    max_eval_samples=50,
                    seed=0,
                    output_dir="results",
                ),
            )

    print("Done. Primary models:", MODELS)


if __name__ == "__main__":
    main()
