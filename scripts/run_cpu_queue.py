"""Resumable CPU experiment queue for small SmolVLMs (train-before-eval order)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CFG = ROOT / "configs"
DONE = ROOT / "results" / "queue_done.txt"
FAILED = ROOT / "results" / "queue_failed.txt"
DONE.parent.mkdir(parents=True, exist_ok=True)


def load_set(path: Path) -> set:
    if not path.exists():
        return set()
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def mark(path: Path, name: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(name + "\n")


def ordered_configs() -> list[Path]:
    """Zero-shot first, then for each model/method/seed: train then its evals."""
    out: list[Path] = []
    seen = set()

    def add(pat: str):
        for p in sorted(CFG.glob(pat)):
            if "smolvlm_2b" in p.name:
                continue
            if p.name in seen:
                continue
            seen.add(p.name)
            out.append(p)

    # 1) Core zero-shot on VQA-RAD (already have results; kept for resume)
    add("eval_smolvlm_*_zeroshot_*_vqa_rad.yaml")

    models = ["smolvlm_256m", "smolvlm_500m"]
    methods = ["lora", "lora_r16", "adapter", "ia3", "dora"]
    seeds = [0, 1, 2]
    quants = ["fp32", "int8", "int4"]

    # 2) Train then eval on VQA-RAD FIRST (main paper results)
    for m in models:
        for method in methods:
            for seed in seeds:
                train_name = f"train_{m}_{method}_vqarad_s{seed}.yaml"
                add(train_name)
                for q in quants:
                    add(f"eval_{m}_{method}_{q}_vqarad_s{seed}.yaml")

    # 3) Extra datasets last (optional; do not block main VQA-RAD PEFT runs)
    # Set RUN_EXTRA_DATASETS=1 to include SLAKE / PathVQA in the queue.
    import os

    if os.environ.get("RUN_EXTRA_DATASETS", "0") == "1":
        add("eval_smolvlm_*_zeroshot_*_slake.yaml")
        add("eval_smolvlm_*_zeroshot_*_pathvqa.yaml")
        for m in models:
            add(f"eval_{m}_lora_fp32_slake_transfer.yaml")
            add(f"eval_{m}_lora_fp32_pathvqa_transfer.yaml")

    return out


def adapter_ready(cfg: Path) -> bool:
    """Skip PEFT evals whose training checkpoint is missing."""
    if not cfg.name.startswith("eval_"):
        return True
    if "zeroshot" in cfg.name:
        return True
    import yaml

    from src.peft_methods.adapter import is_bottleneck_adapter_dir

    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    ap = data.get("adapter_path")
    if not ap:
        return True
    p = ROOT / ap
    if (p / "adapter" / "adapter_config.json").exists() or (p / "adapter_config.json").exists():
        return True
    return is_bottleneck_adapter_dir(p)


def run_one(cfg: Path) -> int:
    name = cfg.stem
    mode = "train" if name.startswith("train_") else "eval"
    if mode == "eval" and not adapter_ready(cfg):
        print(f"[defer] {name} (adapter not trained yet)")
        return -2
    script = ROOT / "src" / ("train.py" if mode == "train" else "evaluate.py")
    print(f"\n=== [{mode}] {name} ===", flush=True)
    proc = subprocess.run([sys.executable, str(script), "--config", str(cfg)], cwd=str(ROOT))
    return proc.returncode


def main():
    done = load_set(DONE)
    failed = load_set(FAILED)
    configs = ordered_configs()
    print(f"Queue size={len(configs)} done={len(done)} failed={len(failed)}")
    deferred = []
    for cfg in configs:
        name = cfg.stem
        if name in done:
            print(f"[skip] {name}")
            continue
        code = run_one(cfg)
        if code == 0:
            mark(DONE, name)
            print(f"[ok] {name}")
        elif code == -2:
            deferred.append(name)
        else:
            mark(FAILED, name)
            print(f"[fail] {name} exit={code}")
    if deferred:
        print("Deferred (missing adapters):", len(deferred))
    print("QUEUE_DONE")
    subprocess.run([sys.executable, str(ROOT / "analysis" / "make_tables.py")], cwd=str(ROOT))
    subprocess.run([sys.executable, str(ROOT / "analysis" / "make_figures.py")], cwd=str(ROOT))


if __name__ == "__main__":
    main()
