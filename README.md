# Adapting Small Vision–Language Models for Medical VQA on CPUs

CPU-only benchmark of parameter-efficient fine-tuning (PEFT) and low-bit quantization for small vision–language models on medical visual question answering (VQA).

**Repository:** [github.com/abubakarabdullah014/Adapting-Small-Vision-Language-Models-for-Medical-VQA-on-CPUs](https://github.com/abubakarabdullah014/Adapting-Small-Vision-Language-Models-for-Medical-VQA-on-CPUs)

## Authors

1. **Abu Bakar Abdullah** — Air University, Islamabad — abubakar.abdullah549@gmail.com  
2. **Abdul Mohsin** — Chongqing University of Posts and Telecommunications, China — abdulmohsin9696@gmail.com  

## What this repo contains

| Path | Description |
|------|-------------|
| `src/` | Training, evaluation, PEFT methods, quantization, metrics, profiling |
| `configs/` | YAML experiment configs (train / eval) |
| `scripts/` | Resumable CPU queue, pilot, download helpers |
| `analysis/` | Scripts to build tables/figures from logged JSONL |
| `results/` | Aggregate logs (`eval_runs.jsonl`, `train_runs.jsonl`, summaries) |

**Not included (local only):** model checkpoints, Hugging Face cache, Python venv, and manuscript/LaTeX sources.

## Methods

- PEFT: LoRA (r=8, r=16), bottleneck adapters, IA³, DoRA  
- Precision: FP32, INT8, INT4 (weight-only, CPU)  
- Models: SmolVLM-256M-Instruct, SmolVLM-500M-Instruct  
- Primary dataset: VQA-RAD (optional SLAKE / PathVQA transfer)

## Hardware target

- Intel Core i7-class laptop CPU (tested on i7-1185G7)  
- ~16 GB RAM  
- No CUDA / GPU required  

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Install CPU torch if needed: https://pytorch.org

# Resumable experiment queue (skips jobs listed in results/queue_done.txt)
$env:HF_HUB_DISABLE_SYMLINKS_WARNING=1
python scripts\run_cpu_queue.py

# Summaries from logs
python analysis\make_tables.py
python analysis\make_figures.py
```

## Key results (logged)

Aggregate metrics are in `results/eval_runs.jsonl` and `results/eval_summary.csv`.  
Example (SmolVLM-500M, VQA-RAD, FP32, seed 0): LoRA-r16 overall ≈ **0.52** vs zero-shot ≈ **0.49**.

## License / citation

Code and logged results accompany the paper *Adapting Small Vision–Language Models for Medical VQA on CPUs*. Please cite the published version when available.
