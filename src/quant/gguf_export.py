"""GGUF / llama.cpp export helpers and documentation for deployment case study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def export_gguf_instructions(
    model_hf_id: str,
    out_dir: str = "checkpoints/gguf",
    quants=("Q8_0", "Q4_K_M"),
) -> Dict:
    """
    Write a reproducible shell script describing how to convert a HF VLM
    (or its LLM backbone) to GGUF with llama.cpp.

    Full multimodal GGUF conversion depends on llama.cpp vision support for
    the specific architecture; this function documents the exact commands used
    in the paper's deployment case study.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    script = out / "convert_gguf.sh"
    lines = [
        "#!/usr/bin/env bash",
        "# GGUF conversion case study for CPU deployment",
        "# Requires: llama.cpp built with CPU support",
        "set -euo pipefail",
        f'MODEL="{model_hf_id}"',
        f'OUT="{out.as_posix()}"',
        'LLAMA_CPP="${LLAMA_CPP:-./llama.cpp}"',
        'python "$LLAMA_CPP/convert_hf_to_gguf.py" "$MODEL" --outdir "$OUT"',
    ]
    for q in quants:
        lines.append(
            f'"$LLAMA_CPP/llama-quantize" "$OUT"/model-f16.gguf "$OUT"/model-{q}.gguf {q}'
        )
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "model_hf_id": model_hf_id,
        "out_dir": str(out),
        "quants": list(quants),
        "script": str(script),
        "notes": (
            "Measure latency with llama-cli -m model-Q4_K_M.gguf -p '<prompt>' "
            "and record tokens/s on the same CPU laptop used for PyTorch runs."
        ),
    }
    (out / "gguf_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
