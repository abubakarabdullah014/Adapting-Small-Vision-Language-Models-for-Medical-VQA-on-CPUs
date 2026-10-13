"""Finalize experiment grid from pilot_report.json (gates 3B training, etc.)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    pilot_path = ROOT / "results" / "pilot_report.json"
    out = ROOT / "results" / "experiment_grid.json"
    if not pilot_path.exists():
        grid = {
            "status": "pilot_missing",
            "train_models": ["internvl25_1b", "qwen2_vl_2b", "smolvlm_2b"],
            "eval_only_models": ["qwen25_vl_3b"],
            "note": "Run scripts/pilot.py before full training. 3B gated by default.",
        }
    else:
        reports = json.loads(pilot_path.read_text(encoding="utf-8"))
        train, eval_only = [], []
        for r in reports:
            m = r["model"]
            if r.get("train_feasible"):
                train.append(m)
            else:
                eval_only.append(m)
        # Always keep known-large model gated if absent from pilot
        if "qwen25_vl_3b" not in train and "qwen25_vl_3b" not in eval_only:
            eval_only.append("qwen25_vl_3b")
        grid = {
            "status": "ready",
            "train_models": train,
            "eval_only_models": eval_only,
            "pilot": reports,
        }
    out.write_text(json.dumps(grid, indent=2), encoding="utf-8")
    print(json.dumps(grid, indent=2)[:2000])
    print("Wrote", out)


if __name__ == "__main__":
    main()
