# Results integrity

- `eval_runs.jsonl` / `train_runs.jsonl` — **real** logged experiments (source of truth for the paper).
- `eval_runs_smoke.jsonl` / `train_runs_smoke.jsonl` — synthetic pipeline checks only; never cite as scientific findings.
- `pilot_report.json` / `pilot_report.md` — CPU feasibility pilot (RAM + step time).
- `queue_done.txt` / `queue_failed.txt` — resumable experiment queue state.

`analysis/make_tables.py` prefers real JSONL files and falls back to smoke only when real logs are absent.
