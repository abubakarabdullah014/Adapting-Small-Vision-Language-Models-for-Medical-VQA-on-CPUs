"""Generate LaTeX tables from results/*.jsonl (source of truth)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "paper" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def eval_df() -> pd.DataFrame:
    rows = load_jsonl(RESULTS / "eval_runs.jsonl")
    # Fall back to smoke pipeline output only if no real runs exist yet
    if not rows:
        rows = load_jsonl(RESULTS / "eval_runs_smoke.jsonl")
        if rows:
            print("[make_tables] Using smoke results (no real eval_runs.jsonl yet).")

    flat = []
    for r in rows:
        m = r.get("metrics", {})
        lat = r.get("latency", {})
        flat.append(
            {
                "run_name": r.get("run_name"),
                "model": r.get("model"),
                "method": r.get("method"),
                "quant": r.get("quant"),
                "dataset": r.get("dataset"),
                "seed": r.get("seed"),
                "closed_acc": m.get("closed_acc"),
                "open_f1": m.get("open_f1"),
                "overall": m.get("overall"),
                "overall_ci_lo": (r.get("overall_ci95") or [None, None])[0],
                "overall_ci_hi": (r.get("overall_ci95") or [None, None])[1],
                "p50_e2e_s": lat.get("p50_e2e_s"),
                "p95_e2e_s": lat.get("p95_e2e_s"),
                "tok_s": lat.get("mean_tok_s"),
                "peak_rss_gb": r.get("peak_rss_gb"),
                "disk_mb": r.get("disk_size_mb"),
                "energy_kwh": r.get("energy_kwh"),
            }
        )
    return pd.DataFrame(flat)


def train_df() -> pd.DataFrame:
    rows = load_jsonl(RESULTS / "train_runs.jsonl")
    if not rows:
        rows = load_jsonl(RESULTS / "train_runs_smoke.jsonl")
        if rows:
            print("[make_tables] Using smoke train results (no real train_runs.jsonl yet).")

    flat = []
    for r in rows:
        flat.append(
            {
                "run_name": r.get("run_name"),
                "model": r.get("model"),
                "method": r.get("method"),
                "dataset": r.get("dataset"),
                "seed": r.get("seed"),
                "steps": r.get("steps"),
                "wall_clock_s": r.get("wall_clock_s"),
                "peak_rss_gb": r.get("peak_rss_gb"),
                "trainable_pct": r.get("trainable_pct"),
                "energy_kwh": r.get("energy_kwh"),
            }
        )
    return pd.DataFrame(flat)


def latex_escape(s: str) -> str:
    return (
        str(s)
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def df_to_latex(df: pd.DataFrame, caption: str, label: str, float_cols=None) -> str:
    if df.empty:
        return f"% Empty table: {label}\n\\begin{{table}}[t]\\caption{{{caption}}}\\label{{{label}}}\\centering\n(No results yet.)\\end{{table}}\n"
    float_cols = float_cols or []
    show = df.copy()
    for c in float_cols:
        if c in show.columns:
            show[c] = show[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "--")
    # Manual booktabs-ish table
    cols = list(show.columns)
    header = " & ".join(latex_escape(c) for c in cols) + " \\\\"
    lines = [
        "\\begin{table}[t]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{" + ("l" * len(cols)) + "}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    for _, row in show.iterrows():
        lines.append(" & ".join(latex_escape(row[c]) for c in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main():
    edf = eval_df()
    tdf = train_df()
    edf.to_csv(RESULTS / "eval_summary.csv", index=False)
    tdf.to_csv(RESULTS / "train_summary.csv", index=False)

    # Main accuracy table: model x method x quant on vqa_rad
    if not edf.empty:
        main = edf[edf["dataset"] == "vqa_rad"].copy()
        if not main.empty and "seed" in main.columns:
            agg = (
                main.groupby(["model", "method", "quant"], dropna=False)
                .agg(
                    overall=("overall", "mean"),
                    closed_acc=("closed_acc", "mean"),
                    open_f1=("open_f1", "mean"),
                    p50=("p50_e2e_s", "mean"),
                    rss=("peak_rss_gb", "mean"),
                )
                .reset_index()
            )
        else:
            agg = main
    else:
        agg = pd.DataFrame()

    tab1 = df_to_latex(
        agg,
        caption="Accuracy and efficiency on VQA-RAD (CPU). Means over seeds when available.",
        label="tab:main_vqarad",
        float_cols=["overall", "closed_acc", "open_f1", "p50", "rss"],
    )
    (TABLES / "tab_main_vqarad.tex").write_text(tab1, encoding="utf-8")

    # Efficiency table
    if not edf.empty:
        eff = (
            edf.groupby(["model", "quant"], dropna=False)
            .agg(p50=("p50_e2e_s", "mean"), p95=("p95_e2e_s", "mean"), tok_s=("tok_s", "mean"), rss=("peak_rss_gb", "mean"), disk=("disk_mb", "mean"))
            .reset_index()
        )
    else:
        eff = pd.DataFrame()
    tab2 = df_to_latex(
        eff,
        caption="Inference efficiency by model and precision on CPU.",
        label="tab:efficiency",
        float_cols=["p50", "p95", "tok_s", "rss", "disk"],
    )
    (TABLES / "tab_efficiency.tex").write_text(tab2, encoding="utf-8")

    tab3 = df_to_latex(
        tdf,
        caption="Training cost of PEFT methods on CPU (VQA-RAD).",
        label="tab:train_cost",
        float_cols=["wall_clock_s", "peak_rss_gb", "trainable_pct", "energy_kwh"],
    )
    (TABLES / "tab_train_cost.tex").write_text(tab3, encoding="utf-8")

    # Recommendation stub table
    reco = pd.DataFrame(
        [
            {"Scenario": "Highest accuracy (CPU RAM OK)", "Recommendation": "LoRA r=16 + FP32/INT8"},
            {"Scenario": "Memory-constrained clinic PC", "Recommendation": "LoRA r=8 + INT8"},
            {"Scenario": "Lowest latency CPU deploy", "Recommendation": "INT8 (or GGUF Q8\\_0); avoid INT4 if dequant is slow"},
            {"Scenario": "Fastest adaptation", "Recommendation": "IA3 (fewest trainable params)"},
        ]
    )
    (TABLES / "tab_recommendations.tex").write_text(
        df_to_latex(reco, "Deployment recommendations from the CPU benchmark.", "tab:reco"),
        encoding="utf-8",
    )
    print("Wrote tables to", TABLES)


if __name__ == "__main__":
    main()
