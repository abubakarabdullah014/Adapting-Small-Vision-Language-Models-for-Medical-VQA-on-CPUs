"""Generate publication figures (PDF, 300 dpi) from results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "paper" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def load_eval() -> pd.DataFrame:
    csv = RESULTS / "eval_summary.csv"
    if csv.exists():
        return pd.read_csv(csv)
    # Build from jsonl via make_tables logic
    from analysis.make_tables import eval_df

    return eval_df()


def pareto_plot(df: pd.DataFrame, x: str, y: str, hue: str, out: Path, xlabel: str, ylabel: str, title: str):
    if df.empty or x not in df.columns or y not in df.columns:
        _placeholder(out, title)
        return
    d = df.dropna(subset=[x, y]).copy()
    if d.empty:
        _placeholder(out, title)
        return
    plt.figure(figsize=(6.2, 4.2))
    sns.scatterplot(data=d, x=x, y=y, hue=hue if hue in d.columns else None, style="method" if "method" in d.columns else None, s=80)
    # Pareto front (maximize y, minimize x)
    pts = d[[x, y]].to_numpy()
    order = np.argsort(pts[:, 0])
    best = -np.inf
    front = []
    for i in order:
        if pts[i, 1] > best:
            front.append(pts[i])
            best = pts[i, 1]
    if front:
        front = np.array(front)
        plt.plot(front[:, 0], front[:, 1], "k--", lw=1.2, label="Pareto front")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def latency_bars(df: pd.DataFrame, out: Path):
    if df.empty or "p50_e2e_s" not in df.columns:
        _placeholder(out, "Latency")
        return
    d = df.dropna(subset=["p50_e2e_s"]).copy()
    if d.empty:
        _placeholder(out, "Latency")
        return
    g = d.groupby(["model", "quant"], as_index=False)["p50_e2e_s"].mean()
    plt.figure(figsize=(7, 4))
    sns.barplot(data=g, x="model", y="p50_e2e_s", hue="quant")
    plt.ylabel("Median end-to-end latency (s)")
    plt.xlabel("Model")
    plt.title("CPU inference latency by precision")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def radar_methods(df: pd.DataFrame, out: Path):
    """Simple multi-metric radar for methods on one model if data exist."""
    if df.empty:
        _placeholder(out, "Radar")
        return
    d = df[df["dataset"] == "vqa_rad"].copy() if "dataset" in df.columns else df.copy()
    if d.empty:
        _placeholder(out, "Radar")
        return
    metrics = ["overall", "closed_acc", "open_f1"]
    for m in metrics:
        if m not in d.columns:
            _placeholder(out, "Radar")
            return
    # Invert latency into a score
    if "p50_e2e_s" in d.columns and d["p50_e2e_s"].notna().any():
        d["speed"] = 1.0 / d["p50_e2e_s"].clip(lower=1e-3)
        d["speed"] = d["speed"] / d["speed"].max()
        metrics = metrics + ["speed"]
    g = d.groupby("method")[metrics].mean()
    if g.empty:
        _placeholder(out, "Radar")
        return
    labels = metrics
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    for method, row in g.iterrows():
        vals = row.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, label=str(method))
        ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Method trade-offs (normalized)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def _placeholder(out: Path, title: str):
    plt.figure(figsize=(5, 3))
    plt.text(0.5, 0.5, f"{title}\n(awaiting experimental results)", ha="center", va="center")
    plt.axis("off")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def memory_accuracy(df: pd.DataFrame, out: Path):
    pareto_plot(
        df,
        x="peak_rss_gb",
        y="overall",
        hue="quant",
        out=out,
        xlabel="Peak RSS (GB)",
        ylabel="Overall score",
        title="Accuracy–memory Pareto (CPU)",
    )


def latency_accuracy(df: pd.DataFrame, out: Path):
    pareto_plot(
        df,
        x="p50_e2e_s",
        y="overall",
        hue="quant",
        out=out,
        xlabel="Median latency (s)",
        ylabel="Overall score",
        title="Accuracy–latency Pareto (CPU)",
    )


def main():
    # Ensure CSV exists
    try:
        import analysis.make_tables as mt

        mt.main()
    except Exception:
        pass
    df = load_eval()
    memory_accuracy(df, FIGS / "fig_pareto_memory.png")
    latency_accuracy(df, FIGS / "fig_pareto_latency.png")
    latency_bars(df, FIGS / "fig_latency_bars.png")
    radar_methods(df, FIGS / "fig_radar.png")
    print("Wrote figures to", FIGS)


if __name__ == "__main__":
    main()
