"""Memory, latency, and energy profiling utilities."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import psutil


@dataclass
class LatencyStats:
    ttft_s: float
    tokens_per_s: float
    e2e_s: float
    n_tokens: int


@dataclass
class ProfileReport:
    peak_rss_gb: float
    mean_rss_gb: float
    latencies: List[Dict[str, Any]]
    p50_e2e_s: float
    p95_e2e_s: float
    mean_ttft_s: float
    mean_tok_s: float
    energy_kwh: Optional[float] = None
    duration_s: Optional[float] = None


def current_rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


class PeakRSSTracker:
    def __init__(self):
        self.peak = current_rss_gb()
        self.samples: List[float] = []

    def tick(self) -> float:
        r = current_rss_gb()
        self.samples.append(r)
        self.peak = max(self.peak, r)
        return r


@contextmanager
def energy_tracker(project_name: str = "vlm_cpu_benchmark", output_dir: str = "results/energy"):
    """Optional CodeCarbon tracker; yields dict with energy_kwh."""
    os.makedirs(output_dir, exist_ok=True)
    tracker = None
    info: Dict[str, Any] = {"energy_kwh": None, "duration_s": None}
    try:
        from codecarbon import EmissionsTracker

        tracker = EmissionsTracker(
            project_name=project_name,
            output_dir=output_dir,
            save_to_file=True,
            log_level="error",
            measure_power_secs=30,
        )
        tracker.start()
        t0 = time.perf_counter()
        yield info
        info["duration_s"] = time.perf_counter() - t0
        emissions = tracker.stop()
        # CodeCarbon returns kg CO2e; energy is in tracker.final_emissions_data
        try:
            info["energy_kwh"] = float(tracker.final_emissions_data.energy_consumed)
        except Exception:
            info["energy_kwh"] = None
        info["emissions_kg"] = float(emissions) if emissions is not None else None
    except Exception as exc:
        info["error"] = str(exc)
        yield info


def measure_generation_latency(
    generate_fn,
    n_new_tokens_fn=None,
) -> LatencyStats:
    """
    Measure TTFT and tokens/s for one generation call.

    generate_fn should return (output, n_tokens) or just output.
    If streaming is unavailable, TTFT approximates full E2E (documented limitation).
    """
    t0 = time.perf_counter()
    out = generate_fn()
    t1 = time.perf_counter()
    e2e = t1 - t0
    if isinstance(out, tuple):
        _, n_tokens = out
    elif n_new_tokens_fn is not None:
        n_tokens = int(n_new_tokens_fn(out))
    else:
        n_tokens = 1
    n_tokens = max(int(n_tokens), 1)
    # Without token streaming on CPU generate(), TTFT ~= E2E for first token batch
    ttft = e2e
    tok_s = n_tokens / e2e if e2e > 0 else 0.0
    return LatencyStats(ttft_s=ttft, tokens_per_s=tok_s, e2e_s=e2e, n_tokens=n_tokens)


def summarize_latencies(latencies: List[LatencyStats]) -> ProfileReport:
    import numpy as np

    e2e = np.array([x.e2e_s for x in latencies], dtype=np.float64)
    ttft = np.array([x.ttft_s for x in latencies], dtype=np.float64)
    toks = np.array([x.tokens_per_s for x in latencies], dtype=np.float64)
    rss = current_rss_gb()
    return ProfileReport(
        peak_rss_gb=rss,
        mean_rss_gb=rss,
        latencies=[asdict(x) for x in latencies],
        p50_e2e_s=float(np.percentile(e2e, 50)) if len(e2e) else float("nan"),
        p95_e2e_s=float(np.percentile(e2e, 95)) if len(e2e) else float("nan"),
        mean_ttft_s=float(ttft.mean()) if len(ttft) else float("nan"),
        mean_tok_s=float(toks.mean()) if len(toks) else float("nan"),
    )
