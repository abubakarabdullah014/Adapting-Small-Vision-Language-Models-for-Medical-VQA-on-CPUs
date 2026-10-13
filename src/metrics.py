"""Accuracy metrics, bootstrap CIs, and McNemar's test for medical VQA."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))


def token_f1(pred: str, gold: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = {}
    for t in p:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in g:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    prec = overlap / len(p)
    rec = overlap / len(g)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def recall_at_tokens(pred: str, gold: str) -> float:
    p = set(normalize_text(pred).split())
    g = set(normalize_text(gold).split())
    if not g:
        return 1.0 if not p else 0.0
    return len(p & g) / len(g)


def score_predictions(
    preds: Sequence[str],
    golds: Sequence[str],
    answer_types: Sequence[str],
) -> Dict[str, float]:
    """Aggregate closed accuracy, open recall/F1, and overall score."""
    closed_correct: List[float] = []
    open_f1: List[float] = []
    open_rec: List[float] = []
    overall: List[float] = []

    for pred, gold, atype in zip(preds, golds, answer_types):
        if atype == "closed":
            em = exact_match(pred, gold)
            closed_correct.append(em)
            overall.append(em)
        else:
            f1 = token_f1(pred, gold)
            rec = recall_at_tokens(pred, gold)
            open_f1.append(f1)
            open_rec.append(rec)
            overall.append(f1)

    def mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "closed_acc": mean(closed_correct),
        "n_closed": len(closed_correct),
        "open_f1": mean(open_f1),
        "open_recall": mean(open_rec),
        "n_open": len(open_f1),
        "overall": mean(overall),
        "n_total": len(overall),
    }


def per_sample_scores(
    preds: Sequence[str],
    golds: Sequence[str],
    answer_types: Sequence[str],
) -> np.ndarray:
    scores = []
    for pred, gold, atype in zip(preds, golds, answer_types):
        if atype == "closed":
            scores.append(exact_match(pred, gold))
        else:
            scores.append(token_f1(pred, gold))
    return np.asarray(scores, dtype=np.float64)


def bootstrap_ci(
    scores: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) for a 1-D score array."""
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    n = scores.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        means.append(scores[idx].mean())
    means = np.sort(means)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(scores.mean()), lo, hi


def mcnemar_closed(
    preds_a: Sequence[str],
    preds_b: Sequence[str],
    golds: Sequence[str],
    answer_types: Sequence[str],
) -> Dict[str, float]:
    """McNemar's test on closed-form questions between two systems."""
    correct_a = []
    correct_b = []
    for a, b, g, t in zip(preds_a, preds_b, golds, answer_types):
        if t != "closed":
            continue
        correct_a.append(exact_match(a, g) > 0.5)
        correct_b.append(exact_match(b, g) > 0.5)
    if not correct_a:
        return {"b": 0, "c": 0, "statistic": float("nan"), "pvalue": float("nan")}
    a_arr = np.asarray(correct_a, dtype=bool)
    b_arr = np.asarray(correct_b, dtype=bool)
    # b: A wrong, B correct; c: A correct, B wrong
    b = int((~a_arr & b_arr).sum())
    c = int((a_arr & ~b_arr).sum())
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "pvalue": 1.0}
    # Continuity-corrected McNemar
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(1 - stats.chi2.cdf(stat, df=1))
    return {"b": b, "c": c, "statistic": float(stat), "pvalue": p}
