from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score


def _auc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def core_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    """y in {0:E,1:B,2:S}; score higher = more sarcomatoid."""
    y, score = np.asarray(y), np.asarray(score)
    m = {
        "auc_E_vs_nonE": _auc(y > 0, score),
        "auc_nonS_vs_S": _auc(y == 2, score),
    }
    es = np.isin(y, [0, 2])
    m["auc_E_vs_S"] = _auc(y[es] == 2, score[es]) if es.sum() else float("nan")
    m["kendall_tau"] = float(kendalltau(y, score).statistic) if len(np.unique(y)) > 1 else float("nan")
    return m


def selection_metric(y, score) -> float:
    m = core_metrics(y, score)
    v = [m["auc_E_vs_nonE"], m["auc_nonS_vs_S"]]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def bootstrap_ci(y, score, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    point = core_metrics(y, score)
    samples = {k: [] for k in point}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        for k, v in core_metrics(y[i], score[i]).items():
            samples[k].append(v)
    out = {}
    for k, v in point.items():
        arr = np.array(samples[k], dtype=float)
        arr = arr[np.isfinite(arr)]
        lo, hi = (np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if len(arr) else (np.nan, np.nan))
        out[k] = {"value": v, "ci_low": float(lo), "ci_high": float(hi)}
    return out
