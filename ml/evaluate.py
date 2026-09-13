"""Evaluation metrics — the single code path every reported number comes from.

Anomaly detection: precision, recall, F1 (thresholded) and PR-AUC via
average_precision_score (primary metric — anomalies are rare).
Forecasting: MAE, RMSE, conditional MAPE (only rows with |actual| >= epsilon; the
excluded-row count is always reported) and pinball loss per quantile plus the mean
across quantiles.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score


def precision_recall_f1(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def pr_auc(y_true, scores) -> float:
    return float(
        average_precision_score(np.asarray(y_true, dtype=bool), np.asarray(scores, dtype=float))
    )


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape_conditional(y_true, y_pred, epsilon: float) -> dict:
    """MAPE over rows with |actual| >= epsilon only."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) >= epsilon
    if not mask.any():
        return {"mape": None, "n_used": 0, "n_total": int(len(y_true))}
    value = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
    return {"mape": value, "n_used": int(mask.sum()), "n_total": int(len(y_true))}


def pinball_loss(y_true, y_pred, tau: float) -> float:
    """Mean pinball (quantile) loss: tau*d if d = y - pred >= 0 else (1-tau)*|d|."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def forecast_metrics(y_true, per_quantile_preds: dict[float, np.ndarray], epsilon: float) -> dict:
    """Full metric set for one horizon; MAPE is computed on the median prediction."""
    quantiles = sorted(per_quantile_preds)
    pinballs = {f"pinball_q{q}": pinball_loss(y_true, per_quantile_preds[q], q) for q in quantiles}
    median_key = min(quantiles, key=lambda q: abs(q - 0.5))
    median_pred = per_quantile_preds[median_key]
    mape_info = mape_conditional(y_true, median_pred, epsilon)
    return {
        **pinballs,
        "pinball_mean": float(np.mean(list(pinballs.values()))),
        "mae": mae(y_true, median_pred),
        "rmse": rmse(y_true, median_pred),
        "mape": mape_info["mape"],
        "mape_n_used": mape_info["n_used"],
        "mape_n_total": mape_info["n_total"],
    }


def aggregate_folds(per_fold: list) -> dict | None:
    """Aggregate a list of per-fold metric dicts (recursively) or scalar values.

    Returns {metric: {"mean", "std"}}; keys with no numeric values are dropped (None).
    """
    if not per_fold:
        return {}
    if isinstance(per_fold[0], dict):
        out: dict = {}
        for key in per_fold[0]:
            merged = aggregate_folds([fold[key] for fold in per_fold if key in fold])
            if merged is not None:
                out[key] = merged
        return out
    nums = [v for v in per_fold if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not nums:
        return None
    mean = sum(nums) / len(nums)
    var = sum((v - mean) ** 2 for v in nums) / len(nums)
    return {"mean": mean, "std": math.sqrt(var)}
