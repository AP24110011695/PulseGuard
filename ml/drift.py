"""Drift detection (PSI + residual monitoring) and the promotion gate (plan §10–§11).

Pure functions only — the DB wiring lives in the backend service layer and the CLI/demo
orchestration in scripts/. Conventions:

- PSI: reference = the champion's training-time feature distributions (quantile bin
  edges + frequencies, captured at registration and stored as a model artifact and in
  the drift_references table). Thresholds: <0.1 stable, 0.1–0.2 warn, >0.2 drift.
- Residual monitor: rolling mean absolute error of stored median forecasts vs actuals,
  compared to the champion's healthy-period baseline; drift when the last `sustained`
  consecutive windows all exceed `multiplier` × baseline.
- Promotion gate: the challenger is compared against the champion on the frozen golden
  set through the same evaluation code path; promotion requires the measured primary
  improvement (plus secondary/sanity rules) — a challenger never replaces the champion
  automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PSI_WARN = 0.10
PSI_DRIFT = 0.20


def psi_bin_edges(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Quantile bin edges from the reference distribution (deduplicated)."""
    reference = np.asarray(reference, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:  # constant feature: single bin spanning the value
        edges = np.array([reference.min() - 0.5, reference.max() + 0.5])
    return edges


def bin_frequencies(values: np.ndarray, edges: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Frequencies of values across the bins defined by `edges`; out-of-range values
    clip into the boundary bins, frequencies are epsilon-smoothed."""
    values = np.asarray(values, dtype=float)
    idx = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    counts = np.bincount(idx, minlength=len(edges) - 1)
    freqs = counts / max(len(values), 1)
    return np.clip(freqs, epsilon, None)


def psi(expected: np.ndarray, actual: np.ndarray, epsilon: float = 1e-6) -> float:
    """Population Stability Index over bin frequency vectors: Σ (a−e)·ln(a/e)."""
    e = np.clip(np.asarray(expected, dtype=float), epsilon, None)
    a = np.clip(np.asarray(actual, dtype=float), epsilon, None)
    e = e / e.sum()
    a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def feature_reference(X: np.ndarray, feature_names: list[str], n_bins: int = 10) -> dict:
    """Per-feature reference distributions (bin edges + frequencies) for a feature matrix."""
    out = {}
    for i, name in enumerate(feature_names):
        edges = psi_bin_edges(X[:, i], n_bins)
        out[name] = {
            "edges": edges.tolist(),
            "freqs": bin_frequencies(X[:, i], edges).tolist(),
        }
    return out


def build_reference_set(
    frames,
    feature_cfg,
    points_per_day: int,
    window_end: int,
    window_days: int,
    n_bins: int = 10,
) -> dict:
    """Per-series feature references over the trailing `window_days` of rows before
    `window_end` — the regime the model was calibrated on.

    Calendar encodings (hour_*/dow_*) are excluded: their distribution depends on the
    window's day composition, not on data health, and would fire spuriously when the
    check window is shorter than the reference window.
    """
    from ml.features import build_features, feature_names

    names = [
        n
        for n in feature_names(feature_cfg)
        if not (n.startswith("hour_") or n.startswith("dow_"))
    ]
    out = {}
    for name, frame in sorted(frames.items()):
        features = build_features(frame["value"], feature_cfg, frame["ts"])
        row_index = features.index.to_numpy()
        mask = (row_index >= feature_cfg.max_lookback) & (row_index < window_end)
        positions = np.flatnonzero(mask)[-window_days * points_per_day :]
        out[name] = feature_reference(features.to_numpy()[positions], names, n_bins)
    return out


def compute_feature_psi(
    reference: dict, X: np.ndarray, feature_names: list[str]
) -> dict[str, float]:
    """PSI per feature between the stored reference and the current-window matrix."""
    out: dict[str, float] = {}
    for i, name in enumerate(feature_names):
        ref = reference.get(name)
        if ref is None:
            continue
        actual = bin_frequencies(X[:, i], np.asarray(ref["edges"], dtype=float))
        out[name] = psi(np.asarray(ref["freqs"], dtype=float), actual)
    return out


def psi_status(value: float) -> str:
    if value < PSI_WARN:
        return "ok"
    if value < PSI_DRIFT:
        return "warn"
    return "drift"


def residual_drift(
    actuals,
    medians,
    baseline_mae: float | None,
    window: int,
    multiplier: float,
    sustained: int,
) -> dict:
    """Rolling residual monitor over consecutive windows of stored forecasts.

    Drift when the last `sustained` windows all exceed multiplier × baseline_mae;
    warn when only the most recent window breaches; ok otherwise; unknown when no
    baseline or not enough forecasts exist.
    """
    errors = np.abs(np.asarray(actuals, dtype=float) - np.asarray(medians, dtype=float))
    n_windows = len(errors) // window
    window_maes = [
        float(errors[i * window : (i + 1) * window].mean()) for i in range(n_windows)
    ]
    if baseline_mae is None or n_windows == 0:
        return {"status": "unknown", "window_maes": window_maes, "baseline_mae": baseline_mae}
    breaches = [m > multiplier * baseline_mae for m in window_maes]
    tail = breaches[-sustained:]
    if len(tail) == sustained and all(tail):
        status = "drift"
    elif breaches and breaches[-1]:
        status = "warn"
    else:
        status = "ok"
    return {
        "status": status,
        "window_maes": window_maes,
        "baseline_mae": baseline_mae,
        "multiplier": multiplier,
        "sustained": sustained,
        "breaches": breaches,
    }


@dataclass
class GateDecision:
    decision: str  # "promoted" | "rejected"
    reason: str
    rule_trace: dict = field(default_factory=dict)


def evaluate_gate(
    task: str,
    champion_metrics: dict,
    challenger_metrics: dict,
    gate_cfg: dict,
    challenger_train_rows: int,
) -> GateDecision:
    """Measured champion-vs-challenger comparison on the frozen golden set.

    Forecast: promote iff mean-pinball improvement ≥ min_relative_improvement AND MAE
    regression ≤ max_regression AND band coverage within tolerance of nominal.
    Anomaly: promote iff PR-AUC ≥ champion − auc_tolerance AND F1 ≥ champion −
    f1_tolerance. A challenger below min_train_rows is rejected outright.
    """
    if challenger_train_rows < int(gate_cfg.get("min_train_rows", 0)):
        return GateDecision(
            "rejected",
            f"insufficient challenger training data ({challenger_train_rows} rows < "
            f"{int(gate_cfg.get('min_train_rows', 0))})",
            {"min_train_rows": gate_cfg.get("min_train_rows", 0), "actual": challenger_train_rows},
        )

    if task == "forecast":
        champ_pinball = float(champion_metrics["mean_pinball"])
        chall_pinball = float(challenger_metrics["mean_pinball"])
        improvement = (champ_pinball - chall_pinball) / champ_pinball
        primary_ok = improvement >= float(gate_cfg["min_relative_improvement"])

        champ_mae = float(champion_metrics["mae"])
        chall_mae = float(challenger_metrics["mae"])
        mae_regression = (chall_mae - champ_mae) / champ_mae
        secondary_ok = mae_regression <= float(gate_cfg["max_regression"])

        nominal = float(gate_cfg["band_nominal_coverage"])
        tolerance = float(gate_cfg["coverage_tolerance"])
        coverage = float(challenger_metrics["band_coverage"])
        sanity_ok = abs(coverage - nominal) <= tolerance

        trace = {
            "primary": {
                "rule": f"mean pinball improvement >= {gate_cfg['min_relative_improvement']}",
                "improvement": improvement,
                "passed": primary_ok,
            },
            "secondary": {
                "rule": f"MAE regression <= {gate_cfg['max_regression']}",
                "regression": mae_regression,
                "passed": secondary_ok,
            },
            "sanity": {
                "rule": f"band coverage within ±{tolerance} of {nominal}",
                "coverage": coverage,
                "passed": sanity_ok,
            },
        }
        if primary_ok and secondary_ok and sanity_ok:
            return GateDecision(
                "promoted",
                f"mean pinball improved {improvement:.2%} on the golden set",
                trace,
            )
        failed = [k for k, v in trace.items() if not v["passed"]]
        return GateDecision(
            "rejected", f"gate rules failed: {', '.join(failed)}", trace
        )

    champ_auc = float(champion_metrics["pr_auc"])
    chall_auc = float(challenger_metrics["pr_auc"])
    auc_ok = chall_auc >= champ_auc - float(gate_cfg["auc_tolerance"])
    champ_f1 = float(champion_metrics["f1"])
    chall_f1 = float(challenger_metrics["f1"])
    f1_ok = chall_f1 >= champ_f1 - float(gate_cfg["f1_tolerance"])
    trace = {
        "primary": {
            "rule": f"PR-AUC >= champion - {gate_cfg['auc_tolerance']}",
            "champion": champ_auc,
            "challenger": chall_auc,
            "passed": bool(auc_ok),
        },
        "secondary": {
            "rule": f"F1 >= champion - {gate_cfg['f1_tolerance']}",
            "champion": champ_f1,
            "challenger": chall_f1,
            "passed": bool(f1_ok),
        },
    }
    if auc_ok and f1_ok:
        delta = chall_auc - champ_auc
        return GateDecision(
            "promoted", f"PR-AUC change {delta:+.4f} within tolerance on the golden set", trace
        )
    failed = [k for k, v in trace.items() if not v["passed"]]
    return GateDecision("rejected", f"gate rules failed: {', '.join(failed)}", trace)
