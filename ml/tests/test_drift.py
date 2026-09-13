import numpy as np
import pytest

from ml.drift import (
    GateDecision,
    bin_frequencies,
    compute_feature_psi,
    evaluate_gate,
    feature_reference,
    psi,
    psi_bin_edges,
    psi_status,
    residual_drift,
)


def test_psi_identical_distributions_is_zero():
    rng = np.random.default_rng(0)
    reference = rng.normal(0.0, 1.0, 5000)
    edges = psi_bin_edges(reference, n_bins=10)
    expected = bin_frequencies(reference, edges)
    actual = bin_frequencies(rng.normal(0.0, 1.0, 5000), edges)
    assert psi(expected, actual) < 0.01


def test_psi_detects_a_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(0.0, 1.0, 5000)
    edges = psi_bin_edges(reference, n_bins=10)
    expected = bin_frequencies(reference, edges)
    shifted = bin_frequencies(rng.normal(3.0, 1.0, 5000), edges)
    value = psi(expected, shifted)
    assert value > 0.2
    assert psi_status(value) == "drift"


def test_psi_hand_computed_two_bins():
    # expected [0.5, 0.5], actual [0.9, 0.1]: sum (a-e) ln(a/e)
    expected = np.array([0.5, 0.5])
    actual = np.array([0.9, 0.1])
    expected_value = (0.9 - 0.5) * np.log(0.9 / 0.5) + (0.1 - 0.5) * np.log(0.1 / 0.5)
    assert psi(expected, actual) == pytest.approx(expected_value)


def test_psi_status_thresholds():
    assert psi_status(0.05) == "ok"
    assert psi_status(0.15) == "warn"
    assert psi_status(0.25) == "drift"


def test_bin_frequencies_sum_to_one_and_clip():
    edges = np.array([0.0, 1.0, 2.0])  # two bins: [0,1) and [1,2]
    freqs = bin_frequencies(np.array([-5.0, 0.5, 1.5, 7.0]), edges)
    assert freqs.sum() == pytest.approx(1.0)
    assert freqs[0] == pytest.approx(0.5)  # -5 clips into the first bin
    assert freqs[1] == pytest.approx(0.5)  # 7.0 clips into the last bin


def test_constant_feature_reference_has_two_bins():
    edges = psi_bin_edges(np.full(100, 3.0))
    assert len(edges) == 2


def test_compute_feature_psi_reports_per_feature():
    rng = np.random.default_rng(1)
    X_ref = np.column_stack([rng.normal(0, 1, 4000), rng.normal(0, 1, 4000)])
    ref = feature_reference(X_ref, ["a", "b"], n_bins=10)
    X_now = np.column_stack([rng.normal(4, 1, 2000), rng.normal(0, 1, 2000)])
    psis = compute_feature_psi(ref, X_now, ["a", "b"])
    assert psis["a"] > 0.2
    assert psis["b"] < 0.1


def test_residual_drift_fires_on_sustained_breaches():
    baseline = 1.0
    window = 3
    errors_ok = [0.9] * 6
    errors_bad = [2.0] * 9  # three consecutive windows above 1.5x
    result = residual_drift(
        actuals=errors_ok + errors_bad,
        medians=[0.0] * 15,
        baseline_mae=baseline,
        window=window,
        multiplier=1.5,
        sustained=3,
    )
    assert result["status"] == "drift"


def test_residual_drift_warn_on_single_breach():
    result = residual_drift(
        actuals=[0.9, 0.9, 0.9, 2.0, 2.0, 2.0],
        medians=[0.0] * 6,
        baseline_mae=1.0,
        window=3,
        multiplier=1.5,
        sustained=3,  # only ONE window of data → cannot be sustained
    )
    assert result["status"] == "warn"


def test_residual_drift_ok_and_unknown():
    ok = residual_drift([0.9] * 6, [0.0] * 6, 1.0, 3, 1.5, 2)
    assert ok["status"] == "ok"
    unknown = residual_drift([0.9] * 6, [0.0] * 6, None, 3, 1.5, 2)
    assert unknown["status"] == "unknown"


def _forecast_metrics(pinball: float, mae: float, coverage: float) -> dict:
    return {"mean_pinball": pinball, "mae": mae, "band_coverage": coverage}


GATE = {
    "min_relative_improvement": 0.02,
    "max_regression": 0.01,
    "band_nominal_coverage": 0.90,
    "coverage_tolerance": 0.05,
    "auc_tolerance": 0.005,
    "f1_tolerance": 0.02,
    "min_train_rows": 100,
}


def test_gate_forecast_promotes_on_improvement():
    decision = evaluate_gate(
        "forecast",
        _forecast_metrics(1.0, 4.0, 0.90),
        _forecast_metrics(0.90, 3.9, 0.92),
        GATE,
        challenger_train_rows=5000,
    )
    assert decision.decision == "promoted"
    assert decision.rule_trace["primary"]["passed"] is True


def test_gate_forecast_rejects_without_improvement():
    decision = evaluate_gate(
        "forecast",
        _forecast_metrics(1.0, 4.0, 0.90),
        _forecast_metrics(1.0, 4.0, 0.90),
        GATE,
        challenger_train_rows=5000,
    )
    assert decision.decision == "rejected"
    assert "primary" in decision.reason


def test_gate_forecast_boundary_exact_threshold_promotes():
    # improvement exactly at the 2% threshold satisfies ">=" (plan §11)
    decision = evaluate_gate(
        "forecast",
        _forecast_metrics(1.0, 4.0, 0.90),
        _forecast_metrics(0.98, 4.0, 0.90),
        GATE,
        challenger_train_rows=5000,
    )
    assert decision.decision == "promoted"


def test_gate_forecast_rejects_on_mae_regression():
    decision = evaluate_gate(
        "forecast",
        _forecast_metrics(1.0, 4.0, 0.90),
        _forecast_metrics(0.90, 4.2, 0.90),  # MAE +5% > 1% allowed
        GATE,
        challenger_train_rows=5000,
    )
    assert decision.decision == "rejected"
    assert "secondary" in decision.reason


def test_gate_forecast_rejects_on_band_coverage():
    decision = evaluate_gate(
        "forecast",
        _forecast_metrics(1.0, 4.0, 0.90),
        _forecast_metrics(0.90, 3.9, 0.70),  # 20pp off nominal > 5pp tolerance
        GATE,
        challenger_train_rows=5000,
    )
    assert decision.decision == "rejected"
    assert "sanity" in decision.reason


def test_gate_rejects_insufficient_training_data():
    decision = evaluate_gate(
        "forecast",
        _forecast_metrics(1.0, 4.0, 0.90),
        _forecast_metrics(0.5, 2.0, 0.90),
        GATE,
        challenger_train_rows=50,
    )
    assert decision.decision == "rejected"
    assert "insufficient" in decision.reason


def test_gate_anomaly_promotes_and_rejects():
    champion = {"pr_auc": 0.60, "f1": 0.40}
    better = {"pr_auc": 0.75, "f1": 0.55}
    promoted = evaluate_gate("anomaly", champion, better, GATE, 5000)
    assert promoted.decision == "promoted"

    worse = {"pr_auc": 0.55, "f1": 0.30}
    rejected = evaluate_gate("anomaly", champion, worse, GATE, 5000)
    assert rejected.decision == "rejected"

    # within tolerance (PR-AUC −0.003, F1 −0.01) → still promoted
    near = {"pr_auc": 0.597, "f1": 0.39}
    assert evaluate_gate("anomaly", champion, near, GATE, 5000).decision == "promoted"


def test_gate_returns_gate_decision_type():
    decision = evaluate_gate(
        "forecast", _forecast_metrics(1.0, 4.0, 0.9), _forecast_metrics(1.0, 4.0, 0.9), GATE, 5000
    )
    assert isinstance(decision, GateDecision)
    assert set(decision.rule_trace) == {"primary", "secondary", "sanity"}
