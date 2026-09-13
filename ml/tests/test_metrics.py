import numpy as np
import pytest
from sklearn.metrics import average_precision_score

import ml.evaluate as ev


def test_pinball_hand_computed():
    # y >= pred: loss = tau * (y - pred)
    assert ev.pinball_loss([10.0], [8.0], 0.5) == 1.0
    assert ev.pinball_loss([10.0], [8.0], 0.05) == 0.1
    assert ev.pinball_loss([10.0], [8.0], 0.95) == 1.9
    # y < pred: loss = (1 - tau) * (pred - y)
    assert ev.pinball_loss([10.0], [12.0], 0.5) == 1.0
    assert ev.pinball_loss([10.0], [12.0], 0.05) == 1.9
    assert ev.pinball_loss([10.0], [12.0], 0.95) == pytest.approx(0.1)


def test_pinball_penalizes_wrong_side_asy():
    # over-predicting at a low quantile is punished more than under-predicting
    low_over = ev.pinball_loss([10.0], [12.0], 0.05)
    low_under = ev.pinball_loss([10.0], [8.0], 0.05)
    assert low_over > low_under


def test_mae_rmse_hand_computed():
    y = [1.0, 2.0, 3.0]
    pred = [2.0, 2.0, 5.0]
    assert ev.mae(y, pred) == pytest.approx((1.0 + 0.0 + 2.0) / 3)
    assert ev.rmse(y, pred) == pytest.approx(np.sqrt((1.0 + 0.0 + 4.0) / 3))


def test_mape_conditional_excludes_small_actuals():
    y = [10.0, 0.5, 2.0]
    pred = [11.0, 1.0, 1.0]
    result = ev.mape_conditional(y, pred, epsilon=1.0)
    assert result["n_total"] == 3
    assert result["n_used"] == 2  # |0.5| < epsilon excluded
    assert result["mape"] == pytest.approx((0.1 + 0.5) / 2)


def test_mape_conditional_all_excluded():
    result = ev.mape_conditional([0.0], [1.0], epsilon=1.0)
    assert result["mape"] is None
    assert result["n_used"] == 0


def test_precision_recall_f1_hand_computed():
    metrics = ev.precision_recall_f1([1, 1, 0, 0], [1, 0, 1, 0])
    assert metrics["tp"] == 1 and metrics["fp"] == 1 and metrics["fn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_precision_recall_f1_zero_division_is_zero():
    metrics = ev.precision_recall_f1([0, 0], [1, 1])
    assert metrics["precision"] == 0.0 and metrics["f1"] == 0.0


def test_pr_auc_matches_sklearn_reference():
    y = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    assert ev.pr_auc(y, scores) == pytest.approx(average_precision_score(y, scores))


def test_forecast_metrics_bundle():
    y = np.array([10.0, 12.0, 8.0])
    preds = {
        0.05: np.array([9.0, 11.0, 7.0]),
        0.5: np.array([10.0, 12.0, 9.0]),
        0.95: np.array([11.0, 13.0, 10.0]),
    }
    metrics = ev.forecast_metrics(y, preds, epsilon=1.0)
    assert metrics["pinball_q0.5"] == pytest.approx(ev.pinball_loss(y, preds[0.5], 0.5))
    assert metrics["pinball_mean"] == pytest.approx(
        np.mean([metrics["pinball_q0.05"], metrics["pinball_q0.5"], metrics["pinball_q0.95"]])
    )
    assert metrics["mae"] == pytest.approx(np.mean([0.0, 0.0, 1.0]))
    assert metrics["mape_n_used"] == 3


def test_aggregate_folds_mean_and_std():
    folds = [{"f1": 0.2, "nested": {"mae": 1.0}}, {"f1": 0.4, "nested": {"mae": 3.0}}]
    agg = ev.aggregate_folds(folds)
    assert agg["f1"]["mean"] == pytest.approx(0.3)
    assert agg["f1"]["std"] == pytest.approx(np.std([0.2, 0.4]))
    assert agg["nested"]["mae"]["mean"] == pytest.approx(2.0)
