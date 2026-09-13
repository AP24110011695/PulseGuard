import numpy as np
import pandas as pd
import pytest

from ml.features import FeatureConfig, build_features, feature_names

N = 400


@pytest.fixture
def series():
    rng = np.random.default_rng(0)
    ts = pd.date_range("2026-01-01", periods=N, freq="min")
    values = pd.Series(rng.normal(50.0, 5.0, N))
    return values, ts


def test_feature_names_and_count(series):
    values, ts = series
    cfg = FeatureConfig()
    feats = build_features(values, cfg, ts)
    assert list(feats.columns) == feature_names(cfg)
    expected = (
        1
        + len(cfg.lags)
        + len(cfg.rolling_windows) * len(cfg.rolling_stats)
        + len(cfg.roc_offsets)
        + 1  # z-score
        + 4  # calendar
    )
    assert feats.shape[1] == expected


def test_warmup_rows_dropped_and_no_nan(series):
    values, ts = series
    cfg = FeatureConfig()
    feats = build_features(values, cfg, ts)
    assert len(feats) == N - cfg.max_lookback
    assert feats.index[0] == cfg.max_lookback  # positional labels preserved
    assert not feats.isna().any().any()


def test_lags_use_only_past(series):
    values, ts = series
    feats = build_features(values, FeatureConfig(), ts)
    for lag in (1, 5, 60):
        expected = values.shift(lag)[feats.index]
        assert np.allclose(feats[f"lag_{lag}"], expected)


def test_rolling_windows_end_at_current_row(series):
    values, ts = series
    feats = build_features(values, FeatureConfig(), ts)
    t = 250
    window = values.iloc[t - 4 : t + 1]  # trailing, inclusive of the current point
    assert feats["roll_5_mean"].loc[t] == pytest.approx(window.mean())
    assert feats["roll_5_min"].loc[t] == pytest.approx(window.min())
    assert feats["roll_5_max"].loc[t] == pytest.approx(window.max())


def test_zscore_excludes_current_point_from_context(series):
    values, ts = series
    feats = build_features(values, FeatureConfig(), ts)
    t = 250
    past = values.iloc[t - 60 : t]  # trailing window WITHOUT the current point
    expected = (values.iloc[t] - past.mean()) / (past.std(ddof=0) + 1e-9)
    assert feats["zscore_60"].loc[t] == pytest.approx(expected)


def test_no_future_information_leaks(series):
    """Perturbing the future must not change features computed from the past."""
    values, ts = series
    cfg = FeatureConfig()
    feats_before = build_features(values, cfg, ts)
    perturbed = values.copy()
    perturbed.iloc[200:] += 1000.0
    feats_after = build_features(perturbed, cfg, ts)

    past_rows = feats_before.index[feats_before.index < 200]
    pd.testing.assert_frame_equal(feats_before.loc[past_rows], feats_after.loc[past_rows])
    future_rows = feats_before.index[feats_before.index >= 200]
    assert not np.allclose(feats_before.loc[future_rows], feats_after.loc[future_rows])


def test_feature_labels_refer_to_original_positions(series):
    """Regression guard for the alignment convention: feature row labels are ORIGINAL
    series positions, so the row labeled L must describe the state at values[L].
    Training/eval code relies on this when pairing X rows with label-indexed targets."""
    values, ts = series
    cfg = FeatureConfig(lags=(1,), rolling_windows=(), roc_offsets=(), calendar=False)
    feats = build_features(values, cfg, ts)
    for label in feats.index[::50]:
        assert feats.loc[label, "value"] == values.iloc[label]
        assert feats.loc[label, "lag_1"] == values.iloc[label - 1]


def test_calendar_features_encoded(series):
    values, ts = series
    feats = build_features(values, FeatureConfig(), ts)
    hours = ts.hour + ts.minute / 60.0  # DatetimeIndex accessors
    expected_sin = np.sin(2 * np.pi * hours / 24.0)
    assert np.allclose(feats["hour_sin"], expected_sin[feats.index])
    assert ((feats["hour_sin"] >= -1.0) & (feats["hour_sin"] <= 1.0)).all()
