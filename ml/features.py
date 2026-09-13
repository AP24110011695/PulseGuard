"""Trailing-only feature engineering — the single source of features for training AND serving.

Conventions (PULSEGUARD.md §3.3 leakage rules):
- Row t describes everything known AT time t: the current observation, past lags,
  rolling statistics over windows that END at t, rates of change, and calendar
  encodings. Nothing from the future is ever touched (no centered windows, no
  negative shifts).
- The z-score feature compares the current value against a trailing window that
  EXCLUDES the current point, so point anomalies stand out against their context.
- Window sizes are in ROWS (samples). On 1-minute synthetic data rows == minutes;
  other datasets must set sizes accordingly (documented in the train configs).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    lags: tuple[int, ...] = (1, 2, 3, 5, 15, 30, 60)
    rolling_windows: tuple[int, ...] = (5, 15, 60)
    rolling_stats: tuple[str, ...] = ("mean", "std", "min", "max")
    roc_offsets: tuple[int, ...] = (1, 60)
    calendar: bool = True

    @property
    def max_lookback(self) -> int:
        lookbacks = [*self.lags, *self.rolling_windows, *self.roc_offsets]
        return max(lookbacks, default=1)

    def validate(self) -> None:
        if not self.lags:
            raise ValueError("FeatureConfig.lags must not be empty")
        bad = set(self.rolling_stats) - {"mean", "std", "min", "max"}
        if bad:
            raise ValueError(f"unsupported rolling stats: {sorted(bad)}")


def feature_names(cfg: FeatureConfig) -> list[str]:
    names = ["value", *(f"lag_{lag}" for lag in cfg.lags)]
    for window in cfg.rolling_windows:
        names.extend(f"roll_{window}_{stat}" for stat in cfg.rolling_stats)
    names.extend(f"roc_{off}" for off in cfg.roc_offsets)
    if cfg.rolling_windows:
        names.append(f"zscore_{max(cfg.rolling_windows)}")
    if cfg.calendar:
        names.extend(["hour_sin", "hour_cos", "dow_sin", "dow_cos"])
    return names


def build_features(values: pd.Series, cfg: FeatureConfig, ts: pd.Series) -> pd.DataFrame:
    """Build the feature matrix for one series.

    `values` and `ts` must share a RangeIndex of positions 0..n-1. The returned frame
    keeps that positional labeling and drops the warmup rows (the first
    `max_lookback` positions), so row labels still refer to positions in the
    original series.
    """
    cfg.validate()
    v = values.astype(float).reset_index(drop=True)
    ts = pd.Series(ts).reset_index(drop=True)
    feats: dict[str, pd.Series] = {"value": v}

    for lag in cfg.lags:
        feats[f"lag_{lag}"] = v.shift(lag)

    for window in cfg.rolling_windows:
        roll = v.rolling(window, min_periods=window)
        stats = {
            "mean": roll.mean(),
            "std": roll.std(ddof=0),
            "min": roll.min(),
            "max": roll.max(),
        }
        for stat in cfg.rolling_stats:
            feats[f"roll_{window}_{stat}"] = stats[stat]

    for offset in cfg.roc_offsets:
        feats[f"roc_{offset}"] = v - v.shift(offset)

    if cfg.rolling_windows:
        window = max(cfg.rolling_windows)
        past = v.shift(1).rolling(window, min_periods=window)
        feats[f"zscore_{window}"] = (v - past.mean()) / (past.std(ddof=0) + 1e-9)

    if cfg.calendar:
        hours = ts.dt.hour + ts.dt.minute / 60.0
        dows = ts.dt.dayofweek.astype(float)
        feats["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
        feats["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
        feats["dow_sin"] = np.sin(2.0 * np.pi * dows / 7.0)
        feats["dow_cos"] = np.cos(2.0 * np.pi * dows / 7.0)

    out = pd.DataFrame(feats)
    out = out.iloc[cfg.max_lookback :]
    if out.isna().any().any():
        raise ValueError("feature matrix contains NaN after the warmup drop")
    return out
