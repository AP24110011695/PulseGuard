"""Config-driven, seeded synthetic metric generator with ground-truth anomaly labels.

Per-series components: trend, daily seasonality, weekend factor, Gaussian noise,
spikes, permanent level shifts, temporary variance changes, and contextual anomalies
(off-peak-level values injected at peak time, anomalous only given their context).

Labeling convention (PULSEGUARD.md §4):
- permanent level shifts label only their transition window (the new level becomes
  the new normal afterwards);
- temporary regimes (spikes, variance-change windows, contextual anomalies) label the
  affected points.

Determinism: fixed start timestamp + seed in the config produce byte-identical output.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

POINTS_PER_DAY = 1440  # 1-minute resolution
_EDGE_MARGIN_MINUTES = POINTS_PER_DAY  # keep anomaly placement away from the stream edges


@dataclass(frozen=True)
class SpikesConfig:
    count: int
    magnitude: tuple[float, float]
    width_minutes: tuple[int, int]


@dataclass(frozen=True)
class LevelShiftConfig:
    count: int
    magnitude: float  # signed: applied as-is to every point from the shift onward
    transition_minutes: int  # only this window is labeled anomalous
    at_day: float | None = None  # fixed placement (day index); None → seeded random


@dataclass(frozen=True)
class VarianceChangeConfig:
    count: int
    sigma_multiplier: float
    duration_minutes: int


@dataclass(frozen=True)
class ContextualAnomaliesConfig:
    count: int
    # Deviation as a fraction of the daily amplitude, removed at peak-time points.
    magnitude_fraction: tuple[float, float]


@dataclass(frozen=True)
class SeriesConfig:
    name: str
    unit: str
    description: str = ""
    base: float = 0.0
    trend_per_day: float = 0.0
    daily_amplitude: float = 0.0
    daily_phase_hours: float = 0.0
    weekend_factor: float = 1.0
    noise_sigma: float = 1.0
    spikes: SpikesConfig = field(default_factory=lambda: SpikesConfig(0, (0.0, 0.0), (1, 1)))
    level_shifts: LevelShiftConfig = field(default_factory=lambda: LevelShiftConfig(0, 0.0, 1))
    variance_changes: VarianceChangeConfig = field(
        default_factory=lambda: VarianceChangeConfig(0, 1.0, 1)
    )
    contextual: ContextualAnomaliesConfig = field(
        default_factory=lambda: ContextualAnomaliesConfig(0, (0.0, 0.0))
    )


@dataclass(frozen=True)
class SimulatorConfig:
    seed: int
    start: str  # fixed ISO timestamp; required for byte-identical output
    days: int
    series: list[SeriesConfig]


def _pair(raw: list | tuple, cast=float) -> tuple:
    if len(raw) != 2:
        raise ValueError(f"expected a [low, high] pair, got {raw!r}")
    return (cast(raw[0]), cast(raw[1]))


def _series_from_dict(raw: dict) -> SeriesConfig:
    return SeriesConfig(
        name=str(raw["name"]),
        unit=str(raw.get("unit", "")),
        description=str(raw.get("description", "")),
        base=float(raw.get("base", 0.0)),
        trend_per_day=float(raw.get("trend_per_day", 0.0)),
        daily_amplitude=float(raw.get("daily_amplitude", 0.0)),
        daily_phase_hours=float(raw.get("daily_phase_hours", 0.0)),
        weekend_factor=float(raw.get("weekend_factor", 1.0)),
        noise_sigma=float(raw.get("noise_sigma", 1.0)),
        spikes=SpikesConfig(
            count=int(raw.get("spikes", {}).get("count", 0)),
            magnitude=_pair(raw.get("spikes", {}).get("magnitude", [0.0, 0.0])),
            width_minutes=_pair(raw.get("spikes", {}).get("width_minutes", [1, 1]), int),
        ),
        level_shifts=LevelShiftConfig(
            count=int(raw.get("level_shifts", {}).get("count", 0)),
            magnitude=float(raw.get("level_shifts", {}).get("magnitude", 0.0)),
            transition_minutes=int(raw.get("level_shifts", {}).get("transition_minutes", 1)),
            at_day=(
                float(raw["level_shifts"]["at_day"])
                if raw.get("level_shifts", {}).get("at_day") is not None
                else None
            ),
        ),
        variance_changes=VarianceChangeConfig(
            count=int(raw.get("variance_changes", {}).get("count", 0)),
            sigma_multiplier=float(raw.get("variance_changes", {}).get("sigma_multiplier", 1.0)),
            duration_minutes=int(raw.get("variance_changes", {}).get("duration_minutes", 1)),
        ),
        contextual=ContextualAnomaliesConfig(
            count=int(raw.get("contextual", {}).get("count", 0)),
            magnitude_fraction=_pair(
                raw.get("contextual", {}).get("magnitude_fraction", [0.0, 0.0])
            ),
        ),
    )


def load_config(path: str | Path) -> SimulatorConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SimulatorConfig(
        seed=int(raw["seed"]),
        start=str(raw["start"]),
        days=int(raw["days"]),
        series=[_series_from_dict(s) for s in raw["series"]],
    )


def _find_start(rng: np.random.Generator, n: int, length: int, occupied: np.ndarray) -> int:
    """Pick a start index whose window touches no already-labeled point; -1 if none found."""
    lo, hi = _EDGE_MARGIN_MINUTES, n - length - _EDGE_MARGIN_MINUTES
    if hi <= lo:
        return -1
    for _ in range(200):
        s = int(rng.integers(lo, hi))
        if not occupied[s : s + length].any():
            return s
    return -1


def _mark(is_anomaly: np.ndarray, anomaly_type: np.ndarray, idx: np.ndarray, kind: str) -> None:
    """Label points; the first anomaly type applied to a point wins."""
    for i in idx:
        if not is_anomaly[i]:
            is_anomaly[i] = True
            anomaly_type[i] = kind


def generate_series(cfg: SeriesConfig, sim: SimulatorConfig, index: int) -> pd.DataFrame:
    rng = np.random.default_rng([sim.seed, index])
    n = sim.days * POINTS_PER_DAY
    start = pd.Timestamp(sim.start)
    ts = pd.date_range(start, periods=n, freq="min")

    minutes = np.arange(n, dtype=np.float64)
    hour_of_day = (start.hour + start.minute / 60.0 + minutes / 60.0) % 24.0
    day_index = (
        int(start.dayofweek)
        + (int(start.hour * POINTS_PER_DAY + start.minute) + minutes.astype(np.int64))
        // POINTS_PER_DAY
    ) % 7
    weekend = (day_index >= 5).astype(np.float64)
    days_elapsed = minutes / POINTS_PER_DAY

    seasonal_daily = cfg.daily_amplitude * np.cos(
        2.0 * np.pi * (hour_of_day - cfg.daily_phase_hours) / 24.0
    )
    noise = rng.normal(0.0, cfg.noise_sigma, n)
    value = (cfg.base + cfg.trend_per_day * days_elapsed + seasonal_daily) * np.where(
        weekend > 0, cfg.weekend_factor, 1.0
    ) + noise

    is_anomaly = np.zeros(n, dtype=bool)
    anomaly_type = np.full(n, "", dtype=object)

    # Spikes: short, high-magnitude bursts.
    for _ in range(cfg.spikes.count):
        width = int(rng.integers(cfg.spikes.width_minutes[0], cfg.spikes.width_minutes[1] + 1))
        s = _find_start(rng, n, width, is_anomaly)
        if s < 0:
            continue
        value[s : s + width] += float(rng.uniform(*cfg.spikes.magnitude))
        _mark(is_anomaly, anomaly_type, np.arange(s, s + width), "spike")

    # Permanent level shifts: only the transition window is labeled anomalous.
    for _ in range(cfg.level_shifts.count):
        if cfg.level_shifts.at_day is not None:
            s = int(cfg.level_shifts.at_day * POINTS_PER_DAY)
            upper = n - cfg.level_shifts.transition_minutes - _EDGE_MARGIN_MINUTES
            s = min(max(s, _EDGE_MARGIN_MINUTES), max(_EDGE_MARGIN_MINUTES, upper))
            if is_anomaly[s : s + cfg.level_shifts.transition_minutes].any():
                continue
        else:
            s = _find_start(rng, n, cfg.level_shifts.transition_minutes, is_anomaly)
            if s < 0:
                continue
        value[s:] += cfg.level_shifts.magnitude
        _mark(
            is_anomaly,
            anomaly_type,
            np.arange(s, s + cfg.level_shifts.transition_minutes),
            "level_shift",
        )

    # Temporary variance changes: extra noise of inflated sigma across a window.
    for _ in range(cfg.variance_changes.count):
        duration = cfg.variance_changes.duration_minutes
        s = _find_start(rng, n, duration, is_anomaly)
        if s < 0:
            continue
        extra_sigma = cfg.noise_sigma * (cfg.variance_changes.sigma_multiplier - 1.0)
        value[s : s + duration] += rng.normal(0.0, extra_sigma, duration)
        _mark(is_anomaly, anomaly_type, np.arange(s, s + duration), "variance_change")

    # Contextual anomalies: trough-level values injected at peak time — plausible in
    # absolute terms, anomalous given the time-of-day context.
    if cfg.daily_amplitude > 0 and cfg.contextual.count > 0:
        peak = seasonal_daily > 0.5 * cfg.daily_amplitude
        candidates = np.flatnonzero(peak & ~is_anomaly)
        count = min(cfg.contextual.count, len(candidates))
        if count > 0:
            chosen = rng.choice(candidates, size=count, replace=False)
            fractions = rng.uniform(*cfg.contextual.magnitude_fraction, count)
            value[chosen] -= cfg.daily_amplitude * fractions
            _mark(is_anomaly, anomaly_type, chosen, "contextual")

    return pd.DataFrame(
        {
            "series_name": cfg.name,
            "ts": ts,
            "value": np.round(value, 6),
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
        }
    )


def generate_dataset(sim: SimulatorConfig) -> pd.DataFrame:
    frames = [generate_series(sc, sim, i) for i, sc in enumerate(sim.series)]
    return pd.concat(frames, ignore_index=True)


def dataset_summary(df: pd.DataFrame) -> dict:
    out: dict = {"series": {}}
    for name, group in df.groupby("series_name", sort=True):
        by_type: dict[str, int] = {}
        labeled = group[group["is_anomaly"]]
        for t, n in labeled.groupby("anomaly_type").size().items():
            by_type[str(t)] = int(n)
        out["series"][str(name)] = {
            "rows": int(len(group)),
            "anomalies": int(len(labeled)),
            "anomaly_rate": round(float(len(labeled)) / len(group), 6),
            "by_type": by_type,
        }
    out["total_rows"] = int(len(df))
    out["total_anomalies"] = int(df["is_anomaly"].sum())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ml.simulator", description=__doc__)
    parser.add_argument("--config", required=True, help="path to a simulator YAML config")
    parser.add_argument(
        "--out", type=Path, default=None, help="write the combined dataset CSV here"
    )
    parser.add_argument(
        "--ingest", action="store_true", help="POST the generated series to the PulseGuard API"
    )
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--chunk-size", type=int, default=10_000)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    df = generate_dataset(cfg)
    print(json.dumps(dataset_summary(df), indent=2))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
        print(f"wrote {args.out}")

    if args.ingest:
        from ml.api_client import ingest_dataframe

        meta = {
            s.name: {"unit": s.unit, "description": s.description or f"Simulated {s.name}"}
            for s in cfg.series
        }
        t0 = time.perf_counter()
        totals = ingest_dataframe(args.api_url, df, series_meta=meta, chunk_size=args.chunk_size)
        elapsed = time.perf_counter() - t0
        totals["ingest_elapsed_seconds"] = round(elapsed, 3)
        print(json.dumps(totals, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
