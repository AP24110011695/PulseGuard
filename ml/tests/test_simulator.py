from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from ml.simulator import generate_dataset, load_config

CONFIG_PATH = Path(__file__).resolve().parents[2] / "ml" / "configs" / "simulator_default.yaml"


@pytest.fixture(scope="module")
def default_config():
    return load_config(CONFIG_PATH)


def test_default_config_loads(default_config):
    assert default_config.seed == 42
    assert default_config.days == 28
    assert [s.name for s in default_config.series] == [
        "api_latency_p95_ms",
        "cpu_usage_pct",
        "request_rate_rps",
    ]
    for s in default_config.series:
        assert s.spikes.count >= 1
        assert s.level_shifts.count >= 1
        assert s.variance_changes.count >= 1
        assert s.contextual.count >= 1


def test_same_seed_is_byte_identical(default_config):
    df1 = generate_dataset(default_config)
    df2 = generate_dataset(default_config)
    pd.testing.assert_frame_equal(df1, df2, check_exact=True)


def test_different_seed_changes_values(default_config):
    df1 = generate_dataset(default_config)
    df2 = generate_dataset(replace(default_config, seed=default_config.seed + 1))
    assert not df1["value"].equals(df2["value"])


def test_dataset_structure(default_config):
    df = generate_dataset(default_config)
    expected_rows = default_config.days * 1440
    for name, group in df.groupby("series_name"):
        assert len(group) == expected_rows, name
        ts = pd.DatetimeIndex(group["ts"])
        assert ts.is_monotonic_increasing
        assert (ts[1:] - ts[:-1] == pd.Timedelta(minutes=1)).all()
        assert group["ts"].dt.tz is not None
        assert group["value"].notna().all()
        assert group["is_anomaly"].dtype == bool
        # every labeled point carries a type; normal points have none
        assert (group.loc[~group["is_anomaly"], "anomaly_type"] == "").all()
        assert (group.loc[group["is_anomaly"], "anomaly_type"] != "").all()


def test_labels_present_with_expected_types_and_rate(default_config):
    df = generate_dataset(default_config)
    for name, group in df.groupby("series_name"):
        rate = float(group["is_anomaly"].mean())
        assert 0.0 < rate < 0.03, f"{name}: anomaly rate {rate}"
        types = set(group.loc[group["is_anomaly"], "anomaly_type"].unique())
        assert {"spike", "level_shift", "variance_change", "contextual"} <= types, name


def test_level_shift_labels_only_transition(default_config):
    df = generate_dataset(default_config)
    for _, group in df.groupby("series_name"):
        shifts = group.index[group["anomaly_type"] == "level_shift"]
        assert len(shifts) > 0
        transition = group.loc[shifts, "ts"]
        expected = int(
            next(
                s.level_shifts.transition_minutes
                for s in default_config.series
                if s.name == group["series_name"].iloc[0]
            )
        )
        assert len(transition) == expected
