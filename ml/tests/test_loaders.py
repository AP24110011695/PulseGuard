import json
from pathlib import Path

import pytest

from ml.loaders import load_nab_series


@pytest.fixture
def nab_files(tmp_path: Path):
    rows = ["timestamp,value"]
    for i in range(10):
        rows.append(f"2026-01-01 00:0{i}:00.000000,{100.0 + i}")
    csv_path = tmp_path / "dummy_metric.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    labels = {
        "realAWSCloudwatch/dummy_metric.csv": [
            ["2026-01-01 00:02:00.000000", "2026-01-01 00:03:00.000000"]
        ]
    }
    labels_path = tmp_path / "combined_windows.json"
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    return csv_path, labels_path


def test_load_nab_without_labels(nab_files):
    csv_path, _ = nab_files
    df = load_nab_series(csv_path, series_name="my_series")
    assert len(df) == 10
    assert set(df.columns) == {"series_name", "ts", "value", "is_anomaly", "anomaly_type"}
    assert (df["series_name"] == "my_series").all()
    assert not df["is_anomaly"].any()
    assert df["ts"].dt.tz is not None


def test_load_nab_with_labels(nab_files):
    csv_path, labels_path = nab_files
    df = load_nab_series(csv_path, labels_path=labels_path)
    labeled = df[df["is_anomaly"]]
    assert list(labeled["value"]) == [102.0, 103.0]
    assert (labeled["anomaly_type"] == "nab_label").all()


def test_missing_csv_raises_helpful_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Download a labeled series"):
        load_nab_series(tmp_path / "nope.csv")


def test_missing_labels_file_raises(tmp_path: Path, nab_files):
    csv_path, _ = nab_files
    with pytest.raises(FileNotFoundError, match="combined_windows.json"):
        load_nab_series(csv_path, labels_path=tmp_path / "missing.json")


def test_labels_key_not_found_raises(nab_files):
    csv_path, labels_path = nab_files
    # a labels file with no matching basename must fail loudly instead of silently
    # returning unlabeled data
    labels_path.write_text(json.dumps({"other/thing.csv": []}), encoding="utf-8")
    with pytest.raises(KeyError):
        load_nab_series(csv_path, labels_path=labels_path)


def test_missing_columns_raise(tmp_path: Path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("time,v\n2026-01-01,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing expected NAB columns"):
        load_nab_series(bad_csv)
