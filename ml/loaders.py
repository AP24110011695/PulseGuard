"""Public benchmark dataset loaders (secondary sanity check — see PULSEGUARD.md §4B).

Currently supports the Numenta Anomaly Benchmark (NAB). Download labeled series manually:

    data/raw/nab/realAWSCloudwatch/ec2_cpu_utilization_24ae8bf.csv   (from NAB data/)
    data/raw/nab/labels/combined_windows.json                        (from NAB labels/)

This module standardizes them to the PulseGuard schema: series_name, ts (UTC), value,
is_anomaly, anomaly_type. Public data is a sanity check only — never proof of real-world
validity, and its statistics are measured, never invented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

NAB_TIME_COL = "timestamp"
NAB_VALUE_COL = "value"
NAB_URL = "https://github.com/numenta/NAB"


def _label_windows(labels_path: Path, csv_path: Path) -> list[list[str]]:
    """Extract the label windows for one CSV from NAB's combined_windows.json."""
    if not labels_path.exists():
        raise FileNotFoundError(
            f"NAB labels file not found at {labels_path}. Download labels/combined_windows.json "
            f"from {NAB_URL} into data/raw/nab/labels/."
        )
    all_windows = json.loads(labels_path.read_text(encoding="utf-8"))
    for key, windows in all_windows.items():
        if Path(key).name == csv_path.name:
            return windows
    raise KeyError(f"no label windows for '{csv_path.name}' in {labels_path}")


def load_nab_series(
    csv_path: str | Path,
    labels_path: str | Path | None = None,
    series_name: str | None = None,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"NAB csv not found at {csv_path}. Download a labeled series from {NAB_URL} "
            "(data/realAWSCloudwatch/*.csv) into data/raw/nab/."
        )
    raw = pd.read_csv(csv_path)
    missing = {NAB_TIME_COL, NAB_VALUE_COL} - set(raw.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing expected NAB columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "series_name": series_name or csv_path.stem,
            "ts": pd.to_datetime(raw[NAB_TIME_COL], utc=True),
            "value": raw[NAB_VALUE_COL].astype(float),
            "is_anomaly": False,
            "anomaly_type": "",
        }
    )

    if labels_path is not None:
        windows = _label_windows(Path(labels_path), csv_path)
        mask = pd.Series(False, index=out.index)
        for window in windows:
            w_start = pd.Timestamp(window[0], tz="UTC")
            w_end = pd.Timestamp(window[1], tz="UTC")
            mask |= (out["ts"] >= w_start) & (out["ts"] <= w_end)
        out.loc[mask, "is_anomaly"] = True
        out.loc[mask, "anomaly_type"] = "nab_label"

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ml.loaders", description=__doc__)
    parser.add_argument("--csv", required=True, help="path to a NAB series CSV")
    parser.add_argument("--labels", default=None, help="path to NAB combined_windows.json")
    parser.add_argument("--name", default=None, help="override the series name")
    parser.add_argument("--out", type=Path, default=None, help="write the standardized CSV here")
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--chunk-size", type=int, default=10_000)
    args = parser.parse_args(argv)

    df = load_nab_series(args.csv, labels_path=args.labels, series_name=args.name)
    labeled = int(df["is_anomaly"].sum())
    print(
        f"loaded {len(df)} points for '{df['series_name'].iloc[0]}' "
        f"({labeled} labeled anomalous, {labeled / len(df):.4%})"
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
        print(f"wrote {args.out}")

    if args.ingest:
        from ml.api_client import ingest_dataframe

        meta = {
            str(df["series_name"].iloc[0]): {
                "unit": None,
                "description": f"NAB benchmark series {Path(args.csv).name}",
                "source": "nab",
            }
        }
        result = ingest_dataframe(args.api_url, df, series_meta=meta, chunk_size=args.chunk_size)
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
