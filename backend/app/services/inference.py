"""Model serving: champion resolution, feature building from stored data, prediction
persistence and model-version tracing (plan §9).

The ModelManager resolves each task's `champion` alias through the MLflow registry,
loads the pyfunc artifact, and checks the alias version on every use so an alias move
is picked up without a restart (or force one via POST /api/v1/models/reload). Feature
configuration comes from the champion's run params, so serving always reproduces the
features the registered model was trained with.

Feature frames are built with the shared `ml.features` code over the stored history and
indexed by row position into the fetched point list — the same positional convention as
training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.pyfunc import PyFuncModel
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.models.metric import MetricPoint
from backend.app.models.prediction import ActiveModel, AnomalyResult, StoredForecast
from backend.app.services.alerts import raise_anomaly_alerts
from ml.features import FeatureConfig, build_features

CHAMPION_ALIAS = "champion"
TASK_MODEL_NAMES = {
    "forecast": "pulseguard-forecaster",
    "anomaly": "pulseguard-anomaly-detector",
}
LOOKBACK_BUFFER = 240  # extra history rows beyond the feature warmup


class ModelUnavailable(Exception):
    """Raised when a champion is required but cannot be resolved or loaded."""


@dataclass
class LoadedChampion:
    task: str
    name: str
    version: int
    model: PyFuncModel
    feature_config: dict
    horizons: tuple[int, ...] = ()
    quantiles: tuple[float, ...] = ()
    threshold: float | None = None
    loaded_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def feature_settings(self) -> FeatureConfig:
        return FeatureConfig(**self.feature_config)


def _client() -> MlflowClient:
    return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)


def _parse_json_param(value: str | None, fallback):
    if value is None:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


class ModelManager:
    """Caches loaded champions and keeps them in sync with the registry alias."""

    def __init__(self) -> None:
        self._champions: dict[str, LoadedChampion] = {}

    def get(self, task: str) -> LoadedChampion:
        champion = self._champions.get(task)
        if champion is None:
            raise ModelUnavailable(
                f"no champion loaded for task '{task}'; the model may not be registered yet"
            )
        return champion

    def ensure_current(self, task: str) -> LoadedChampion:
        """Return the loaded champion after checking the alias for a version change."""
        name, version = self.resolve_alias(task)
        champion = self._champions.get(task)
        if champion is None or champion.version != version or champion.name != name:
            return self.load(task, name, version)
        return champion

    def resolve_alias(self, task: str) -> tuple[str, int]:
        name = TASK_MODEL_NAMES[task]
        try:
            version = _client().get_model_version_by_alias(name, CHAMPION_ALIAS)
        except Exception as exc:  # mlflow raises many types; wrap for uniform handling
            raise ModelUnavailable(
                f"cannot resolve '{CHAMPION_ALIAS}' alias for {name}: {exc}"
            ) from exc
        return name, int(version.version)

    def load(self, task: str, name: str, version: int) -> LoadedChampion:
        try:
            model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
        except Exception as exc:
            raise ModelUnavailable(f"cannot load {name} version {version}: {exc}") from exc

        run_params: dict = {}
        model_version = _client().get_model_version(name, str(version))
        if model_version.run_id:
            run_params = _client().get_run(model_version.run_id).data.params
        threshold_raw = run_params.get("threshold")
        champion = LoadedChampion(
            task=task,
            name=name,
            version=version,
            model=model,
            feature_config=_parse_json_param(run_params.get("feature_config"), {}),
            horizons=tuple(int(h) for h in _parse_json_param(run_params.get("horizons"), [])),
            quantiles=tuple(float(q) for q in _parse_json_param(run_params.get("quantiles"), [])),
            threshold=float(threshold_raw) if threshold_raw is not None else None,
        )
        self._champions[task] = champion
        return champion

    def reload(self, task: str) -> LoadedChampion:
        name, version = self.resolve_alias(task)
        return self.load(task, name, version)


model_manager = ModelManager()


def _fetch_tail_points(db: Session, series_id: int, limit: int) -> list[MetricPoint]:
    rows = db.execute(
        select(MetricPoint)
        .where(MetricPoint.series_id == series_id)
        .order_by(MetricPoint.ts.desc())
        .limit(limit)
    ).scalars().all()
    return list(reversed(rows))


def _to_frame(points: list[MetricPoint]) -> pd.DataFrame:
    return pd.DataFrame({"value": [p.value for p in points], "ts": [p.ts for p in points]})


def list_registered_models() -> list[dict]:
    """Serialize registered models + versions from the registry."""
    client = _client()
    results = []
    for rm in client.search_registered_models():
        versions = []
        for mv in client.search_model_versions(f"name='{rm.name}'"):
            versions.append(
                {
                    "version": int(mv.version),
                    "status": str(mv.status),
                    "run_id": mv.run_id,
                    "aliases": sorted(mv.aliases) if mv.aliases else [],
                }
            )
        versions.sort(key=lambda v: v["version"])
        results.append({"name": rm.name, "versions": versions})
    results.sort(key=lambda r: r["name"])
    return results


def sync_active_model(db: Session, champion: LoadedChampion) -> None:
    """Mirror the resolved champion into active_models for API traceability."""
    metrics = {
        "feature_config": champion.feature_config,
        "horizons": list(champion.horizons),
        "quantiles": list(champion.quantiles),
        "threshold": champion.threshold,
    }
    stmt = pg_insert(ActiveModel).values(
        task=champion.task,
        alias=CHAMPION_ALIAS,
        mlflow_model_name=champion.name,
        model_version=champion.version,
        metrics=metrics,
        promoted_at=datetime.now(UTC),
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_active_models_task_alias",
        set_={
            "mlflow_model_name": stmt.excluded.mlflow_model_name,
            "model_version": stmt.excluded.model_version,
            "metrics": stmt.excluded.metrics,
            "promoted_at": stmt.excluded.promoted_at,
        },
    )
    db.execute(stmt)
    db.commit()


def forecast_series(db: Session, series_id: int, horizon_minutes: int) -> StoredForecast:
    """Build features from the stored history, predict the band for one horizon, and
    persist the forecast with the champion's model version."""
    champion = model_manager.ensure_current("forecast")
    feature_cfg = champion.feature_settings()
    if horizon_minutes not in champion.horizons:
        raise ValueError(
            f"unsupported horizon {horizon_minutes}; champion supports {list(champion.horizons)}"
        )

    lookback = feature_cfg.max_lookback + LOOKBACK_BUFFER
    points = _fetch_tail_points(db, series_id, lookback)
    if len(points) < feature_cfg.max_lookback + 1:
        raise ValueError(
            f"insufficient history: {len(points)} points stored, "
            f"{feature_cfg.max_lookback + 1} required"
        )
    frame = _to_frame(points)
    features = build_features(frame["value"], feature_cfg, frame["ts"])
    latest = features.iloc[[-1]].copy()
    latest["horizon"] = horizon_minutes

    prediction = champion.model.predict(latest)
    quantiles = {str(q): float(prediction.iloc[0][f"q{q}"]) for q in champion.quantiles}
    forecast = StoredForecast(
        series_id=series_id,
        horizon_minutes=horizon_minutes,
        target_ts=points[-1].ts + timedelta(minutes=horizon_minutes),
        quantiles=quantiles,
        model_name=champion.name,
        model_version=champion.version,
    )
    db.add(forecast)
    db.commit()
    db.refresh(forecast)
    return forecast


def score_series_anomalies(
    db: Session,
    series_id: int,
    start: datetime | None,
    end: datetime | None,
    limit: int,
) -> list[AnomalyResult]:
    """Score a window of stored points with the champion detector and persist one
    result row per (point, model version) — idempotent on re-scoring."""
    champion = model_manager.ensure_current("anomaly")
    feature_cfg = champion.feature_settings()
    lookback = feature_cfg.max_lookback + LOOKBACK_BUFFER

    end_bound = end or db.scalar(
        select(func.max(MetricPoint.ts)).where(MetricPoint.series_id == series_id)
    )
    if end_bound is None:
        raise ValueError("no points stored for this series")
    start_bound = start or end_bound - timedelta(minutes=limit)
    history_start = start_bound - timedelta(minutes=feature_cfg.max_lookback + 1)

    points = db.execute(
        select(MetricPoint)
        .where(
            MetricPoint.series_id == series_id,
            MetricPoint.ts >= history_start,
            MetricPoint.ts <= end_bound,
        )
        .order_by(MetricPoint.ts.asc())
        .limit(lookback + limit)
    ).scalars().all()
    window = [p for p in points if p.ts >= start_bound][-limit:]
    if not window:
        raise ValueError("no points fall inside the requested window")

    frame = _to_frame(points)
    features = build_features(frame["value"], feature_cfg, frame["ts"])
    # window points are the contiguous tail of `points`; features keep positional labels
    n_total, n_window = len(points), len(window)
    row_index = features.index.to_numpy()
    window_rows = row_index[row_index >= n_total - n_window]
    selected = features.loc[window_rows]

    prediction = champion.model.predict(selected)
    results = [
        AnomalyResult(
            series_id=series_id,
            ts=points[pos].ts,
            score=float(score),
            is_anomaly=bool(flag),
            threshold=champion.threshold,
            model_name=champion.name,
            model_version=champion.version,
        )
        for pos, score, flag in zip(
            window_rows,
            prediction["score"].to_numpy(),
            prediction["is_anomaly"].to_numpy(),
            strict=True,
        )
    ]
    _upsert_anomaly_results(db, results)
    raise_anomaly_alerts(db, series_id, results)
    # return the persisted rows so ids/timestamps reflect the database state
    saved = db.execute(
        select(AnomalyResult)
        .where(
            AnomalyResult.series_id == series_id,
            AnomalyResult.model_name == champion.name,
            AnomalyResult.model_version == champion.version,
            AnomalyResult.ts >= window[0].ts,
            AnomalyResult.ts <= window[-1].ts,
        )
        .order_by(AnomalyResult.ts.asc())
    ).scalars().all()
    return list(saved)


def _upsert_anomaly_results(db: Session, results: list[AnomalyResult]) -> None:
    if not results:
        return
    values = [
        {
            "series_id": r.series_id,
            "ts": r.ts,
            "score": r.score,
            "is_anomaly": r.is_anomaly,
            "threshold": r.threshold,
            "model_name": r.model_name,
            "model_version": r.model_version,
        }
        for r in results
    ]
    stmt = pg_insert(AnomalyResult).values(values)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_anomaly_results_key",
        set_={
            "score": stmt.excluded.score,
            "is_anomaly": stmt.excluded.is_anomaly,
            "threshold": stmt.excluded.threshold,
            "created_at": datetime.now(UTC),
        },
    )
    db.execute(stmt)
    db.commit()
