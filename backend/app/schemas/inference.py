from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class StoredForecastOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    series_id: int
    created_at: datetime
    horizon_minutes: int
    target_ts: datetime
    quantiles: dict[str, float]
    model_name: str
    model_version: int


class StoredForecastsPage(BaseModel):
    series_id: int
    count: int
    forecasts: list[StoredForecastOut]


class AnomalyScoreRequest(BaseModel):
    start: datetime | None = None
    end: datetime | None = None
    limit: int = Field(default=500, ge=1, le=5000)


class AnomalyResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    series_id: int
    ts: datetime
    score: float
    is_anomaly: bool
    threshold: float | None
    model_name: str
    model_version: int
    created_at: datetime


class AnomalyScoreResponse(BaseModel):
    scored: int
    flagged: int
    model_name: str
    model_version: int
    results: list[AnomalyResultOut]


class AnomalyResultsPage(BaseModel):
    series_id: int
    count: int
    results: list[AnomalyResultOut]


class ModelVersionInfo(BaseModel):
    version: int
    status: str
    run_id: str | None
    aliases: list[str]


class RegisteredModelInfo(BaseModel):
    name: str
    versions: list[ModelVersionInfo]


class ChampionInfo(BaseModel):
    task: str
    model_name: str
    model_version: int
    promoted_at: datetime | None
    metrics: dict | None
    loaded: bool


class ReloadResponse(BaseModel):
    models: dict[str, ChampionInfo]
