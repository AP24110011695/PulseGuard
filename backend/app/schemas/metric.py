import math
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.core.config import settings


class SeriesCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    unit: str | None = Field(default=None, max_length=32)
    description: str | None = Field(default=None, max_length=500)
    source: str = Field(default="api", max_length=32)
    tags: dict[str, str] | None = None


class SeriesOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    unit: str | None
    description: str | None
    source: str
    tags: dict[str, str] | None
    created_at: datetime


class SeriesDetailOut(SeriesOut):
    point_count: int
    first_ts: datetime | None
    last_ts: datetime | None


class PointIn(BaseModel):
    ts: datetime
    value: float
    source_label: bool | None = None

    @field_validator("value")
    @classmethod
    def value_must_be_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("value must be a finite number")
        return v


class PointsIngest(BaseModel):
    points: list[PointIn] = Field(min_length=1, max_length=settings.ingest_max_points_per_request)


class IngestResult(BaseModel):
    series_id: int
    received: int
    inserted: int
    duplicates: int


class PointOut(BaseModel):
    ts: datetime
    value: float
    source_label: bool | None


class PointsPage(BaseModel):
    series_id: int
    start: datetime | None
    end: datetime | None
    max_points: int
    downsampled: bool
    bucket_seconds: int | None
    count: int
    points: list[PointOut]
