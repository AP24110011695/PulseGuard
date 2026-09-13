from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DriftEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    series_id: int
    detected_at: datetime
    kind: str
    status: str
    feature: str | None
    psi_value: float | None
    threshold: float | None
    details: dict


class DriftEventsPage(BaseModel):
    count: int
    events: list[DriftEventOut]


class DriftCheckRequest(BaseModel):
    series_id: int | None = None
    task: str = Field(default="forecast")
    window_points: int = Field(default=2880, ge=100, le=20_000)


class DriftCheckResponse(BaseModel):
    checks: list[dict]
    events_recorded: int


class DriftStatusEntry(BaseModel):
    series_id: int
    series_name: str
    psi: dict | None
    residual: dict | None
    overall: str


class DriftStatusResponse(BaseModel):
    series: list[DriftStatusEntry]


class PromotionDecisionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    decided_at: datetime
    task: str
    champion_name: str
    champion_version: int
    challenger_name: str
    challenger_version: int | None
    decision: str
    reason: str
    metrics: dict


class PromotionsPage(BaseModel):
    count: int
    decisions: list[PromotionDecisionOut]


class RetrainingTriggerRequest(BaseModel):
    task: str = Field(default="forecast")
    config: str = Field(default="ml/configs/train_synthetic.yaml")
    challenger_window: str = Field(default="expanding")


class RetrainingTriggerResponse(BaseModel):
    status: str
    task: str
    config: str
    challenger_window: str
