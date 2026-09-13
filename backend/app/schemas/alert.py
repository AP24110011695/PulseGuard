from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    series_id: int
    created_at: datetime
    kind: str
    severity: str
    message: str
    payload: dict
    acknowledged: bool
    dedupe_key: str


class AlertsPage(BaseModel):
    count: int
    unacknowledged: int
    alerts: list[AlertOut]


class AckResponse(BaseModel):
    id: int
    acknowledged: bool


__all__ = ["AlertOut", "AlertsPage", "AckResponse", "Field"]
