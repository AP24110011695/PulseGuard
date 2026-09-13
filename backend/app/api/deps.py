from fastapi import Depends
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.core.errors import ApiError
from backend.app.models.metric import MetricSeries


def get_series_or_404(series_id: int, db: Session = Depends(get_db)) -> MetricSeries:
    series = db.get(MetricSeries, series_id)
    if series is None:
        raise ApiError(404, "series_not_found", f"metric series {series_id} not found")
    return series
