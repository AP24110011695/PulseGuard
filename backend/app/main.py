import logging
import math
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api import alerts, drift, health, inference, models, points, series
from backend.app.core.config import settings
from backend.app.core.errors import ApiError

logger = logging.getLogger("pulseguard")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Best-effort champion load at startup: the API stays up (endpoints return 503)
    when the MLflow registry or a champion is not available yet."""
    from backend.app.core.database import SessionLocal
    from backend.app.services.inference import model_manager

    if not hasattr(app.state, "session_factory"):
        app.state.session_factory = SessionLocal  # tests pre-set their own
    for task in ("forecast", "anomaly"):
        try:
            champion = model_manager.reload(task)
            logger.info("loaded %s champion version %s", champion.name, champion.version)
        except Exception as exc:
            logger.warning("champion not loaded for task '%s': %s", task, exc)
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _json_safe(value):
    """Make pydantic error payloads JSON-encodable: non-finite floats and exception
    objects (which strict JSON encoding or json.dumps reject) become strings."""
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.message, "code": exc.code}
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": _json_safe(exc.errors()), "code": "validation_error"},
    )


api_v1 = APIRouter(prefix="/api/v1")
api_v1.include_router(health.router)
api_v1.include_router(series.router)
api_v1.include_router(points.router)
api_v1.include_router(models.router)
api_v1.include_router(inference.router)
api_v1.include_router(drift.router)
api_v1.include_router(alerts.router)
app.include_router(api_v1)
