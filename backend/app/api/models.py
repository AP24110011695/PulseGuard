"""Model registry endpoints: list models, champion info, forced reload."""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db
from backend.app.core.errors import ApiError
from backend.app.models.prediction import ActiveModel
from backend.app.schemas.inference import (
    ChampionInfo,
    RegisteredModelInfo,
    ReloadResponse,
)
from backend.app.services.inference import (
    TASK_MODEL_NAMES,
    ModelUnavailable,
    list_registered_models,
    model_manager,
    sync_active_model,
)

router = APIRouter(tags=["models"])

VALID_TASKS = tuple(TASK_MODEL_NAMES)


def _champion_info(task: str, db: Session) -> ChampionInfo:
    # always resolve the alias live (cheap registry call) so the mirror cannot go stale
    # after promotions performed outside this API process
    try:
        sync_active_model(db, model_manager.ensure_current(task))
    except ModelUnavailable:
        raise ApiError(
            404, "no_champion", f"no champion registered for task '{task}'"
        ) from None
    row = db.scalar(
        select(ActiveModel).where(ActiveModel.task == task, ActiveModel.alias == "champion")
    )
    try:
        loaded = model_manager.get(task)
        is_loaded = loaded.version == row.model_version
    except ModelUnavailable:
        is_loaded = False
    return ChampionInfo(
        task=task,
        model_name=row.mlflow_model_name,
        model_version=row.model_version,
        promoted_at=row.promoted_at,
        metrics=row.metrics,
        loaded=is_loaded,
    )


@router.get("/models", response_model=list[RegisteredModelInfo])
def list_models() -> list[RegisteredModelInfo]:
    try:
        return [RegisteredModelInfo(**entry) for entry in list_registered_models()]
    except Exception as exc:
        raise ApiError(
            503, "mlflow_unavailable", f"cannot reach the MLflow registry: {exc}"
        ) from exc


@router.get("/models/{task}/champion", response_model=ChampionInfo)
def get_champion(task: str, db: Session = Depends(get_db)) -> ChampionInfo:
    if task not in VALID_TASKS:
        raise ApiError(422, "invalid_task", f"task must be one of {list(VALID_TASKS)}")
    return _champion_info(task, db)


@router.post("/models/reload", response_model=ReloadResponse)
def reload_models(db: Session = Depends(get_db)) -> ReloadResponse:
    out: dict[str, ChampionInfo] = {}
    for task in VALID_TASKS:
        try:
            champion = model_manager.reload(task)
        except ModelUnavailable as exc:
            out[task] = ChampionInfo(
                task=task,
                model_name=TASK_MODEL_NAMES[task],
                model_version=0,
                promoted_at=None,
                metrics={"error": str(exc)},
                loaded=False,
            )
            continue
        sync_active_model(db, champion)
        out[task] = ChampionInfo(
            task=task,
            model_name=champion.name,
            model_version=champion.version,
            promoted_at=None,
            metrics={"horizons": list(champion.horizons), "quantiles": list(champion.quantiles)},
            loaded=True,
        )
    return ReloadResponse(models=out)
