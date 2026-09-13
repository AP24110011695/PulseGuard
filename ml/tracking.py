"""MLflow tracking and model-registry helpers (Phase 3).

The registered champions are pooled (series-agnostic) models: one
`pulseguard-forecaster` and one `pulseguard-anomaly-detector`, trained on the full
initial training region of every dataset series. The champion is resolved through the
`champion` alias; run params carry the horizons, quantiles and feature config that
serving needs to reproduce training-time features.
"""

from __future__ import annotations

import json
import os

import mlflow
from mlflow import MlflowClient

CHAMPION_ALIAS = "champion"
EXPERIMENT_NAME = "pulseguard"
TASK_MODEL_NAMES = {
    "forecast": "pulseguard-forecaster",
    "anomaly": "pulseguard-anomaly-detector",
}


def tracking_uri() -> str:
    """Compose sets MLFLOW_TRACKING_URI=http://mlflow:5000; local default is the
    host-mapped server on the IPv4 loopback (see .env.example). Tests override with a
    sqlite:/// URI via the same env var."""
    return os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def client() -> MlflowClient:
    return MlflowClient(tracking_uri=tracking_uri())


def _pip_requirements() -> list[str]:
    import lightgbm
    import numpy
    import pandas
    import sklearn

    return [
        f"scikit-learn=={sklearn.__version__}",
        f"numpy=={numpy.__version__}",
        f"pandas=={pandas.__version__}",
        f"lightgbm=={lightgbm.__version__}",
    ]


def register_champion(
    task: str,
    wrapper,
    *,
    seed: int,
    config_sha256: str,
    feature_config: dict,
    horizons: tuple[int, ...] = (),
    quantiles: tuple[float, ...] = (),
    threshold: float | None = None,
    metrics: dict[str, float] | None = None,
    report: dict | None = None,
    reference: dict | None = None,
    run_name: str | None = None,
) -> tuple[str, int, str]:
    """Log the wrapper as a run artifact, register it, and move the champion alias.

    Returns (model_name, version, run_id). Run params carry everything serving needs:
    task, config sha, feature config, horizons/quantiles (forecast) and the calibrated
    threshold (anomaly).
    """
    if task not in TASK_MODEL_NAMES:
        raise ValueError(f"unknown task {task!r}")
    name = TASK_MODEL_NAMES[task]

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name or f"{task}-champion-candidate") as run:
        run_id = run.info.run_id
        mlflow.set_tags({"pulseguard.task": task, "pulseguard.config_sha256": config_sha256})
        params = {
            "task": task,
            "config_sha256": config_sha256,
            "seed": seed,
            "feature_config": json.dumps(feature_config, sort_keys=True),
        }
        if horizons:
            params["horizons"] = json.dumps(list(horizons))
        if quantiles:
            params["quantiles"] = json.dumps(list(quantiles))
        if threshold is not None:
            params["threshold"] = str(threshold)
        mlflow.log_params(params)
        for key, value in (metrics or {}).items():
            mlflow.log_metric(key, value)
        if report is not None:
            mlflow.log_dict(report, "eval_report.json")
        if reference is not None:
            mlflow.log_dict(reference, "drift_reference.json")

        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=wrapper,
            pip_requirements=_pip_requirements(),
        )

        mlflow_client = client()
        version = getattr(info, "registered_model_version", None)
        if version is None:
            # create_model_version does not auto-create the registered model
            if not mlflow_client.search_registered_models(filter_string=f"name='{name}'"):
                mlflow_client.create_registered_model(name)
            model_version = mlflow_client.create_model_version(
                name=name,
                source=info.model_uri,
                run_id=run_id,
                model_id=getattr(info, "model_id", None),
            )
            version = model_version.version
        mlflow_client.set_registered_model_alias(name, CHAMPION_ALIAS, str(version))
    return name, int(version), run_id


def get_champion(task: str) -> tuple[str, int]:
    """Resolve the champion alias; raises MlflowException when absent."""
    name = TASK_MODEL_NAMES[task]
    version = client().get_model_version_by_alias(name, CHAMPION_ALIAS)
    return name, int(version.version)


def load_champion(task: str):
    """Load the champion pyfunc model; returns (name, version, model)."""
    name, version = get_champion(task)
    model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    return name, version, model


def list_registered_models() -> list[dict]:
    mlflow_client = client()
    results = []
    for rm in mlflow_client.search_registered_models():
        versions = []
        for mv in mlflow_client.search_model_versions(f"name='{rm.name}'"):
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


def flatten_metrics(per_series: dict, task: str) -> dict[str, float]:
    """Flatten per-series aggregate metrics into MLflow metric keys."""
    out: dict[str, float] = {}
    for series, entry in per_series.items():
        safe = series.replace(" ", "_")
        if task == "anomaly":
            agg = entry["aggregate"]
            for key in ("precision", "recall", "f1"):
                out[f"{safe}.{key}_mean"] = float(agg[key]["mean"])
            out[f"{safe}.pr_auc_pooled"] = float(agg["pr_auc_pooled"])
        else:
            for horizon, agg in entry["aggregate"].items():
                for key in ("pinball_mean", "mae", "rmse"):
                    out[f"{safe}.{horizon}.{key}_mean"] = float(agg[key]["mean"])
    return out
