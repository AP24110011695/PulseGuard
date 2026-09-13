from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, read from the environment and an optional local .env file.

    Real environment variables take priority over .env values (pydantic-settings
    precedence), which is how the Docker Compose api service points DATABASE_URL at "db".
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "PulseGuard"

    database_url: str = "postgresql+psycopg://pulseguard:pulseguard_dev@localhost:5432/pulseguard"

    # MLflow tracking server (registry + runs). Tests override this with a sqlite URI.
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"

    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://localhost:4173",
        "http://localhost:3000",
    ]

    ingest_max_points_per_request: int = 50_000
    ingest_chunk_size: int = 5_000
    query_default_max_points: int = 2_000
    query_max_points_ceiling: int = 20_000


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
