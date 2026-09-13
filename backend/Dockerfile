FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /code

# libgomp1: required by LightGBM's OpenMP runtime on slim images
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY backend ./backend
COPY ml ./ml

RUN pip install --no-cache-dir .

EXPOSE 8000

# Apply migrations, then serve. The image also carries the `ml` package so training
# jobs can run via `docker compose run --rm api python -m ml.<module>` in later phases.
CMD ["sh", "-c", "python -m alembic -c backend/alembic.ini upgrade head && uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"]
