-- Creates the mlflow database up front so the Phase 3 compose service can attach to it.
SELECT 'CREATE DATABASE mlflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec
