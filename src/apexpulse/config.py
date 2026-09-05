"""Typed runtime configuration, sourced from the environment and `.env`.

Every subsystem reads its settings from the single :func:`get_settings` accessor so
that configuration stays declarative, validated once at startup, and trivially
overridable inside tests.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application-wide settings resolved from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APEXPULSE_",
        extra="ignore",
    )

    # -- General ------------------------------------------------------------
    environment: str = Field(default="local", description="Deployment environment name.")
    log_level: str = Field(default="INFO", description="Root logging level.")

    # -- Kafka / Redpanda ---------------------------------------------------
    kafka_bootstrap_servers: str = Field(
        default="localhost:19092",
        description="Comma-separated Kafka bootstrap servers.",
    )
    kafka_telemetry_topic: str = Field(
        default="apexpulse.telemetry.ticks",
        description="Topic carrying raw per-tick telemetry events.",
    )
    kafka_prediction_topic: str = Field(
        default="apexpulse.predictions",
        description="Topic carrying emitted win-probability predictions.",
    )
    kafka_consumer_group: str = Field(
        default="apexpulse-stream",
        description="Consumer group id for the streaming processor.",
    )

    # -- Redis --------------------------------------------------------------
    redis_url: RedisDsn = Field(
        default=RedisDsn("redis://localhost:6379/0"),
        description="Redis connection URL backing the live match state store.",
    )
    redis_state_ttl_seconds: int = Field(
        default=3600,
        ge=1,
        description="Expiry applied to live match snapshots.",
    )

    # -- Simulation ---------------------------------------------------------
    tick_rate_hz: float = Field(
        default=8.0,
        gt=0,
        le=128,
        description="Telemetry ticks emitted per second by the replay simulator.",
    )

    # -- Storage paths ------------------------------------------------------
    data_dir: Path = Field(default=PROJECT_ROOT / "data", description="Telemetry data root.")
    model_dir: Path = Field(default=PROJECT_ROOT / "models", description="Model artifact root.")
    duckdb_path: Path = Field(
        default=PROJECT_ROOT / "data" / "processed" / "apexpulse.duckdb",
        description="DuckDB database file for historical telemetry.",
    )

    # -- API ----------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0", description="Bind address for the FastAPI server.")
    api_port: int = Field(default=8000, ge=1, le=65535, description="FastAPI server port.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
