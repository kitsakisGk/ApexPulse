"""Smoke tests asserting the package scaffold imports and configures cleanly."""

from __future__ import annotations

import apexpulse
from apexpulse.config import Settings, get_settings


def test_package_exposes_version() -> None:
    assert apexpulse.__version__ == "0.1.0"


def test_settings_expose_documented_defaults() -> None:
    settings = Settings()

    assert settings.environment == "local"
    assert settings.kafka_telemetry_topic == "apexpulse.telemetry.ticks"
    assert settings.tick_rate_hz == 8.0


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_environment_overrides_are_applied(monkeypatch) -> None:
    monkeypatch.setenv("APEXPULSE_TICK_RATE_HZ", "32")

    assert Settings().tick_rate_hz == 32.0


def test_default_backends_require_no_external_services() -> None:
    """The out-of-the-box configuration must run on a machine without Docker."""
    settings = Settings()

    assert settings.broker_backend == "memory"
    assert settings.state_backend == "sqlite"
    assert settings.requires_external_services is False


def test_selecting_kafka_or_redis_flags_external_dependencies() -> None:
    assert Settings(broker_backend="kafka").requires_external_services is True
    assert Settings(state_backend="redis").requires_external_services is True


def test_invalid_backend_is_rejected_at_startup() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(broker_backend="rabbitmq")  # type: ignore[arg-type]
