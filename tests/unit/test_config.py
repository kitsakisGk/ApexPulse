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
