"""Tests asserting the compose stack and application settings stay in agreement.

The infrastructure is declarative, so these guard the seam that actually breaks in
practice: ports and topic names drifting between `docker-compose.yml` and `Settings`.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import pytest

from apexpulse.config import PROJECT_ROOT, Settings

yaml = pytest.importorskip("yaml")

COMPOSE_PATH = PROJECT_ROOT / "docker-compose.yml"


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


def _published_host_ports(service: dict) -> set[int]:
    return {int(str(mapping).split(":")[0]) for mapping in service.get("ports", [])}


def test_compose_file_exists() -> None:
    assert COMPOSE_PATH.is_file()


def test_core_services_are_declared(compose: dict) -> None:
    assert {"redpanda", "redis", "redpanda-init", "console"} <= set(compose["services"])


def test_stateful_services_declare_healthchecks(compose: dict) -> None:
    for name in ("redpanda", "redis"):
        assert "healthcheck" in compose["services"][name], f"{name} needs a healthcheck"


def test_dependents_wait_for_healthy_broker(compose: dict) -> None:
    for name in ("redpanda-init", "console"):
        condition = compose["services"][name]["depends_on"]["redpanda"]["condition"]
        assert condition == "service_healthy"


def test_kafka_port_matches_settings(compose: dict) -> None:
    _, _, port = Settings().kafka_bootstrap_servers.partition(":")

    assert int(port) in _published_host_ports(compose["services"]["redpanda"])


def test_redis_port_matches_settings(compose: dict) -> None:
    port = urlparse(str(Settings().redis_url)).port

    assert port in _published_host_ports(compose["services"]["redis"])


def test_bootstrap_creates_every_configured_topic(compose: dict) -> None:
    command = compose["services"]["redpanda-init"]["command"]
    settings = Settings()

    assert settings.kafka_telemetry_topic in command
    assert settings.kafka_prediction_topic in command


def test_host_ports_are_unique(compose: dict) -> None:
    claimed: list[int] = []
    for service in compose["services"].values():
        claimed.extend(_published_host_ports(service))

    assert len(claimed) == len(set(claimed)), "two services publish the same host port"


def test_named_volumes_are_declared(compose: dict) -> None:
    declared = set(compose.get("volumes") or {})
    for service in compose["services"].values():
        for mapping in service.get("volumes", []):
            source = str(mapping).split(":")[0]
            if not source.startswith((".", "/", "~")):
                assert source in declared, f"undeclared volume {source}"


def test_wait_script_is_present() -> None:
    assert Path(PROJECT_ROOT / "scripts" / "wait_for_services.py").is_file()
