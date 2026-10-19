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


# -- Application services -----------------------------------------------------


def _healthcheck_source(compose: dict, service: str) -> str | None:
    """Return where ``service``'s healthcheck is defined, or None if nowhere.

    Compose honours a HEALTHCHECK baked into an image, so a healthcheck can live
    in either place. Both count; neither means `service_healthy` waits forever.
    """
    definition = compose["services"][service]
    if "healthcheck" in definition:
        return "compose"
    build = definition.get("build")
    if isinstance(build, dict) and "dockerfile" in build:
        dockerfile = (PROJECT_ROOT / build["dockerfile"]).read_text(encoding="utf-8")
        if "HEALTHCHECK" in dockerfile:
            return "dockerfile"
    return None


def test_application_services_are_declared(compose: dict) -> None:
    assert {"api", "frontend"} <= set(compose["services"])


def test_every_healthy_dependency_can_report_health(compose: dict) -> None:
    """A `service_healthy` wait on a service with no healthcheck never resolves."""
    for name, service in compose["services"].items():
        for dependency, condition in (service.get("depends_on") or {}).items():
            if condition.get("condition") == "service_healthy":
                assert _healthcheck_source(compose, dependency), (
                    f"{name} waits for {dependency} to be healthy, "
                    f"but {dependency} defines no healthcheck"
                )


def test_the_api_reaches_infrastructure_by_service_name(compose: dict) -> None:
    """Inside the network, localhost is the container itself, not the broker."""
    environment = compose["services"]["api"]["environment"]

    assert environment["APEXPULSE_BROKER_BACKEND"] == "kafka"
    assert environment["APEXPULSE_STATE_BACKEND"] == "redis"
    assert environment["APEXPULSE_KAFKA_BOOTSTRAP_SERVERS"].startswith("redpanda:")
    assert "redis:6379" in environment["APEXPULSE_REDIS_URL"]


def test_the_dashboard_points_the_browser_at_a_host_address(compose: dict) -> None:
    """NEXT_PUBLIC_* run in the viewer's browser, where `api` does not resolve."""
    args = compose["services"]["frontend"]["build"]["args"]

    for key in ("NEXT_PUBLIC_API_URL", "NEXT_PUBLIC_WS_URL"):
        assert "localhost" in args[key], f"{key} must be reachable from the browser"
        assert "//api" not in args[key]


def test_the_api_healthcheck_does_not_require_a_model() -> None:
    """/health is 503 without a checkpoint; probing it would restart-loop the API."""
    dockerfile = (PROJECT_ROOT / "docker" / "api.Dockerfile").read_text(encoding="utf-8")
    healthcheck = dockerfile[dockerfile.index("HEALTHCHECK") :]

    assert "/ready" in healthcheck
    assert "/health" not in healthcheck.split("CMD", 1)[1].splitlines()[0]


def test_model_checkpoints_are_mounted_read_only(compose: dict) -> None:
    volumes = compose["services"]["api"]["volumes"]

    assert any(volume.endswith("/app/models:ro") for volume in volumes)


def test_the_build_context_excludes_heavy_directories() -> None:
    ignored = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    for path in (".venv", "frontend/node_modules", "data", ".git"):
        assert path in ignored, f"{path} would be shipped into every build context"
