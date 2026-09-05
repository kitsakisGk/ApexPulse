"""Block until the ApexPulse infrastructure dependencies accept connections.

Compose healthchecks gate container start-up, but host-side processes still race
the published ports. Run this before any producer or consumer that talks to the
stack from outside Docker.

Usage:
    uv run python scripts/wait_for_services.py [--timeout 60]
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from urllib.parse import urlparse

from apexpulse.config import get_settings


def _wait_for_port(host: str, port: int, label: str, deadline: float) -> bool:
    """Poll ``host:port`` until it accepts a TCP connection or ``deadline`` passes."""
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            with socket.create_connection((host, port), timeout=2.0):
                print(f"  [ok]   {label:<10} {host}:{port} (attempt {attempt})")
                return True
        except OSError:
            time.sleep(1.0)
    print(f"  [fail] {label:<10} {host}:{port} unreachable")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=60.0, help="Seconds to wait per service.")
    args = parser.parse_args()

    settings = get_settings()
    broker = settings.kafka_bootstrap_servers.split(",")[0].strip()
    kafka_host, _, kafka_port = broker.rpartition(":")
    redis = urlparse(str(settings.redis_url))

    targets = [
        (kafka_host or "localhost", int(kafka_port or 19092), "redpanda"),
        (redis.hostname or "localhost", redis.port or 6379, "redis"),
    ]

    print("Waiting for ApexPulse infrastructure...")
    deadline = time.monotonic() + args.timeout
    results = [_wait_for_port(host, port, label, deadline) for host, port, label in targets]

    if all(results):
        print("All services are reachable.")
        return 0

    print(
        "\nOne or more services did not come up. Try: docker compose up -d --wait", file=sys.stderr
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
