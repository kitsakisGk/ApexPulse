"""Command-line entrypoint for ApexPulse services.

Subcommands are registered here as each subsystem lands; today the CLI exposes the
project metadata needed to verify a working installation.
"""

from __future__ import annotations

import typer

from apexpulse import __version__
from apexpulse.config import get_settings
from apexpulse.logging import configure_logging, get_logger

app = typer.Typer(
    name="apexpulse",
    help="Real-time esports telemetry and live win-probability engine.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the installed ApexPulse version."""
    typer.echo(f"ApexPulse {__version__}")


@app.command()
def config() -> None:
    """Print the resolved runtime configuration."""
    configure_logging()
    settings = get_settings()
    get_logger(__name__).info(
        "resolved_configuration",
        environment=settings.environment,
        kafka_bootstrap_servers=settings.kafka_bootstrap_servers,
        telemetry_topic=settings.kafka_telemetry_topic,
        redis_url=str(settings.redis_url),
        tick_rate_hz=settings.tick_rate_hz,
    )


if __name__ == "__main__":  # pragma: no cover
    app()
