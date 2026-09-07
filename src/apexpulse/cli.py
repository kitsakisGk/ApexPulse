"""Command-line entrypoint for ApexPulse services.

Subcommands are registered here as each subsystem lands; today the CLI exposes the
project metadata and environment diagnostics needed to verify an installation.
"""

from __future__ import annotations

import asyncio

import typer

from apexpulse import __version__
from apexpulse.broker import create_broker
from apexpulse.config import get_settings
from apexpulse.logging import configure_logging, get_logger
from apexpulse.storage import create_state_store

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
        broker_backend=settings.broker_backend,
        state_backend=settings.state_backend,
        kafka_bootstrap_servers=settings.kafka_bootstrap_servers,
        telemetry_topic=settings.kafka_telemetry_topic,
        redis_url=str(settings.redis_url),
        tick_rate_hz=settings.tick_rate_hz,
    )


@app.command()
def doctor() -> None:
    """Check that the configured backends are usable on this machine.

    Exits non-zero when a selected backend cannot be reached, so the same command
    works as a CI gate and as a local sanity check.
    """
    settings = get_settings()
    ok = True

    typer.echo(f"ApexPulse {__version__} environment check\n")
    typer.echo(f"  environment     {settings.environment}")
    typer.echo(f"  broker backend  {settings.broker_backend}")
    typer.echo(f"  state backend   {settings.state_backend}")
    typer.echo(
        "  external deps   "
        + ("redpanda/redis required" if settings.requires_external_services else "none")
    )
    typer.echo("")

    async def _probe() -> tuple[bool, bool]:
        broker_ok = False
        state_ok = False
        try:
            async with create_broker(settings) as broker:
                await broker.publish(settings.kafka_telemetry_topic, b"doctor")
                await broker.flush(timeout=2.0)
                broker_ok = True
        except Exception as exc:  # report any failure to the operator
            typer.echo(f"  [FAIL] broker    {type(exc).__name__}: {exc}")

        try:
            async with create_state_store(settings) as store:
                await store.set("apexpulse:doctor", {"ok": True}, ttl=10)
                state_ok = await store.get("apexpulse:doctor") is not None
                await store.delete("apexpulse:doctor")
        except Exception as exc:  # report any failure to the operator
            typer.echo(f"  [FAIL] state     {type(exc).__name__}: {exc}")

        return broker_ok, state_ok

    broker_ok, state_ok = asyncio.run(_probe())

    if broker_ok:
        typer.echo(f"  [ok]   broker    {settings.broker_backend} publish/flush succeeded")
    if state_ok:
        typer.echo(f"  [ok]   state     {settings.state_backend} round-trip succeeded")

    ok = broker_ok and state_ok
    typer.echo("")
    typer.echo("All configured backends are operational." if ok else "One or more backends failed.")
    raise typer.Exit(0 if ok else 1)


if __name__ == "__main__":  # pragma: no cover
    app()
