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


@app.command()
def simulate(
    seed: int = typer.Option(42, help="RNG seed; the same seed replays the same match."),
    tick_rate: float = typer.Option(8.0, help="Snapshots emitted per simulated second."),
    map_name: str = typer.Option("de_mirage", help="Map to play."),
) -> None:
    """Simulate one match and print a summary of what happened.

    Runs entirely in-process — no broker, no services — so it is the quickest way
    to see the telemetry the pipeline is built around.
    """
    from collections import Counter

    from apexpulse.producer import MatchSimulator
    from apexpulse.schemas.enums import MapName
    from apexpulse.schemas.events import (
        BombPlantedEvent,
        KillEvent,
        MatchEndEvent,
        RoundEndEvent,
    )

    simulator = MatchSimulator(seed=seed, tick_rate_hz=tick_rate, map_name=MapName(map_name))

    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    rounds: list[RoundEndEvent] = []
    plants = kills = 0
    final: MatchEndEvent | None = None

    for event in simulator.run():
        counts[event.event_type.value] += 1
        if isinstance(event, RoundEndEvent):
            rounds.append(event)
            reasons[event.reason.value] += 1
        elif isinstance(event, BombPlantedEvent):
            plants += 1
        elif isinstance(event, KillEvent):
            kills += 1
        elif isinstance(event, MatchEndEvent):
            final = event

    total = sum(counts.values())
    typer.echo(f"\nMatch {simulator.match_id} on {map_name}  (seed={seed}, {tick_rate} Hz)")
    typer.echo("=" * 58)

    if final is not None:
        typer.echo(f"  Final score    CT {final.score_ct} - {final.score_t} T")
        typer.echo(f"  Winner         {final.winner.value}")

    typer.echo(f"  Rounds played  {len(rounds)}")
    typer.echo(f"  Total events   {total:,}  ({counts['tick']:,} ticks, {kills} kills)")

    if rounds:
        ct_wins = sum(1 for event in rounds if event.winner.value == "CT")
        typer.echo(f"  CT win rate    {ct_wins / len(rounds):.1%}")
        typer.echo(f"  Plant rate     {plants / len(rounds):.1%}")

    typer.echo("\n  Round outcomes")
    for reason, count in reasons.most_common():
        bar = "#" * count
        typer.echo(f"    {reason:16} {count:3}  {bar}")

    typer.echo("\n  Scoreline")
    score_ct = score_t = 0
    for event in rounds:
        marker = "CT" if event.score_ct > score_ct else "T "
        score_ct, score_t = event.score_ct, event.score_t
        typer.echo(
            f"    R{event.round_number:<3} {marker} wins  {score_ct:>2}-{score_t:<2}"
            f"  ({event.reason.value})"
        )
    typer.echo("")


@app.command()
def replay(
    seed: int = typer.Option(42, help="RNG seed."),
    tick_rate: float = typer.Option(8.0, help="Snapshots per simulated second."),
    speed: float = typer.Option(
        0.0, help="Wall-clock multiplier; 1.0 is real time, 0.0 is as fast as possible."
    ),
    max_events: int = typer.Option(0, help="Stop after N events; 0 means run the full match."),
) -> None:
    """Publish a simulated match to the broker and consume it back.

    Exercises the real path a live match takes — producer, transport, consumer —
    against whichever backend is configured.
    """
    from apexpulse.broker import create_broker
    from apexpulse.producer import MatchSimulator, TelemetryReplayer
    from apexpulse.schemas.events import parse_event

    configure_logging()
    settings = get_settings()
    typer.echo(f"broker: {settings.broker_backend}   topic: {settings.kafka_telemetry_topic}\n")

    async def _run() -> None:
        received: list[str] = []

        async with create_broker(settings) as broker:

            async def _consume() -> None:
                async for message in broker.consume(
                    settings.kafka_telemetry_topic, group="cli-replay"
                ):
                    received.append(parse_event(message.value).event_type.value)

            consumer = asyncio.create_task(_consume())
            await asyncio.sleep(0.05)

            replayer = TelemetryReplayer(broker=broker, settings=settings, speed=speed)
            stats = await replayer.replay(
                MatchSimulator(seed=seed, tick_rate_hz=tick_rate),
                max_events=max_events or None,
            )
            await asyncio.sleep(0.1)
            consumer.cancel()

        typer.echo(f"  published   {stats.events_published:,} events")
        typer.echo(f"  consumed    {len(received):,} events")
        typer.echo(f"  ticks       {stats.ticks_published:,}")
        typer.echo(f"  rounds      {stats.rounds_completed}")
        typer.echo(f"  duration    {stats.duration_seconds:.2f}s")
        typer.echo(f"  throughput  {stats.events_per_second:,.0f} events/s")

    asyncio.run(_run())


@app.command()
def live(
    seed: int = typer.Option(42, help="RNG seed."),
    tick_rate: float = typer.Option(8.0, help="Snapshots per simulated second."),
    speed: float = typer.Option(0.0, help="Wall-clock multiplier; 1.0 is real time."),
    max_events: int = typer.Option(2000, help="Stop after N events; 0 runs the full match."),
) -> None:
    """Run the full pipeline and print the live match state as it updates.

    Producer to broker to consumer to state store, then reads the stored snapshot
    back — the same path a real match takes.
    """
    from apexpulse.broker import create_broker
    from apexpulse.producer import MatchSimulator, TelemetryReplayer
    from apexpulse.storage import create_state_store
    from apexpulse.stream import MatchTracker, TelemetryConsumer

    configure_logging()
    settings = get_settings()

    async def _run() -> None:
        async with create_broker(settings) as broker, create_state_store(settings) as store:
            tracker = MatchTracker(store=store, settings=settings)
            consumer = TelemetryConsumer(broker=broker, settings=settings)
            consumer.on(None, tracker.handle)

            task = asyncio.create_task(consumer.run(max_events=max_events or None))
            await asyncio.sleep(0.05)

            replayer = TelemetryReplayer(broker=broker, settings=settings, speed=speed)
            await replayer.replay(
                MatchSimulator(seed=seed, tick_rate_hz=tick_rate),
                max_events=max_events or None,
            )
            await asyncio.sleep(0.2)
            consumer.stop()
            stats = await task

            typer.echo("")
            typer.echo(f"  consumed    {stats.events_consumed:,} events")
            typer.echo(f"  malformed   {stats.events_failed}")
            typer.echo(f"  handler err {stats.handler_errors}")

            for match_id in await tracker.live_match_ids():
                snapshot = await tracker.snapshot(match_id)
                if snapshot is None:
                    continue

                state = snapshot.state
                round_state = state.round_state
                momentum = snapshot.momentum
                history = await tracker.history(match_id)

                typer.echo(f"\n  Live state for {match_id}")
                typer.echo("  " + "-" * 46)
                typer.echo(f"    score          CT {state.score_ct} - {state.score_t} T")
                typer.echo(
                    f"    round          {round_state.round_number}  ({round_state.phase.value})"
                )
                typer.echo(f"    clock          {round_state.seconds_remaining:.1f}s")
                typer.echo(f"    bomb           {'PLANTED' if round_state.bomb_planted else '-'}")
                typer.echo(f"    alive          CT {state.alive_ct} vs T {state.alive_t}")
                typer.echo(f"    man advantage  {state.man_advantage:+d}")
                typer.echo(
                    f"    economy        CT ${state.economy_ct.money:,}"
                    f"  vs  T ${state.economy_t.money:,}"
                )
                typer.echo(
                    f"    momentum       kill delta {momentum.kill_delta:+d}"
                    f"  over {momentum.window_seconds}s"
                )

                streak = history.current_streak()
                if streak is not None:
                    side, length = streak
                    typer.echo(
                        f"    streak         {side.value} has won {length} in a row"
                        f"  ({history.rounds_played} rounds recorded)"
                    )
            typer.echo("")

    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    app()
