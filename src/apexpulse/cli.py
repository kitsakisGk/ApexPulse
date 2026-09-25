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


@app.command()
def dataset(
    matches: int = typer.Option(20, help="Matches to simulate."),
    tick_rate: float = typer.Option(4.0, help="Snapshots per simulated second."),
    seed: int = typer.Option(0, help="First RNG seed; each match uses seed + n."),
    output: str = typer.Option("", help="Parquet destination; defaults to data/processed."),
) -> None:
    """Generate a labelled training dataset and write it to DuckDB and Parquet.

    Each tick is stored with the outcome of the round it belonged to, which is the
    supervised label the win-probability model learns from.
    """
    from pathlib import Path

    from apexpulse.producer import MatchSimulator
    from apexpulse.storage import DuckDBSink

    configure_logging()
    settings = get_settings()
    destination = Path(output) if output else settings.data_dir / "processed" / "training.parquet"

    async def _run() -> None:
        async with DuckDBSink(settings=settings, batch_size=2_000) as sink:
            for offset in range(matches):
                simulator = MatchSimulator(
                    match_id=f"apex-{seed + offset:04d}",
                    seed=seed + offset,
                    tick_rate_hz=tick_rate,
                )
                for event in simulator.run():
                    await sink.handle(event)
            await sink.flush()

            rows = sink.count("training_data")
            ct_wins, t_wins = sink.label_balance()
            path = sink.export_parquet(destination)

            typer.echo(f"\n  matches        {sink.count('matches'):,}")
            typer.echo(f"  rounds         {sink.count('rounds'):,}")
            typer.echo(f"  raw ticks      {sink.count('ticks'):,}")
            typer.echo(f"  training rows  {rows:,}")
            if rows:
                typer.echo(f"  label balance  CT {ct_wins / rows:.1%}  vs  T {t_wins / rows:.1%}")
            typer.echo(f"  database       {settings.duckdb_path}")
            typer.echo(f"  parquet        {path}  ({path.stat().st_size / 1024:,.0f} KB)\n")

    asyncio.run(_run())


@app.command()
def train(
    test_fraction: float = typer.Option(0.2, help="Share of matches held out for evaluation."),
    rounds: int = typer.Option(400, help="Maximum boosting rounds."),
    seed: int = typer.Option(42, help="Seeds the booster for a reproducible run."),
    save: bool = typer.Option(True, help="Write the checkpoint to the model directory."),
    calibrate: bool = typer.Option(
        False,
        help="Fit an isotonic calibrator. Off by default: measured to worsen both "
        "skill and calibration on this model.",
    ),
) -> None:
    """Train the win-probability model on the stored dataset.

    Reads the labelled ``training_data`` view from DuckDB, holds out whole
    matches, trains a gradient-boosted classifier, and reports how well its
    probabilities hold up.
    """
    from apexpulse.ml import (
        assess_calibration,
        format_reliability_table,
        save_model,
        train_model,
    )
    from apexpulse.storage import DuckDBSink

    configure_logging()
    settings = get_settings()

    async def _run() -> None:
        async with DuckDBSink(settings=settings) as sink:
            rows = sink.count("training_data")
            if rows == 0:
                typer.echo(
                    "No training data found. Generate some first:\n  apexpulse dataset --matches 40"
                )
                raise typer.Exit(1)
            frame = sink.training_frame()

        typer.echo(f"\nTraining on {rows:,} rows from {frame['match_id'].nunique()} matches")
        typer.echo("=" * 62)

        booster, calibrator, result = train_model(
            frame,
            test_fraction=test_fraction,
            num_rounds=rounds,
            seed=seed,
            calibrate=calibrate,
        )
        metrics = result.metrics

        typer.echo("\n  Dataset")
        typer.echo(
            f"    train          {result.train_rows:,} rows / {result.train_matches} matches"
        )
        typer.echo(f"    test           {result.test_rows:,} rows / {result.test_matches} matches")
        typer.echo(f"    CT base rate   {metrics.base_rate:.1%}")

        typer.echo("\n  Held-out performance")
        typer.echo(
            f"    log loss       {metrics.log_loss:.4f}  (baseline {metrics.baseline_log_loss:.4f})"
        )
        typer.echo(
            f"    skill score    {metrics.skill_score:+.1%}  over always guessing the base rate"
        )
        typer.echo(f"    ROC AUC        {metrics.roc_auc:.4f}")
        typer.echo(f"    Brier score    {metrics.brier_score:.4f}")
        typer.echo(f"    accuracy       {metrics.accuracy:.1%}")
        typer.echo(f"    best iteration {result.best_iteration}")

        typer.echo("\n  Feature importance (gain)")
        for name, gain in list(result.feature_importance.items())[:10]:
            bar = "#" * max(1, round(gain * 60)) if gain > 0 else ""
            typer.echo(f"    {name:20} {gain:>6.1%}  {bar}")

        # Calibration is the measure that matters for a broadcast gauge: a "70%"
        # must be right about 70% of the time.
        import xgboost as xgb

        from apexpulse.features import FEATURE_NAMES, build_training_set, split_by_match

        _, test_frame = split_by_match(frame, test_fraction=test_fraction)
        test_features, test_labels = build_training_set(test_frame)

        matrix = xgb.DMatrix(test_features, feature_names=list(FEATURE_NAMES))
        raw = booster.predict(matrix, iteration_range=(0, booster.best_iteration + 1))
        calibrated = calibrator.apply(raw)

        report = assess_calibration(test_labels, calibrated)
        raw_report = assess_calibration(test_labels, raw)

        typer.echo("\n  Calibration (does a stated 70% actually win 70% of the time?)")
        typer.echo(format_reliability_table(report))
        if not calibrator.is_identity:
            typer.echo(
                f"    raw model      {raw_report.expected_calibration_error:.2%}"
                f"   ->  calibrated  {report.expected_calibration_error:.2%}"
            )
        verdict = "well calibrated" if report.is_well_calibrated else "needs calibration"
        typer.echo(f"    verdict        {verdict}")

        if save:
            model_path, calibrator_path, metadata_path = save_model(
                booster, result, calibrator, settings=settings
            )
            typer.echo(f"\n  Saved model      {model_path}")
            typer.echo(f"  Saved calibrator {calibrator_path}")
            typer.echo(f"  Saved metadata   {metadata_path}\n")
        else:
            typer.echo("\n  (not saved; pass --save to write the checkpoint)\n")

    asyncio.run(_run())


@app.command()
def predict(
    seed: int = typer.Option(42, help="RNG seed for the simulated match."),
    tick_rate: float = typer.Option(8.0, help="Snapshots per simulated second."),
    rounds: int = typer.Option(3, help="Rounds to score before stopping."),
    max_trees: int = typer.Option(0, help="Cap trees per prediction; 0 uses the checkpoint."),
) -> None:
    """Score a simulated match tick by tick and print the win probability.

    Runs the real serving path — feature extraction, model, calibrator — and
    reports the per-tick latency the dashboard would experience.
    """
    from apexpulse.inference import InferenceEngine
    from apexpulse.schemas.events import RoundEndEvent, TickEvent
    from apexpulse.stream.window import TelemetryWindow

    configure_logging()

    try:
        engine = InferenceEngine.from_checkpoint(max_trees=max_trees or None)
    except FileNotFoundError as exc:
        typer.echo(f"{exc}")
        raise typer.Exit(1) from exc

    from apexpulse.producer import MatchSimulator

    simulator = MatchSimulator(seed=seed, tick_rate_hz=tick_rate)
    window = TelemetryWindow(span_seconds=15.0)

    typer.echo(f"\nScoring match {simulator.match_id}  (seed={seed}, {tick_rate} Hz)")
    typer.echo(f"trees per prediction: {engine.tree_count}")
    typer.echo("=" * 72)
    typer.echo("  round  clock   alive      CT win%   bar                        latency")
    typer.echo("  " + "-" * 70)

    completed = 0
    emitted = 0

    for event in simulator.run():
        window.observe(event)

        if isinstance(event, RoundEndEvent):
            completed += 1
            typer.echo(
                f"  -- round {event.round_number} to {event.winner.value}"
                f" ({event.reason.value}), score {event.score_ct}-{event.score_t} --"
            )
            if completed >= rounds:
                break
            continue

        if not isinstance(event, TickEvent):
            continue

        prediction = engine.predict(event.state, window.metrics())
        if not prediction.scored:
            continue

        # One line per simulated second keeps the output readable.
        emitted += 1
        if emitted % max(1, int(tick_rate)) != 0:
            continue

        state = event.state
        probability = prediction.ct_win_probability
        filled = round(probability * 24)
        bar = "#" * filled + "." * (24 - filled)
        bomb = " BOMB" if state.round_state.bomb_planted else "     "

        typer.echo(
            f"  {state.round_state.round_number:>5}"
            f"  {state.round_state.seconds_remaining:>5.1f}"
            f"  {state.alive_ct}v{state.alive_t}{bomb}"
            f"  {probability:>7.1%}   {bar}  {prediction.latency_ms:>6.2f} ms"
        )

    stats = engine.stats
    typer.echo("\n  Latency over " + f"{stats.count:,} scored ticks")
    typer.echo(f"    mean   {stats.mean_ms:>7.3f} ms")
    typer.echo(f"    p50    {stats.p50_ms:>7.3f} ms")
    typer.echo(f"    p95    {stats.p95_ms:>7.3f} ms")
    typer.echo(f"    p99    {stats.p99_ms:>7.3f} ms")

    from apexpulse.inference.engine import LATENCY_BUDGET_MS

    verdict = "within budget" if stats.p99_ms < LATENCY_BUDGET_MS else "OVER BUDGET"
    typer.echo(f"    budget {LATENCY_BUDGET_MS:>7.3f} ms  ->  {verdict}")
    typer.echo(f"    skipped {stats.skipped:,} unscoreable ticks (freezetime)\n")


@app.command()
def validate(
    test_fraction: float = typer.Option(0.2, help="Share of matches held out."),
) -> None:
    """Benchmark the saved model by situation and check for feature drift.

    A single held-out score says the model works on average. This reports where
    it works and where it does not, then compares the live feature distribution
    against the one it was trained on.
    """
    import xgboost as xgb

    from apexpulse.features import FEATURE_NAMES, build_training_set, split_by_match
    from apexpulse.ml import (
        DriftDetector,
        benchmark_by_situation,
        format_benchmark_table,
        format_drift_table,
        load_model,
    )
    from apexpulse.storage import DuckDBSink

    configure_logging()
    settings = get_settings()

    async def _run() -> None:
        async with DuckDBSink(settings=settings) as sink:
            if sink.count("training_data") == 0:
                typer.echo("No data found. Run 'apexpulse dataset --matches 40' first.")
                raise typer.Exit(1)
            frame = sink.training_frame()

        try:
            booster, calibrator, metadata = load_model(settings=settings)
        except FileNotFoundError as exc:
            typer.echo(f"{exc}")
            raise typer.Exit(1) from exc

        train_frame, test_frame = split_by_match(frame, test_fraction=test_fraction)
        test_features, test_labels = build_training_set(test_frame)

        best = metadata.get("best_iteration")
        iteration_range = (0, int(best) + 1) if best is not None else None
        matrix = xgb.DMatrix(test_features, feature_names=list(FEATURE_NAMES))
        raw = (
            booster.predict(matrix, iteration_range=iteration_range)
            if iteration_range
            else booster.predict(matrix)
        )
        probabilities = calibrator.apply(raw)

        typer.echo(f"\nValidating on {len(test_frame):,} held-out rows")
        typer.echo("=" * 62)

        report = benchmark_by_situation(test_frame, test_labels, probabilities)
        typer.echo("\n  Accuracy by match situation")
        typer.echo(format_benchmark_table(report))

        failing = report.failing_slices
        if failing:
            typer.echo(
                f"\n  {len(failing)} situation(s) add no skill: "
                + ", ".join(item.name for item in failing)
            )

        train_features, _ = build_training_set(train_frame)
        detector = DriftDetector.from_frame(train_features)
        live = {name: test_features[name].tolist() for name in test_features.columns}

        typer.echo("\n  Feature drift, held-out against training")
        typer.echo(format_drift_table(detector.detect(live)))
        typer.echo("")

    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    app()
