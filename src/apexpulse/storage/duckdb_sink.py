"""Historical telemetry persistence via DuckDB.

Live state answers "what is happening now" and expires; this sink answers "what
happened" and does not. It is also where the training set comes from: every tick
is stored with the outcome of the round it belonged to, so a supervised dataset is
a single query rather than a replay.

Writes are batched. A tick is a few hundred bytes and arrives 8x a second per
match, so per-row inserts would dominate the consumer's time budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any, Final, Self

from apexpulse.config import get_settings
from apexpulse.logging import get_logger
from apexpulse.schemas.enums import Team

if TYPE_CHECKING:
    from pathlib import Path

    import duckdb

    from apexpulse.config import Settings
    from apexpulse.schemas.events import MatchEndEvent, RoundEndEvent, TickEvent

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 500
"""Ticks buffered before a flush; roughly one minute of a single 8 Hz match."""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ticks (
    match_id           VARCHAR NOT NULL,
    sequence           BIGINT  NOT NULL,
    timestamp          TIMESTAMPTZ NOT NULL,
    map_name           VARCHAR NOT NULL,
    round_number       INTEGER NOT NULL,
    phase              VARCHAR NOT NULL,
    seconds_remaining  DOUBLE  NOT NULL,
    bomb_planted       BOOLEAN NOT NULL,
    bomb_seconds       DOUBLE,
    score_ct           INTEGER NOT NULL,
    score_t            INTEGER NOT NULL,
    alive_ct           INTEGER NOT NULL,
    alive_t            INTEGER NOT NULL,
    health_ct          INTEGER NOT NULL,
    health_t           INTEGER NOT NULL,
    money_ct           INTEGER NOT NULL,
    money_t            INTEGER NOT NULL,
    equipment_ct       INTEGER NOT NULL,
    equipment_t        INTEGER NOT NULL,
    losses_ct          INTEGER NOT NULL,
    losses_t           INTEGER NOT NULL
);

-- No primary key on `ticks` deliberately. It is append-only and hot: a key would
-- make every insert probe an index, which measured ~130 rows/s against ~200k for
-- a bulk append. Replays are de-duplicated by the training view instead.

CREATE TABLE IF NOT EXISTS rounds (
    match_id      VARCHAR NOT NULL,
    round_number  INTEGER NOT NULL,
    winner        VARCHAR NOT NULL,
    reason        VARCHAR NOT NULL,
    score_ct      INTEGER NOT NULL,
    score_t       INTEGER NOT NULL,
    timestamp     TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (match_id, round_number)
);

CREATE TABLE IF NOT EXISTS matches (
    match_id    VARCHAR PRIMARY KEY,
    winner      VARCHAR NOT NULL,
    score_ct    INTEGER NOT NULL,
    score_t     INTEGER NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL
);
"""

TRAINING_VIEW_SQL = """
CREATE OR REPLACE VIEW training_data AS
WITH deduped AS (
    SELECT DISTINCT ON (match_id, sequence) *
    FROM ticks
    ORDER BY match_id, sequence
)
SELECT
    t.*,
    r.winner  AS round_winner,
    r.reason  AS round_reason,
    CAST(r.winner = 'CT' AS INTEGER) AS ct_won
FROM deduped AS t
JOIN rounds AS r
  ON t.match_id = r.match_id
 AND t.round_number = r.round_number
WHERE t.phase <> 'freezetime';
"""
"""Joins each tick to its round's outcome, which is the supervised label.

Freezetime ticks are excluded: nothing has happened yet, so they carry no signal
and would teach the model that the pre-round state predicts the result.
"""


_ARROW_SCHEMA_FIELDS: Final = (
    ("match_id", "string"),
    ("sequence", "int64"),
    ("timestamp", "timestamp"),
    ("map_name", "string"),
    ("round_number", "int32"),
    ("phase", "string"),
    ("seconds_remaining", "float64"),
    ("bomb_planted", "bool"),
    ("bomb_seconds", "float64"),
    ("score_ct", "int32"),
    ("score_t", "int32"),
    ("alive_ct", "int32"),
    ("alive_t", "int32"),
    ("health_ct", "int32"),
    ("health_t", "int32"),
    ("money_ct", "int32"),
    ("money_t", "int32"),
    ("equipment_ct", "int32"),
    ("equipment_t", "int32"),
    ("losses_ct", "int32"),
    ("losses_t", "int32"),
)
"""Column order and type of a tick row, matching the ``ticks`` table."""


@cache
def _arrow_schema() -> Any:
    """Build the Arrow schema once; constructing it per flush is wasteful."""
    import pyarrow as pa

    mapping = {
        "string": pa.string(),
        "int64": pa.int64(),
        "int32": pa.int32(),
        "float64": pa.float64(),
        "bool": pa.bool_(),
        "timestamp": pa.timestamp("us", tz="UTC"),
    }
    return pa.schema([pa.field(name, mapping[kind]) for name, kind in _ARROW_SCHEMA_FIELDS])


@dataclass
class SinkStats:
    """Counters describing a sink's lifetime."""

    ticks_written: int = 0
    rounds_written: int = 0
    matches_written: int = 0
    flushes: int = 0

    @property
    def pending_flush_size(self) -> float:
        """Mean ticks per flush; 0.0 before the first flush."""
        return 0.0 if self.flushes == 0 else self.ticks_written / self.flushes


@dataclass
class DuckDBSink:
    """Append-only historical store for telemetry.

    Args:
        path: Database file, or ``:memory:`` for an ephemeral store.
        settings: Runtime configuration; defaults to the process settings.
        batch_size: Ticks buffered before an automatic flush.
    """

    path: Path | str | None = None
    settings: Settings = field(default_factory=get_settings)
    batch_size: int = DEFAULT_BATCH_SIZE

    _conn: duckdb.DuckDBPyConnection | None = field(default=None, init=False, repr=False)
    _buffer: list[tuple[Any, ...]] = field(default_factory=list, init=False, repr=False)
    _stats: SinkStats = field(default_factory=SinkStats, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

    @property
    def stats(self) -> SinkStats:
        """Counters for this sink."""
        return self._stats

    @property
    def pending(self) -> int:
        """Ticks buffered but not yet written."""
        return len(self._buffer)

    @property
    def _db(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            raise RuntimeError("sink is not started")
        return self._conn

    async def start(self) -> None:
        """Open the database and create the schema if it does not exist."""
        import duckdb

        target = self.path if self.path is not None else self.settings.duckdb_path
        if target != ":memory:":
            from pathlib import Path as _Path

            resolved = _Path(target)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            target = str(resolved)

        self._conn = duckdb.connect(str(target))
        self._db.execute(SCHEMA_SQL)
        self._db.execute(TRAINING_VIEW_SQL)
        logger.info("duckdb_sink_started", path=str(target))

    async def stop(self) -> None:
        """Flush buffered ticks and close the database."""
        if self._conn is None:
            return
        await self.flush()
        self._conn.close()
        self._conn = None
        logger.info(
            "duckdb_sink_stopped",
            ticks=self._stats.ticks_written,
            rounds=self._stats.rounds_written,
            matches=self._stats.matches_written,
        )

    # -- Writes ---------------------------------------------------------------

    async def write_tick(self, tick: TickEvent) -> None:
        """Buffer ``tick``, flushing once the batch is full."""
        self._buffer.append(self._tick_row(tick))
        if len(self._buffer) >= self.batch_size:
            await self.flush()

    async def write_round(self, event: RoundEndEvent) -> None:
        """Record a completed round, which supplies the training label.

        Rounds are written immediately rather than buffered: a tick is worthless
        for training until its round's outcome is known, so the label must never
        lag behind the features it explains.
        """
        self._db.execute(
            """
            INSERT INTO rounds VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (match_id, round_number) DO UPDATE SET
                winner = excluded.winner,
                reason = excluded.reason,
                score_ct = excluded.score_ct,
                score_t = excluded.score_t,
                timestamp = excluded.timestamp
            """,
            (
                event.match_id,
                event.round_number,
                event.winner.value,
                event.reason.value,
                event.score_ct,
                event.score_t,
                event.timestamp,
            ),
        )
        self._stats.rounds_written += 1

    async def write_match(self, event: MatchEndEvent) -> None:
        """Record a finished match and flush any ticks still buffered."""
        await self.flush()
        self._db.execute(
            """
            INSERT INTO matches VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (match_id) DO UPDATE SET
                winner = excluded.winner,
                score_ct = excluded.score_ct,
                score_t = excluded.score_t,
                timestamp = excluded.timestamp
            """,
            (
                event.match_id,
                event.winner.value,
                event.score_ct,
                event.score_t,
                event.timestamp,
            ),
        )
        self._stats.matches_written += 1

    async def flush(self) -> int:
        """Write buffered ticks; return how many rows were persisted.

        Rows go in as a single Arrow table rather than row-wise parameters.
        DuckDB is columnar: an ``executemany`` upsert probes the primary-key index
        once per row and measured ~130 rows/s, while a bulk append of the same
        data is three orders of magnitude faster.
        """
        if not self._buffer:
            return 0

        rows = self._buffer
        self._buffer = []

        import pyarrow as pa

        table = pa.Table.from_arrays(
            [pa.array(column) for column in zip(*rows, strict=True)],
            schema=_arrow_schema(),
        )
        # Registered as a view so the INSERT reads it without a copy.
        self._db.register("_pending_ticks", table)
        try:
            self._db.execute("INSERT INTO ticks SELECT * FROM _pending_ticks")
        finally:
            self._db.unregister("_pending_ticks")

        self._stats.ticks_written += len(rows)
        self._stats.flushes += 1
        return len(rows)

    # -- Reads ----------------------------------------------------------------

    def count(self, table: str) -> int:
        """Return the row count of ``table``.

        Args:
            table: One of ``ticks``, ``rounds``, ``matches``, or ``training_data``.
        """
        if table not in {"ticks", "rounds", "matches", "training_data"}:
            raise ValueError(f"unknown table: {table!r}")
        result = self._db.execute(f"SELECT count(*) FROM {table}").fetchone()
        return int(result[0]) if result else 0

    def training_frame(self) -> Any:
        """Return the labelled training set as a pandas DataFrame."""
        return self._db.execute("SELECT * FROM training_data").df()

    def export_parquet(self, destination: Path | str, *, table: str = "training_data") -> Path:
        """Write ``table`` to a Parquet file and return its path."""
        from pathlib import Path as _Path

        if table not in {"ticks", "rounds", "matches", "training_data"}:
            raise ValueError(f"unknown table: {table!r}")

        target = _Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._db.execute(
            f"COPY (SELECT * FROM {table}) TO ? (FORMAT PARQUET)",
            (str(target),),
        )
        logger.info("parquet_exported", table=table, path=str(target))
        return target

    def label_balance(self) -> tuple[int, int]:
        """Return ``(ct_wins, t_wins)`` across labelled training rows.

        A wildly skewed split means the model can score well by always guessing
        one side, so this is checked before training rather than after.
        """
        result = self._db.execute(
            "SELECT sum(ct_won), count(*) - sum(ct_won) FROM training_data"
        ).fetchone()
        if result is None or result[0] is None:
            return 0, 0
        return int(result[0]), int(result[1])

    # -- Event routing --------------------------------------------------------

    async def handle(self, event: Any) -> None:
        """Route a telemetry event to the appropriate table.

        Registered against the consumer with ``consumer.on(None, sink.handle)``.
        """
        from apexpulse.schemas.events import MatchEndEvent, RoundEndEvent, TickEvent

        if isinstance(event, TickEvent):
            await self.write_tick(event)
        elif isinstance(event, RoundEndEvent):
            await self.write_round(event)
        elif isinstance(event, MatchEndEvent):
            await self.write_match(event)

    # -- Internals ------------------------------------------------------------

    @staticmethod
    def _tick_row(tick: TickEvent) -> tuple[Any, ...]:
        """Flatten a tick into the column order of the ``ticks`` table."""
        state = tick.state
        round_state = state.round_state
        ct_players = state.players_on(Team.CT)
        t_players = state.players_on(Team.T)

        return (
            tick.match_id,
            tick.sequence,
            tick.timestamp,
            state.map_name.value,
            round_state.round_number,
            round_state.phase.value,
            round_state.seconds_remaining,
            round_state.bomb_planted,
            round_state.bomb_seconds_remaining,
            state.score_ct,
            state.score_t,
            state.alive_ct,
            state.alive_t,
            sum(player.health for player in ct_players),
            sum(player.health for player in t_players),
            state.economy_ct.money,
            state.economy_t.money,
            state.economy_ct.equipment_value,
            state.economy_t.equipment_value,
            state.economy_ct.consecutive_losses,
            state.economy_t.consecutive_losses,
        )

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()
