# ApexPulse

**Real-time esports telemetry and live win-probability engine for CS2.**

ApexPulse ingests high-frequency match telemetry, processes it as a stream, scores a
live win-probability model on every tick, and broadcasts the result to a low-latency
dashboard.

> 🚧 Under active construction. The full architecture diagram, benchmarks, and
> quickstart land with the Day 14 documentation pass.

## Stack

| Layer | Technology |
| --- | --- |
| Ingestion & streaming | Redpanda (Kafka API) |
| Live state | Redis |
| Historical storage | DuckDB + Parquet |
| ML inference | XGBoost |
| API & broadcast | FastAPI + WebSockets |
| Frontend | Next.js + Tailwind CSS |
| Packaging | uv, Docker Compose, GitHub Actions |

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync --all-extras          # install runtime + dev dependencies
cp .env.example .env          # seed local configuration
uv run apexpulse version      # verify the installation
uv run pytest                 # run the test suite
uv run ruff check .           # lint
uv run mypy                   # type-check
```

## Layout

```
src/apexpulse/
├── schemas/     # pydantic domain models
├── producer/    # telemetry replay simulator + kafka producer
├── stream/      # streaming consumer + windowed aggregation
├── storage/     # redis live state + duckdb historical sink
├── features/    # real-time feature extraction
├── ml/          # offline training and calibration
├── inference/   # low-latency scoring worker
└── api/         # fastapi rest + websocket broadcast
```

## License

MIT © George Kitsakis
