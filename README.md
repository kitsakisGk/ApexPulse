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
| Ingestion & streaming | Redpanda (Kafka API), or an in-process broker |
| Live state | Redis, or SQLite |
| Historical storage | DuckDB + Parquet |
| ML inference | XGBoost |
| API & broadcast | FastAPI + WebSockets |
| Frontend | Next.js + Tailwind CSS |
| Packaging | uv, Docker Compose, GitHub Actions |

## Running modes

ApexPulse selects its transport and state store at startup, so the entire pipeline
runs with **no external services** — useful on machines without Docker, and the
reason the test suite needs no containers.

| | `broker_backend` | `state_backend` | Requires |
| --- | --- | --- | --- |
| **Standalone** (default) | `memory` | `sqlite` | nothing |
| **Full stack** | `kafka` | `redis` | Redpanda + Redis |

Application code is written against the `EventBroker` and `StateStore` interfaces
only, so switching is purely configuration:

```bash
uv run apexpulse doctor    # verify the configured backends are reachable
```

```bash
# opt into the container stack
APEXPULSE_BROKER_BACKEND=kafka APEXPULSE_STATE_BACKEND=redis uv run apexpulse doctor
```

## Try it

No services required — these run on the standalone backends.

```bash
uv run apexpulse doctor                    # verify the configured backends
uv run apexpulse simulate --seed 7         # simulate a match, print the scoreline
uv run apexpulse replay --tick-rate 4      # publish through the broker and read it back
uv run apexpulse replay --speed 1.0        # replay in real time, as a live match would arrive
```

`simulate` runs the match generator in-process and summarises it. `replay` exercises
the real path — producer, transport, consumer — against whichever backend is
configured, reporting published vs. consumed counts and throughput.

## Infrastructure (optional)

The container stack is only needed for the `kafka`/`redis` backends. Requires
Docker Desktop or any Docker Engine with Compose v2.

```bash
make up              # start redpanda + redis, wait for healthchecks
make ps              # service status
make topics          # list the bootstrapped kafka topics
make down            # stop the stack (volumes preserved)
make clean           # stop and destroy volumes
```

`make up` blocks until every healthcheck reports healthy, then a one-shot
`redpanda-init` container creates the telemetry and prediction topics.

| Service | Host endpoint | Purpose |
| --- | --- | --- |
| Redpanda | `localhost:19092` | Kafka API — telemetry and prediction topics |
| Redpanda Admin | `localhost:9644` | Admin API and Prometheus metrics |
| Redis | `localhost:6379` | Live match state store |
| Console | http://localhost:8080 | Topic and message browser |

To confirm host-side reachability before running a producer or consumer:

```bash
uv run python scripts/wait_for_services.py
```

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

Or run every gate at once with `make check`.

## Layout

```
src/apexpulse/
├── broker/      # pluggable transport: in-memory | kafka
│   ├── base.py      # EventBroker interface
│   ├── memory.py    # asyncio-queue implementation
│   └── kafka.py     # confluent-kafka implementation
├── schemas/     # pydantic domain models
├── producer/    # telemetry replay simulator + event publisher
├── stream/      # streaming consumer + windowed aggregation
├── storage/     # state store (memory | sqlite | redis) + duckdb sink
├── features/    # real-time feature extraction
├── ml/          # offline training and calibration
├── inference/   # low-latency scoring worker
└── api/         # fastapi rest + websocket broadcast
```

## License

MIT © George Kitsakis
