# ApexPulse API: FastAPI, the inference engine, and the WebSocket broadcaster.
#
# Two stages. The builder resolves dependencies with uv against the lockfile; the
# runtime image copies only the finished virtualenv and the source, so the
# compilers and caches uv needed never ship.

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /usr/local/bin/uv

WORKDIR /app

# Copy files outside the build cache's hot path first, so a source edit does not
# invalidate the dependency layer and force a full reinstall.
COPY pyproject.toml uv.lock README.md ./

# `--no-install-project` installs dependencies only; the project itself goes in
# after the source is copied, in a layer that rebuilds on every code change.
RUN uv sync --frozen --no-dev --no-install-project \
        --extra api --extra stream --extra storage --extra ml

COPY src ./src
RUN uv sync --frozen --no-dev \
        --extra api --extra stream --extra storage --extra ml


FROM python:3.12-slim AS runtime

# Run as an unprivileged user: the API has no reason to own the filesystem.
RUN useradd --create-home --uid 10001 apexpulse

WORKDIR /app
COPY --from=builder --chown=apexpulse:apexpulse /app /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    APEXPULSE_ENVIRONMENT=container \
    APEXPULSE_API_HOST=0.0.0.0 \
    APEXPULSE_API_PORT=8000

# Model checkpoints are build artifacts, not source, so they are mounted at run
# time rather than baked in. Without one the API still serves live match state
# and reports the absence on /health.
RUN mkdir -p /app/models /app/data/processed && chown -R apexpulse:apexpulse /app/models /app/data

USER apexpulse
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)"

CMD ["apexpulse", "serve"]
