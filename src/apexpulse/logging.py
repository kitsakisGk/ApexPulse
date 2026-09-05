"""Structured logging setup shared by every ApexPulse service.

Local runs get colourised, human-readable output; anything else emits JSON lines so
container logs stay machine-parseable.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from apexpulse.config import get_settings


def configure_logging(*, force_json: bool | None = None) -> None:
    """Configure ``structlog`` and the stdlib logging bridge.

    Args:
        force_json: Override the automatic renderer choice. ``None`` selects JSON
            for every environment except ``local``.
    """
    settings = get_settings()
    use_json = force_json if force_json is not None else settings.environment != "local"

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer() if use_json else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structured logger for ``name``."""
    return structlog.stdlib.get_logger(name)
