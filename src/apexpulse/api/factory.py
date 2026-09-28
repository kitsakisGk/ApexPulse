"""Module-level application instance.

Uvicorn's reloader and most production runners take an import string rather than
an object, because they re-import the module in a fresh process. This gives them
a stable target:

    uvicorn apexpulse.api.factory:app

Everything else should call :func:`~apexpulse.api.app.create_app` directly, so it
can pass its own settings instead of sharing this instance.
"""

from __future__ import annotations

from apexpulse.api.app import create_app

app = create_app()

__all__ = ["app"]
