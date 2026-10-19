"""Shared test configuration."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip APEXPULSE_* variables so tests see the code's defaults.

    CI exports APEXPULSE_STATE_BACKEND=memory for the CLI steps it runs. Without
    this, a test asserting the default backend would be testing the workflow's
    configuration rather than the code, and passing or failing by where it ran.
    Tests that need a setting pass it to Settings explicitly.
    """
    for name in list(os.environ):
        if name.startswith("APEXPULSE_"):
            monkeypatch.delenv(name, raising=False)
