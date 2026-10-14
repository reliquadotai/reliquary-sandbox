"""Session-wide test fixtures."""

import os
import sys
from pathlib import Path

import pytest

# Tests run as a normal user: scoring processes keep that uid (the service refuses to
# score unprivileged without this flag), and the forkserver starts through the repo's
# wrapper with this interpreter (the image ships the wrapper and its own venv).
os.environ.setdefault("RELIQUARY_SCORING_UNPRIVILEGED_OK", "1")
os.environ.setdefault("RELIQUARY_SANDBOX_SCORING_WRAPPER",
                      str(Path(__file__).resolve().parents[1] / "scripts" / "scoring-python"))
os.environ.setdefault("RELIQUARY_SCORING_PYTHON", sys.executable)


@pytest.fixture(autouse=True, scope="session")
def _stop_the_scoring_forkserver():
    """Scoring processes (the verifiers bridge) and the service's lifespan start a
    multiprocessing forkserver: stopped when the session ends, never left behind."""
    yield
    from reliquary_sandbox_service import scoring_server

    scoring_server.stop()
