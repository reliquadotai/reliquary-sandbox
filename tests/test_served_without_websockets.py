"""The gateway serves HTTP only: no WebSocket protocol is loaded (the lock carries
websockets 16 for verifiers' sake; the service never needs it)."""

import json
import re
from pathlib import Path

from uvicorn.main import main as uvicorn_cli

from reliquary_sandbox_service.episodes.local_gateway import server_config

ROOT = Path(__file__).resolve().parents[1]


def test_the_local_gateway_server_loads_no_websocket_protocol():
    config = server_config(object(), 8123)
    assert config.ws == "none"
    assert (config.host, config.port, config.lifespan) == ("127.0.0.1", 8123, "on")


def test_the_controller_image_serves_without_websockets():
    dockerfile = (ROOT / "Dockerfile").read_text()
    stage = dockerfile[dockerfile.index("AS controller\n"):]
    command = json.loads(re.search(r"^CMD (\[.*\])$", stage, re.M).group(1))
    assert command[0] == "uvicorn"
    context = uvicorn_cli.make_context("uvicorn", command[1:])
    assert context.params["ws"] == "none"
    assert context.params["factory"] is True
