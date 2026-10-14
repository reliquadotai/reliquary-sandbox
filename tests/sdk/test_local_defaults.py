"""Local endpoint, truthful runtime selection, and CLI behavior."""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from reliquary_sandbox import APIClient, AsyncSandboxClient, Config, CreateSandboxRequest, cli
from reliquary_sandbox.models import CommandResponse
from reliquary_sandbox.sandbox import SandboxAuthCache, SandboxClient


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in tuple(os.environ):
        if name.startswith("RELIQUARY_SANDBOX_"):
            monkeypatch.delenv(name)


def test_default_config_is_local_and_does_not_read_other_config(tmp_path):
    assert Config().base_url == "http://127.0.0.1:8000"
    assert Config().api_key == ""
    assert Config().config_dir == tmp_path / ".reliquary-sandbox"


@pytest.mark.parametrize("runtime,expected", [("container", False), ("vm", True)])
def test_sync_create_sends_explicit_runtime(monkeypatch, runtime, expected):
    api = APIClient(api_key="test-key")
    client = SandboxClient(api, runtime=runtime)
    api.request = MagicMock(return_value={})
    monkeypatch.setattr("reliquary_sandbox.sandbox.Sandbox.model_validate", lambda data: data)
    client.create(CreateSandboxRequest(name="test", docker_image="python:3.12-slim"))
    assert api.request.call_args.kwargs["json"]["vm"] is expected
    api.client.close()


@pytest.mark.asyncio
async def test_async_default_runtime_is_container(monkeypatch):
    async with AsyncSandboxClient(api_key="test-key") as client:
        client.client.request = AsyncMock(return_value={})
        monkeypatch.setattr("reliquary_sandbox.sandbox.Sandbox.model_validate", lambda data: data)
        await client.create(CreateSandboxRequest(name="test", docker_image="python:3.12-slim"))
        assert client.client.request.await_args.kwargs["json"]["vm"] is False


def test_cached_bearer_token_is_private(tmp_path):
    path = tmp_path / "config" / "auth.json"
    cache = SandboxAuthCache(path, MagicMock())
    cache.set(
        "sandbox-1",
        {"token": "test", "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()},
    )
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_cached_bearer_replaces_old_public_file(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text("{}")
    path.chmod(0o644)
    cache = SandboxAuthCache(path, MagicMock())
    cache.set("sandbox-1", {"token": "fresh"})
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())["sandbox-1"]["token"] == "fresh"


def test_cli_run_preserves_environment_and_exit_code(monkeypatch, capsys):
    api = MagicMock()
    sandbox = MagicMock()
    sandbox.execute_command.return_value = CommandResponse(stdout="output", stderr="error", exit_code=7)
    monkeypatch.setattr(cli, "APIClient", lambda **kwargs: api)
    monkeypatch.setattr(cli, "SandboxClient", lambda *args, **kwargs: sandbox)
    code = cli.main(["run", "sandbox-1", "--env", "VALUE=a=b", "--json", "echo hello"])
    assert code == 7
    assert json.loads(capsys.readouterr().out)["exit_code"] == 7
    sandbox.execute_command.assert_called_once_with(
        "sandbox-1", "echo hello", working_dir=None, env={"VALUE": "a=b"}, timeout=300
    )
    api.client.close.assert_called_once()


def test_cli_reports_invalid_environment_as_json(monkeypatch, capsys):
    monkeypatch.setattr(cli, "APIClient", lambda **kwargs: MagicMock())
    monkeypatch.setattr(cli, "SandboxClient", lambda *args, **kwargs: MagicMock())
    assert cli.main(["run", "sandbox-1", "--env", "bad", "--json", "true"]) == 1
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"
