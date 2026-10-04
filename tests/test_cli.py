"""Command-line requests, restore constraints, output, and process exit codes."""

import json
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from reliquary_sandbox import CommandResponse, SandboxCheckpoint, cli


class SandboxResult(BaseModel):
    id: str = "guest-id"
    status: str = "RUNNING"


def checkpoint() -> SandboxCheckpoint:
    now = datetime.now(timezone.utc)
    return SandboxCheckpoint(
        id="checkpoint-id",
        sandbox_id="guest-id",
        state="DURABLE",
        depth=0,
        docker_image="approved-guest",
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def clients(monkeypatch):
    monkeypatch.delenv("RELIQUARY_SANDBOX_DEFAULT_IMAGE", raising=False)
    api = Mock()
    client = Mock()
    client.create.return_value = SandboxResult()
    api_factory = Mock(return_value=api)
    sandbox_factory = Mock(return_value=client)
    monkeypatch.setattr(cli, "APIClient", api_factory)
    monkeypatch.setattr(cli, "SandboxClient", sandbox_factory)
    return api, client, api_factory, sandbox_factory


def test_create_defaults_use_bounded_container_request(clients, capsys):
    api, client, _, factory = clients
    assert cli.main(["create", "--json"]) == 0
    factory.assert_called_once_with(api, runtime="container")
    request = client.create.call_args.args[0]
    assert request.name == "eval-workspace"
    assert request.docker_image == "reliquary-sandbox-guest:0.1.0"
    assert (request.cpu_cores, request.memory_gb, request.disk_size_gb) == (1, 1, 5)
    assert request.timeout_minutes == 60
    assert json.loads(capsys.readouterr().out)["id"] == "guest-id"
    api.client.close.assert_called_once()


@pytest.mark.parametrize("guest_code,expected", [(0, 0), (7, 7), (255, 125), (-1, 1)])
def test_run_preserves_output_and_returns_guest_exit_status(clients, capsys, guest_code, expected):
    api, client, _, _ = clients
    client.execute_command.return_value = CommandResponse(
        stdout="evaluation output\n", stderr="evaluation diagnostic\n", exit_code=guest_code
    )
    assert cli.main(["run", "guest-id", "--env", "MODE=eval", "python", "eval.py"]) == expected
    client.execute_command.assert_called_once_with(
        "guest-id", "python eval.py", working_dir=None, env={"MODE": "eval"}, timeout=300
    )
    captured = capsys.readouterr()
    assert captured.out == "evaluation output\n"
    assert captured.err == "evaluation diagnostic\n"
    api.client.close.assert_called_once()


def test_checkpoint_wait_and_model_list_emit_json(clients, capsys):
    _, client, _, _ = clients
    durable = checkpoint()
    client.checkpoint.return_value = durable
    client.wait_for_checkpoint.return_value = durable
    assert cli.main(["checkpoint", "guest-id", "--wait", "--timeout", "10", "--json"]) == 0
    client.checkpoint.assert_called_once_with("guest-id")
    client.wait_for_checkpoint.assert_called_once_with("checkpoint-id", timeout_seconds=10)
    assert json.loads(capsys.readouterr().out)["state"] == "DURABLE"
    client.list_checkpoints.return_value = [durable]
    assert cli.main(["checkpoints", "guest-id", "--json"]) == 0
    client.list_checkpoints.assert_called_once_with("guest-id")
    assert json.loads(capsys.readouterr().out)[0]["id"] == "checkpoint-id"


def test_restore_omits_image_and_disk_source_fields(clients, capsys):
    api, client, _, factory = clients
    assert cli.main([
        "restore", "checkpoint-id", "--name", "restored-eval", "--cpu", "0.5",
        "--memory", "0.5", "--timeout", "10", "--json",
    ]) == 0
    factory.assert_called_once_with(api, runtime="container")
    request = client.create.call_args.args[0]
    assert request.checkpoint_id == "checkpoint-id"
    assert request.name == "restored-eval"
    assert request.docker_image is None
    assert request.disk_size_gb is None
    assert "docker_image" not in request.model_fields_set
    assert "disk_size_gb" not in request.model_fields_set
    assert (request.cpu_cores, request.memory_gb, request.timeout_minutes) == (0.5, 0.5, 10)
    assert json.loads(capsys.readouterr().out)["id"] == "guest-id"


def test_delete_checkpoint_uses_local_extension(clients, capsys):
    api, _, _, _ = clients
    api.request.return_value = {"success": True}
    assert cli.main(["delete-checkpoint", "checkpoint-id", "--json"]) == 0
    api.request.assert_called_once_with("DELETE", "/sandbox/checkpoints/checkpoint-id")
    assert json.loads(capsys.readouterr().out) == {"success": True}
    api.client.close.assert_called_once()


def test_api_failure_returns_error_and_closes_connection(clients, capsys):
    api, client, _, _ = clients
    client.create.side_effect = RuntimeError("sandbox runtime unavailable")
    assert cli.main(["create", "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": "RuntimeError", "message": "sandbox runtime unavailable"
    }
    api.client.close.assert_called_once()
