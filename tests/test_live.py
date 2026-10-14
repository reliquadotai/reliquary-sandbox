"""Functional integration against an ephemeral local Linux Docker host."""

import json
import os
import secrets
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import docker
import httpx
import pytest

from reliquary_sandbox import (
    APIClient,
    AsyncSandboxClient,
    CreateSandboxRequest,
    SandboxClient,
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("RUN_SANDBOX_LIVE") != "1", reason="set RUN_SANDBOX_LIVE=1 on a Linux test host"
    ),
]

GUEST_IMAGE = "reliquary-sandbox-guest:0.1.0"


@dataclass
class LiveController:
    url: str
    tenant_key: str
    other_key: str
    daemon: docker.DockerClient


@pytest.fixture(scope="module")
def live_controller(tmp_path_factory):
    if sys.platform != "linux":
        pytest.fail("live integration requires Linux")
    daemon = docker.from_env()
    assert "runsc" in daemon.info().get("Runtimes", {}), "runsc must be registered with Docker"
    daemon.images.get(GUEST_IMAGE)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    tenant_key = secrets.token_urlsafe(32)
    other_key = secrets.token_urlsafe(32)
    directory = tmp_path_factory.mktemp("live-controller")
    environment = {
        **os.environ,
        "RELIQUARY_SANDBOX_TOKENS": json.dumps({"checks": tenant_key, "other": other_key}),
        "RELIQUARY_SANDBOX_STATE_DIR": str(directory / "state"),
        "RELIQUARY_SANDBOX_PUBLIC_BASE_URL": url,
        "RELIQUARY_SANDBOX_DOCKER_RUNTIME": "runsc",
        "RELIQUARY_SANDBOX_REQUIRE_GVISOR": "true",
        "RELIQUARY_SANDBOX_ALLOWED_IMAGES": json.dumps([GUEST_IMAGE]),
        "RELIQUARY_SANDBOX_DEFAULT_IMAGE": GUEST_IMAGE,
        "RELIQUARY_SANDBOX_REAPER_INTERVAL_SECONDS": "0.5",
    }
    with (directory / "controller.log").open("wb") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "reliquary_sandbox_service.app:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--no-proxy-headers",
            ],
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail("the local integration controller exited before readiness")
                try:
                    response = httpx.get(f"{url}/readyz", timeout=1)
                    if response.status_code == 200:
                        assert response.json()["runtime"] == "runsc"
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.1)
            else:
                pytest.fail("the local integration controller did not become ready")
            yield LiveController(url, tenant_key, other_key, daemon)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            daemon.close()


@pytest.fixture
def sdk_environment(live_controller, monkeypatch):
    monkeypatch.setenv("RELIQUARY_SANDBOX_API_BASE_URL", live_controller.url)
    monkeypatch.setenv("RELIQUARY_SANDBOX_API_KEY", live_controller.tenant_key)
    monkeypatch.setenv("RELIQUARY_SANDBOX_TEAM_ID", "")
    monkeypatch.delenv("RELIQUARY_SANDBOX_CONTEXT", raising=False)


def request(name: str, **overrides) -> CreateSandboxRequest:
    return CreateSandboxRequest(
        name=name,
        docker_image=GUEST_IMAGE,
        cpu_cores=1,
        memory_gb=1,
        disk_size_gb=0.25,
        timeout_minutes=5,
        **overrides,
    )


def test_sdk_lifecycle_commands_files_and_checkpoint(live_controller, sdk_environment, tmp_path):
    api = APIClient()
    api.config.config_dir = tmp_path / "sdk-cache"
    client = SandboxClient(api)
    active = []
    try:
        sandbox = client.create(request("integration-eval"))
        active.append(sandbox.id)
        client.wait_for_creation(sandbox.id, max_attempts=10)
        assert client.get(sandbox.id).status == "RUNNING"
        assert sandbox.id in [item.id for item in client.list().sandboxes]

        containers = live_controller.daemon.containers.list(
            filters={"label": f"reliquary.sandbox.id={sandbox.id}"}
        )
        assert len(containers) == 1
        container = containers[0]
        container.reload()
        config = container.attrs["HostConfig"]
        assert config["Runtime"] == "runsc"
        assert config["NetworkMode"] == "none"
        assert config["ReadonlyRootfs"] is True
        assert config["Privileged"] is False
        assert config["CapDrop"] == ["ALL"]
        assert config["Memory"] == 1024**3
        assert config["NanoCpus"] == 1_000_000_000
        assert container.attrs["Config"]["User"] == "1000:1000"

        result = client.execute_command(sandbox.id, "python -c 'print(sum(range(100)))'", timeout=10)
        assert (result.exit_code, result.stdout, result.stderr) == (0, "4950\n", "")
        failed = client.execute_command(
            sandbox.id,
            "python -c 'import sys; sys.stderr.write(\"expected failure\\n\"); sys.exit(7)'",
            timeout=10,
        )
        assert failed.exit_code == 7
        assert failed.stderr == "expected failure\n"

        source = tmp_path / "input.bin"
        source.write_bytes(b"\x00binary-evaluation-input\xff\n")
        client.upload_file(sandbox.id, "/workspace/input.bin", str(source))
        destination = tmp_path / "download.bin"
        client.download_file(sandbox.id, "/workspace/input.bin", str(destination))
        assert destination.read_bytes() == source.read_bytes()
        client.execute_command(sandbox.id, "printf 'checkpoint content' > /workspace/answer.txt")
        assert client.read_file(sandbox.id, "/workspace/answer.txt").content == "checkpoint content"

        checkpoint = client.checkpoint(sandbox.id)
        durable = client.wait_for_checkpoint(checkpoint.id, timeout_seconds=30)
        assert durable.state == "DURABLE"
        client.delete(sandbox.id)
        active.remove(sandbox.id)
        assert not live_controller.daemon.containers.list(
            all=True, filters={"label": f"reliquary.sandbox.id={sandbox.id}"}
        )

        restored = client.create(
            CreateSandboxRequest(
                name="restored-eval",
                checkpoint_id=checkpoint.id,
                cpu_cores=1,
                memory_gb=1,
                timeout_minutes=5,
            )
        )
        active.append(restored.id)
        client.wait_for_creation(restored.id, max_attempts=10)
        assert client.read_file(restored.id, "/workspace/answer.txt").content == "checkpoint content"
        client.download_file(restored.id, "/workspace/input.bin", str(destination))
        assert destination.read_bytes() == source.read_bytes()

        headers = {"Authorization": f"Bearer {live_controller.other_key}"}
        response = httpx.get(
            f"{live_controller.url}/api/v1/sandbox/{restored.id}", headers=headers
        )
        assert response.status_code == 404
        response = httpx.get(f"{live_controller.url}/api/v1/sandbox", headers=headers)
        response.raise_for_status()
        assert response.json()["sandboxes"] == []
    finally:
        for sandbox_id in active:
            client.delete(sandbox_id)
        api.client.close()


@pytest.mark.asyncio
async def test_async_sdk_commands_and_cleanup(live_controller, sdk_environment, tmp_path):
    async with AsyncSandboxClient() as client:
        client._auth_cache._cache_file = Path(tmp_path / "async-auth.json")
        sandbox = await client.create(request("async-integration-eval"))
        try:
            await client.wait_for_creation(sandbox.id, max_attempts=10)
            result = await client.execute_command(sandbox.id, "printf 'async complete'", timeout=10)
            assert result.exit_code == 0
            assert result.stdout == "async complete"
        finally:
            await client.delete(sandbox.id)


def test_command_deadline_removes_guest(live_controller, sdk_environment, tmp_path):
    api = APIClient()
    api.config.config_dir = tmp_path / "timeout-cache"
    client = SandboxClient(api)
    sandbox = client.create(request("deadline-eval"))
    try:
        response = httpx.post(
            f"{live_controller.url}/api/v1/sandbox/{sandbox.id}/command",
            headers={"Authorization": f"Bearer {live_controller.tenant_key}"},
            json={"command": "python -c 'import time; time.sleep(30)'", "timeout": 0.2},
            timeout=10,
        )
        response.raise_for_status()
        assert response.json()["timed_out"] is True
        assert client.get(sandbox.id).status == "TIMEOUT"
        assert not live_controller.daemon.containers.list(
            all=True, filters={"label": f"reliquary.sandbox.id={sandbox.id}"}
        )
    finally:
        client.delete(sandbox.id)
        api.client.close()
