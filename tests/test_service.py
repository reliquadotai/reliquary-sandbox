"""Lifecycle, ownership, and wire-contract integration."""

import io
import socket
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pytest
import uvicorn
from fastapi.testclient import TestClient

from reliquary_sandbox import APIClient, CreateSandboxRequest, SandboxClient
from reliquary_sandbox_service.app import create_app
from reliquary_sandbox_service.manager import CreateRequest, Manager
from reliquary_sandbox_service.settings import Settings


class FakeRuntime:
    def __init__(self):
        self.containers = {}
        self.calls = 0
        self.available = True

    def readiness(self):
        if not self.available:
            raise RuntimeError
        return {"ready": True}

    def create(self, sandbox_id, request):
        self.calls += 1
        self.containers[sandbox_id] = {}
        return sandbox_id

    def delete(self, sandbox_id):
        self.containers.pop(sandbox_id, None)

    def status(self, sandbox_id):
        return "running" if sandbox_id in self.containers else "deleted"

    def managed_sandboxes(self):
        return {key: key for key in self.containers}

    def execute(self, sandbox_id, command, timeout, cwd=None, env=None):
        assert timeout > 0
        return {"stdout": "wire contract\n", "stderr": "", "exit_code": 0}

    def upload(self, sandbox_id, path, data):
        self.containers[sandbox_id][path] = data

    def download(self, sandbox_id, path):
        if path not in self.containers[sandbox_id]:
            raise FileNotFoundError
        return self.containers[sandbox_id][path]

    def export_workspace(self, sandbox_id):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            for path, content in self.containers[sandbox_id].items():
                archive.writestr(path.removeprefix("/workspace/"), content)
        return data.getvalue()

    def import_workspace(self, sandbox_id, snapshot):
        with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
            self.containers[sandbox_id] = {"/workspace/" + name: archive.read(name) for name in archive.namelist()}


@pytest.fixture
def setup(tmp_path):
    config = Settings(tokens={"evals": "e" * 32, "miners": "m" * 32}, state_dir=tmp_path,
                      max_sandboxes_per_tenant=2, reaper_interval_seconds=3600)
    runtime = FakeRuntime()
    with TestClient(create_app(config, runtime)) as client:
        yield client, runtime, config


KEY = {"Authorization": "Bearer " + "e" * 32}
OTHER = {"Authorization": "Bearer " + "m" * 32}


def request(**overrides):
    return {"name": "evaluation", "docker_image": "reliquary-sandbox-guest:0.1.0", **overrides}


def test_lifecycle_idempotency_and_tenant_scopes(setup):
    client, runtime, _ = setup
    payload = request(idempotency_key="same-request")
    first = client.post("/api/v1/sandbox", json=payload, headers=KEY)
    assert first.status_code == 200, first.text
    sandbox_id = first.json()["id"]
    assert client.post("/api/v1/sandbox", json=payload, headers=KEY).json()["id"] == sandbox_id
    assert runtime.calls == 1
    assert client.post("/api/v1/sandbox", json={**payload, "name": "different"}, headers=KEY).status_code == 409
    assert client.get(f"/api/v1/sandbox/{sandbox_id}", headers=OTHER).status_code == 404
    assert client.get("/api/v1/sandbox", headers=OTHER).json()["sandboxes"] == []
    assert client.get("/api/v1/sandbox", headers=KEY).json()["total"] == 1
    assert client.post("/api/v1/sandbox/status:batchGet", json={"sandbox_ids": [sandbox_id]}, headers=KEY).json()["statuses"][0]["status"] == "RUNNING"
    assert client.delete(f"/api/v1/sandbox/{sandbox_id}", headers=KEY).status_code == 200
    assert client.delete(f"/api/v1/sandbox/{sandbox_id}", headers=KEY).status_code == 200
    assert sandbox_id not in runtime.containers


def test_auth_capabilities_and_quotas(setup):
    client, _, _ = setup
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/sandbox").status_code == 401
    assert client.get("/api/v1/capabilities", headers=KEY).json()["vm"] is False
    assert client.post("/api/v1/sandbox", json=request(vm=True), headers=KEY).status_code == 501
    assert client.post("/api/v1/sandbox", json=request(timeout_minutes=-1), headers=KEY).status_code == 422
    assert client.post("/api/v1/sandbox", json=request(cpu_cores=-1), headers=KEY).status_code == 422
    for _ in range(2):
        assert client.post("/api/v1/sandbox", json=request(), headers=KEY).status_code == 200
    assert client.post("/api/v1/sandbox", json=request(), headers=KEY).status_code == 429


def test_durable_checkpoint_and_scoped_gateway(setup):
    client, _, _ = setup
    sandbox_id = client.post("/api/v1/sandbox", json=request(), headers=KEY).json()["id"]
    auth = client.post(f"/api/v1/sandbox/{sandbox_id}/auth", headers=KEY).json()
    gateway = f"/gateway/evals/{sandbox_id}"
    scoped = {"Authorization": "Bearer " + auth["token"]}
    assert client.post(gateway + "/upload", params={"path": "/workspace/input"}, files={"file": ("input", b"payload")}, headers=scoped).status_code == 200
    assert client.get(gateway + "/download", params={"path": "/workspace/input"}, headers=scoped).content == b"payload"
    assert client.get(gateway + "/download", params={"path": "/workspace/input"}, headers=KEY).status_code == 401
    checkpoint = client.post(f"/api/v1/sandbox/{sandbox_id}/checkpoints", headers=KEY).json()
    assert checkpoint["state"] == "DURABLE"
    client.delete(f"/api/v1/sandbox/{sandbox_id}", headers=KEY)
    restored = client.post("/api/v1/sandbox", json={"name": "restore", "checkpoint_id": checkpoint["id"]}, headers=KEY)
    assert restored.status_code == 200, restored.text
    assert client.get(f'/api/v1/sandbox/checkpoints/{checkpoint["id"]}', headers=OTHER).status_code == 404
    new_id = restored.json()["id"]
    assert client.get(f"/gateway/evals/{new_id}/download", params={"path": "/workspace/input"}, headers=scoped).status_code == 401


def test_concurrent_idempotency_and_recovery(tmp_path):
    config = Settings(tokens={"evals": "e" * 32}, state_dir=tmp_path)
    runtime = FakeRuntime()
    manager = Manager(config, runtime)
    payload = CreateRequest(**request(idempotency_key="parallel-create"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: manager.create("evals", payload), range(8)))
    assert len({r["id"] for r in results}) == 1
    assert runtime.calls == 1
    sandbox_id = results[0]["id"]
    manager.store.close()
    recovered = Manager(config, runtime)
    recovered.reconcile()
    assert recovered.owned("evals", sandbox_id)["status"] == "RUNNING"
    record = recovered.owned("evals", sandbox_id)
    record["expires_at"] = 0
    recovered.store.save(record)
    recovered.reap()
    assert recovered.owned("evals", sandbox_id)["status"] == "TIMEOUT"
    assert sandbox_id not in runtime.containers
    recovered.store.close()


def test_readiness_tracks_runtime(setup):
    client, runtime, _ = setup
    assert client.get("/readyz").status_code == 200
    runtime.available = False
    assert client.get("/readyz").status_code == 503


def test_original_command_wire_with_sync_sdk(tmp_path, monkeypatch):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    config = Settings(tokens={"evals": "e" * 32}, state_dir=tmp_path / "state", public_base_url=f"http://127.0.0.1:{port}")
    app = create_app(config, FakeRuntime())
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        monkeypatch.setenv("RELIQUARY_SANDBOX_API_BASE_URL", config.public_base_url)
        monkeypatch.setenv("RELIQUARY_SANDBOX_TEAM_ID", "")
        monkeypatch.delenv("RELIQUARY_SANDBOX_CONTEXT", raising=False)
        api = APIClient(api_key="e" * 32)
        api.config.config_dir = tmp_path / "sdk"
        sdk = SandboxClient(api)
        box = sdk.create(CreateSandboxRequest(**request()))
        result = sdk.execute_command(box.id, "echo hello", timeout=5)
        assert result.stdout == "wire contract\n"
        assert result.exit_code == 0
        sdk.upload_bytes(box.id, "/workspace/source.txt", b"source", "source.txt")
        assert sdk.read_file(box.id, "/workspace/source.txt").content == "source"
        snapshot = sdk.checkpoint(box.id)
        assert sdk.get_checkpoint(snapshot.id).state == "DURABLE"
        assert sdk.list_checkpoints(box.id)[0].id == snapshot.id
        sdk.delete(box.id)
        api.client.close()
    finally:
        server.should_exit = True
        thread.join(timeout=5)
