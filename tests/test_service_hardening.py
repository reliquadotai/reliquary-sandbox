"""Verified controller admission, cleanup, and retained-history behavior."""

import base64
import hashlib
import hmac
import json
import struct
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from test_service import FakeRuntime

from reliquary_sandbox.rpc_command_session import build_command_session_start_request
from reliquary_sandbox_service.app import create_app
from reliquary_sandbox_service.manager import CreateRequest, Manager
from reliquary_sandbox_service.settings import Settings
from reliquary_sandbox_service.store import Store

KEY = {"Authorization": "Bearer " + "e" * 32}


def settings(tmp_path, **kwargs):
    return Settings(tokens={"evals": "e" * 32}, state_dir=tmp_path,
                    reaper_interval_seconds=3600, **kwargs)


def request(**kwargs):
    return CreateRequest(name="evaluation", docker_image="reliquary-sandbox-guest:0.1.0", **kwargs)


def test_existing_state_directory_and_database_are_private(tmp_path):
    directory = tmp_path / "state"
    directory.mkdir(mode=0o755)
    store = Store(directory)
    try:
        assert directory.stat().st_mode & 0o777 == 0o700
        assert (directory / "state.sqlite3").stat().st_mode & 0o777 == 0o600
    finally:
        store.close()


@pytest.mark.parametrize("payload", [[], "invalid", {"exp": "invalid"}])
def test_gateway_requires_well_formed_signed_claims(tmp_path, payload):
    config = settings(tmp_path)
    with TestClient(create_app(config, FakeRuntime())) as client:
        sandbox_id = client.post("/api/v1/sandbox", json=request().model_dump(), headers=KEY).json()["id"]
        claims = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        signature = hmac.new(config.tokens["evals"].encode(), claims.encode(), hashlib.sha256).hexdigest()
        response = client.get(f"/gateway/evals/{sandbox_id}/download", params={"path": "/workspace/input"},
                              headers={"Authorization": f"Bearer {claims}.{signature}"})
        assert response.status_code == 401


def test_gateway_rejects_invalid_signature_before_parsing(tmp_path):
    config = settings(tmp_path)
    with TestClient(create_app(config, FakeRuntime())) as client:
        response = client.get("/gateway/evals/unknown/download", params={"path": "/workspace/input"},
                              headers={"Authorization": "Bearer invalid.signature"})
        assert response.status_code == 401
        assert client.get("/api/v1/sandbox", headers={"Authorization": "e" * 32}).status_code == 401


def test_expired_sandbox_rejects_file_checkpoint_and_new_auth(tmp_path):
    config = settings(tmp_path)
    runtime = FakeRuntime()
    with TestClient(create_app(config, runtime)) as client:
        sandbox_id = client.post("/api/v1/sandbox", json=request().model_dump(), headers=KEY).json()["id"]
        auth = client.post(f"/api/v1/sandbox/{sandbox_id}/auth", headers=KEY).json()
        store = Store(tmp_path)
        record = store.get(sandbox_id)
        record["expires_at"] = 0
        store.save(record)
        store.close()
        response = client.get(f"/gateway/evals/{sandbox_id}/download", params={"path": "/workspace/input"},
                              headers={"Authorization": "Bearer " + auth["token"]})
        assert response.status_code == 409
        assert client.post(f"/api/v1/sandbox/{sandbox_id}/checkpoints", headers=KEY).status_code == 409
        assert client.post(f"/api/v1/sandbox/{sandbox_id}/auth", headers=KEY).status_code == 409
        assert sandbox_id not in runtime.containers


def test_idle_deadline_is_enforced_without_waiting_for_reaper(tmp_path):
    backend = FakeRuntime()
    manager = Manager(settings(tmp_path), backend)
    try:
        sandbox_id = manager.create("evals", request(idle_timeout_minutes=1))["id"]
        record = manager.owned("evals", sandbox_id)
        record["last_activity"] = time.time() - 61
        manager.store.save(record)
        with pytest.raises(HTTPException) as error:
            manager.file("evals", sandbox_id, "/workspace/input", b"data")
        assert error.value.status_code == 409
        assert manager.owned("evals", sandbox_id)["status"] == "TIMEOUT"
    finally:
        manager.store.close()


def test_failed_provisioning_cleanup_is_retried_and_counted_in_quota(tmp_path):
    class RecoverableRuntime(FakeRuntime):
        fail_cleanup = True

        def create(self, sandbox_id, payload):
            super().create(sandbox_id, payload)
            raise RuntimeError("unavailable")

        def delete(self, sandbox_id):
            if self.fail_cleanup:
                raise RuntimeError("unavailable")
            super().delete(sandbox_id)

    runtime = RecoverableRuntime()
    manager = Manager(settings(tmp_path, max_sandboxes_per_tenant=1), runtime)
    try:
        with pytest.raises(HTTPException) as error:
            manager.create("evals", request())
        assert error.value.status_code == 503
        record = manager.store.all("evals")[0]
        assert record["cleanup_pending"] is True
        with pytest.raises(HTTPException) as error:
            manager.create("evals", request())
        assert error.value.status_code == 429
        runtime.fail_cleanup = False
        manager.reap()
        assert manager.owned("evals", record["id"])["cleanup_pending"] is False
        assert not runtime.containers
    finally:
        manager.store.close()


def test_unavailable_file_operation_updates_lifecycle_and_cleans_runtime(tmp_path):
    class UnavailableFiles(FakeRuntime):
        def download(self, sandbox_id, path):
            raise RuntimeError("unavailable")

    runtime = UnavailableFiles()
    manager = Manager(settings(tmp_path), runtime)
    try:
        sandbox_id = manager.create("evals", request())["id"]
        with pytest.raises(HTTPException) as error:
            manager.file("evals", sandbox_id, "/workspace/input")
        assert error.value.status_code == 503
        assert manager.owned("evals", sandbox_id)["status"] == "ERROR"
        assert sandbox_id not in runtime.containers
    finally:
        manager.store.close()


def test_reaper_continues_other_cleanup_after_one_runtime_failure(tmp_path):
    class PartlyAvailableRuntime(FakeRuntime):
        failed_id = None

        def delete(self, sandbox_id):
            if sandbox_id == self.failed_id:
                raise RuntimeError("unavailable")
            super().delete(sandbox_id)

    runtime = PartlyAvailableRuntime()
    manager = Manager(settings(tmp_path), runtime)
    try:
        first = manager.create("evals", request())["id"]
        second = manager.create("evals", request())["id"]
        runtime.failed_id = first
        for sandbox_id in (first, second):
            record = manager.owned("evals", sandbox_id)
            record["expires_at"] = 0
            manager.store.save(record)
        with pytest.raises(RuntimeError):
            manager.reap()
        assert manager.owned("evals", first)["cleanup_pending"] is True
        assert manager.owned("evals", second)["status"] == "TIMEOUT"
        assert second not in runtime.containers
    finally:
        manager.store.close()


def test_busy_sandbox_fails_admission_without_waiting(tmp_path):
    manager = Manager(settings(tmp_path), FakeRuntime())
    started, release = threading.Event(), threading.Event()
    sandbox_id = manager.create("evals", request())["id"]

    def held_operation():
        with manager.operation_lock(sandbox_id):
            started.set()
            release.wait(5)

    worker = threading.Thread(target=held_operation)
    worker.start()
    try:
        assert started.wait(1)
        with pytest.raises(HTTPException) as error:
            manager.execute("evals", sandbox_id, "true", 1)
        assert error.value.status_code == 409
    finally:
        release.set()
        worker.join(1)
        manager.store.close()


def test_gateway_output_limit_returns_protocol_error(tmp_path):
    class LimitedOutput(FakeRuntime):
        def execute(self, sandbox_id, command, timeout, cwd=None, env=None):
            return {"stdout": "partial", "stderr": "", "exit_code": 0, "truncated": True}

    with TestClient(create_app(settings(tmp_path), LimitedOutput())) as client:
        sandbox_id = client.post("/api/v1/sandbox", json=request().model_dump(), headers=KEY).json()["id"]
        auth = client.post(f"/api/v1/sandbox/{sandbox_id}/auth", headers=KEY).json()
        payload = build_command_session_start_request(command="echo hello", working_dir=None, env=None).SerializeToString()
        response = client.post(f"/gateway/evals/{sandbox_id}/command_session.CommandSession/Start",
                               content=b"\x00" + struct.pack(">I", len(payload)) + payload,
                               headers={"Authorization": "Bearer " + auth["token"],
                                        "Content-Type": "application/connect+proto"})
        assert response.status_code == 200
        position = 0
        end = None
        while position < len(response.content):
            flag = response.content[position]
            size = struct.unpack(">I", response.content[position + 1:position + 5])[0]
            value = response.content[position + 5:position + 5 + size]
            if flag == 2:
                end = json.loads(value)
            position += 5 + size
        assert end["error"]["code"] == "resource_exhausted"


@pytest.mark.parametrize("fail", [False, True])
def test_checkpoint_export_allows_lifetime_cleanup_and_preserves_terminal_state(tmp_path, fail):
    started, release = threading.Event(), threading.Event()

    class PausedExport(FakeRuntime):
        def export_workspace(self, sandbox_id):
            started.set()
            release.wait(5)
            if fail:
                raise RuntimeError("unavailable")
            return b"checkpoint"

    runtime = PausedExport()
    manager = Manager(settings(tmp_path), runtime)
    sandbox_id = manager.create("evals", request())["id"]
    errors = []

    def checkpoint():
        try:
            manager.checkpoint("evals", sandbox_id)
        except HTTPException as error:
            errors.append(error.status_code)

    worker = threading.Thread(target=checkpoint)
    worker.start()
    try:
        assert started.wait(1)
        record = manager.owned("evals", sandbox_id)
        record["expires_at"] = 0
        manager.store.save(record)
        manager.reap()
        assert sandbox_id not in runtime.containers
        release.set()
        worker.join(2)
        assert not worker.is_alive()
        assert errors == [503 if fail else 409]
        assert manager.owned("evals", sandbox_id)["status"] == "TIMEOUT"
        assert not manager.store.checkpoints("evals")
    finally:
        release.set()
        worker.join(2)
        manager.store.close()


def test_terminal_history_and_operation_locks_are_bounded(tmp_path):
    manager = Manager(settings(tmp_path, max_terminal_records_per_tenant=2), FakeRuntime())
    try:
        for _ in range(6):
            sandbox_id = manager.create("evals", request())["id"]
            manager.operation_lock(sandbox_id)
            manager.terminate("evals", sandbox_id)
        assert len(manager.store.all("evals")) == 2
        assert len(manager.operation_locks) == 2
        with pytest.raises(HTTPException):
            manager.operation_lock("unknown")
        assert len(manager.operation_locks) == 2
        record = manager.store.all("evals")[0]
        record["terminated_at"] = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
        manager.store.save(record)
        manager.prune_history()
        assert len(manager.store.all("evals")) == 1
    finally:
        manager.store.close()


@pytest.mark.parametrize("entry, allow", [
    ("verifiers:boom-pkg==unpackaged", False),
    ("verifiers:boom-pkg==1.0", True),
])
def test_startup_refuses_an_ungated_verifiers_entry_before_any_import(tmp_path, monkeypatch,
                                                                       entry, allow):
    import sys

    from episode_fixtures import episode_settings

    from reliquary_sandbox_service import scoring_server

    (tmp_path / "boom_pkg.py").write_text(
        "import os\nopen(os.environ['BOOM_LOG'], 'a').write('imported')\nraise RuntimeError\n")
    monkeypatch.setenv("BOOM_LOG", str(tmp_path / "boom.log"))
    monkeypatch.syspath_prepend(str(tmp_path))
    preloaded = []
    monkeypatch.setattr(scoring_server, "start", lambda preload=(): preloaded.append(preload))
    monkeypatch.setattr(scoring_server, "check_preload", lambda modules: preloaded.append(modules))
    config, _ = episode_settings(tmp_path, episode_envs={"boom": entry},
                                 episode_allow_unpackaged_verifiers=allow)
    with pytest.raises(ValueError), TestClient(create_app(config, FakeRuntime())):
        pass
    assert preloaded == [] and "boom_pkg" not in sys.modules
    assert not (tmp_path / "boom.log").exists()
