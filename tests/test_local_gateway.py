"""local_gateway: a complete episode gateway on 127.0.0.1."""

import hashlib
import threading
from contextlib import contextmanager

import httpx
import pytest
from episode_fixtures import IMAGE, budgets
from fake_boxes import FakeBoxes
from gateway_fixtures import fake_gateway

from reliquary_sandbox.episode_task import TaskLimits
from reliquary_sandbox_service.episodes import local_gateway as local_gateway_module
from reliquary_sandbox_service.episodes.local_gateway import default_budgets, local_gateway

GIB = 1024**3
WRITE = {"turn": 0, "k": 0, "tool": "bash", "arguments": {"command": "write /work/answer.txt 42"}}


def test_a_local_gateway_serves_a_verifiable_episode_over_http(tmp_path):
    with fake_gateway(tmp_path) as (gateway, _):
        token = gateway.issue(env="fake-env", split="train", index=7, image=IMAGE,
                              budgets=budgets())
        with httpx.Client(base_url=gateway.url, timeout=30) as http:
            opened = http.post("/episodes", json={"token": token}).json()
            auth = {"Authorization": f"Bearer {opened['episode_key']}"}
            episode = opened["episode_id"]
            assert http.post(f"/episodes/{episode}/calls", headers=auth, json=WRITE).status_code == 200
            final = http.post(f"/episodes/{episode}/finish", headers=auth).json()["record"]["body"]
            transcript = http.get(f"/episodes/{episode}", headers=auth).json()["transcript"]
    assert final["status"] == "graded" and final["reward"] == 1.0
    assert gateway.verify(transcript).ok


def test_the_gateway_stops_when_the_block_exits(tmp_path):
    with fake_gateway(tmp_path) as (gateway, _):
        assert httpx.get(f"{gateway.url}/capacity", timeout=10).status_code == 200
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"{gateway.url}/capacity", timeout=5)


def test_default_budgets_meet_the_task_limits():
    limits = TaskLimits(memory_bytes=6 * GIB, pids=512, wall_s=900, per_call_timeout_s=180)
    chosen = default_budgets(limits)
    assert (chosen.memory_bytes, chosen.pids, chosen.wall_s, chosen.per_call_timeout_s) == (
        6 * GIB, 512, 900, 180)
    assert default_budgets().memory_bytes == 2 * GIB
    assert default_budgets(limits, wall_s=60).wall_s == 60
    assert limits.shortfall(default_budgets(limits)) == []


# -- fix round 1: key files, cleanup of real boxes --------------------------------


def test_key_files_are_created_private_and_never_overwritten(tmp_path):
    with fake_gateway(tmp_path) as (gateway, _):
        key = tmp_path / "gateway" / "machine.pem"
        assert key.stat().st_mode & 0o777 == 0o600
        assert gateway.settings.machine_key_file == key
    with pytest.raises(FileExistsError):
        with fake_gateway(tmp_path):
            pass


class FakeDockerClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def owned_runtime(monkeypatch, tmp_path, *, fail_on=()):
    """Stand in for the Docker-backed runtime local_gateway builds itself."""
    boxes = FakeBoxes([IMAGE])
    client = FakeDockerClient()
    original = boxes.remove

    def remove(box_id):
        if box_id in fail_on:
            raise RuntimeError("docker down")
        original(box_id)

    boxes.remove = remove
    monkeypatch.setattr(local_gateway_module, "_docker_task_runtime",
                        lambda settings: (boxes, client))
    return boxes, client


def gateway_with_own_runtime(tmp_path):
    helper = tmp_path / "busybox"
    helper.write_bytes(b"not really busybox")
    return local_gateway(tmp_path / "gateway", envs={"fake-env": "fake_episode_env:sandbox_task"},
                         images=[IMAGE], episode_allow_unquoted_disk=True,
                         episode_helper_busybox=helper,
                         episode_helper_busybox_sha256=hashlib.sha256(b"not really busybox")
                         .hexdigest())


def test_cleanup_removes_every_box_despite_a_failure_and_closes_docker(tmp_path, monkeypatch,
                                                                         caplog):
    boxes, client = owned_runtime(monkeypatch, tmp_path, fail_on={"b-1"})
    with gateway_with_own_runtime(tmp_path):
        for box_id in ("b-1", "b-2", "b-3"):
            boxes.boxes[box_id] = {"episode": box_id, "files": {}, "cpu": 0, "dirs": set()}
    assert set(boxes.removed) == {"b-2", "b-3"} and "b-1" in boxes.boxes
    assert client.closed
    assert any("b-1" in r.getMessage() for r in caplog.records)


def test_no_cleanup_while_the_server_is_still_alive(tmp_path, monkeypatch, caplog):
    boxes, client = owned_runtime(monkeypatch, tmp_path)
    alive = threading.Event()
    stuck = threading.Thread(target=alive.wait, daemon=True)
    stuck.start()

    @contextmanager
    def never_stops(app, port):
        yield local_gateway_module._Served(f"http://127.0.0.1:{port}", stuck)

    monkeypatch.setattr(local_gateway_module, "_serve", never_stops)
    try:
        with gateway_with_own_runtime(tmp_path):
            boxes.boxes["b-1"] = {"episode": "b-1", "files": {}, "cpu": 0, "dirs": set()}
        assert boxes.removed == [] and not client.closed
        assert any("still running" in r.getMessage() for r in caplog.records)
    finally:
        alive.set()


def test_docker_host_is_honoured(tmp_path, monkeypatch):
    import docker

    made = {}

    class Recorder:
        def __init__(self, **kwargs):
            made.update(kwargs)

    monkeypatch.setattr(docker, "DockerClient", Recorder)
    monkeypatch.setattr(docker, "from_env", lambda **kwargs: pytest.fail("used DOCKER_HOST"))
    monkeypatch.setattr(local_gateway_module, "TaskBoxRuntime",
                        lambda settings, *, client, namespace: ("runtime", client))
    settings = type("S", (), {"docker_host": "unix:///run/test-docker.sock"})()
    runtime, client = local_gateway_module._docker_task_runtime(settings)
    assert made["base_url"] == "unix:///run/test-docker.sock" and isinstance(client, Recorder)
