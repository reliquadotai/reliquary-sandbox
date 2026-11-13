"""Gateway-wide limits: API rate per credential and address, the free-disk floor at open,
per-box I/O caps and the inode cap's commands (live checks: the VM runbook in docs/egress.md)."""

import asyncio
import os
from types import SimpleNamespace

import pytest
from episode_manager_fixtures import build
from fastapi import HTTPException

from reliquary_sandbox_service.rate_limit import RateLimit
from reliquary_sandbox_service.task_runtime import InfraFault, TaskBoxRuntime


async def _ok(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b""})


def _hit(app, client="1.2.3.4", auth=None, xff=None):
    headers = [] if auth is None else [(b"authorization", auth)]
    if xff is not None:
        headers.append((b"x-forwarded-for", xff))
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b""}
    asyncio.run(app({"type": "http", "client": (client, 1), "headers": headers}, receive, send))
    return sent[0]["status"], dict(sent[0]["headers"])


def test_rate_per_key_and_per_address():
    now = [0.0]
    app = RateLimit(_ok, rate_per_key=1, burst_per_key=2, rate_per_ip=100, burst_per_ip=3,
                    clock=lambda: now[0])
    assert [_hit(app, auth=b"Bearer a")[0] for _ in range(3)] == [200, 200, 429]
    assert _hit(app, auth=b"Bearer b", client="5.6.7.8")[0] == 200       # another key
    status, headers = _hit(app, client="9.9.9.9")                         # per-address burst 3
    assert status == 200
    assert [_hit(app, client="9.9.9.9")[0] for _ in range(3)] == [200, 200, 429]
    now[0] += 1.0
    assert _hit(app, auth=b"Bearer a", client="7.7.7.7")[0] == 200       # refilled


def test_address_behind_caddy_is_the_last_forwarded_entry():
    app = RateLimit(_ok, rate_per_ip=0.001, burst_per_ip=1, clock=lambda: 0.0)
    assert _hit(app, client="127.0.0.1", xff=b"6.6.6.6, 1.1.1.1")[0] == 200
    assert _hit(app, client="127.0.0.1", xff=b"spoofed, 1.1.1.1")[0] == 429   # same last hop
    assert _hit(app, client="127.0.0.1", xff=b"2.2.2.2")[0] == 200
    status, headers = _hit(app, client="127.0.0.1", xff=b"2.2.2.2")
    assert status == 429 and int(headers[b"retry-after"]) >= 1


def test_off_by_default():
    app = RateLimit(_ok)
    assert all(_hit(app, auth=b"x")[0] == 200 for _ in range(500))


def test_open_refused_below_the_free_disk_floor(tmp_path):
    h = build(tmp_path, episode_min_free_disk_bytes=10 * 1024**3)
    h.boxes.disk_free_bytes = lambda: 1024**3
    with pytest.raises(HTTPException) as error:
        h.open()
    assert error.value.status_code == 503 and "disk" in str(error.value.detail)
    h.boxes.disk_free_bytes = lambda: 20 * 1024**3
    assert h.open("s-2")


def _runtime(**settings):
    rt = TaskBoxRuntime.__new__(TaskBoxRuntime)
    rt.settings = SimpleNamespace(episode_allow_unquoted_disk=False, **settings)
    rt._docker_root_dir, rt._docker_block_device = "/var/lib/docker", "/dev/loop0"
    return rt


def test_io_max_line(tmp_path, monkeypatch):
    assert _runtime()._io_max() is None
    rt = _runtime(episode_disk_write_bps=100 << 20, episode_disk_read_iops=4000)
    monkeypatch.setattr("os.stat", lambda path: SimpleNamespace(st_rdev=os.makedev(7, 0)))
    assert rt._io_max() == f"7:0 wbps={100 << 20} riops=4000"
    scope = tmp_path / "scope"
    scope.mkdir()
    rt._cgroup_scope = lambda container_id: scope
    rt._limit_io(SimpleNamespace(id="c1"))
    assert (scope / "io.max").read_text() == f"7:0 wbps={100 << 20} riops=4000\n"
    rt._cgroup_scope = lambda container_id: tmp_path / "missing"
    with pytest.raises(InfraFault):
        rt._limit_io(SimpleNamespace(id="c1"))


def test_inode_cap_commands(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        out = "projid = 42\n" if argv[0] == "xfs_io" else ""
        return SimpleNamespace(returncode=0, stdout=out, stderr="")
    import reliquary_sandbox_service.task_runtime as mod
    monkeypatch.setattr(mod.subprocess, "run", run)
    container = SimpleNamespace(reload=lambda: None,
                                attrs={"GraphDriver": {"Data": {"UpperDir": "/var/lib/docker/x/diff"}}})
    _runtime(episode_box_inodes=1_000_000)._cap_inodes(container)
    assert calls == [["xfs_io", "-c", "lsproj", "/var/lib/docker/x/diff"],
                     ["xfs_quota", "-x", "-c", "limit -p ihard=1000000 42", "/var/lib/docker"]]
    calls.clear()
    _runtime()._cap_inodes(container)                       # off: no command
    assert calls == []
    monkeypatch.setattr(mod.subprocess, "run",
                        lambda argv, **kw: SimpleNamespace(returncode=0, stdout="projid = 0\n"))
    with pytest.raises(InfraFault):                          # no project quota: ours, never a box without the cap
        _runtime(episode_box_inodes=1000)._cap_inodes(container)


def test_capacity_report_lists_images_still_pulling(tmp_path):
    from episode_fixtures import IMAGE

    from reliquary_sandbox_service.episodes.capacity import capacity_report
    other = "other@sha256:" + "3" * 64
    h = build(tmp_path, episode_images=[IMAGE, other])
    document = capacity_report(h.manager)["document"]
    assert document["images"] == [IMAGE] and document["images_pending"] == [other]
    (tmp_path / "all").mkdir()
    h2 = build(tmp_path / "all")
    assert "images_pending" not in capacity_report(h2.manager)["document"]
