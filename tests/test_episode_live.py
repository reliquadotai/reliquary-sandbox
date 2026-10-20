"""Episode boxes on a real Docker host with gVisor. Opt-in: never run on a
production machine. See docs/deployment.md, "Episode hosts: testing".

    RUN_EPISODE_LIVE=1                 enable
    DOCKER_HOST=unix:///run/reliquary-xfs-docker.sock   (scripts/test-xfs-docker.sh env)
    EPISODE_LIVE_XFS=1                 the daemon's data-root is overlay2 on xfs with pquota
    EPISODE_LIVE_DOCKER_PIDFILE=...    that daemon's pidfile (default /var/run/docker.pid)
    EPISODE_LIVE_CPU_SOURCE=cgroupfs   read box CPU from cgroupfs instead of Docker stats
    EPISODE_LIVE_BUSYBOX=/usr/bin/busybox   the static helper busybox (sha256 computed here)
    EPISODE_LIVE_IMAGE=repo@sha256:... default: the local ubuntu:22.04, by its RepoDigest
    EPISODE_LIVE_METRICS=path          append the measurements as JSON lines

Run sequentially (`-p no:randomly`, never `-n`): the host's CPU is shared.
"""

import errno
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import docker
import pytest
from docker.errors import ImageNotFound
from episode_fixtures import budgets, claims, directory_for, episode_settings
from fastapi import HTTPException

from reliquary_sandbox.attest import (
    Ed25519TokenVerifier,
    Expected,
    Signer,
    issue_session_token,
    load_private_key,
    verify_transcript,
)
from reliquary_sandbox_service.episodes.manager import EpisodeManager
from reliquary_sandbox_service.episodes.registry import TaskRegistry
from reliquary_sandbox_service.runtime import RuntimeUnavailable
from reliquary_sandbox_service.store import Store
from reliquary_sandbox_service.task_runtime import HELPER, BoxFault, BoxSpec, TaskBoxRuntime

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("RUN_EPISODE_LIVE") != "1",
                       reason="set RUN_EPISODE_LIVE=1 on the episode test host"),
]
XFS = os.getenv("EPISODE_LIVE_XFS") == "1"
CPU_SOURCE = os.getenv("EPISODE_LIVE_CPU_SOURCE", "docker")
BUSYBOX = Path(os.getenv("EPISODE_LIVE_BUSYBOX", "/usr/bin/busybox"))
PIDFILE = Path(os.getenv("EPISODE_LIVE_DOCKER_PIDFILE", "/var/run/docker.pid"))
METRICS = os.getenv("EPISODE_LIVE_METRICS")
TESTS = Path(__file__).resolve().parent
EXPECTED = dict(hotkey="5Hotkey", engagement="corpus:job-1:7", env="live-env", split="train",
                index=7, checkpoint="ck-1")
LISTING = "for p in /proc/[0-9]*; do tr '\\0' ' ' < $p/cmdline 2>/dev/null; echo; done"


def report(name, **values):
    """Observed values for the task report: printed (`-s`) and appended to METRICS."""
    line = json.dumps({"check": name, **values}, sort_keys=True)
    print("LIVE", line)
    if METRICS:
        with open(METRICS, "a") as out:
            out.write(line + "\n")


@pytest.fixture(scope="module")
def client():
    daemon = docker.from_env(timeout=120)
    assert "runsc" in daemon.info().get("Runtimes", {}), "register runsc first"
    return daemon


@pytest.fixture(scope="module")
def image(client):
    if os.getenv("EPISODE_LIVE_IMAGE"):
        return os.environ["EPISODE_LIVE_IMAGE"]
    try:
        found = client.images.get("ubuntu:22.04")
    except ImageNotFound:
        found = client.images.pull("ubuntu", tag="22.04")
    digest = next(d for d in found.attrs["RepoDigests"] if d.startswith("ubuntu@"))
    os.environ["EPISODE_LIVE_IMAGE"] = digest  # read by live_episode_env
    return digest


def live_settings(tmp_path, image, **overrides):
    values = dict(
        episode_images=[image], episode_envs={"live-env": "live_episode_env:sandbox_task"},
        episode_allow_unquoted_disk=not XFS, episode_cpu_source=CPU_SOURCE,
        episode_helper_busybox=BUSYBOX,
        episode_helper_busybox_sha256=hashlib.sha256(BUSYBOX.read_bytes()).hexdigest(),
        episode_docker_pidfile=PIDFILE, episode_nano_cpus=1_000_000_000,
    )
    values.update(overrides)
    return episode_settings(tmp_path, **values)


@pytest.fixture
def boxes(tmp_path, client, image):
    settings, _ = live_settings(tmp_path, image)
    runtime = TaskBoxRuntime(settings, client=client, namespace=uuid.uuid4().hex)
    yield runtime
    for box_id in runtime.managed():
        runtime.remove(box_id)


def start(boxes, image, disk=1024**3, memory=1024**3, pids=256):
    box_id = uuid.uuid4().hex[:12]
    boxes.create(box_id, box_id, BoxSpec(image=image, workdir="/root", env={"LANG": "C.UTF-8"},
                                         user=None, memory_bytes=memory,
                                         nano_cpus=1_000_000_000, pids=pids, disk_bytes=disk,
                                         lifetime_s=600))
    return box_id


def bash(boxes, box_id, command, timeout_s=30, limit=65536):
    return boxes.bash(box_id, uuid.uuid4().hex, command, timeout_s=timeout_s, output_limit=limit,
                      env={}, workdir="/root", user=None)


def processes(boxes, box_id):
    return bash(boxes, box_id, LISTING).stdout.decode()


# -- the runtime on its own --------------------------------------------------


def test_preflight_matches_this_daemon(boxes, client):
    info = client.info()
    report("daemon", driver=info.get("Driver"), driver_status=info.get("DriverStatus"), xfs=XFS)
    if XFS:
        facts = boxes.preflight()
        report("preflight", **facts)
        assert facts["disk_quota"] == "xfs-pquota"
        assert facts["runsc_version"] == "release-20260928.0"
        return
    assert boxes.preflight()["disk_quota"] == "disabled"
    boxes.settings.episode_allow_unquoted_disk = False
    with pytest.raises(RuntimeUnavailable, match="overlay2"):
        boxes.preflight()


def test_task_image_runs_as_is_under_gvisor_without_network(boxes, client, image):
    box_id = start(boxes, image)
    attrs = client.containers.get(boxes._name(box_id)).attrs
    assert attrs["HostConfig"]["Runtime"] == "runsc"
    assert attrs["HostConfig"]["NetworkMode"] == "none"
    assert attrs["HostConfig"]["ReadonlyRootfs"] is False
    result = bash(boxes, box_id, "id -u; test -e /usr/local/bin/python3 || echo no-python; "
                                 "(exec 3<>/dev/tcp/1.1.1.1/53) 2>/dev/null && echo net-open || echo net-closed")
    assert result.stdout.decode().split() == ["0", "no-python", "net-closed"]
    assert boxes.box_user(box_id) == "0:0"


def test_timeout_kills_the_call_group_and_keeps_the_box(boxes, image):
    box_id = start(boxes, image)
    assert bash(boxes, box_id, "nohup sleep 1001 >/dev/null 2>&1 & echo started").stdout == b"started\n"
    timed = bash(boxes, box_id, "echo before; sleep 1002; echo never", timeout_s=2)
    assert timed.timed_out and timed.stdout == b"before\n"
    time.sleep(0.5)
    lines = processes(boxes, box_id)
    assert "sleep 1001" in lines and "sleep 1002" not in lines


def test_long_calls_background_jobs_and_escapees(boxes, image):
    box_id = start(boxes, image)
    began = time.monotonic()
    timed = bash(boxes, box_id, "sleep 611", timeout_s=3)  # PID 1 is `sleep 600`
    sleep_611_s = time.monotonic() - began
    assert timed.timed_out and sleep_611_s < 20
    assert bash(boxes, box_id, "echo alive").stdout == b"alive\n"
    began = time.monotonic()
    background = bash(boxes, box_id, "sleep 30 & echo started")
    background_s = time.monotonic() - began
    assert background.stdout == b"started\n" and not background.timed_out
    began = time.monotonic()
    escaped = bash(boxes, box_id, "setsid sleep 1004 & echo escaped")
    escapee_s = time.monotonic() - began
    assert escaped.stdout == b"escaped\n" and not escaped.timed_out
    lines = processes(boxes, box_id)
    assert "sleep 30" in lines and "sleep 1004" in lines and "sleep 611" not in lines
    flood = bash(boxes, box_id, "yes", timeout_s=3, limit=65536)
    assert flood.timed_out and flood.truncated and len(flood.stdout) == 65536
    assert "yes" not in processes(boxes, box_id).split()
    report("timeouts", sleep_611_call_s=round(sleep_611_s, 2),
           background_call_s=round(background_s, 2), escapee_call_s=round(escapee_s, 2))
    assert background_s < 5 and escapee_s < 5


def test_stopping_the_agents_processes_keeps_the_box(boxes, image):
    """Before extract: everything the agent left (background jobs, escapees,
    a fork that respawns) is killed; PID 1 and the entrypoint `sleep` stay."""
    box_id = start(boxes, image)
    bash(boxes, box_id, "nohup sleep 1011 >/dev/null 2>&1 & setsid sleep 1012 & "
                        "(while :; do sleep 1013; done) & echo started")
    boxes.stop_processes(box_id, timeout_s=30)
    assert boxes.status(box_id) == "running"
    lines = processes(boxes, box_id)
    assert not any(f"sleep {n}" in lines for n in (1011, 1012, 1013)), lines
    assert bash(boxes, box_id, "echo alive").stdout == b"alive\n"


def test_output_is_truncated_and_flagged(boxes, image):
    box_id = start(boxes, image)
    result = bash(boxes, box_id, "head -c 200000 /dev/zero | tr '\\0' a", limit=65536)
    assert result.truncated and len(result.stdout) == 65536 and result.exit_code == 0


def test_cpu_is_measured_per_call(boxes, image):
    box_id = start(boxes, image)
    readings = [boxes.cpu_ns(box_id)]
    began = time.monotonic()
    bash(boxes, box_id, "i=0; while [ $i -lt 400000 ]; do i=$((i+1)); done", timeout_s=60)
    wall = time.monotonic() - began
    readings.append(boxes.cpu_ns(box_id))
    busy = readings[-1] - readings[-2]
    bash(boxes, box_id, "sleep 1")
    readings.append(boxes.cpu_ns(box_id))
    idle = readings[-1] - readings[-2]
    for _ in range(3):
        readings.append(boxes.cpu_ns(box_id))
    report("cpu", source=CPU_SOURCE, busy_ms=busy // 10**6, idle_ms=idle // 10**6,
           busy_wall_ms=int(wall * 1000), busy_per_wall=round(busy / 1e9 / wall, 2))
    assert readings == sorted(readings), readings
    assert busy > 200_000_000, busy
    assert idle < busy / 4, (idle, busy)
    assert busy <= wall * 1e9 * 1.25 + 200_000_000, (busy, wall)  # one CPU


def test_binary_files_round_trip_through_exec(boxes, image):
    box_id = start(boxes, image)
    data = os.urandom(1024**2)
    boxes.write_file(box_id, "/root/blob.bin", data)
    assert boxes.read_file(box_id, "/root/blob.bin", 2 * 1024**2) == data
    with pytest.raises(FileNotFoundError):
        boxes.read_file(box_id, "/root/missing", 10)


def test_fast_exec_output_is_never_lost(boxes, image):
    """docker-py's exec stream lost a first frame that arrived with the upgrade
    headers (~2 % of fast execs). Our own transport must lose nothing."""
    box_id = start(boxes, image)
    outputs = [boxes.exec(box_id, [HELPER, "head", "-c", "100", "/etc/hostname"], timeout_s=30,
                          output_limit=1024, user="0", workdir="/").stdout for _ in range(300)]
    report("fast_exec_probe", execs=300, lost=300 - outputs.count(b"sandbox\n"))
    assert outputs == [b"sandbox\n"] * 300


def test_helper_reads_and_programs_lose_nothing(boxes, image):
    """Framed helpers and env programs over many fast runs: nothing lost."""
    box_id = start(boxes, image)
    user = boxes.box_user(box_id)
    wrong = 0
    for n in range(200):
        data = f"value {n}\n".encode()
        boxes.write_file(box_id, "/root/a.txt", data)
        wrong += boxes.read_file(box_id, "/root/a.txt", 100) != data
    outputs = [boxes.run_program(box_id, ["cat", "/root/a.txt"], timeout_s=30, output_limit=100,
                                 env={}, workdir="/root", user=user).stdout for _ in range(50)]
    empty = boxes.run_program(box_id, ["true"], timeout_s=30, output_limit=100, env={},
                              workdir="/root", user=user)
    report("output_loss", wrong_reads=wrong, program_outputs_ok=outputs.count(b"value 199\n"))
    assert wrong == 0
    assert outputs == [b"value 199\n"] * 50 and empty.stdout == b"" and empty.exit_code == 0


def test_etc_files_are_read_only_binds(boxes, image):
    box_id = start(boxes, image)
    for path in ("/etc/hosts", "/etc/hostname", "/etc/resolv.conf"):
        result = bash(boxes, box_id, f"dd if=/dev/zero of={path} bs=1k count=1 2>&1; echo rc=$?; "
                                     f"echo x >> {path}; echo rc=$?")
        assert b"rc=0" not in result.stdout, (path, result.stdout)
    assert bash(boxes, box_id, "cat /etc/hostname").stdout == b"sandbox\n"


def test_the_helper_directory_is_read_only(boxes, image):
    box_id = start(boxes, image)
    pinned = hashlib.sha256(BUSYBOX.read_bytes()).hexdigest()
    attempts = ["mv /.reliquary /x", "mv /.reliquary/bin /.reliquary/b2",
                "rm -f /.reliquary/bin/busybox", "rm -rf /.reliquary",
                "cp /bin/true /.reliquary/bin/busybox", "echo x > /.reliquary/bin/busybox",
                "chmod 000 /.reliquary/bin/busybox", "ln -sf /bin/true /.reliquary/bin/busybox",
                "touch /.reliquary/new", "mount -t tmpfs none /.reliquary", "umount /.reliquary"]
    outcomes = {}
    for attempt in attempts:
        result = bash(boxes, box_id, f"{attempt} 2>&1; echo rc=$?")
        outcomes[attempt] = result.stdout.decode().strip().splitlines()[-2:]
        assert b"rc=0" not in result.stdout, (attempt, result.stdout)
    report("helper_dir", **{a: " | ".join(o) for a, o in outcomes.items()})
    found = bash(boxes, box_id, f"sha256sum {HELPER}").stdout.split()[0].decode()
    assert found == pinned
    boxes.write_file(box_id, "/root/after.txt", b"still works\n")
    assert boxes.read_file(box_id, "/root/after.txt", 100) == b"still works\n"
    assert boxes.file_kind(box_id, "/root/after.txt") == "file"
    assert bash(boxes, box_id, "echo ok").stdout == b"ok\n"


def _report_quota_failure(boxes, box_id, began):
    """Capture this test box's exit evidence before fixture cleanup, without identifiers."""
    try:
        container = boxes._container(box_id)
        container.reload()
        raw = container.attrs["State"]
        state = {key: raw.get(key) for key in (
            "Status", "ExitCode", "OOMKilled", "StartedAt", "FinishedAt")}
        state["Error"] = bool(raw.get("Error"))
        report("quota_failure", elapsed_s=time.monotonic() - began, state=state,
               cgroup={name: _cgroup_file(boxes, box_id, name) for name in (
                   "memory.events", "memory.current", "pids.current", "pids.events")})
    except Exception as exc:
        report("quota_failure_inspection", error=type(exc).__name__)


@pytest.mark.skipif(not XFS, reason="needs a Docker data-root on xfs with pquota")
def test_writable_root_is_capped_by_the_project_quota(boxes, image):
    assert boxes.preflight()["disk_quota"] == "xfs-pquota"
    began = time.monotonic()
    box_id = start(boxes, image, disk=256 * 1024**2)
    try:
        result = bash(boxes, box_id, "dd if=/dev/zero of=/big bs=1M count=400 2>&1; echo rc=$?",
                      timeout_s=120)
        assert b"rc=0" not in result.stdout, result.stdout
        assert b"No space left on device" in result.stdout, result.stdout
        assert boxes.status(box_id) == "running"
        stuck = bash(boxes, box_id, "sleep 1003", timeout_s=2)
        assert stuck.timed_out
        assert "sleep 1003" not in processes(boxes, box_id)
        with pytest.raises(OSError) as full:
            boxes.write_file(box_id, "/root/more.bin", os.urandom(64 * 1024**2))
        assert full.value.errno == errno.ENOSPC, full.value
        full_write = f"OSError errno={full.value.errno}"
        assert bash(boxes, box_id, "rm -f /big /root/more.bin && echo ok").stdout == b"ok\n"
        again = bash(boxes, box_id, "dd if=/dev/zero of=/again bs=1M count=100 2>&1; echo rc=$?",
                     timeout_s=120)
        report("quota", dd_400m=result.stdout.decode()[-200:], write_file_when_full=full_write,
               dd_100m_after_rm=again.stdout.decode()[-120:])
        assert b"rc=0" in again.stdout, again.stdout
    except Exception:
        _report_quota_failure(boxes, box_id, began)
        raise


def _cgroup_file(boxes, box_id, name):
    try:
        return (boxes._cgroup_scope(boxes._container(box_id).id) / name).read_text()
    except Exception as exc:
        return f"unreadable: {type(exc).__name__}"


def test_memory_hog_is_the_agents_fault(boxes, client, image):
    box_id = start(boxes, image, memory=256 * 1024**2)
    container_id = boxes._container(box_id).id
    with pytest.raises(BoxFault) as hog:
        bash(boxes, box_id, "head -c 2000000000 /dev/zero | tail > /dev/null; echo rc=$?",
             timeout_s=60)
    try:
        events = (boxes._cgroup_scope(container_id) / "memory.events").read_text()
    except OSError as exc:
        events = f"unreadable: {type(exc).__name__}"
    state = client.containers.get(boxes._name(box_id)).attrs["State"]
    report("memory_hog", fault=str(hog.value), memory_events=events, status=state.get("Status"),
           oom_killed=state.get("OOMKilled"), error=state.get("Error"))
    # Under runsc the cgroup OOM kill takes the whole sandbox: the box died of
    # its own limit during the call, which is the agent's fault, never ours.
    assert state.get("OOMKilled") is True


# -- the gateway over real boxes ---------------------------------------------


@dataclass
class Gateway:
    manager: EpisodeManager
    boxes: TaskBoxRuntime
    validator: Signer
    machine: Signer
    image: str

    def open(self, session_id, **budget_overrides):
        values = {"per_call_timeout_s": 10, **budget_overrides}
        token = issue_session_token(claims(session_id, env="live-env", image=self.image,
                                           budgets=budgets(**values)), self.validator)
        return self.manager.open(token)["episode_key"]

    def call(self, session_id, key, turn, tool, arguments):
        """The call's record body, or the final body when the call ended the episode."""
        try:
            return self.manager.call(session_id, key, turn, 0, tool, arguments)["record"]["body"]
        except HTTPException as exc:
            assert isinstance(exc.detail, dict) and exc.detail.get("final"), exc.detail
            return exc.detail["final"]["body"]

    def verify(self, session_id, key, require_graded=True):
        transcript = self.manager.transcript(session_id, key)["transcript"]
        return verify_transcript(transcript, directory_for(self.machine),
                                 Ed25519TokenVerifier({"v-1": self.validator.public_key_b64}),
                                 Expected(**EXPECTED, require_graded=require_graded))


def manager_for(settings, client, namespace):
    machine = Signer(settings.machine_key_id, load_private_key(settings.machine_key_file))
    boxes = TaskBoxRuntime(settings, client=client, namespace=namespace)
    return EpisodeManager(settings, boxes, Store(settings.state_dir), machine,
                          TaskRegistry(settings.episode_envs)), boxes, machine


@pytest.fixture
def gateway(tmp_path, client, image):
    settings, validator = live_settings(tmp_path, image)
    manager, boxes, machine = manager_for(settings, client, uuid.uuid4().hex)
    yield Gateway(manager, boxes, validator, machine, image)
    for box_id in boxes.managed():
        boxes.remove(box_id)


def test_full_signed_episode_on_a_real_box(gateway):
    g = gateway
    began = time.monotonic()
    key = g.open("s-1")
    open_s = time.monotonic() - began
    assert g.call("s-1", key, 0, "bash", {"command": "echo 41 > answer.txt"})["exit_code"] == 0
    assert g.call("s-1", key, 1, "edit", {"path": "answer.txt", "old_str": "41", "new_str": "42"}
                  )["output"] == "Edited answer.txt"
    assert g.call("s-1", key, 2, "bash", {"command": "sleep 60"})["timed_out"]
    began = time.monotonic()
    final = g.manager.finish("s-1", key)["record"]["body"]
    finish_s = time.monotonic() - began
    assert final["status"] == "graded" and final["reward"] == 1.0, json.dumps(final)
    result = g.verify("s-1", key)
    assert result.ok, result.reasons
    wrong = g.open("s-2")
    g.call("s-2", wrong, 0, "bash", {"command": "echo 41 > answer.txt"})
    final = g.manager.finish("s-2", wrong)["record"]["body"]
    assert final["status"] == "graded" and final["reward"] == 0.0, final
    assert g.verify("s-2", wrong).ok
    assert g.boxes.managed() == {}
    report("episode", open_s=round(open_s, 2), finish_s=round(finish_s, 2))


RUNTIME_TEXTS = ("OCI runtime", "urpc", "containerManager")


def assert_no_runtime_text(g, session_id, key):
    records = g.manager.transcript(session_id, key)["transcript"]["records"]
    for record in records:
        text = json.dumps(record["body"])
        assert not any(t in text for t in RUNTIME_TEXTS), text


def test_kill_all_mid_call_ends_box_failed_at_that_call(gateway):
    g = gateway
    key = g.open("s-kill")
    g.call("s-kill", key, 0, "bash", {"command": "echo before"})
    first = g.call("s-kill", key, 1, "bash", {"command": "sleep 0.2; kill -9 -1"})
    report("kill_all", at_call=first.get("kind"), status=first.get("status"),
           reason=first.get("reason"), i=first.get("i"))
    assert first["kind"] == "final" and first["status"] == "box_failed", first
    assert first["i"] == 2  # record 0, the first call, then the final: the kill is not recorded
    assert_no_runtime_text(g, "s-kill", key)
    result = g.verify("s-kill", key, require_graded=False)
    assert result.ok, result.reasons


def test_busy_calls_record_their_cpu(gateway):
    g = gateway
    key = g.open("s-cpu")
    record = g.call("s-cpu", key, 0, "bash",
                    {"command": "i=0; while [ $i -lt 600000 ]; do i=$((i+1)); done"})
    report("call_cpu", cpu_ms=record["cpu_ms"], wall_ms=record["wall_ms"])
    assert record["cpu_ms"] >= 0.5 * record["wall_ms"] * 1, record  # min(cpus, 1) = 1


def test_fork_bomb_at_the_pids_limit(gateway):
    g = gateway
    key = g.open("s-fork", pids=64, per_call_timeout_s=5)
    bomb = g.call("s-fork", key, 0, "bash", {"command": "b(){ b|b& }; b"})
    after = bomb if bomb["kind"] == "final" else g.call("s-fork", key, 1, "bash",
                                                         {"command": "echo hi"})
    report("fork_bomb", bomb=bomb.get("status") or {"exit": bomb.get("exit_code"),
                                                    "timed_out": bomb.get("timed_out")},
           after=after.get("status") or after.get("output"), reason=after.get("reason"))
    assert after.get("status") == "box_failed", after
    assert_no_runtime_text(g, "s-fork", key)


def test_a_missing_workdir_is_a_tool_error(gateway):
    g = gateway
    key = g.open("s-rmwd")
    assert g.call("s-rmwd", key, 0, "bash", {"command": "cd / && rm -rf /root"})["exit_code"] == 0
    edit = g.call("s-rmwd", key, 1, "edit", {"path": "answer.txt", "old_str": "4", "new_str": "2"})
    assert edit["kind"] == "call" and edit["exit_code"] == 1, edit
    after = g.call("s-rmwd", key, 2, "bash", {"command": "echo hi"})
    assert after["kind"] == "call" and after["exit_code"] == 1, after
    final = g.manager.finish("s-rmwd", key)["record"]["body"]
    report("missing_workdir", edit=edit["output"], bash=after["output"], final=final["status"],
           reward=final["reward"])
    assert final["status"] == "graded" and final["reward"] == 0.0


def test_a_replaced_shell_breaks_only_the_agents_commands(gateway):
    g = gateway
    key = g.open("s-sh")
    g.call("s-sh", key, 0, "bash", {"command": "echo 41 > answer.txt"})
    sh = g.call("s-sh", key, 1, "bash", {"command": "rm -f /bin/sh && cp /bin/false /bin/sh && "
                                                    "sh -c 'echo still'; echo rc=$?"})
    assert "still" not in sh["output"] and "rc=1" in sh["output"], sh
    g.call("s-sh", key, 2, "bash", {"command": "rm /usr/bin/bash && cp /usr/bin/false /usr/bin/bash"})
    broken = g.call("s-sh", key, 3, "bash", {"command": "echo hi"})
    assert broken["kind"] == "call" and broken["exit_code"] != 0 and "hi" not in broken["output"]
    edit = g.call("s-sh", key, 4, "edit", {"path": "answer.txt", "old_str": "41", "new_str": "42"})
    assert edit["output"] == "Edited answer.txt", edit
    final = g.manager.finish("s-sh", key)["record"]["body"]
    report("replaced_shell", sh=sh["output"], bash_exit=broken["exit_code"], final=final["status"],
           reward=final["reward"])
    assert final["status"] == "graded" and final["reward"] == 1.0, json.dumps(final)


def test_a_memory_hog_ends_the_episode_box_failed(gateway):
    g = gateway
    key = g.open("s-oom", memory_bytes=256 * 1024**2)
    hog = g.call("s-oom", key, 0, "bash",
                 {"command": "head -c 2000000000 /dev/zero | tail > /dev/null; echo rc=$?"})
    report("memory_hog_episode", kind=hog["kind"], status=hog.get("status"),
           reason=hog.get("reason"))
    assert hog["kind"] == "final" and hog["status"] == "box_failed", hog
    assert_no_runtime_text(g, "s-oom", key)


CHILD = r"""
import json, sys
import docker
from reliquary_sandbox.attest import Signer, load_private_key
from reliquary_sandbox_service.episodes.manager import EpisodeManager
from reliquary_sandbox_service.episodes.registry import TaskRegistry
from reliquary_sandbox_service.settings import Settings
from reliquary_sandbox_service.store import Store
from reliquary_sandbox_service.task_runtime import TaskBoxRuntime

config = json.loads(sys.argv[1])
settings = Settings(**config["settings"])
boxes = TaskBoxRuntime(settings, client=docker.from_env(timeout=120), namespace=config["namespace"])
signer = Signer(settings.machine_key_id, load_private_key(settings.machine_key_file))
manager = EpisodeManager(settings, boxes, Store(settings.state_dir), signer,
                         TaskRegistry(settings.episode_envs))
key = manager.open(config["token"])["episode_key"]
manager.call(config["session"], key, 0, 0, "bash", {"command": "echo 41 > answer.txt"})
print(json.dumps({"key": key}), flush=True)
sys.stdin.read()
"""


def test_a_killed_controller_resumes_the_episode(tmp_path, client, image):
    settings, validator = live_settings(tmp_path, image)
    namespace = uuid.uuid4().hex
    token = issue_session_token(claims("s-restart", env="live-env", image=image,
                                       budgets=budgets(per_call_timeout_s=10)), validator)
    config = {"settings": settings.model_dump(mode="json"), "namespace": namespace,
              "token": token, "session": "s-restart"}
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(TESTS), os.environ.get("PYTHONPATH", "")])}
    child = subprocess.Popen([sys.executable, "-c", CHILD, json.dumps(config)], env=environment,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        line = child.stdout.readline()
        assert line, "the controller process died before its first call"
        key = json.loads(line)["key"]
    finally:
        child.kill()  # SIGKILL our own controller process, mid-episode
        child.wait(timeout=30)
    manager, boxes, machine = manager_for(settings, client, namespace)
    try:
        manager.reconcile()
        assert boxes.box_user("s-restart") == "0:0"
        g = Gateway(manager, boxes, validator, machine, image)
        assert g.call("s-restart", key, 1, "bash", {"command": "cat answer.txt"})["output"] == "41\n"
        g.call("s-restart", key, 2, "edit", {"path": "answer.txt", "old_str": "41", "new_str": "42"})
        final = manager.finish("s-restart", key)["record"]["body"]
        assert final["status"] == "graded" and final["reward"] == 1.0, json.dumps(final)
        result = g.verify("s-restart", key)
        assert result.ok, result.reasons
    finally:
        for box_id in boxes.managed():
            boxes.remove(box_id)


def test_measurements(gateway):
    g = gateway
    opens, finishes, calls, execs, idle_memory = [], [], [], [], []
    for n in range(3):
        session = f"s-m{n}"
        began = time.monotonic()
        key = g.open(session)
        opens.append(time.monotonic() - began)
        for turn in range(5):
            began = time.monotonic()
            g.call(session, key, turn, "bash", {"command": "true"})
            calls.append(time.monotonic() - began)
        for _ in range(5):
            began = time.monotonic()
            g.boxes.exec(session, [HELPER, "true"], timeout_s=30, output_limit=1024, user="0",
                         workdir="/")
            execs.append(time.monotonic() - began)
        time.sleep(2)
        container = g.boxes._container(session)
        stats = g.boxes.client.api.stats(container.id, stream=False, one_shot=True)
        idle_memory.append({"docker_usage": stats.get("memory_stats", {}).get("usage"),
                            "cgroup_current": _cgroup_file(g.boxes, session, "memory.current").strip(),
                            "cgroup_peak": _cgroup_file(g.boxes, session, "memory.peak").strip()})
        g.call(session, key, 5, "bash", {"command": "echo 42 > answer.txt"})
        began = time.monotonic()
        final = g.manager.finish(session, key)["record"]["body"]
        finishes.append(time.monotonic() - began)
        assert final["reward"] == 1.0

    def median_ms(values):
        return int(statistics.median(values) * 1000)

    report("measurements", open_ms=median_ms(opens), bash_call_ms=median_ms(calls),
           raw_exec_ms=median_ms(execs), finish_ms=median_ms(finishes), idle_memory=idle_memory)
