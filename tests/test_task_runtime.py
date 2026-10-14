"""TaskBoxRuntime against a fake Docker client (no containers)."""

import errno
import hashlib
import itertools
import socket as socket_module
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from docker.errors import APIError, ImageNotFound, NotFound
from episode_fixtures import IMAGE

from reliquary_sandbox_service.episodes.exec_transport import ExecStream
from reliquary_sandbox_service.runtime import (
    DockerRuntime,
    RuntimeUnavailable,
)
from reliquary_sandbox_service.task_runtime import (
    BASH_RUNNER,
    CLEAR_ROOT,
    FRAMED_SCRIPTS,
    HELPER,
    HELPER_ROOT,
    KILL_GROUP,
    MAKE_DIRS,
    OUTPUT_ATTEMPTS,
    READ_FILE,
    REFUSED_MARKER,
    REMOVE_RESTORED,
    RESOLVE_EXISTING,
    RESOLVE_LINKS,
    RUN_PROGRAM,
    SCRUBBED,
    STOP_PROCESSES,
    TAR_CREATE,
    TAR_EXTRACT,
    TASK_CAPABILITIES,
    WRITE_FILE,
    BoxFault,
    BoxSpec,
    ExecResult,
    HelperBudgetTimeout,
    InfraFault,
    NotRegularFile,
    RuntimeExecFailed,
    TaskBoxRuntime,
    frame_markers,
)


class SocketProxy:
    """A client socket that records settimeout calls (docker-py sets a timeout)."""

    def __init__(self, sock):
        self._inner = sock
        self.timeouts = []

    def settimeout(self, value):
        self.timeouts.append(value)
        self._inner.settimeout(value)

    def fileno(self):
        return self._inner.fileno()

    def __getattr__(self, name):
        return getattr(self._inner, name)


UNSTARTED = "unstarted"


def default_handler(argv, stdin):
    if argv[1:] == ["id"]:
        return [(b"uid=1000(user) gid=1000(user) groups=1000(user)\n", None)], 0, 0
    return [(b"ok\n", None)], 0, 0


def api_error(message, status):
    response = requests.Response()
    response.status_code = status
    response.reason = "x"
    response.url = "http://docker/x"
    return APIError(message, response=response, explanation=message)


class Box:
    def __init__(self, labels):
        self.id = "cid-" + labels.get(TaskBoxRuntime.BOX_LABEL, "probe")
        self.labels = labels
        self.status = "created"
        self.removed = False
        self.attrs = {"State": {"Error": "", "OOMKilled": False,
                                "StartedAt": "2026-10-05T10:00:00.123456789Z"}}

    def start(self):
        self.status = "running"

    def reload(self):
        pass

    def remove(self, force=False):
        self.removed = True


class FakeDocker:
    def __init__(self):
        self.info_value = {"OSType": "linux", "Runtimes": {"runsc": {"path": "/usr/bin/runsc"}}, "Driver": "overlay2", "NCPU": 8,
                           "DriverStatus": [["Backing Filesystem", "xfs"], ["Supports d_type", "true"]]}
        self.images_value = {IMAGE}
        self.created = []
        self.boxes = {}
        self.last_box = None
        self.execs = {}
        self.handler = default_handler
        self.probe_error = None
        self.lose_output = 0  # the next N framed helpers lose their whole stdout
        self.lose_prefix = 0  # the next N framed helpers lose their first write
        self.drop_trailer = 0  # the next N bash runners end without their trailer
        self.sockets = []
        self.stats_value = {"cpu_stats": {"cpu_usage": {"total_usage": 7_000_000}}}
        self.containers = SimpleNamespace(create=self.create, get=self.get, list=self.list)
        self.images = SimpleNamespace(get=self.image, list=self.image_list)
        self.api = SimpleNamespace(exec_create=self.exec_create, exec_start=self.exec_start,
                                   exec_inspect=self.exec_inspect, stats=self.stats)

    def info(self):
        return self.info_value

    def image(self, name):
        if name not in self.images_value:
            raise ImageNotFound("missing")
        return SimpleNamespace(id="sha256:imageid")

    def image_list(self):
        return [SimpleNamespace(id="sha256:imageid", attrs={"RepoDigests": [name]})
                for name in sorted(self.images_value)]

    def create(self, **kwargs):
        if kwargs.get("labels", {}).get(TaskBoxRuntime.LABEL) == "probe" and self.probe_error:
            raise self.probe_error
        self.created.append(kwargs)
        box = Box(kwargs["labels"])
        self.last_box = box
        if "name" in kwargs:
            self.boxes[kwargs["name"]] = box
        return box

    def get(self, name):
        box = self.boxes.get(name)
        if box is None or box.removed:
            raise NotFound("gone")
        return box

    def list(self, all=False, filters=None):
        wanted = [item.split("=", 1) for item in filters["label"]]
        return [b for b in self.boxes.values()
                if not b.removed and all_labels(b.labels, wanted)]

    def exec_create(self, container_id, argv, **kwargs):
        exec_id = f"exec-{len(self.execs)}"
        self.execs[exec_id] = {"argv": argv, "kwargs": kwargs, "exit": None}
        return {"Id": exec_id}

    def exec_start(self, exec_id, stream=False, demux=False, socket=False):
        assert socket is True
        item = self.execs[exec_id]
        # A 4th value `UNSTARTED`: the runtime never started the command (Docker wrote
        # its own error as the whole stream), so no helper marker or trailer either.
        chunks, code, delay, *unstarted = self.handler(item["argv"], item.get("stdin"))
        item["exit"] = code
        argv = item["argv"] if not unstarted else []
        if len(argv) > 5 and argv[3] in FRAMED_SCRIPTS:
            # What the real helper prints: its output between the nonce's markers.
            opened, closed = frame_markers(argv[5])
            chunks = [(opened, None), *chunks, (closed, None)]
            if self.lose_output:
                self.lose_output -= 1
                chunks = [(None, err) for _, err in chunks if err]
            elif self.lose_prefix:
                self.lose_prefix -= 1
                chunks = chunks[1:]
        if len(argv) > 6 and argv[3] == BASH_RUNNER:
            # The real runner's last write: its trailer with the exit status.
            if self.drop_trailer:
                self.drop_trailer -= 1
            else:
                chunks = itertools.chain(chunks, [(f"RQ-END {argv[6]} {code}".encode(), None)])
        client, server = socket_module.socketpair()
        client = SocketProxy(client)
        self.sockets.append(client)

        def feed():
            try:
                for out, err in chunks:
                    for number, data in ((1, out), (2, err)):
                        if data:
                            server.sendall(struct.pack(">BxxxL", number, len(data)) + data)
                if delay:
                    threading.Event().wait(delay)
            except OSError:
                pass
            finally:
                server.close()

        threading.Thread(target=feed, daemon=True).start()
        return client

    def exec_inspect(self, exec_id):
        exit_code = self.execs[exec_id]["exit"]
        return {"Running": False, "ExitCode": 0 if exit_code is None else exit_code}

    def stats(self, container_id, stream=True, one_shot=None):
        assert stream is False and one_shot is True
        return self.stats_value


class FakeTransport:
    """The exec stream from FakeDocker's socket pair (the real Unix-socket
    transport is tested in test_exec_transport.py)."""

    def __init__(self, docker):
        self.docker = docker

    def start(self, exec_id, on_connect=None):
        sock = self.docker.exec_start(exec_id, socket=True)
        if on_connect is not None:
            on_connect(sock)
        return ExecStream(sock, b"")


def all_labels(labels, wanted):
    return all(labels.get(key) == value for key, value in wanted)


def make_helper(tmp_path):
    path = tmp_path / "busybox"
    path.write_bytes(b"static busybox")
    path.chmod(0o755)
    return path


def settings(**overrides):
    values = dict(state_dir=None, episode_helper_busybox=None, episode_helper_busybox_sha256=None,
                  episode_docker_pidfile="/nonexistent", episode_fault_settle_s=0, episode_runsc_version="release-20260928.0", docker_runtime="runsc", require_gvisor=True, episode_images=[IMAGE],
                  episode_allow_unquoted_disk=False, episode_cpu_source="docker",
                  episode_cgroup_root="/nonexistent", episode_capacity=2,
                  episode_nano_cpus=2_000_000_000)
    values.update(overrides)
    return SimpleNamespace(**values)


def spec(**overrides):
    values = dict(image=IMAGE, workdir="/testbed", env={"A": "1"}, user=None,
                  memory_bytes=2 * 1024**3, nano_cpus=2_000_000_000, pids=512,
                  disk_bytes=10 * 1024**3, lifetime_s=3700)
    values.update(overrides)
    return BoxSpec(**values)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    helper = make_helper(tmp_path)
    built = TaskBoxRuntime(
        settings(state_dir=tmp_path, episode_helper_busybox=helper,
                 episode_helper_busybox_sha256=hashlib.sha256(helper.read_bytes()).hexdigest()),
        client=FakeDocker(), namespace="n" * 32)
    built.exec_transport = FakeTransport(built.client)
    monkeypatch.setattr(built, "_runsc_version", lambda path: "release-20260928.0")
    monkeypatch.setattr(built, "_daemon_started_at", lambda: 1.0)
    monkeypatch.setattr(built, "_helper_version", lambda path: (1, 37, 0))
    return built


def test_task_box_runs_the_image_as_is_with_a_quota_capped_writable_root(runtime):
    runtime.create("s-1", "s-1", spec())
    config = runtime.client.created[-1]
    assert config["runtime"] == "runsc"
    assert config["network_mode"] == "none"
    assert config["read_only"] is False
    assert config["storage_opt"] == {"size": str(10 * 1024**3)}
    assert config["user"] is None
    assert config["working_dir"] == "/testbed"
    assert config["entrypoint"] == ["sleep", "3700"]
    assert config["cap_drop"] == ["ALL"] and config["cap_add"] == TASK_CAPABILITIES
    assert "NET_RAW" not in config["cap_add"]
    assert config["mem_limit"] == config["memswap_limit"] == 2 * 1024**3
    assert config["pids_limit"] == 512
    assert config["hostname"] == "sandbox"
    assert "tmpfs" not in config and "volumes" not in config


def test_test_flag_drops_the_quota(runtime):
    runtime.settings.episode_allow_unquoted_disk = True
    runtime.create("s-1", "s-1", spec())
    assert "storage_opt" not in runtime.client.created[-1]


def test_unapproved_or_missing_images_never_start(runtime):
    with pytest.raises(ValueError):
        runtime.create("s-1", "s-1", spec(image="other@sha256:" + "b" * 64))
    runtime.client.images_value.clear()
    with pytest.raises(RuntimeUnavailable):
        runtime.create("s-1", "s-1", spec())
    assert runtime.client.created == []


def test_preflight_accepts_overlay2_on_xfs_with_working_quota(runtime):
    assert runtime.preflight() == {"runtime": "runsc", "images": [IMAGE], "disk_quota": "xfs-pquota",
                                   "runsc_version": "release-20260928.0",
                                   "busybox_version": "1.37.0", "host_cpus": 8,
                                   "cpu_overcommitted": False}
    probe = runtime.client.created[-1]
    assert probe["storage_opt"] == {"size": str(64 * 1024**2)}


@pytest.mark.parametrize("change", [
    lambda c: c.info_value.update(Driver="overlayfs"),
    lambda c: c.info_value.update(DriverStatus=[["Backing Filesystem", "extfs"]]),
    lambda c: setattr(c, "probe_error", APIError("--storage-opt is supported only for overlay over xfs with 'pquota'")),
    lambda c: c.images_value.clear(),
    lambda c: c.info_value.update(Runtimes={"runc": {}}),
])
def test_preflight_refuses_hosts_that_cannot_enforce_limits(runtime, change):
    change(runtime.client)
    with pytest.raises(RuntimeUnavailable):
        runtime.preflight()


def test_preflight_test_flag_skips_quota_checks(runtime):
    runtime.settings.episode_allow_unquoted_disk = True
    runtime.client.info_value.update(Driver="overlayfs")
    assert runtime.preflight()["disk_quota"] == "disabled"


def test_episode_boxes_and_tenant_guests_never_reconcile_each_other(runtime, tmp_path):
    runtime.create("s-1", "s-1", spec())
    tenant = DockerRuntime(SimpleNamespace(state_dir=tmp_path), client=runtime.client)
    assert tenant.managed_sandboxes() == {}
    assert runtime.managed() == {"s-1": "s-1"}


def test_exec_caps_each_stream_and_flags_truncation(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: ([(b"abcdef", b"xy")], 3, 0)
    result = runtime.exec("s-1", ["true"], timeout_s=5, output_limit=4)
    assert (result.stdout, result.stderr, result.exit_code) == (b"abcd", b"xy", 3)
    assert result.truncated and not result.timed_out


def test_exec_timeout_keeps_the_box(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: ([(b"partial", None)], 0, 1.0)
    result = runtime.exec("s-1", ["sleep", "9"], timeout_s=0.05, output_limit=100)
    assert result.timed_out and result.exit_code == -1
    assert runtime.status("s-1") == "running"


def test_bash_reports_unkillable_commands(runtime, monkeypatch):
    runtime.create("s-1", "s-1", spec())
    monkeypatch.setattr("reliquary_sandbox_service.task_runtime.time.sleep", lambda _: None)
    runtime.client.handler = lambda argv, stdin: ([], 1 if argv[3] == KILL_GROUP else 0,
                                                  0 if argv[3] == KILL_GROUP else 1.0)
    with pytest.raises(BoxFault):
        runtime.bash("s-1", "b" * 32, "sleep 100", timeout_s=0.05, output_limit=100,
                     env={}, workdir="/", user=None)


@pytest.mark.parametrize("code,error", [(3, FileNotFoundError), (4, IsADirectoryError), (5, ValueError), (1, PermissionError)])
def test_read_file_maps_failures(runtime, code, error):
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: ([], code, 0)
    with pytest.raises(error):
        runtime.read_file("s-1", "/x", 10)


def test_read_file_refuses_oversized_files(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: ([(b"x" * 11, None)], 0, 0)
    with pytest.raises(ValueError):
        runtime.read_file("s-1", "/x", 10)


def test_write_file_streams_stdin(runtime, monkeypatch):
    runtime.create("s-1", "s-1", spec())
    sent = {}

    def stream(exec_id, stdin):
        sent["stdin"] = stdin
        yield (None, None)

    monkeypatch.setattr(runtime, "_exec_stream", stream)
    runtime.write_file("s-1", "/testbed/a.py", b"print(1)\n")
    item = list(runtime.client.execs.values())[-1]
    assert item["argv"][-1] == "/testbed/a.py" and item["kwargs"]["stdin"] is True
    assert sent["stdin"] == b"print(1)\n"


def test_cpu_from_docker_stats(runtime):
    runtime.create("s-1", "s-1", spec())
    assert runtime.cpu_ns("s-1") == 7_000_000
    runtime.client.stats_value = {"cpu_stats": {}}
    with pytest.raises(InfraFault):
        runtime.cpu_ns("s-1")


def test_cpu_from_cgroupfs(runtime, tmp_path):
    runtime.create("s-1", "s-1", spec())
    runtime.settings.episode_cpu_source = "cgroupfs"
    runtime.settings.episode_cgroup_root = tmp_path
    scope = tmp_path / "system.slice" / "docker-cid-s-1.scope"
    scope.mkdir(parents=True)
    (scope / "cpu.stat").write_text("usage_usec 1500\nuser_usec 1000\n")
    assert runtime.cpu_ns("s-1") == 1_500_000


def test_a_box_whose_tools_dir_cannot_be_made_never_serves_and_is_ours_to_blame(runtime):
    runtime.client.handler = lambda argv, stdin: ([], 1, 0)
    with pytest.raises(InfraFault):
        runtime.create("s-1", "s-1", spec())
    assert runtime.client.last_box.removed


def test_boxes_only_run_under_runsc(runtime):
    runtime.settings.docker_runtime = "runc"
    with pytest.raises(RuntimeUnavailable):
        runtime.create("s-1", "s-1", spec())


def test_preflight_pins_the_runsc_version_and_refuses_ignore_cgroups(runtime, monkeypatch):
    monkeypatch.setattr(runtime, "_runsc_version", lambda path: "release-20250101.0")
    with pytest.raises(RuntimeUnavailable):
        runtime.preflight()
    monkeypatch.setattr(runtime, "_runsc_version", lambda path: "release-20260928.0")
    runtime.client.info_value["Runtimes"]["runsc"]["runtimeArgs"] = ["--ignore-cgroups"]
    with pytest.raises(RuntimeUnavailable):
        runtime.preflight()


def test_preflight_warns_when_agent_and_grading_boxes_could_overcommit_the_cpus(runtime, caplog):
    runtime.settings.episode_capacity = 3  # 3 episodes x 2 boxes x 2 CPUs = 12 > 8
    with caplog.at_level("WARNING"):
        facts = runtime.preflight()
    assert facts["cpu_overcommitted"] is True and facts["host_cpus"] == 8
    assert any("CPU" in r.getMessage() and r.levelname == "WARNING" for r in caplog.records)


def test_probe_container_is_always_removed(runtime):
    runtime.preflight()
    assert runtime.client.last_box.removed


def test_the_probe_failure_is_an_infra_fault(runtime):
    runtime.client.probe_error = APIError("no pquota")
    with pytest.raises(InfraFault):
        runtime.preflight()


def test_a_box_the_agent_killed_is_the_agents_fault(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.boxes[runtime._name("s-1")].status = "exited"

    def fail(argv, stdin):
        raise APIError("container is not running")

    runtime.client.exec_create = lambda *a, **k: (_ for _ in ()).throw(APIError("not running"))
    runtime.client.api.exec_create = runtime.client.exec_create
    with pytest.raises(BoxFault):
        runtime.exec("s-1", ["true"], timeout_s=5, output_limit=10)


def test_a_deleted_workdir_is_the_agents_fault(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.api.exec_create = lambda *a, **k: (_ for _ in ()).throw(
        APIError("OCI runtime exec failed: chdir: no such file or directory"))
    with pytest.raises(BoxFault):
        runtime.exec("s-1", ["true"], timeout_s=5, output_limit=10, workdir="/gone")


def test_a_box_at_its_pids_limit_is_the_agents_fault(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.stats_value = {"pids_stats": {"current": 512, "limit": 512},
                                  "cpu_stats": {"cpu_usage": {"total_usage": 1}}}
    assert runtime.pids_exhausted("s-1")
    runtime.client.api.exec_create = lambda *a, **k: (_ for _ in ()).throw(APIError("cannot fork"))
    with pytest.raises(BoxFault):
        runtime.exec("s-1", ["true"], timeout_s=5, output_limit=10)
    runtime.client.stats_value = {"pids_stats": {"current": 511, "limit": 512}}
    assert not runtime.pids_exhausted("s-1")
    runtime.client.stats_value = {}
    assert not runtime.pids_exhausted("s-1")


def test_pids_from_cgroupfs(runtime, tmp_path):
    runtime.create("s-1", "s-1", spec())
    runtime.settings.episode_cpu_source = "cgroupfs"
    runtime.settings.episode_cgroup_root = tmp_path
    scope = tmp_path / "system.slice" / "docker-cid-s-1.scope"
    scope.mkdir(parents=True)
    (scope / "pids.max").write_text("512\n")
    (scope / "pids.current").write_text("512\n")
    assert runtime.pids_exhausted("s-1")
    (scope / "pids.current").write_text("3\n")
    assert not runtime.pids_exhausted("s-1")
    (scope / "pids.max").write_text("max\n")
    (scope / "pids.current").write_text("999999\n")
    assert not runtime.pids_exhausted("s-1")


def test_cpu_of_a_dead_box_is_its_last_value_or_a_box_fault(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.create("s-2", "s-2", spec())
    assert runtime.cpu_ns("s-1") == 7_000_000
    for name in ("s-1", "s-2"):
        runtime.client.boxes[runtime._name(name)].status = "exited"
    assert runtime.cpu_ns("s-1") == 7_000_000
    with pytest.raises(BoxFault):
        runtime.cpu_ns("s-2")


def test_write_to_a_full_disk_is_enospc(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: ([(None, b"cat: write error: No space left on device\n")], 1, 0)
    with pytest.raises(OSError) as caught:
        runtime.write_file("s-1", "/testbed/a", b"x")
    assert caught.value.errno == errno.ENOSPC


def test_an_unreachable_daemon_is_our_fault(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.api.exec_create = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down"))

    def down(name):
        raise ConnectionError("down")

    runtime.client.containers.get = down
    with pytest.raises(InfraFault):
        runtime.exec("s-1", ["true"], timeout_s=5, output_limit=10)


def test_a_runner_refusal_is_a_box_fault_not_an_exit_code(runtime):
    runtime.create("s-1", "s-1", spec())
    marker = f"{REFUSED_MARKER} {'d' * 32}\n".encode()
    runtime.client.handler = lambda argv, stdin: ([(None, marker)], 125, 0)
    with pytest.raises(BoxFault):
        runtime.bash("s-1", "d" * 32, "ls", timeout_s=5, output_limit=100, env={},
                     workdir="/", user=None)
    runtime.client.handler = lambda argv, stdin: ([(b"x", None)], 125, 0)
    assert runtime.bash("s-1", "e" * 32, "exit 125", timeout_s=5, output_limit=100, env={},
                        workdir="/", user=None).exit_code == 125


def test_a_timed_out_exec_releases_its_socket(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: ([(b"p", None)], 0, 5.0)
    result = runtime.exec("s-1", ["sleep", "9"], timeout_s=0.1, output_limit=10)
    assert result.timed_out
    assert runtime.client.sockets[-1].fileno() == -1
    assert runtime._connections == {}


def test_write_survives_the_reader_quitting_early(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: ([(None, b"cat: Permission denied\n")], 1, 0)
    with pytest.raises(PermissionError):
        runtime.write_file("s-1", "/testbed/big", b"x" * (8 * 1024 * 1024))


def run_bash(runtime, call_id="a" * 32, **kwargs):
    values = dict(timeout_s=5, output_limit=100, env={"A": "1"}, workdir="/testbed", user="1000")
    values.update(kwargs)
    return runtime.bash("s-1", call_id, "ls -la", **values)


def test_bash_timeout_kills_only_the_call_group_as_the_calls_uid_from_slash(runtime):
    runtime.create("s-1", "s-1", spec())
    seen = []

    def handler(argv, stdin):
        seen.append(argv)
        if argv[3] == KILL_GROUP:
            return [], 0, 0
        return [(b"started\n", None)], 0, 1.0

    runtime.client.handler = handler
    result = run_bash(runtime, "a" * 32, timeout_s=0.05, user="1000:1001")
    assert result.timed_out and result.exit_code == -1 and result.stdout == b"started\n"
    assert seen[-1] == [HELPER, "sh", "-c", KILL_GROUP, "reliquary-kill", "a" * 32]
    kill = list(runtime.client.execs.values())[-1]["kwargs"]
    assert kill["user"] == "1000:1001" and kill["workdir"] == "/"
    assert runtime.status("s-1") == "running"


def test_every_helper_runs_from_the_controllers_busybox_in_slash_as_a_numeric_user(runtime):
    runtime.create("s-1", "s-1", spec(user="appuser"))
    run_bash(runtime, user="appuser")
    runtime.client.handler = lambda argv, stdin: ([(b"x", None)], 0, 0)
    runtime.read_file("s-1", "/x", 10, user="appuser")
    runtime.write_file("s-1", "/x", b"1", user="appuser")
    runtime.file_kind("s-1", "/x", user="appuser")
    for item in runtime.client.execs.values():
        if item["argv"][1:] == ["id"]:
            continue  # the one-time lookup of the user name, before any agent code ran
        assert item["argv"][0] == HELPER, item["argv"]
        assert item["kwargs"]["workdir"] == "/"
        assert item["kwargs"]["user"] in ("0", "1000:1000"), item["kwargs"]
    runner = [i for i in runtime.client.execs.values() if "reliquary-bash" in i["argv"]][0]
    assert runner["argv"][-4:] == [str(100), "a" * 32, "ls -la", "/testbed"]
    assert runner["kwargs"]["environment"] == {"A": "1"}


def test_the_busybox_is_bind_mounted_read_only_with_the_etc_files(runtime, tmp_path):
    runtime.create("s-1", "s-1", spec())
    mounts = {m["Target"]: m for m in runtime.client.created[-1]["mounts"]}
    assert set(mounts) == {"/etc/hosts", "/etc/hostname", "/etc/resolv.conf", HELPER_ROOT}
    for mount in mounts.values():
        assert mount["ReadOnly"] is True and mount["Type"] == "bind"
    # the whole directory is mounted, holding only a copy of the binary
    source = tmp_path / "episode-helper"
    assert mounts[HELPER_ROOT]["Source"] == str(source)
    assert [str(p.relative_to(source)) for p in source.rglob("*")] == ["bin", "bin/busybox"]
    assert (source / "bin" / "busybox").read_bytes() == (tmp_path / "busybox").read_bytes()
    for target in ("/etc/hosts", "/etc/hostname", "/etc/resolv.conf"):
        assert mounts[target]["Source"].startswith(str(tmp_path / "episode-etc"))
    assert open(mounts["/etc/resolv.conf"]["Source"]).read() == ""
    runtime.remove("s-1")
    assert not list((tmp_path / "episode-etc").iterdir())


def test_tools_dir_and_user_are_prepared_at_creation_before_the_agent_runs(runtime):
    runtime.create("s-1", "s-1", spec(user=None))
    first, second = list(runtime.client.execs.values())[:2]
    assert first["argv"][0] == HELPER and first["kwargs"]["user"] == "0"
    assert "1777" in first["argv"][3] and first["argv"][-1] == "/dev/shm/.reliquary-tools"
    assert second["argv"] == [HELPER, "id"]


def test_missing_or_modified_busybox_refuses_preflight_and_creation(runtime, tmp_path):
    runtime.preflight()
    (tmp_path / "busybox").write_bytes(b"swapped")
    for call in (runtime.preflight, lambda: runtime.create("s-1", "s-1", spec())):
        with pytest.raises(RuntimeUnavailable):
            call()
    (tmp_path / "busybox").unlink()
    with pytest.raises(RuntimeUnavailable):
        runtime.preflight()


def test_a_non_executable_busybox_refuses_preflight(runtime, tmp_path):
    (tmp_path / "busybox").chmod(0o644)
    with pytest.raises(RuntimeUnavailable):
        runtime.preflight()


def kill_box(runtime, **state):
    runtime.client.boxes[runtime._name("s-1")].status = "exited"
    runtime.client.boxes[runtime._name("s-1")].attrs["State"].update(state)


@pytest.mark.parametrize("route", ["read", "write", "kind"])
@pytest.mark.parametrize("result", [([], 127, 0), ([], 126, 0), ([(None, b"OCI runtime exec failed: x\n")], 1, 0),
                                    ([], 0, 2.0)])
def test_a_broken_helper_stream_or_timeout_on_a_running_box_is_the_agents_fault(runtime, monkeypatch, route, result):
    runtime.create("s-1", "s-1", spec())
    monkeypatch.setattr("reliquary_sandbox_service.task_runtime.time.sleep", lambda _: None)
    runtime.client.handler = lambda argv, stdin: result
    calls = {
        "read": lambda: runtime.read_file("s-1", "/x", 10),
        "write": lambda: runtime.write_file("s-1", "/x", b"1"),
        "kind": lambda: runtime.file_kind("s-1", "/x"),
    }
    original = runtime.exec

    def short(box_id, argv, **kwargs):
        kwargs["timeout_s"] = min(kwargs["timeout_s"], 0.1)
        return original(box_id, argv, **kwargs)

    monkeypatch.setattr(runtime, "exec", short)
    with pytest.raises(BoxFault):
        calls[route]()


def test_exec_exceptions_that_are_oci_start_failures_are_the_agents_fault(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.api.exec_create = lambda *a, **k: (_ for _ in ()).throw(
        api_error("OCI runtime exec failed: chdir: no such file or directory", 500))
    with pytest.raises(BoxFault):
        runtime.exec("s-1", ["true"], timeout_s=5, output_limit=10)


@pytest.mark.parametrize("error", [api_error("containerd: context deadline exceeded", 500),
                                   NotFound("gone"), ConnectionError("down")])
def test_docker_api_failures_are_ours(runtime, error):
    runtime.create("s-1", "s-1", spec())
    runtime.client.api.exec_create = lambda *a, **k: (_ for _ in ()).throw(error)
    with pytest.raises(InfraFault) as caught:
        runtime.exec("s-1", ["true"], timeout_s=5, output_limit=10)
    assert not isinstance(caught.value, BoxFault)


def test_a_box_that_stopped_for_the_agents_reasons_is_the_agents_fault(runtime):
    runtime.create("s-1", "s-1", spec())
    kill_box(runtime)  # PID 1 exited, State.Error empty
    assert isinstance(runtime.classify_failure("s-1", "x"), BoxFault)
    kill_box(runtime, OOMKilled=True)
    fault = runtime.classify_failure("s-1", "x")
    assert isinstance(fault, BoxFault) and "out of memory" in str(fault)


def test_a_box_stopped_by_a_runtime_error_is_ours(runtime):
    runtime.create("s-1", "s-1", spec())
    kill_box(runtime, Error="runsc: sentry panic")
    assert isinstance(runtime.classify_failure("s-1", "x"), InfraFault)
    assert not isinstance(runtime.classify_failure("s-1", "x"), BoxFault)


def test_a_box_stopped_by_a_daemon_restart_is_ours(runtime, monkeypatch):
    runtime.create("s-1", "s-1", spec())
    kill_box(runtime)
    started = 1_791_194_400.0  # 2026-10-05T10:00:00Z
    monkeypatch.setattr(runtime, "_daemon_started_at", lambda: started + 600)
    assert not isinstance(runtime.classify_failure("s-1", "x"), BoxFault)
    monkeypatch.setattr(runtime, "_daemon_started_at", lambda: started - 600)
    assert isinstance(runtime.classify_failure("s-1", "x"), BoxFault)
    monkeypatch.setattr(runtime, "_daemon_started_at", lambda: None)
    assert isinstance(runtime.classify_failure("s-1", "x"), BoxFault)


def test_an_unreachable_daemon_during_classification_is_ours(runtime):
    runtime.create("s-1", "s-1", spec())

    def down(name):
        raise ConnectionError("down")

    runtime.client.containers.get = down
    assert isinstance(runtime.classify_failure("s-1", "x"), InfraFault)


def test_the_runner_caps_output_inside_the_box(runtime):
    runtime.create("s-1", "s-1", spec())
    run_bash(runtime, output_limit=1234)
    runner = [i for i in runtime.client.execs.values() if "reliquary-bash" in i["argv"]][0]
    assert runner["argv"][-4] == "1234"


def test_preflight_refuses_when_the_daemon_start_time_is_unreadable(runtime, monkeypatch):
    monkeypatch.setattr(runtime, "_daemon_started_at", lambda: None)
    with pytest.raises(RuntimeUnavailable):
        runtime.preflight()


def test_cpu_stats_failure_on_a_box_that_just_died_is_the_dead_box_path(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.create("s-2", "s-2", spec())
    assert runtime.cpu_ns("s-1") == 7_000_000

    def dying(*args, **kwargs):
        for name in ("s-1", "s-2"):
            runtime.client.boxes[runtime._name(name)].status = "exited"
        raise APIError("stats failed")

    runtime.client.api.stats = dying
    assert runtime.cpu_ns("s-1") == 7_000_000  # cached
    runtime.client.boxes[runtime._name("s-1")].status = "running"
    runtime.client.boxes[runtime._name("s-2")].status = "running"
    with pytest.raises(BoxFault):
        runtime.cpu_ns("s-2")  # never measured, died during the failing read


def test_cpu_stats_failure_on_a_running_box_or_unreachable_daemon_is_ours(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.api.stats = lambda *a, **k: (_ for _ in ()).throw(APIError("boom"))
    with pytest.raises(InfraFault) as caught:
        runtime.cpu_ns("s-1")
    assert not isinstance(caught.value, BoxFault)

    def down(name):
        raise ConnectionError("down")

    runtime.client.containers.get = down
    runtime.client.api.stats = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down"))
    runtime._cpu_last.clear()
    with pytest.raises(InfraFault):
        runtime.cpu_ns("s-1")


@pytest.mark.parametrize("error", [api_error("containerd: context deadline exceeded", 500),
                                   NotFound("gone"), ConnectionError("down"), RuntimeError("bug")])
def test_a_box_that_stopped_cleanly_is_the_agents_whatever_the_exception(runtime, error):
    runtime.create("s-1", "s-1", spec())
    kill_box(runtime)
    assert isinstance(runtime.classify_failure("s-1", "x", error), BoxFault)


def test_an_unexpected_exception_type_on_a_running_box_is_our_bug_and_is_logged(runtime, caplog):
    runtime.create("s-1", "s-1", spec())
    fault = runtime.classify_failure("s-1", "x", RuntimeError("bug"))
    assert isinstance(fault, InfraFault) and not isinstance(fault, BoxFault)
    assert "our bug" in caplog.text


def test_the_resolved_user_is_kept_returned_and_never_looked_up_again(runtime, tmp_path):
    runtime.create("s-1", "s-1", spec(user="appuser"))
    assert runtime.box_user("s-1") == "1000:1000"
    ids_before = [i for i in runtime.client.execs.values() if i["argv"][1:] == ["id"]]
    assert len(ids_before) == 1
    run_bash(runtime, user="appuser")
    runtime.read_file("s-1", "/x", 10, user="appuser")
    assert [i for i in runtime.client.execs.values() if i["argv"][1:] == ["id"]] == ids_before
    with pytest.raises(ValueError):
        runtime.read_file("s-1", "/x", 10, user="someoneelse")
    fresh = TaskBoxRuntime(runtime.settings, client=runtime.client, namespace="n" * 32,
                           exec_transport=runtime.exec_transport)
    with pytest.raises(InfraFault):
        fresh.read_file("s-1", "/x", 10, user="appuser")
    fresh.restore_user("s-1", "1000:1000", "appuser")
    fresh.read_file("s-1", "/x", 10, user="appuser")
    assert [i for i in runtime.client.execs.values() if i["argv"][1:] == ["id"]] == ids_before


@pytest.mark.parametrize("events,expected", [("oom 3\noom_kill 3\n", BoxFault),
                                             ("oom 0\noom_kill 2\n", InfraFault), (None, BoxFault)])
def test_oom_counts_against_the_agent_only_when_its_own_limit_was_hit(runtime, tmp_path, events, expected):
    runtime.create("s-1", "s-1", spec())
    runtime.settings.episode_cgroup_root = tmp_path
    if events is not None:
        scope = tmp_path / "system.slice" / "docker-cid-s-1.scope"
        scope.mkdir(parents=True)
        (scope / "memory.events").write_text(events)
    kill_box(runtime, OOMKilled=True)
    fault = runtime.classify_failure("s-1", "x")
    assert isinstance(fault, expected)
    assert isinstance(fault, BoxFault) == (expected is BoxFault)


def test_large_writes_clear_the_socket_timeout_first(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.write_file("s-1", "/x", b"data")
    assert runtime.client.sockets[-1].timeouts == [None]


def test_oci_text_on_stdout_counts_only_when_the_helper_failed(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: ([(b"OCI runtime exec failed\n", None)], 0, 0)
    assert runtime.read_file("s-1", "/x", 100) == b"OCI runtime exec failed\n"
    runtime.client.handler = lambda argv, stdin: ([(b"OCI runtime exec failed\n", None)], 1, 0)
    with pytest.raises(BoxFault):
        runtime.read_file("s-1", "/x", 100)


def flip_on_second_reload(runtime, name="s-1"):
    box = runtime.client.boxes[runtime._name(name)]
    original, calls = box.reload, []

    def reload():
        calls.append(1)
        original()
        if len(calls) >= 2:
            box.status = "exited"

    box.reload = reload
    return calls


def test_a_box_dying_during_the_settle_period_is_the_agents_not_ours(runtime):
    runtime.create("s-1", "s-1", spec())
    calls = flip_on_second_reload(runtime)
    fault = runtime.classify_failure("s-1", "x", api_error("containerd: context deadline exceeded", 500))
    assert isinstance(fault, BoxFault) and len(calls) >= 2


def test_a_box_that_stays_running_through_the_settle_period_stays_ours(runtime):
    runtime.create("s-1", "s-1", spec())
    fault = runtime.classify_failure("s-1", "x", api_error("containerd: context deadline exceeded", 500))
    assert isinstance(fault, InfraFault) and not isinstance(fault, BoxFault)


def test_a_box_dying_during_the_settle_period_with_a_runtime_error_is_still_ours(runtime):
    runtime.create("s-1", "s-1", spec())
    box = runtime.client.boxes[runtime._name("s-1")]
    original = box.reload

    def reload():
        original()
        box.status = "exited"
        box.attrs["State"]["Error"] = "sentry panic"

    calls = []
    box.reload = lambda: (calls.append(1), reload() if len(calls) >= 2 else original())
    fault = runtime.classify_failure("s-1", "x", api_error("500", 500))
    assert isinstance(fault, InfraFault) and not isinstance(fault, BoxFault)


def test_the_settle_period_also_applies_to_a_cpu_stats_failure(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.api.stats = lambda *a, **k: (_ for _ in ()).throw(APIError("boom"))
    calls = []
    box = runtime.client.boxes[runtime._name("s-1")]
    original = box.reload

    def reload():
        calls.append(1)
        original()
        if len(calls) >= 3:  # cpu_ns reload, failure reload, settle reload
            box.status = "exited"

    box.reload = reload
    with pytest.raises(BoxFault):
        runtime.cpu_ns("s-1")


def test_concurrent_creates_share_one_helper_copy(runtime, tmp_path):
    errors = []

    def create(name):
        try:
            runtime.create(name, name, spec())
        except Exception as exc:  # pragma: no cover - the assertion below reports it
            errors.append(exc)

    threads = [threading.Thread(target=create, args=(f"s-{i}",)) for i in range(2, 8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert sorted(p.name for p in (tmp_path / "episode-helper" / "bin").iterdir()) == ["busybox"]


def without_nonce(argv):
    return argv[:5] + argv[6:]


def capture_execs(runtime, monkeypatch, result=None):
    seen = []

    def fake_exec(box_id, argv, **kwargs):
        seen.append((argv, kwargs))
        found = result or ExecResult(b"", b"", 0, False, False)
        if len(argv) > 5 and argv[3] in FRAMED_SCRIPTS and not found.timed_out:
            opened, closed = frame_markers(argv[5])
            found = ExecResult(opened + found.stdout + closed, found.stderr,
                               found.exit_code, False, found.truncated)
        return found

    monkeypatch.setattr(runtime, "exec", fake_exec)
    return seen


def test_helper_timeouts_are_cut_to_the_callers_deadline(runtime, monkeypatch):
    runtime.create("s-1", "s-1", spec())
    seen = capture_execs(runtime, monkeypatch)
    runtime.read_file("s-1", "/x", 10, timeout_s=2.5)
    runtime.write_file("s-1", "/x", b"1", timeout_s=2.5)
    runtime.file_kind("s-1", "/x", timeout_s=2.5)
    runtime.make_dirs("s-1", "/a/b", timeout_s=2.5)
    runtime.tar_create("s-1", ["app"], 100, timeout_s=2.5)
    runtime.tar_extract("s-1", b"tar", "/app", timeout_s=2.5)
    runtime.read_file("s-1", "/x", 10)
    assert [kwargs["timeout_s"] for _, kwargs in seen] == [2.5] * 6 + [60]
    for argv, kwargs in seen:
        assert argv[0] == HELPER and kwargs["workdir"] == "/" and kwargs["user"] == "1000:1000"


def test_state_helpers_run_busybox_tar_without_following_links(runtime, monkeypatch):
    runtime.create("s-1", "s-1", spec())
    seen = capture_execs(runtime, monkeypatch)
    runtime.make_dirs("s-1", "/a/b")
    runtime.tar_create("s-1", ["app", "logs/x"], 100)
    runtime.tar_extract("s-1", b"tar", "/dest")
    assert seen[0][0] == [HELPER, "sh", "-c", MAKE_DIRS, "mkdirs", "/a/b"]
    assert without_nonce(seen[1][0]) == [HELPER, "sh", "-c", TAR_CREATE, "archive", "app", "logs/x"]
    assert seen[1][1]["output_limit"] == 100 + sum(map(len, frame_markers(seen[1][0][5])))
    assert seen[2][0] == [HELPER, "sh", "-c", TAR_EXTRACT, "restore", "/dest"]
    assert seen[2][1]["stdin"] == b"tar"
    assert "-h" not in TAR_CREATE and "tar -cf - --" in TAR_CREATE
    assert 'cd -- "$1"' in TAR_EXTRACT and "tar -xof -" in TAR_EXTRACT


def test_tar_create_maps_overflow_and_failure(runtime, monkeypatch):
    runtime.create("s-1", "s-1", spec())
    capture_execs(runtime, monkeypatch, ExecResult(b"x" * 100, b"", 0, False, True))
    with pytest.raises(ValueError):
        runtime.tar_create("s-1", ["app"], 100)
    capture_execs(runtime, monkeypatch, ExecResult(b"", b"tar: app: No such file", 1, False, False))
    with pytest.raises(OSError):
        runtime.tar_create("s-1", ["app"], 100)
    with pytest.raises(OSError):
        runtime.tar_extract("s-1", b"t", "/app")
    with pytest.raises(PermissionError):
        runtime.make_dirs("s-1", "/a")


def test_read_file_flags_non_regular_files(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: ([], 5, 0)
    with pytest.raises(NotRegularFile):
        runtime.read_file("s-1", "/x", 10)


def test_daemon_reachability_is_a_ping(runtime):
    runtime.client.ping = lambda: True
    assert runtime.daemon_reachable()

    def down():
        raise requests.exceptions.ConnectionError("down")

    runtime.client.ping = down
    assert not runtime.daemon_reachable()


BUSYBOX = __import__("shutil").which("busybox")
needs_busybox = pytest.mark.skipif(BUSYBOX is None, reason="busybox is not installed")


def host_script(script):
    import subprocess

    def run(*args, stdin=None, cwd="/"):
        nonce = ["0123456789abcdef"] if script in FRAMED_SCRIPTS else []
        done = subprocess.run([BUSYBOX, "sh", "-c", script.replace(HELPER, BUSYBOX), "x", *nonce,
                               *args], input=stdin, capture_output=True, timeout=30, cwd=cwd)
        if nonce:  # the real busybox prints the frame markers
            opened, closed = frame_markers(nonce[0])
            assert done.stdout.startswith(opened) and done.stdout.endswith(closed)
            done.stdout = done.stdout[len(opened):-len(closed)]
        return done
    return run


def test_preflight_requires_a_recent_helper_busybox(runtime, monkeypatch):
    monkeypatch.setattr(runtime, "_helper_version", lambda path: (1, 35, 0))
    with pytest.raises(RuntimeUnavailable):
        runtime.preflight()

    def unreadable(path):
        raise ValueError("no banner")

    monkeypatch.setattr(runtime, "_helper_version", unreadable)
    with pytest.raises(RuntimeUnavailable):
        runtime.preflight()


@needs_busybox
def test_the_helper_version_is_read_from_the_banner():
    from pathlib import Path

    assert TaskBoxRuntime._helper_version(Path(BUSYBOX)) >= (1, 36)


def test_resolve_existing_runs_the_helper(runtime, monkeypatch):
    runtime.create("s-1", "s-1", spec())
    seen = capture_execs(runtime, monkeypatch, ExecResult(b"/usr/bin\n", b"", 0, False, False))
    assert runtime.resolve_existing("s-1", "/app/lib/x", timeout_s=3) == "/usr/bin"
    assert without_nonce(seen[-1][0]) == [HELPER, "sh", "-c", RESOLVE_EXISTING, "resolve", "/app/lib/x"]
    assert seen[-1][1]["timeout_s"] == 3
    capture_execs(runtime, monkeypatch, ExecResult(b"", b"", 1, False, False))
    with pytest.raises(OSError):
        runtime.resolve_existing("s-1", "/dangling")


def test_resolve_links_runs_one_framed_helper_and_parses_each_record(runtime, monkeypatch):
    runtime.create("s-1", "s-1", spec())
    out = b"/app/real/f\n\0\n\0/tests/x\n\0"
    seen = capture_execs(runtime, monkeypatch, ExecResult(out, b"", 0, False, False))
    assert runtime.resolve_links("s-1", ["/app/in", "/app/dangling", "/app/out"], timeout_s=3) == [
        "/app/real/f", None, "/tests/x"]
    assert without_nonce(seen[-1][0]) == [HELPER, "sh", "-c", RESOLVE_LINKS, "links", "/app/in",
                                          "/app/dangling", "/app/out"]
    assert RESOLVE_LINKS in FRAMED_SCRIPTS and runtime.resolve_links("s-1", []) == []


@pytest.mark.parametrize("out, code", [
    (b"/app/real/f\n\0", 0),  # one record for two links
    (b"/app/real/f\n\0/x\0", 0),  # a record without its newline
    (b"/a\n\0/b\n\0", 1),  # readlink failed
])
def test_resolve_links_refuses_an_output_that_does_not_match(runtime, monkeypatch, out, code):
    runtime.create("s-1", "s-1", spec())
    capture_execs(runtime, monkeypatch, ExecResult(out, b"", code, False, False))
    with pytest.raises(OSError):
        runtime.resolve_links("s-1", ["/app/a", "/app/b"])


def test_remove_restored_runs_rm_with_the_helper(runtime, monkeypatch):
    runtime.create("s-1", "s-1", spec())
    seen = capture_execs(runtime, monkeypatch)
    runtime.remove_restored("s-1", "/app", timeout_s=3)
    assert seen[-1][0] == [HELPER, "sh", "-c", REMOVE_RESTORED, "remove", "/app"]
    capture_execs(runtime, monkeypatch, ExecResult(b"", b"rm: busy", 1, False, False))
    with pytest.raises(OSError):
        runtime.remove_restored("s-1", "/app")


@needs_busybox
def test_real_busybox_resolves_restored_links(tmp_path):
    (tmp_path / "app" / "real").mkdir(parents=True)
    (tmp_path / "app" / "real" / "f").write_bytes(b"x")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "x").write_bytes(b"ref")
    (tmp_path / "app" / "in").symlink_to("real/f")
    (tmp_path / "app" / "out").symlink_to("../tests/x")
    (tmp_path / "app" / "dangling").symlink_to("nowhere")
    (tmp_path / "app" / "loop").symlink_to("loop")
    (tmp_path / "app" / "new\nline").symlink_to("real")
    names = ["in", "out", "dangling", "loop", "new\nline"]
    out = host_script(RESOLVE_LINKS)(*[str(tmp_path / "app" / n) for n in names]).stdout
    assert out.split(b"\0") == [str(tmp_path / "app/real/f").encode() + b"\n",
                                 str(tmp_path / "tests/x").encode() + b"\n", b"\n", b"\n",
                                 str(tmp_path / "app/real").encode() + b"\n", b""]
    host_script(REMOVE_RESTORED)(str(tmp_path / "app" / "out"))
    assert not (tmp_path / "app" / "out").is_symlink() and (tmp_path / "tests" / "x").exists()


@needs_busybox
def test_real_busybox_resolves_the_nearest_existing_ancestor(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "lib").symlink_to(tmp_path / "real")
    (tmp_path / "app" / "dangling").symlink_to(tmp_path / "nowhere")
    run = host_script(RESOLVE_EXISTING)
    assert run(str(tmp_path / "app/lib/new/file")).stdout.decode().strip() == str(tmp_path / "real")
    assert run(str(tmp_path / "app/new/x")).stdout.decode().strip() == str(tmp_path / "app")
    # A dangling link resolves to its target, which the caller then checks.
    assert run(str(tmp_path / "app/dangling/x")).stdout.decode().strip() == str(tmp_path / "nowhere")


@needs_busybox
def test_real_busybox_archives_links_without_following_and_extracts_without_owners(tmp_path):
    import io
    import tarfile

    src = tmp_path / "src" / "app"
    src.mkdir(parents=True)
    (src / "a").write_bytes(b"a")
    (src / "out").symlink_to("/etc")
    member = str(src).lstrip("/")
    archived = host_script(TAR_CREATE)(member)
    assert archived.returncode == 0, archived.stderr
    members = {m.name: m for m in tarfile.open(fileobj=io.BytesIO(archived.stdout))}
    assert members[member + "/out"].issym() and members[member + "/out"].linkname == "/etc"
    assert not any(name.startswith(member + "/out/") for name in members)  # never followed

    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        info = tarfile.TarInfo("app/a")
        info.size, info.mode = 1, 0o755
        tar.addfile(info, io.BytesIO(b"a"))
    result = host_script(TAR_EXTRACT)(str(tmp_path / "dest"), stdin=data.getvalue())
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "dest" / "app" / "a").read_bytes() == b"a"


def test_clear_root_runs_the_helper_and_refuses_a_symlinked_root(runtime, monkeypatch):
    runtime.create("s-1", "s-1", spec())
    seen = capture_execs(runtime, monkeypatch)
    runtime.clear_root("s-1", "/app", timeout_s=4)
    assert seen[-1][0] == [HELPER, "sh", "-c", CLEAR_ROOT, "clear", "/app"]
    assert seen[-1][1]["timeout_s"] == 4 and seen[-1][1]["user"] == "1000:1000"
    capture_execs(runtime, monkeypatch, ExecResult(b"", b"", 3, False, False))
    with pytest.raises(InfraFault):
        runtime.clear_root("s-1", "/app")
    capture_execs(runtime, monkeypatch, ExecResult(b"", b"rm: denied", 1, False, False))
    with pytest.raises(OSError):
        runtime.clear_root("s-1", "/app")


HELPER_ROUTES = {
    "read": lambda r, **kw: r.read_file("s-1", "/x", 10, **kw),
    "write": lambda r, **kw: r.write_file("s-1", "/x", b"1", **kw),
    "kind": lambda r, **kw: r.file_kind("s-1", "/x", **kw),
    "mkdirs": lambda r, **kw: r.make_dirs("s-1", "/a/b", **kw),
}


@pytest.mark.parametrize("route", sorted(HELPER_ROUTES))
def test_a_helper_cut_by_the_callers_budget_times_out_without_blaming_the_box(
        runtime, monkeypatch, route):
    """The caller's remaining time, not the helper's own bound, stopped it: that
    is a timeout of the caller's step, never a BoxFault (it is not classified)."""
    runtime.create("s-1", "s-1", spec())
    capture_execs(runtime, monkeypatch, ExecResult(b"", b"", -1, True, False))
    monkeypatch.setattr(runtime, "classify_failure", lambda *a, **k: pytest.fail("classified"))
    with pytest.raises(HelperBudgetTimeout) as caught:
        HELPER_ROUTES[route](runtime, timeout_s=2.5)
    assert not isinstance(caught.value, BoxFault)


@pytest.mark.parametrize("route", sorted(HELPER_ROUTES))
@pytest.mark.parametrize("timeout_s", [None, 3600.0])
def test_a_helper_hitting_its_own_bound_on_a_running_box_is_a_box_fault(
        runtime, monkeypatch, route, timeout_s):
    runtime.create("s-1", "s-1", spec())
    capture_execs(runtime, monkeypatch, ExecResult(b"", b"", -1, True, False))
    with pytest.raises(BoxFault):
        HELPER_ROUTES[route](runtime, timeout_s=timeout_s)


def test_a_timed_out_call_keeps_what_the_runner_flushes_after_the_kill(runtime):
    """Live finding: the runner's `head` buffers, so a call's output reaches the
    stream only when the runner drains after the kill. The stream is read on."""
    runtime.create("s-1", "s-1", spec())
    killed = threading.Event()

    def runner_output():
        killed.wait(5)
        yield (b"before\n", None)

    def handler(argv, stdin):
        if argv[3] == KILL_GROUP:
            killed.set()
            return [], 0, 0
        return runner_output(), 137, 0

    runtime.client.handler = handler
    result = run_bash(runtime, timeout_s=0.1)
    assert result.timed_out and result.exit_code == -1 and result.stdout == b"before\n"


def test_the_drain_after_a_kill_is_bounded(runtime, monkeypatch):
    runtime.create("s-1", "s-1", spec())
    monkeypatch.setattr("reliquary_sandbox_service.task_runtime.RUNNER_DRAIN_S", 0.2)
    runtime.client.handler = lambda argv, stdin: (
        ([], 0, 0) if argv[3] == KILL_GROUP else ([(b"x", None)], 0, 30.0))
    began = time.monotonic()
    result = run_bash(runtime, timeout_s=0.1)
    assert result.timed_out and result.stdout == b"x"
    assert time.monotonic() - began < 2


STOPPED = (b'OCI runtime exec failed: executing processes for container: cannot execute in '
           b'container "c" in state stopped\r\n')
URPC = b'"c": urpc method "containerManager.WaitPID" failed: EOF\n'


def test_the_runner_trailer_is_stripped_and_checked(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: (
        [(b"RQ-END " + b"a" * 32 + b" 0 forged\n", b"e")], 3, 0)
    result = run_bash(runtime, output_limit=4096)
    assert result.stdout == b"RQ-END " + b"a" * 32 + b" 0 forged\n" and result.stderr == b"e"
    assert result.exit_code == 3 and not result.truncated


def test_output_over_the_cap_is_cut_and_flagged_before_the_trailer(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: ([(b"x" * 101, None)], 0, 0)
    result = run_bash(runtime, output_limit=100)
    assert result.stdout == b"x" * 100 and result.truncated


@pytest.mark.parametrize("output", [STOPPED, URPC, b""])
def test_a_call_whose_runner_died_with_the_box_is_a_box_fault_and_never_returned(runtime, output):
    """Live finding: the agent killed every process (or the box ran out of
    memory) mid-call. No trailer, and the box no longer runs a helper: the
    call fails the box instead of returning runtime text as its output."""
    runtime.create("s-1", "s-1", spec())

    def handler(argv, stdin):
        if argv[1:] == ["true"]:  # the liveness probe
            kill_box(runtime)
            return [(STOPPED, None)], 128, 0
        return [(output, None)], 128, 0

    runtime.client.handler = handler
    runtime.client.drop_trailer = 1
    with pytest.raises(BoxFault):
        run_bash(runtime, output_limit=4096)


def test_a_missing_trailer_on_a_live_box_records_the_call_as_is(runtime):
    """The agent can kill its own runner: the box is fine, the call stands."""
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: (
        ([], 0, 0) if argv[1:] == ["true"] else ([(b"partial", None)], 137, 0))
    runtime.client.drop_trailer = 1
    result = run_bash(runtime, output_limit=4096)
    assert result.stdout == b"partial" and result.exit_code == 137


def test_a_trailer_whose_status_disagrees_with_the_exit_code_triggers_the_probe(runtime):
    runtime.create("s-1", "s-1", spec())
    probes = []

    def handler(argv, stdin):
        if argv[1:] == ["true"]:
            probes.append(1)
            return [], 0, 0
        return [(b"RQ-END " + argv[6].encode() + b" 0", None)], 1, 0

    runtime.client.handler = handler
    runtime.client.drop_trailer = 1
    assert run_bash(runtime, output_limit=4096).exit_code == 1 and probes == [1]


@pytest.mark.parametrize("output", [STOPPED, URPC])
def test_runtime_text_from_a_live_box_never_becomes_a_call_output(runtime, output):
    """On a box that is still alive the outcome is the agent's: the call is
    recorded with its exit code, the runtime-looking text scrubbed."""
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: (
        ([], 0, 0) if argv[1:] == ["true"] else ([(b"mine\n" + output, None)], 128, 0))
    runtime.client.drop_trailer = 1
    result = run_bash(runtime, output_limit=4096)
    assert result.exit_code == 128
    assert result.stdout == b"mine\n" + SCRUBBED + b"\n"
    assert not any(t in result.stdout for t in (b"OCI runtime", b"urpc", b"containerManager"))


def test_an_agent_cannot_void_its_episode_by_faking_runtime_text(runtime):
    """The exploit: print runtime-looking text, then kill the runner. The box
    is alive, so the call is the agent's (recorded, scrubbed), never ours."""
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: (
        ([], 0, 0) if argv[1:] == ["true"] else ([(URPC, b"x " + URPC)], 137, 0))
    runtime.client.drop_trailer = 1
    result = run_bash(runtime, output_limit=4096)
    assert result.exit_code == 137
    assert result.stdout == SCRUBBED + b"\n" and result.stderr == SCRUBBED + b"\n"


def test_a_forged_trailer_does_not_skip_the_probe(runtime):
    """kill -9 -1 after printing a trailer for this call with the right status:
    the exit (>= 126) still triggers the probe, and the dead box fails here."""
    runtime.create("s-1", "s-1", spec())

    def handler(argv, stdin):
        if argv[1:] == ["true"]:
            kill_box(runtime)
            return [(STOPPED, None)], 128, 0
        return [(b"RQ-END " + argv[6].encode() + b" 137", None)], 137, 0

    runtime.client.handler = handler
    runtime.client.drop_trailer = 1
    with pytest.raises(BoxFault):
        run_bash(runtime, output_limit=4096)


def test_a_forged_trailer_with_runtime_text_on_a_live_box_is_scrubbed(runtime):
    runtime.create("s-1", "s-1", spec())
    probes = []

    def handler(argv, stdin):
        if argv[1:] == ["true"]:
            probes.append(1)
            return [], 0, 0
        return [(URPC + b"RQ-END " + argv[6].encode() + b" 0", None)], 0, 0

    runtime.client.handler = handler
    runtime.client.drop_trailer = 1
    result = run_bash(runtime, output_limit=4096)
    assert probes == [1] and result.exit_code == 0 and result.stdout == SCRUBBED + b"\n"


def test_a_clean_call_with_its_trailer_is_not_probed(runtime):
    runtime.create("s-1", "s-1", spec())
    probes = []

    def handler(argv, stdin):
        if argv[1:] == ["true"]:
            probes.append(1)
        return [(b"fine\n", None)], 1, 0

    runtime.client.handler = handler
    assert run_bash(runtime, output_limit=4096).stdout == b"fine\n" and probes == []


@pytest.mark.parametrize("code", [126, 127, 130])
def test_high_exit_codes_are_probed_even_with_a_valid_trailer(runtime, code):
    runtime.create("s-1", "s-1", spec())
    probes = []

    def handler(argv, stdin):
        if argv[1:] == ["true"]:
            probes.append(1)
            return [], 0, 0
        return [(b"x\n", None)], code, 0

    runtime.client.handler = handler
    assert run_bash(runtime, output_limit=4096).exit_code == code and probes == [1]


def test_the_runner_writes_its_trailer_last_on_its_own_stdout():
    assert BASH_RUNNER.rstrip().endswith('printf \'RQ-END %s %s\' "$id" "$status"\nexit "$status"')


def framed_execs(runtime, script):
    return [i for i in runtime.client.execs.values() if len(i["argv"]) > 3 and i["argv"][3] == script]


@pytest.mark.parametrize("loss", ["lose_output", "lose_prefix"])
def test_a_lost_helper_output_is_read_again(runtime, loss):
    """Output-carrying helpers are framed and re-run (idempotent) when a marker
    is missing: defence in depth (the client bug that lost output is fixed)."""
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: ([(b"value\n", None)], 0, 0)
    setattr(runtime.client, loss, 2)
    assert runtime.read_file("s-1", "/x", 100) == b"value\n"
    assert len(framed_execs(runtime, READ_FILE)) == 3


def test_an_empty_file_is_not_a_lost_output(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: ([], 0, 0)
    assert runtime.read_file("s-1", "/x", 100) == b""
    assert len(framed_execs(runtime, READ_FILE)) == 1


def test_output_lost_on_every_attempt_on_a_live_box_is_the_agents(runtime):
    """Our transport no longer loses output: on a box that still runs, a lost
    output is something the agent did (killed the helper, starved the box)."""
    runtime.create("s-1", "s-1", spec())
    runtime.client.lose_output = OUTPUT_ATTEMPTS
    with pytest.raises(BoxFault):
        runtime.read_file("s-1", "/x", 100)


def test_output_lost_on_a_box_the_runtime_failed_is_ours(runtime):
    runtime.create("s-1", "s-1", spec())

    def handler(argv, stdin):
        kill_box(runtime, Error="sentry panic")
        return [(b"value", None)], 0, 0

    runtime.client.handler = handler
    runtime.client.lose_output = OUTPUT_ATTEMPTS
    with pytest.raises(InfraFault) as caught:
        runtime.read_file("s-1", "/x", 100)
    assert not isinstance(caught.value, BoxFault)


def test_state_and_path_helpers_are_framed_and_retried(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: (
        [(b"/usr/bin\n" if argv[3] == RESOLVE_EXISTING else b"TAR", None)], 0, 0)
    runtime.client.lose_output = 1
    assert runtime.resolve_existing("s-1", "/usr/bin/x") == "/usr/bin"
    runtime.client.lose_output = 1
    assert runtime.tar_create("s-1", ["app"], 100) == b"TAR"
    assert {RESOLVE_EXISTING, TAR_CREATE, READ_FILE} <= set(FRAMED_SCRIPTS)


def test_a_lost_user_lookup_is_asked_again(runtime):
    calls = []

    def handler(argv, stdin):
        if argv[1:] == ["id"]:
            calls.append(1)
            if len(calls) == 1:
                return [], 0, 0  # the output was lost
        return default_handler(argv, stdin)

    runtime.client.handler = handler
    runtime.create("s-1", "s-1", spec())
    assert runtime.box_user("s-1") == "1000:1000" and len(calls) == 2


def test_programs_run_framed_as_given(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: ([(b"out", b"err")], 3, 0)
    result = runtime.run_program("s-1", ["git", "diff", "a b"], timeout_s=5, output_limit=100,
                                 env={"A": "1"}, workdir="/testbed", user="1000:1000")
    assert (result.stdout, result.stderr, result.exit_code) == (b"out", b"err", 3)
    item = list(runtime.client.execs.values())[-1]
    assert without_nonce(item["argv"]) == [HELPER, "sh", "-c", RUN_PROGRAM, "run", "git", "diff", "a b"]
    assert item["kwargs"]["workdir"] == "/testbed" and item["kwargs"]["user"] == "1000:1000"
    assert item["kwargs"]["environment"] == {"A": "1"}


def test_a_lost_program_output_on_a_live_box_is_the_agents_not_an_empty_result(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: ([(b"diff --git a b", None)], 0, 0)
    runtime.client.lose_output = 1
    with pytest.raises(BoxFault):
        runtime.run_program("s-1", ["git", "diff"], timeout_s=5, output_limit=100, env={},
                            workdir="/", user="0")
    # Not re-run: a program may have side effects.
    assert len([i for i in runtime.client.execs.values() if i["argv"][3:4] == [RUN_PROGRAM]]) == 1


@needs_busybox
def test_real_busybox_runs_the_images_program_not_its_own_applet(tmp_path):
    """busybox-static prefers its applets: `ls` in its sh is busybox ls. The
    program runner resolves the name on PATH itself and runs that file."""
    tool = tmp_path / "bin" / "ls"
    tool.parent.mkdir()
    tool.write_text("#!/bin/sh\nprintf '%s|' 'image ls' \"$@\"\n")
    tool.chmod(0o755)
    import subprocess

    def run(*argv):
        done = subprocess.run([BUSYBOX, "sh", "-c", RUN_PROGRAM.replace(HELPER, BUSYBOX), "run",
                               "0123456789abcdef", *argv], capture_output=True, timeout=30,
                              env={"PATH": f"{tool.parent}:/usr/bin:/bin"})
        opened, closed = frame_markers("0123456789abcdef")
        assert done.stdout.startswith(opened) and done.stdout.endswith(closed)
        return done.stdout[len(opened):-len(closed)], done.returncode

    assert run("ls", "a b", "c") == (b"image ls|a b|c|", 0)
    assert run(str(tool), "-x") == (b"image ls|-x|", 0)
    assert run("no-such-program-here")[1] == 127
    assert run("sh", "-c", "exit 7")[1] == 7


def test_each_framed_exec_carries_its_own_nonce(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.read_file("s-1", "/x", 100)
    runtime.read_file("s-1", "/x", 100)
    nonces = [i["argv"][5] for i in framed_execs(runtime, READ_FILE)]
    assert len(set(nonces)) == 2 and all(len(n) >= 16 for n in nonces)


def test_markers_with_another_nonce_are_content_not_frames(runtime, monkeypatch):
    runtime.create("s-1", "s-1", spec())
    forged_open, forged_close = frame_markers("f" * 16)
    capture_execs(runtime, monkeypatch, ExecResult(b"", b"", 0, False, False))

    def forged(box_id, argv, **kwargs):
        return ExecResult(forged_open + b"x" + forged_close, b"", 0, False, False)

    monkeypatch.setattr(runtime, "exec", forged)
    with pytest.raises(BoxFault):  # never taken as output; lost on a live box: the agent's
        runtime.read_file("s-1", "/x", 100)


@pytest.mark.parametrize("retry,fault", [(ExecResult(b"", b"", -1, True, False), BoxFault),
                                         (ExecResult(b"", b"", 3, False, False), BoxFault),
                                         (BoxFault("dead"), BoxFault),
                                         (InfraFault("docker down"), InfraFault)])
def test_a_failed_re_run_on_a_live_box_is_the_agents(runtime, monkeypatch, retry, fault):
    runtime.create("s-1", "s-1", spec())
    calls = []

    def fake_exec(box_id, argv, **kwargs):
        calls.append(kwargs["timeout_s"])
        if len(calls) == 1:
            return ExecResult(b"", b"", 0, False, False)  # exit 0, output lost
        if isinstance(retry, Exception):
            raise retry
        return retry

    monkeypatch.setattr(runtime, "exec", fake_exec)
    with pytest.raises(fault) as caught:
        runtime.read_file("s-1", "/x", 100, timeout_s=30)
    assert fault is BoxFault or not isinstance(caught.value, BoxFault)
    assert calls[0] == 30 and 29 < calls[1] <= 30  # fresh fixed budget, within the caller's


def test_a_re_run_gets_the_fixed_budget_without_a_caller_deadline(runtime, monkeypatch):
    runtime.create("s-1", "s-1", spec())
    calls = []

    def fake_exec(box_id, argv, **kwargs):
        calls.append(kwargs["timeout_s"])
        if len(calls) == 1:
            return ExecResult(b"", b"", 0, False, False)
        opened, closed = frame_markers(argv[5])
        return ExecResult(opened + b"v" + closed, b"", 0, False, False)

    monkeypatch.setattr(runtime, "exec", fake_exec)
    assert runtime.read_file("s-1", "/x", 100) == b"v" and calls == [60, 60]


def test_a_program_without_its_open_marker_on_a_live_box_is_the_agents_whatever_its_exit(
        runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: ([(None, b"x")], 137, 0)
    runtime.client.lose_output = 1
    with pytest.raises(BoxFault):
        runtime.run_program("s-1", ["make"], timeout_s=5, output_limit=100, env={},
                            workdir="/", user="0")


def test_a_program_whose_box_the_runtime_failed_is_ours(runtime):
    runtime.create("s-1", "s-1", spec())

    def handler(argv, stdin):
        kill_box(runtime, Error="sentry panic")
        return [(b"diff", None)], 0, 0

    runtime.client.handler = handler
    runtime.client.lose_output = 1
    with pytest.raises(InfraFault) as caught:
        runtime.run_program("s-1", ["make"], timeout_s=5, output_limit=100, env={},
                            workdir="/", user="0")
    assert not isinstance(caught.value, BoxFault)


# -- stopping the agent's processes before extract ---------------------------


def test_stopping_the_agents_processes_runs_the_helper_as_root_from_slash(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.stop_processes("s-1", timeout_s=20)
    item = list(runtime.client.execs.values())[-1]
    assert item["argv"][:4] == [HELPER, "sh", "-c", STOP_PROCESSES]
    assert item["kwargs"]["user"] == "0" and item["kwargs"]["workdir"] == "/"


def test_the_stop_script_spares_pid_1_itself_and_the_oldest_child_of_pid_1():
    """Never run on this host (it would signal host processes): its rules are
    pinned here and exercised live (test_episode_live)."""
    assert '"$p" = 1' in STOP_PROCESSES and '"$p" = "$$"' in STOP_PROCESSES
    assert "kill -s KILL" in STOP_PROCESSES and "/proc/" in STOP_PROCESSES
    assert "${20}" in STOP_PROCESSES  # field 22 (starttime) after the command name


@needs_busybox
def test_the_stop_script_selects_every_process_but_pid_1_itself_and_the_entrypoint():
    """Dry run on this host's /proc with `kill` replaced by `echo` (nothing is
    signalled): the selection logic, field parsing included, under real busybox."""
    import subprocess

    dry = STOP_PROCESSES.replace('$B kill -s KILL "$p" 2>/dev/null', 'echo "$p"')
    dry = dry.replace('-lt 20', '-lt 1').replace(HELPER, BUSYBOX)
    assert "kill -s" not in dry and " kill " not in dry
    done = subprocess.Popen([BUSYBOX, "sh", "-c", dry, "stop"], stdout=subprocess.PIPE)
    out, _ = done.communicate(timeout=60)
    listed = {int(line) for line in out.split()}
    children = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if fields[1] == "1":
            children.append((int(fields[19]), int(stat.parent.name)))
    entrypoint = min(children)[1]
    assert done.returncode == 3 and len(listed) > 5
    assert not listed & {1, done.pid, entrypoint}


@pytest.mark.parametrize("result,fault", [(([], 126, 0), BoxFault), (([], 3, 0), BoxFault)])
def test_a_stop_that_fails_on_a_live_box_is_the_agents(runtime, result, fault):
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: result
    with pytest.raises(fault):
        runtime.stop_processes("s-1", timeout_s=20)


def test_a_stop_cut_by_the_callers_budget_is_a_budget_timeout(runtime):
    runtime.create("s-1", "s-1", spec())
    runtime.client.handler = lambda argv, stdin: ([], 0, 2.0)
    with pytest.raises(HelperBudgetTimeout):
        runtime.stop_processes("s-1", timeout_s=0.2)


# -- probing a live box before paying the settle period ----------------------


def test_a_live_box_is_probed_without_the_settle_period(runtime, monkeypatch):
    runtime.create("s-1", "s-1", spec())
    settled = []
    monkeypatch.setattr(runtime, "_settled", lambda container: settled.append(1) or True)
    assert runtime._alive("s-1") and settled == []


def test_the_settle_period_is_paid_only_when_the_probe_fails(runtime, monkeypatch):
    runtime.create("s-1", "s-1", spec())
    settled = []
    monkeypatch.setattr(runtime, "_settled", lambda container: settled.append(1) or True)
    runtime.client.handler = lambda argv, stdin: ([], 1, 0)
    assert not runtime._alive("s-1") and settled == [1]


def test_a_program_that_could_not_start_keeps_the_oci_text(runtime, monkeypatch):
    runtime.create("s-1", "s-1", spec())
    capture_execs(runtime, monkeypatch)
    monkeypatch.setattr(runtime, "exec", lambda *a, **k: ExecResult(STOPPED, b"", 126, False, False))
    result = runtime.run_program("s-1", ["make"], timeout_s=5, output_limit=4096, env={},
                                 workdir="/", user="0")
    assert result.exit_code == 126


def test_preflight_refuses_a_daemon_not_on_a_unix_socket(runtime):
    runtime.exec_transport = None
    with pytest.raises(RuntimeUnavailable, match="Unix socket"):
        runtime.preflight()


def test_an_exec_abandoned_during_the_handshake_closes_its_socket(runtime):
    """The socket is registered for `_abort` before the handshake: a daemon that
    never answers the start cannot keep the socket (or the thread) alive."""
    runtime.create("s-1", "s-1", spec())
    ended = threading.Event()
    sockets = []

    class SilentDaemon:
        def start(self, exec_id, on_connect=None):
            ours, theirs = socket_module.socketpair()
            sockets.append((ours, theirs))
            on_connect(ours)
            try:
                ours.recv(1)  # the daemon never answers
                raise ConnectionResetError("closed")
            finally:
                ended.set()

    runtime.exec_transport = SilentDaemon()
    result = runtime.exec("s-1", ["true"], timeout_s=0.2, output_limit=10)
    assert result.timed_out
    assert ended.wait(2), "the handshake was not interrupted"
    assert sockets[0][0].fileno() == -1 and runtime._connections == {}


# -- the runtime's generic start failure (runsc exit 128 on a loaded host) ----------

RUNSC_START = (b"OCI runtime exec failed: /usr/local/bin/runsc did not terminate "
               b"successfully: exit status 128\r\n")
"""What Docker writes on the exec stream (`"%v\\r\\n"`, the text dockerd logged for each
of the 13 failures seen live); the exec's exit code is 128 (Docker >= 24,
setExitCodeFromError's exitUnknown) or 126 (older: the deferred default)."""
START_EXIT = 128


@pytest.fixture
def no_pause(monkeypatch):
    monkeypatch.setattr("reliquary_sandbox_service.task_runtime.RUNTIME_START_PAUSE_S", 0.0)


def runsc_fails(runtime, times, *, script=READ_FILE, output=RUNSC_START, exit_code=START_EXIT,
                stderr=None):
    """The first `times` runs of `script` fail to start; the probe and the rest run."""
    left = [times]

    def handler(argv, stdin):
        if len(argv) > 3 and argv[3] == script and left[0]:
            left[0] -= 1
            return [(output, stderr)], exit_code, 0, UNSTARTED
        return default_handler(argv, stdin)

    runtime.client.handler = handler
    return left


def runs_of(runtime, script):
    return sum(1 for item in runtime.client.execs.values()
               if len(item["argv"]) > 3 and item["argv"][3] == script)


def test_a_read_the_runtime_failed_to_start_is_run_again_and_done(runtime, no_pause):
    runtime.create("s-1", "s-1", spec())
    runsc_fails(runtime, 2)
    assert runtime.read_file("s-1", "/x", 100) == b"ok\n"
    assert runs_of(runtime, READ_FILE) == 3


@pytest.mark.parametrize("path", [b"/usr/bin/runsc", b"runsc",
                                  b"/var/lib/docker/runtimes/runsc.MFRGGZDFMZTWQ2LK===="])
def test_the_start_failure_names_runsc_or_dockers_runtime_args_wrapper(runtime, no_pause, path):
    runtime.create("s-1", "s-1", spec())
    runsc_fails(runtime, 2, output=RUNSC_START.replace(b"/usr/local/bin/runsc", path))
    assert runtime.read_file("s-1", "/x", 100) == b"ok\n"
    assert runs_of(runtime, READ_FILE) == 3


def test_a_read_the_runtime_never_starts_on_a_live_box_is_a_runtime_exec_failure(
        runtime, no_pause):
    runtime.create("s-1", "s-1", spec())
    runsc_fails(runtime, 99)
    with pytest.raises(RuntimeExecFailed) as caught:
        runtime.read_file("s-1", "/x", 100)
    assert isinstance(caught.value, BoxFault)  # every caller not looking for it: as before
    assert runs_of(runtime, READ_FILE) == 3


def test_a_write_the_runtime_failed_to_start_is_never_run_again(runtime, no_pause):
    runtime.create("s-1", "s-1", spec())
    runsc_fails(runtime, 1, script=WRITE_FILE)
    with pytest.raises(RuntimeExecFailed):
        runtime.write_file("s-1", "/x", b"1")
    assert runs_of(runtime, WRITE_FILE) == 1


@pytest.mark.parametrize("output", [
    b"OCI runtime exec failed: chdir: no such file or directory\r\n",  # runsc's reason
    b"OCI runtime exec failed: executing processes for container: cannot execute in "
    b"container in state stopped\r\n"])
def test_a_start_failure_runsc_gives_a_reason_for_stays_the_agents(runtime, no_pause, output):
    runtime.create("s-1", "s-1", spec())
    runsc_fails(runtime, 99, output=output)
    with pytest.raises(BoxFault) as caught:
        runtime.read_file("s-1", "/x", 100)
    assert not isinstance(caught.value, RuntimeExecFailed)
    assert runs_of(runtime, READ_FILE) == 1  # nothing to run again


def test_a_start_failure_at_the_pids_limit_stays_the_agents(runtime, no_pause):
    runtime.create("s-1", "s-1", spec())
    runtime.client.stats_value = {"pids_stats": {"current": 512, "limit": 512},
                                  "cpu_stats": {"cpu_usage": {"total_usage": 1}}}
    runsc_fails(runtime, 99)
    with pytest.raises(BoxFault) as caught:
        runtime.read_file("s-1", "/x", 100)
    assert not isinstance(caught.value, RuntimeExecFailed)
    assert "pids limit" in str(caught.value)


def test_a_start_failure_on_a_box_that_runs_no_helper_stays_the_agents(runtime, no_pause):
    runtime.create("s-1", "s-1", spec())

    def handler(argv, stdin):
        return [(RUNSC_START, None)], START_EXIT, 0, UNSTARTED  # the probe fails to start too

    runtime.client.handler = handler
    with pytest.raises(BoxFault) as caught:
        runtime.read_file("s-1", "/x", 100)
    assert not isinstance(caught.value, RuntimeExecFailed)


def test_a_start_failure_on_a_stopped_box_stays_the_agents(runtime, no_pause):
    runtime.create("s-1", "s-1", spec())

    def handler(argv, stdin):
        if argv[1:] == ["true"]:
            return [], 0, 0  # even a probe that answers does not make it ours
        kill_box(runtime)  # PID 1 exited, State.Error empty: the agent's
        return [(RUNSC_START, None)], START_EXIT, 0, UNSTARTED

    runtime.client.handler = handler
    with pytest.raises(BoxFault) as caught:
        runtime.read_file("s-1", "/x", 100)
    assert not isinstance(caught.value, RuntimeExecFailed)


def test_a_program_the_runtime_failed_to_start_raises_and_is_not_run_again(runtime, no_pause):
    runtime.create("s-1", "s-1", spec())
    runsc_fails(runtime, 1, script=RUN_PROGRAM)  # the wrapper never ran: no open marker
    with pytest.raises(RuntimeExecFailed):
        runtime.run_program("s-1", ["make"], timeout_s=5, output_limit=100, env={},
                            workdir="/", user="0")
    assert runs_of(runtime, RUN_PROGRAM) == 1


def test_a_program_that_could_not_chdir_still_returns_its_failure(runtime, no_pause):
    runtime.create("s-1", "s-1", spec())
    runsc_fails(runtime, 1, script=RUN_PROGRAM, exit_code=127,  # isInvalidCommand's code
                output=b"OCI runtime exec failed: chdir: no such file or directory\r\n")
    result = runtime.run_program("s-1", ["make"], timeout_s=5, output_limit=100, env={},
                                 workdir="/gone", user="0")
    assert result.exit_code == 127 and b"chdir" in result.stdout


def test_an_exec_api_error_with_the_generic_start_failure_is_a_runtime_exec_failure(runtime):
    runtime.create("s-1", "s-1", spec())
    real = runtime.client.api.exec_create

    def create(container_id, argv, **kwargs):
        if argv[1:] != ["true"]:  # the probe starts
            raise api_error(RUNSC_START.decode().strip(), 500)
        return real(container_id, argv, **kwargs)

    runtime.client.api.exec_create = create
    with pytest.raises(RuntimeExecFailed):
        runtime.exec("s-1", ["x"], timeout_s=5, output_limit=10)


@pytest.mark.parametrize("code", [126, 128])
def test_both_codes_docker_gives_the_start_failure_are_recognised(runtime, no_pause, code):
    runtime.create("s-1", "s-1", spec())
    runsc_fails(runtime, 1, exit_code=code)
    assert runtime.read_file("s-1", "/x", 100) == b"ok\n"
    assert runs_of(runtime, READ_FILE) == 2


@pytest.mark.parametrize("code", [1, 125, 127, 137])
def test_the_message_with_another_exit_code_stays_the_agents(runtime, no_pause, code):
    runtime.create("s-1", "s-1", spec())
    runsc_fails(runtime, 99, exit_code=code)
    with pytest.raises(BoxFault) as caught:
        runtime.read_file("s-1", "/x", 100)
    assert not isinstance(caught.value, RuntimeExecFailed)
    assert runs_of(runtime, READ_FILE) == 1


@pytest.mark.parametrize("output", [
    # a reason after the generic text (a shim that appends runsc's stderr)
    RUNSC_START.rstrip() + b": open /proc/self/fd: permission denied\r\n",
    b"x" + RUNSC_START,  # anything before it
    RUNSC_START + b"more\n",  # anything after it
    RUNSC_START.replace(b"/usr/local/bin/runsc", b"/usr/bin/runc"),  # not our runtime
    RUNSC_START.replace(b"/usr/local/bin/runsc", b"/usr/local/bin/xrunsc"),
    RUNSC_START.replace(b"/usr/local/bin/runsc", b"/usr/local/bin/runsc.x y")])
def test_only_dockers_exact_message_is_the_start_failure(runtime, no_pause, output):
    runtime.create("s-1", "s-1", spec())
    runsc_fails(runtime, 99, output=output)
    _only_once(runtime)


def test_the_exact_message_with_anything_on_stderr_is_not_the_start_failure(runtime, no_pause):
    runtime.create("s-1", "s-1", spec())
    runsc_fails(runtime, 99, stderr=b"tar: x\n")  # Docker writes it on stdout alone
    _only_once(runtime)


def _only_once(runtime):
    with pytest.raises(BoxFault) as caught:
        runtime.read_file("s-1", "/x", 100)
    assert not isinstance(caught.value, RuntimeExecFailed)
    assert runs_of(runtime, READ_FILE) == 1


FORGED_NAMES = [b"OCI runtime exec failed: x did not terminate successfully",
                RUNSC_START.rstrip()]
"""File names an agent can give its files (colons and spaces are legal)."""


def tar_refuses(runtime, name, *, script=TAR_CREATE):
    """`script`'s tar prints its output as usual but fails on one file, naming it
    (busybox: `tar: can't open '<name>': Permission denied`, exit 1)."""
    def handler(argv, stdin):
        if len(argv) > 3 and argv[3] == script:
            return ([(b"tar-bytes", None), (None, b"tar: can't open '" + name
                                              + b"': Permission denied\n")], 1, 0)
        return default_handler(argv, stdin)

    runtime.client.handler = handler


@pytest.mark.parametrize("name", FORGED_NAMES)
def test_a_forged_file_name_in_the_archive_stays_a_plain_box_fault_run_once(
        runtime, no_pause, name):
    # The agent's box at extract: an unreadable file named like the runtime's failure.
    runtime.create("s-1", "s-1", spec())
    tar_refuses(runtime, name)
    with pytest.raises(BoxFault) as caught:
        runtime.tar_create("s-1", ["app"], 1000)
    assert not isinstance(caught.value, RuntimeExecFailed)
    assert runs_of(runtime, TAR_CREATE) == 1


@pytest.mark.parametrize("name", FORGED_NAMES)
def test_a_forged_file_name_in_a_grading_restore_stays_a_plain_box_fault_run_once(
        runtime, no_pause, name):
    # A grading box: a restored member tar refuses, its name forged.
    runtime.create("g-1", "g-1", spec())
    tar_refuses(runtime, name, script=TAR_EXTRACT)
    with pytest.raises(BoxFault) as caught:
        runtime.tar_extract("g-1", b"tar", "/work")
    assert not isinstance(caught.value, RuntimeExecFailed)
    assert runs_of(runtime, TAR_EXTRACT) == 1


def test_a_helper_that_ran_and_printed_the_exact_message_is_not_a_start_failure(
        runtime, no_pause):
    # The helper started (its open marker is there), whatever it printed and exited with.
    runtime.create("s-1", "s-1", spec())

    def handler(argv, stdin):
        if len(argv) > 3 and argv[3] == READ_FILE:
            return [(RUNSC_START, None)], START_EXIT, 0
        return default_handler(argv, stdin)

    runtime.client.handler = handler
    with pytest.raises(BoxFault) as caught:
        runtime.read_file("s-1", "/x", 100)
    assert not isinstance(caught.value, RuntimeExecFailed)
    assert runs_of(runtime, READ_FILE) == 1


def test_a_scoring_program_printing_the_exact_message_gets_its_own_result(runtime, no_pause):
    # Agent code under test in a grading box: exit 128 and Docker's text, behind the marker.
    runtime.create("g-1", "g-1", spec())

    def handler(argv, stdin):
        if len(argv) > 3 and argv[3] == RUN_PROGRAM:
            return [(RUNSC_START, None)], START_EXIT, 0
        return default_handler(argv, stdin)

    runtime.client.handler = handler
    result = runtime.run_program("g-1", ["pytest"], timeout_s=5, output_limit=1000, env={},
                                 workdir="/", user="0")
    assert result.exit_code == START_EXIT and result.stdout.startswith(b"OCI runtime")
    assert runs_of(runtime, RUN_PROGRAM) == 1


def test_an_api_error_with_a_reason_after_the_generic_text_stays_the_agents(runtime):
    runtime.create("s-1", "s-1", spec())
    real = runtime.client.api.exec_create

    def create(container_id, argv, **kwargs):
        if argv[1:] != ["true"]:
            raise api_error(RUNSC_START.decode().strip() + ": chdir: permission denied", 500)
        return real(container_id, argv, **kwargs)

    runtime.client.api.exec_create = create
    with pytest.raises(BoxFault) as caught:
        runtime.exec("s-1", ["x"], timeout_s=5, output_limit=10)
    assert not isinstance(caught.value, RuntimeExecFailed)


def first_read_fails_to_start(runtime, then):
    """READ_FILE: its first run fails to start, the next returns `then`; the bounds each
    run was given are recorded."""
    real, bounds = runtime.exec, []

    def exec_(box_id, argv, *, timeout_s, **kwargs):
        if len(argv) > 3 and argv[3] == READ_FILE:
            bounds.append(timeout_s)
            if len(bounds) == 1:
                return ExecResult(RUNSC_START, b"", START_EXIT, False, False)
            return then
        return real(box_id, argv, timeout_s=timeout_s, **kwargs)

    runtime.exec = exec_
    return bounds


def test_a_retry_after_a_start_failure_gets_the_helpers_whole_bound(runtime, no_pause):
    # Timing out with its whole bound is the box's (as on a first run), never a cut one.
    runtime.create("s-1", "s-1", spec())
    bounds = first_read_fails_to_start(runtime, ExecResult(b"", b"", -1, True, False))
    with pytest.raises(BoxFault):
        runtime.read_file("s-1", "/x", 100)
    assert bounds == [60, 60]


def test_a_retry_the_callers_deadline_cut_is_the_callers_timeout_not_the_box(
        runtime, no_pause):
    runtime.create("s-1", "s-1", spec())
    bounds = first_read_fails_to_start(runtime, ExecResult(b"", b"", -1, True, False))
    with pytest.raises(HelperBudgetTimeout):
        runtime.read_file("s-1", "/x", 100, timeout_s=60)  # the helper's bound, at first
    assert bounds[0] == 60 and bounds[1] < 60
