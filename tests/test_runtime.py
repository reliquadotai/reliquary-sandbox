from __future__ import annotations

import io
import subprocess
import sys
import threading
import zipfile
from types import SimpleNamespace

import pytest
from docker.errors import ImageNotFound, NotFound

from reliquary_sandbox_service.runtime import (
    DockerRuntime,
    RuntimeOperationError,
    RuntimeUnavailable,
)


class Container:
    def __init__(self, sandbox_id="test"):
        self.id = "container-id"
        self.labels = {DockerRuntime.LABEL: "true", DockerRuntime.ID_LABEL: sandbox_id}
        self.status = "created"
        self.removed = threading.Event()
        self.remove_calls = []

    def start(self):
        self.status = "running"

    def reload(self):
        pass

    def remove(self, **kwargs):
        self.remove_calls.append(kwargs)
        self.status = "deleted"
        self.removed.set()


class Client:
    def __init__(self):
        self.container = Container()
        self.creation = None
        self.available_images = {"guest:local"}
        self.server_info = {"OSType": "linux", "Runtimes": {"runsc": {}}}
        self.chunks = [(b"hello", b"warning")]
        self.exec_arguments = None
        self.block = False
        self.containers = SimpleNamespace(create=self.create, get=self.get, list=self.list)
        self.images = SimpleNamespace(get=self.image)
        self.api = SimpleNamespace(
            exec_create=self.exec_create,
            exec_start=self.exec_start,
            exec_inspect=lambda _: {"Running": False, "ExitCode": 0},
        )

    def info(self):
        return self.server_info

    def image(self, image):
        if image not in self.available_images:
            raise ImageNotFound("missing image")
        return SimpleNamespace(id="image-id")

    def create(self, **kwargs):
        self.creation = kwargs
        self.container.labels = kwargs["labels"]
        return self.container

    def get(self, name):
        if self.container.removed.is_set():
            raise NotFound("removed")
        return self.container

    def list(self, **kwargs):
        return [] if self.container.removed.is_set() else [self.container]

    def exec_create(self, container_id, command, **kwargs):
        self.exec_arguments = (container_id, command, kwargs)
        return {"Id": "exec-id"}

    def exec_start(self, exec_id, **kwargs):
        if self.block:
            self.container.removed.wait(3)
        yield from self.chunks


@pytest.fixture
def settings():
    return SimpleNamespace(
        docker_runtime="runsc",
        require_gvisor=True,
        default_image="guest:local",
        allowed_images=("guest:local",),
        memory_limit="512m",
        nano_cpus=1_000_000_000,
        workspace_size_bytes=268435456,
        tmp_size_bytes=67108864,
        pids_limit=128,
        default_uid=1000,
        max_file_bytes=262144,
        max_output_bytes=1048576,
        max_snapshot_bytes=524288,
        max_snapshot_files=100,
    )


@pytest.fixture
def runtime(settings):
    return DockerRuntime(settings, client=Client())


def test_creation_applies_isolation_and_resource_bounds(runtime):
    result = runtime.create("test", {"cpu_cores": 0.5, "memory_gb": 0.25, "disk_size_gb": 0.125})
    assert result == "container-id"
    config = runtime.client.creation
    assert config["runtime"] == "runsc"
    assert config["network_mode"] == "none"
    assert config["read_only"] is True
    assert config["user"] == "1000:1000"
    assert config["cap_drop"] == ["ALL"]
    assert config["security_opt"] == ["no-new-privileges:true"]
    assert config["privileged"] is False
    assert config["nano_cpus"] == 500_000_000
    assert config["mem_limit"] == config["memswap_limit"] == 268435456
    assert config["pids_limit"] == 128
    assert "size=134217728" in config["tmpfs"]["/workspace"]
    assert config["log_config"]["Type"] == "none"
    assert "volumes" not in config and "mounts" not in config


@pytest.mark.parametrize(
    "info",
    [
        {"OSType": "windows", "Runtimes": {"runsc": {}}},
        {"OSType": "linux", "Runtimes": {"runc": {}}},
    ],
)
def test_readiness_fails_without_required_isolation(runtime, info):
    runtime.client.server_info = info
    with pytest.raises(RuntimeUnavailable):
        runtime.readiness()


def test_readiness_requires_images_present_locally(runtime):
    runtime.client.available_images.clear()
    with pytest.raises(RuntimeUnavailable):
        runtime.readiness()


@pytest.mark.parametrize(
    "spec",
    [
        {"image": "unknown:image"},
        {"cpu_cores": 100},
        {"memory_gb": 100},
        {"disk_size_gb": 100},
        {"cpu_cores": float("nan")},
        {"memory_gb": 0},
    ],
)
def test_invalid_request_never_creates_a_container(runtime, spec):
    with pytest.raises(ValueError):
        runtime.create("test", spec)
    assert runtime.client.creation is None


def test_execute_keeps_streams_and_limits_output(runtime):
    runtime.settings.max_output_bytes = 8
    result = runtime.execute("test", "printf hello", 1)
    assert result == {
        "stdout": "hello",
        "stderr": "war",
        "exit_code": 0,
        "timed_out": False,
        "truncated": True,
    }
    _, command, config = runtime.client.exec_arguments
    assert command == ["/bin/bash", "-lc", "printf hello"]
    assert config["user"] == "1000:1000"
    assert config["workdir"] == "/workspace"


def test_timeout_removes_entire_container(runtime):
    runtime.client.block = True
    result = runtime.execute("test", "sleep 60", 0.01)
    assert result["timed_out"] is True
    assert result["exit_code"] == -1
    assert runtime.client.container.remove_calls == [{"force": True}]
    assert runtime.status("test") == "deleted"


def test_reconciliation_uses_ownership_labels(runtime):
    assert runtime.managed_sandboxes() == {"test": "container-id"}
    runtime.client.container.labels[DockerRuntime.LABEL] = "false"
    with pytest.raises(RuntimeOperationError):
        runtime.execute("test", "true", 1)


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "/workspace/../file",
        "/workspace//file",
        "/workspace/./file",
        "/workspace",
        "/workspace/file\x00",
    ],
)
def test_file_paths_are_workspace_relative(runtime, path):
    with pytest.raises(ValueError):
        runtime.upload("test", path, b"data")


@pytest.fixture
def guest_files(runtime, tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def guest_run(sandbox_id, command, timeout, **kwargs):
        # The test substitutes a temporary directory for the guest's mount.
        helper_index = command.index("-c") + 1
        helper = command[helper_index].replace(
            "os.open('/workspace',", f"os.open({str(workspace)!r},", 1
        )
        proc = subprocess.run(
            [sys.executable, "-I", "-S", "-c", helper, *command[helper_index + 1 :]],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        limit = kwargs.get("output_limit", runtime.settings.max_output_bytes)
        return {
            "stdout": proc.stdout[:limit],
            "stderr": proc.stderr[:limit],
            "exit_code": proc.returncode,
            "timed_out": False,
            "truncated": len(proc.stdout) + len(proc.stderr) > limit,
        }

    monkeypatch.setattr(runtime, "_run", guest_run)
    return runtime, workspace


def test_chunked_upload_and_binary_download(guest_files):
    runtime, workspace = guest_files
    data = bytes(range(256)) * 600
    runtime.upload("test", "/workspace/nested/binary.dat", data)
    assert runtime.download("test", "/workspace/nested/binary.dat") == data
    assert (workspace / "nested" / "binary.dat").read_bytes() == data
    assert not list((workspace / "nested").glob(".transfer-*"))


def test_empty_file_transfer(guest_files):
    runtime, _ = guest_files
    runtime.upload("test", "/workspace/empty", b"")
    assert runtime.download("test", "/workspace/empty") == b""


def test_file_size_is_bounded(guest_files):
    runtime, workspace = guest_files
    with pytest.raises(ValueError):
        runtime.upload("test", "/workspace/large", b"x" * 262145)
    (workspace / "large").write_bytes(b"x" * 262145)
    with pytest.raises(ValueError):
        runtime.download("test", "/workspace/large")


def test_files_reject_symlink_targets_and_parents(guest_files, tmp_path):
    runtime, workspace = guest_files
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_bytes(b"unchanged")
    (workspace / "linked-file").symlink_to(outside / "file")
    (workspace / "linked-dir").symlink_to(outside, target_is_directory=True)
    for path in ("/workspace/linked-file", "/workspace/linked-dir/file"):
        with pytest.raises(ValueError):
            runtime.upload("test", path, b"changed")
        with pytest.raises(ValueError):
            runtime.download("test", path)
    assert (outside / "file").read_bytes() == b"unchanged"


def test_workspace_checkpoint_roundtrip(guest_files):
    runtime, workspace = guest_files
    runtime.upload("test", "/workspace/nested/result.txt", b"evaluation output")
    (workspace / "empty").mkdir()
    (workspace / "nested" / "result.txt").chmod(0o700)
    archive = runtime.export_workspace("test")
    (workspace / "nested" / "result.txt").unlink()
    (workspace / "nested").rmdir()
    (workspace / "empty").rmdir()
    runtime.import_workspace("test", archive)
    assert runtime.download("test", "/workspace/nested/result.txt") == b"evaluation output"
    assert (workspace / "empty").is_dir()
    assert (workspace / "nested" / "result.txt").stat().st_mode & 0o100
    assert not list(workspace.glob(".restore-*"))


def test_checkpoint_rejects_unsupported_files(guest_files):
    runtime, workspace = guest_files
    (workspace / "linked").symlink_to("/etc/passwd")
    with pytest.raises(ValueError):
        runtime.export_workspace("test")


def test_checkpoint_limits_file_count(guest_files):
    runtime, workspace = guest_files
    runtime.settings.max_snapshot_files = 1
    (workspace / "one").write_bytes(b"1")
    (workspace / "two").write_bytes(b"2")
    with pytest.raises(ValueError):
        runtime.export_workspace("test")


@pytest.mark.parametrize("member", ["../outside", "/outside", "nested/../../outside"])
def test_restore_rejects_invalid_members(guest_files, member):
    runtime, _ = guest_files
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(member, b"content")
    with pytest.raises(ValueError):
        runtime.import_workspace("test", buffer.getvalue())


def test_restore_rejects_compressed_entries(guest_files):
    runtime, _ = guest_files
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("file", b"x" * 100)
    with pytest.raises(ValueError):
        runtime.import_workspace("test", buffer.getvalue())
