"""Bounded Docker execution with a fail-closed gVisor configuration."""

from __future__ import annotations

import base64
import math
import os
import re
import stat
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

import docker
from docker.errors import ImageNotFound, NotFound
from docker.utils import parse_bytes


class RuntimeUnavailable(RuntimeError):
    """The configured execution isolation is unavailable."""


class RuntimeOperationError(RuntimeError):
    """An execution or filesystem operation failed."""


# All filesystem traversal happens inside the guest using directory descriptors.
# Archives are never extracted on the controller host.
_FILES = r"""
import base64, io, os, stat, sys, zipfile
mode, path, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
root = os.open('/workspace', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
def parent(name, create=False):
    if not name.startswith('/workspace/'):
        raise ValueError('files must be inside /workspace')
    parts = name[len('/workspace/'):].split('/')
    if any(p in ('', '.', '..') or '\x00' in p for p in parts):
        raise ValueError('invalid path')
    fd = os.dup(root)
    try:
        for part in parts[:-1]:
            if create:
                try: os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError: pass
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = nxt
        return fd, parts[-1]
    except BaseException:
        os.close(fd); raise
def regular(fd, name):
    try: entry = os.stat(name, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError: return
    if not stat.S_ISREG(entry.st_mode):
        raise ValueError('only regular files are supported')
def read(name):
    fd, leaf = parent(name)
    try:
        fileno = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(fileno, 'rb') as f:
            info = os.fstat(f.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise ValueError('file exceeds limit or is not regular')
            data = f.read(limit + 1)
            if len(data) > limit: raise ValueError('file exceeds limit')
            return data
    finally: os.close(fd)
def write(name, data, permissions=0o600):
    fd, leaf = parent(name, True)
    tmp = '.transfer-' + __import__('uuid').uuid4().hex
    try:
        regular(fd, leaf)
        f = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, permissions, dir_fd=fd)
        with os.fdopen(f, 'wb') as out: out.write(data)
        regular(fd, leaf)
        os.rename(tmp, leaf, src_dir_fd=fd, dst_dir_fd=fd)
    finally:
        try: os.unlink(tmp, dir_fd=fd)
        except FileNotFoundError: pass
        os.close(fd)
if mode == 'upload':
    tmp, offset, final, chunk = sys.argv[4], int(sys.argv[5]), sys.argv[6] == '1', base64.b64decode(sys.argv[7], validate=True)
    fd, leaf = parent(path, True)
    try:
        regular(fd, leaf)
        flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
        flags |= (os.O_CREAT | os.O_EXCL) if offset == 0 else 0
        fileno = os.open(tmp, flags, 0o600, dir_fd=fd)
        with os.fdopen(fileno, 'r+b') as out:
            entry = os.fstat(out.fileno())
            if not stat.S_ISREG(entry.st_mode) or entry.st_size != offset or offset + len(chunk) > limit:
                raise ValueError('invalid upload size')
            out.seek(offset); out.write(chunk)
        if final:
            regular(fd, leaf)
            os.rename(tmp, leaf, src_dir_fd=fd, dst_dir_fd=fd)
    finally: os.close(fd)
elif mode == 'remove':
    fd, leaf = parent(path)
    try: os.unlink(leaf, dir_fd=fd)
    finally: os.close(fd)
elif mode == 'download':
    sys.stdout.buffer.write(read(path))
elif mode == 'export':
    maximum_files, maximum_file = int(sys.argv[4]), int(sys.argv[5]); count = 0; total = 0
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_STORED) as archive:
        for directory, dirs, files, fd in os.fwalk('.', dir_fd=root, follow_symlinks=False):
            prefix = '' if directory == '.' else directory[2:] + '/'
            for name in dirs + files:
                if '\\' in name: raise ValueError('unsupported workspace filename')
                entry = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if not (stat.S_ISDIR(entry.st_mode) or stat.S_ISREG(entry.st_mode)):
                    raise ValueError('workspace contains unsupported file types')
                count += 1
                if count > maximum_files: raise ValueError('workspace contains too many files')
                member = prefix + name
                if stat.S_ISDIR(entry.st_mode): archive.writestr(member + '/', b'')
                else:
                    f = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
                    with os.fdopen(f, 'rb') as source:
                        current = os.fstat(source.fileno())
                        if not stat.S_ISREG(current.st_mode): raise ValueError('unsupported file type')
                        data = source.read(min(limit - total, maximum_file) + 1)
                    if len(data) > maximum_file: raise ValueError('file exceeds limit')
                    total += len(data)
                    if total > limit: raise ValueError('workspace exceeds checkpoint limit')
                    info = zipfile.ZipInfo(member)
                    info.external_attr = (stat.S_IFREG | (current.st_mode & 0o777)) << 16
                    archive.writestr(info, data)
                if out.tell() > limit: raise ValueError('checkpoint exceeds limit')
    data = out.getvalue()
    if len(data) > limit: raise ValueError('checkpoint exceeds limit')
    sys.stdout.buffer.write(data)
elif mode == 'import':
    maximum_files, maximum_file = int(sys.argv[4]), int(sys.argv[5])
    data = read(path)
    with zipfile.ZipFile(io.BytesIO(data), 'r') as archive:
        entries = archive.infolist()
        if len(entries) > maximum_files: raise ValueError('checkpoint contains too many files')
        total = 0; seen = set()
        for info in entries:
            name = info.filename.rstrip('/')
            if name in seen: raise ValueError('duplicate checkpoint path')
            seen.add(name)
            if not name or name.startswith('/') or any(p in ('', '.', '..') for p in name.split('/')):
                raise ValueError('invalid checkpoint path')
            kind = (info.external_attr >> 16) & 0o170000
            if kind not in (0, stat.S_IFREG, stat.S_IFDIR) or info.flag_bits & 1 or info.compress_type != zipfile.ZIP_STORED:
                raise ValueError('unsupported checkpoint entry')
            total += info.file_size
            if info.file_size > maximum_file or total > limit: raise ValueError('checkpoint exceeds limit')
        for info in entries:
            target = '/workspace/' + info.filename.rstrip('/')
            if info.is_dir():
                fd, leaf = parent(target + '/.directory', True); os.close(fd)
            else:
                permissions = 0o600 | ((info.external_attr >> 16) & 0o111)
                write(target, archive.read(info), permissions)
else: raise ValueError('unsupported operation')
os.close(root)
"""
# Stable guest error codes avoid interpreting tracebacks or exposing guest paths.
_FILES = (
    "import sys\ntry:\n"
    + "\n".join("    " + line for line in _FILES.splitlines())
    + "\nexcept FileNotFoundError:\n    sys.exit(44)\n"
    + "except Exception:\n    sys.exit(45)\n"
)


class DockerRuntime:
    """A controller-only Docker client; the socket never reaches a guest."""

    LABEL = "reliquary.sandbox.managed"
    ID_LABEL = "reliquary.sandbox.id"
    LABEL_NAMESPACE = "reliquary.sandbox.namespace"

    def __init__(self, settings: Any, *, client: Any = None):
        self.settings = settings
        try:
            self._namespace = self._load_namespace()
        except RuntimeUnavailable:
            raise
        except (OSError, ValueError, UnicodeError) as exc:
            raise RuntimeUnavailable("Runtime ownership state is unavailable") from exc
        self.client = client or docker.DockerClient(
            base_url=getattr(settings, "docker_host", None),
            timeout=getattr(settings, "docker_connect_timeout", 10),
        )

    def _get(self, name: str, default: Any) -> Any:
        return getattr(self.settings, name, default)

    def _load_namespace(self) -> str:
        directory = Path(self._get("state_dir", Path("data")))
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(descriptor)
            if info.st_uid != os.geteuid():
                raise RuntimeUnavailable("Runtime state directory ownership is invalid")
            os.fchmod(descriptor, 0o700)
            try:
                owner = os.open(
                    ".runtime-owner",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=descriptor,
                )
            except FileExistsError:
                pass
            else:
                with os.fdopen(owner, "w", encoding="ascii") as output:
                    output.write(uuid.uuid4().hex + "\n")
                    output.flush()
                    os.fsync(output.fileno())
                os.fsync(descriptor)
            owner = os.open(
                ".runtime-owner",
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=descriptor,
            )
            with os.fdopen(owner, "r", encoding="ascii") as source:
                info = os.fstat(source.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
                    raise RuntimeUnavailable("Runtime ownership state is invalid")
                os.fchmod(source.fileno(), 0o600)
                namespace = source.read(65).strip()
            if not re.fullmatch(r"[0-9a-f]{32}", namespace):
                raise RuntimeUnavailable("Runtime ownership state is invalid")
            return namespace
        finally:
            os.close(descriptor)

    def _name(self, sandbox_id: str) -> str:
        return "sandbox-" + self._namespace + "-" + sandbox_id

    def readiness(self) -> dict[str, Any]:
        try:
            info = self.client.info()
            if info.get("OSType") != "linux":
                raise RuntimeUnavailable("A Linux Docker daemon is required")
            runtime = self._get("docker_runtime", "runsc")
            if self._get("require_gvisor", True) and runtime != "runsc":
                raise RuntimeUnavailable("gVisor runsc isolation is required")
            if runtime not in info.get("Runtimes", {}):
                raise RuntimeUnavailable("The configured container runtime is unavailable")
            images = tuple(self._get("allowed_images", ()))
            if not images:
                raise RuntimeUnavailable("At least one approved guest image is required")
            for name in images:
                self.client.images.get(name)
            return {"ready": True, "runtime": runtime, "images": len(images)}
        except RuntimeUnavailable:
            raise
        except Exception as exc:
            raise RuntimeUnavailable(
                "Docker or an approved local guest image is unavailable"
            ) from exc

    def create(self, sandbox_id: str, request: Any) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", sandbox_id):
            raise ValueError("Invalid sandbox identifier")
        spec = request.model_dump() if hasattr(request, "model_dump") else dict(request)
        image = spec.get("docker_image") or spec.get("image") or self._get("default_image", None)
        if image not in self._get("allowed_images", ()):
            raise ValueError("The guest image is not approved")
        # Looking up the image does not pull it from a remote registry.
        try:
            approved_image = self.client.images.get(image)
        except ImageNotFound as exc:
            raise RuntimeUnavailable("The approved guest image must be present locally") from exc
        self.readiness()
        memory_cap = parse_bytes(self._get("memory_limit", "512m"))
        memory = self._resource(spec.get("memory_gb"), 1024**3, memory_cap, "memory_gb")
        disk = self._resource(
            spec.get("disk_size_gb"),
            1024**3,
            self._get("workspace_size_bytes", 268435456),
            "disk_size_gb",
        )
        cpu = self._resource(
            spec.get("cpu_cores"), 1_000_000_000, self._get("nano_cpus", 1_000_000_000), "cpu_cores"
        )
        uid = int(self._get("default_uid", 1000))
        if uid <= 0:
            raise ValueError("The guest user must be nonroot")
        maximum_lifetime = float(self._get("max_timeout_minutes", 120))
        requested_lifetime = spec.get("timeout_minutes")
        lifetime = maximum_lifetime if requested_lifetime is None else float(requested_lifetime)
        if not math.isfinite(lifetime) or lifetime <= 0 or lifetime > maximum_lifetime:
            raise ValueError("Sandbox lifetime exceeds the configured resource range")
        container = self.client.containers.create(
            image=approved_image.id,
            command=[
                "/usr/local/bin/python3",
                "-I",
                "-S",
                "-u",
                "-c",
                "import time; time.sleep(" + repr(lifetime * 60) + ")",
            ],
            name=self._name(sandbox_id),
            labels={
                self.LABEL: "true",
                self.ID_LABEL: sandbox_id,
                self.LABEL_NAMESPACE: self._namespace,
            },
            runtime=self._get("docker_runtime", "runsc"),
            user=f"{uid}:{uid}",
            working_dir="/workspace",
            environment=spec.get("environment_vars") or {},
            read_only=True,
            network_mode="none",
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            privileged=False,
            pids_limit=self._get("pids_limit", 128),
            mem_limit=memory,
            memswap_limit=memory,
            nano_cpus=cpu,
            tmpfs={
                "/workspace": f"rw,nosuid,nodev,size={disk},uid={uid},gid={uid},mode=0700",
                "/tmp": f"rw,nosuid,nodev,noexec,size={self._get('tmp_size_bytes', 67108864)},uid={uid},gid={uid},mode=0700",
            },
            init=True,
            detach=True,
            log_config=docker.types.LogConfig(type="none"),
            auto_remove=False,
        )
        try:
            container.start()
            container.reload()
            if container.status != "running":
                raise RuntimeUnavailable("The guest failed to start")
            return container.id
        except Exception:
            container.remove(force=True)
            raise

    @staticmethod
    def _resource(value: Any, scale: int, maximum: int, name: str) -> int:
        if value is None:
            return maximum
        amount = float(value)
        result = int(amount * scale) if math.isfinite(amount) else 0
        if result <= 0 or result > maximum:
            raise ValueError(f"{name} exceeds the configured resource range")
        return result

    def _container(self, sandbox_id: str) -> Any:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", sandbox_id):
            raise ValueError("Invalid sandbox identifier")
        container = self.client.containers.get(self._name(sandbox_id))
        if (
            container.labels.get(self.LABEL) != "true"
            or container.labels.get(self.ID_LABEL) != sandbox_id
            or container.labels.get(self.LABEL_NAMESPACE) != self._namespace
        ):
            raise RuntimeOperationError("Container ownership does not match the sandbox")
        return container

    def managed_sandboxes(self) -> dict[str, str]:
        return {
            item.labels[self.ID_LABEL]: item.id
            for item in self.client.containers.list(
                all=True,
                filters={
                    "label": [self.LABEL + "=true", self.LABEL_NAMESPACE + "=" + self._namespace]
                },
            )
            if self.ID_LABEL in item.labels
            and item.labels.get(self.LABEL_NAMESPACE) == self._namespace
        }

    def status(self, sandbox_id: str) -> str:
        try:
            container = self._container(sandbox_id)
            container.reload()
            return container.status
        except NotFound:
            return "deleted"

    def delete(self, sandbox_id: str) -> None:
        try:
            self._container(sandbox_id).remove(force=True)
        except NotFound:
            return

    def execute(
        self,
        sandbox_id: str,
        command: str,
        timeout: float,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(command, str) or not command or "\x00" in command:
            raise ValueError("A nonempty command is required")
        workdir = self._path(cwd, allow_root=True) if cwd else "/workspace"
        result = self._run(sandbox_id, ["/bin/bash", "-lc", command], timeout, cwd=workdir, env=env)
        return {
            **result,
            "stdout": result["stdout"].decode("utf-8", "replace"),
            "stderr": result["stderr"].decode("utf-8", "replace"),
        }

    def _run(
        self,
        sandbox_id: str,
        command: list[str],
        timeout: float,
        *,
        cwd: str = "/workspace",
        env: dict[str, str] | None = None,
        output_limit: int | None = None,
    ) -> dict[str, Any]:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Execution timeout must be positive and finite")
        container = self._container(sandbox_id)
        cap = output_limit if output_limit is not None else self._get("max_output_bytes", 1048576)
        stdout, stderr = bytearray(), bytearray()
        state: dict[str, Any] = {"truncated": False, "exit_code": -1}
        finished = threading.Event()

        def collect() -> None:
            stream = None
            try:
                uid = int(self._get("default_uid", 1000))
                created = self.client.api.exec_create(
                    container.id,
                    command,
                    stdout=True,
                    stderr=True,
                    stdin=False,
                    tty=False,
                    user=f"{uid}:{uid}",
                    workdir=cwd,
                    environment=env or {},
                )
                stream = self.client.api.exec_start(created["Id"], stream=True, demux=True)
                for chunks in stream:
                    for target, chunk in zip((stdout, stderr), chunks):
                        if chunk:
                            remaining = max(0, cap - len(stdout) - len(stderr))
                            target.extend(chunk[:remaining])
                            state["truncated"] |= len(chunk) > remaining
                while True:
                    inspection = self.client.api.exec_inspect(created["Id"])
                    if not inspection["Running"]:
                        state["exit_code"] = inspection["ExitCode"]
                        break
                    time.sleep(0.02)
            except Exception as exc:
                state["error"] = exc
            finally:
                if stream is not None and hasattr(stream, "close"):
                    try:
                        stream.close()
                    except Exception:
                        pass
                finished.set()

        worker = threading.Thread(target=collect, daemon=True)
        worker.start()
        if not finished.wait(timeout):
            # Removing the entire container terminates the command and its descendants.
            # A timed-out sandbox must not continue accepting work.
            try:
                container.remove(force=True)
            except NotFound:
                pass
            except Exception as exc:
                raise RuntimeOperationError("Unable to terminate a timed-out sandbox") from exc
            finished.wait(min(1.0, timeout))
            return {
                "stdout": bytes(stdout),
                "stderr": bytes(stderr),
                "exit_code": -1,
                "timed_out": True,
                "truncated": state["truncated"],
            }
        if "error" in state:
            raise RuntimeOperationError("Guest execution failed") from state["error"]
        return {
            "stdout": bytes(stdout),
            "stderr": bytes(stderr),
            "exit_code": state["exit_code"],
            "timed_out": False,
            "truncated": state["truncated"],
        }

    @staticmethod
    def _path(path: str, *, allow_root: bool = False) -> str:
        if not isinstance(path, str) or "\x00" in path or "\\" in path:
            raise ValueError("Invalid workspace path")
        parts = path.split("/")
        if ".." in parts or "." in parts or "" in parts[1:]:
            raise ValueError("Invalid workspace path")
        normalized = str(PurePosixPath(path))
        if normalized == "/workspace" and allow_root:
            return normalized
        if not normalized.startswith("/workspace/"):
            raise ValueError("Files must be inside /workspace")
        if len(path.encode("utf-8")) > 4096:
            raise ValueError("Workspace path is too long")
        return normalized

    def _file(self, sandbox_id: str, arguments: list[str], *, output_limit: int = 4096) -> bytes:
        result = self._run(
            sandbox_id,
            ["/usr/local/bin/python3", "-I", "-S", "-c", _FILES, *arguments],
            30,
            env={
                "LD_PRELOAD": "",
                "LD_LIBRARY_PATH": "",
                "LD_AUDIT": "",
                "LD_DEBUG": "",
                "GCONV_PATH": "",
                "LOCPATH": "",
                "GLIBC_TUNABLES": "",
            },
            output_limit=output_limit,
        )
        if result["exit_code"] == 44 and not result["timed_out"]:
            raise FileNotFoundError("Guest file is unavailable")
        if result["exit_code"] == 45 and not result["timed_out"]:
            raise ValueError("Invalid guest filesystem operation or exceeded limit")
        if result["timed_out"] or result["exit_code"] != 0 or result["truncated"]:
            raise RuntimeOperationError("Guest filesystem operation failed or exceeded a limit")
        return result["stdout"]

    def upload(self, sandbox_id: str, path: str, data: bytes) -> None:
        self._upload(sandbox_id, path, data, self._get("max_file_bytes", 16777216))

    def _upload(self, sandbox_id: str, path: str, data: bytes, limit: int) -> None:
        path = self._path(path)
        if not isinstance(data, bytes) or len(data) > limit:
            raise ValueError("File exceeds the configured size limit")
        temporary = ".transfer-" + uuid.uuid4().hex
        chunks = range(0, len(data), 49152) if data else [0]
        try:
            for offset in chunks:
                chunk = data[offset : offset + 49152]
                self._file(
                    sandbox_id,
                    [
                        "upload",
                        path,
                        str(limit),
                        temporary,
                        str(offset),
                        "1" if offset + len(chunk) == len(data) else "0",
                        base64.b64encode(chunk).decode("ascii"),
                    ],
                )
        except Exception:
            temporary_path = str(PurePosixPath(path).parent / temporary)
            try:
                self._file(sandbox_id, ["remove", temporary_path, str(limit)])
            except Exception:
                pass
            raise

    def download(self, sandbox_id: str, path: str) -> bytes:
        limit = self._get("max_file_bytes", 16777216)
        return self._file(
            sandbox_id, ["download", self._path(path), str(limit)], output_limit=limit + 4096
        )

    def export_workspace(self, sandbox_id: str) -> bytes:
        limit = self._get("max_snapshot_bytes", 16777216)
        return self._file(
            sandbox_id,
            [
                "export",
                "/workspace",
                str(limit),
                str(self._get("max_snapshot_files", 1000)),
                str(self._get("max_file_bytes", 16777216)),
            ],
            output_limit=limit + 4096,
        )

    def import_workspace(self, sandbox_id: str, archive: bytes) -> None:
        limit = self._get("max_snapshot_bytes", 16777216)
        path = "/workspace/.restore-" + uuid.uuid4().hex
        self._upload(sandbox_id, path, archive, limit)
        try:
            self._file(
                sandbox_id,
                [
                    "import",
                    path,
                    str(limit),
                    str(self._get("max_snapshot_files", 1000)),
                    str(self._get("max_file_bytes", 16777216)),
                ],
            )
        finally:
            self._file(sandbox_id, ["remove", path, str(limit)])

    def checkpoint(self, sandbox_id: str, checkpoint_id: str) -> bytes:
        return self.export_workspace(sandbox_id)
