"""Fakes for tests that need a real episode gateway without Docker: `FakeBoxes`, an
in-memory stand-in for TaskBoxRuntime (boxes are dicts of files and a tiny command
language), a fake env whose answer is right when /work/answer.txt is "42", and
`fake_gateway`, a `local_gateway` over both. Shipped in the package so downstream
repositories (validators, miners) test against the real routes, manager, grader and signing.
Never used by a production host."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import io
import posixpath
import shlex
import tarfile
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from reliquary_sandbox.episode_task import GradeResult, SandboxTask
from reliquary_sandbox_service.runtime import RuntimeOperationError
from reliquary_sandbox_service.task_runtime import (
    BoxFault,
    ExecResult,
    InfraFault,
    InfraUnavailable,
    NotRegularFile,
    RuntimeExecFailed,
)

# `uid:gid` a box's user resolves to at creation (TaskBoxRuntime.box_user).
USER_IDS = {None: "0:0", "root": "0:0", "agent": "1000:1000"}

CONFORMANCE_DIRECTIVE = "# reliquary-conformance: "
"""The first line of a conformance suite script: FakeBoxes runs what it names (a real
shell reads it as a comment and runs the script below it)."""


CLEAR_AND_KEEP_WORKDIR = 'rm -rf -- "$@" && mkdir -p -- "$0"'
"""The verifiers bridge's clearing command (reliquary_sandbox_verifiers.bridge), which
FakeBoxes runs: absent artifact roots removed, the workdir ($0) recreated."""


class FakeBoxes:
    def __init__(self, images):
        self.images = list(images)
        self.boxes: dict[str, dict] = {}
        self.removed: list[str] = []
        self.fail_create = False
        self.fail_remove = False
        self.broken: set[str] = set()  # gone under us: InfraFault (ours)
        self.dead: set[str] = set()  # killed by the agent: `exited`, helpers raise BoxFault
        self.fail_stop: Exception | None = None  # raised by stop_processes
        self.fail_start: dict[tuple[str, str], int] = {}
        """(box id prefix, argv[0]) -> how many more runs of that command in such a box
        the runtime fails to start (RuntimeExecFailed, as TaskBoxRuntime classifies runsc's
        generic start failure on a live box); `g-` prefixes the grading boxes."""
        self.specs: dict[str, object] = {}
        self.users: dict[str, str] = {}
        self.restored: list[tuple[str, str, str | None]] = []
        self.exec_log: list[tuple[str, list[str], str | None, str | None]] = []
        self.helper_calls: list[tuple[str, str, dict]] = []  # (helper, box, kwargs)
        self.reachable = True  # the Docker daemon answers
        self.extracted: list[tuple[str, str, bytes]] = []  # (box, dir, tar) given to tar_extract
        self.preflight_calls = 0
        self.block = threading.Event()
        self.create_delay = 0.0

    # lifecycle
    def preflight(self, *, probe=True):
        self.preflight_calls += 1
        return {"runtime": "runsc", "images": self.images, "disk_quota": "disabled",
                "runsc_version": "release-20260928.0", "busybox_version": "1.37.0"}

    def images_present(self):
        return list(self.images)

    def image_present(self, image):
        return image in self.images_present()

    def create(self, box_id, episode_id, spec):
        if self.fail_create:
            raise InfraUnavailable("cannot start")
        time.sleep(self.create_delay)
        dirs = {"/", "/work"}
        path = getattr(spec, "workdir", None) or "/"
        while path not in ("/", ""):  # Docker creates the image's WorkingDir
            dirs.add(path)
            path = posixpath.dirname(path)
        self.boxes[box_id] = {"episode": episode_id, "spec": spec, "files": {}, "cpu": 0,
                             "dirs": dirs}
        self.specs[box_id] = spec
        self.users[box_id] = USER_IDS.get(getattr(spec, "user", None), "1001:1001")
        return "cid-" + box_id

    def remove(self, box_id):
        if self.fail_remove:
            raise InfraFault("docker down")
        self.boxes.pop(box_id, None)
        self.users.pop(box_id, None)
        self.removed.append(box_id)

    def box_user(self, box_id):
        return self.users.get(box_id)

    def restore_user(self, box_id, uid_gid, spec_user=None):
        self.restored.append((box_id, uid_gid, spec_user))
        self.users[box_id] = uid_gid

    def status(self, box_id):
        if box_id not in self.boxes:
            return "deleted"
        return "exited" if box_id in self.dead else "running"

    def managed(self):
        return {box_id: box["episode"] for box_id, box in self.boxes.items()}

    def _box(self, box_id):
        if box_id in self.broken or box_id not in self.boxes:
            raise InfraFault("The box is gone")
        if box_id in self.dead:
            raise BoxFault("The box is exited")
        if box_id not in self.users:
            # Like TaskBoxRuntime._numeric_user: after a restart, until restore_user().
            raise InfraFault("The box user was never resolved: the caller must restore_user() it")
        return self.boxes[box_id]

    # execution
    def exec(self, box_id, argv, *, timeout_s, output_limit, stdin=None, env=None, workdir=None, user=None):
        box = self._box(box_id)
        self.exec_log.append((box_id, list(argv), workdir, user))
        for (prefix, command), left in self.fail_start.items():
            if left and box_id.startswith(prefix) and argv[:1] == [command]:
                self.fail_start[(prefix, command)] = left - 1
                raise RuntimeExecFailed(f"{command}: the runtime failed to start the command; "
                                        "the box is running")
        files = box["files"]
        if workdir and workdir not in box["dirs"]:
            # like runsc: a command cannot start in a directory that no longer exists
            return ExecResult(b"", b"OCI runtime exec failed: failed to find initial working "
                              b"directory\n", 127, False, False)
        if (len(argv) >= 5 and argv[:3] == ["sh", "-c", CLEAR_AND_KEEP_WORKDIR]
                and not box.get("shimmed")):
            self._file_command(box, ["rm", "-rf", "--", *argv[4:]])
            return self._file_command(box, ["mkdir", "-p", "--", argv[3]])
        if box.get("shimmed") and argv[:1] not in (["test"], ["rm"]):
            # the conformance `shim`: every image tool prints pass and exits 0
            return ExecResult(b"pass\n", b"", 0, False, False)
        if argv == ["prepare"]:
            files["/work/prepared"] = b"1"
            return ExecResult(b"prepared\n", b"", 0, False, False)
        if argv == ["check"]:
            return ExecResult(b"", b"", 0 if files.get("/work/answer.txt") == b"42\n" else 1, False, False)
        if argv == ["true"]:
            return ExecResult(b"", b"", 0, False, False)
        if len(argv) == 3 and argv[:2] in (["test", "-e"], ["test", "-L"]):
            links = box.setdefault("links", {})
            found = argv[2] in links if argv[1] == "-L" else self._exists(box, argv[2])
            return ExecResult(b"", b"", 0 if found else 1, False, False)
        if len(argv) == 4 and argv[:3] == ["rm", "-rf", "--"]:
            root = argv[3]
            for table in (files, box.setdefault("links", {})):
                for path in [p for p in table if p == root or p.startswith(root + "/")]:
                    del table[path]
            box["dirs"] = {d for d in box["dirs"] if not (d == root or d.startswith(root + "/"))}
            return ExecResult(b"", b"", 0, False, False)
        if argv == ["ls-files"]:  # every regular file of the box, one per line
            return ExecResult("".join(f"{p}\n" for p in sorted(files)).encode(), b"", 0,
                              False, False)
        if argv == ["die"]:  # the command under test kills the box
            self.dead.add(box_id)
            return ExecResult(b"", b"", 137, False, False)
        if argv == ["slow"]:
            return ExecResult(b"", b"", -1, True, False)
        if argv == ["flood"]:
            return ExecResult(b"x" * (output_limit), b"", 0, False, True)
        if len(argv) == 3 and argv[0] == "emit":  # emit <hex byte> <count>: both streams
            data = bytes.fromhex(argv[1]) * int(argv[2])
            return ExecResult(data[:output_limit], data[:output_limit], 0, False,
                              len(data) > output_limit)
        return ExecResult(b"", b"unknown\n", 127, False, False)

    def run_program(self, box_id, argv, *, timeout_s, output_limit, env, workdir, user):
        self.helper_calls.append(("run_program", box_id, {"env": dict(env), "workdir": workdir,
                                                          "user": user}))
        return self.exec(box_id, argv, timeout_s=timeout_s, output_limit=output_limit, env=env,
                         workdir=workdir, user=user)

    def file_kind(self, box_id, path, *, user=None, timeout_s=None):
        box = self._box(box_id)
        self.helper_calls.append(("file_kind", box_id, {"user": user, "timeout_s": timeout_s}))
        if path in box["files"]:
            return "file"
        special = path in box.get("special", set())
        return "other" if special or self._exists(box, path) else "absent"

    def bash(self, box_id, call_id, command, *, timeout_s, output_limit, env, workdir, user):
        box = self._box(box_id)
        if command.startswith(CONFORMANCE_DIRECTIVE):
            return self._conformance(box, command, output_limit)
        if " && " in command:  # stops at the first command that fails, as sh does
            for part in command.split(" && "):
                result = self.bash(box_id, call_id, part, timeout_s=timeout_s,
                                   output_limit=output_limit, env=env, workdir=workdir,
                                   user=user)
                if result.exit_code:
                    break
            return result
        words = command.split(" ", 2)
        name = words[0]
        if name in ("rm", "mkdir", "ln"):
            return self._file_command(box, shlex.split(command))
        if name == "echo":
            out = (command[5:] + "\n").encode()
            return ExecResult(out[:output_limit], b"", 0, False, len(out) > output_limit)
        if name == "sleep":
            if float(words[1]) > timeout_s:
                return ExecResult(b"", b"", -1, True, False)
            return ExecResult(b"", b"", 0, False, False)
        if name == "burn":
            box["cpu"] += int(words[1]) * 1_000_000
            return ExecResult(b"", b"", 0, False, False)
        if name == "write":
            box["files"][words[1]] = (words[2] + "\n").encode()
            return ExecResult(b"", b"", 0, False, False)
        if name == "cat":
            data = box["files"].get(words[1])
            if data is None:
                return ExecResult(b"", b"cat: no such file\n", 1, False, False)
            return ExecResult(data[:output_limit], b"", 0, False, len(data) > output_limit)
        if name == "big":
            data = b"y" * int(words[1])
            return ExecResult(data[:output_limit], b"", 0, False, len(data) > output_limit)
        if name == "binary":
            return ExecResult(b"\xff\xfeok\n", b"", 0, False, False)
        if name == "fail":
            return ExecResult(b"out\n", b"boom\n", 2, False, False)
        if name == "hang":
            self.block.wait(5)
            return ExecResult(b"", b"", 0, False, False)
        return ExecResult(b"", b"unknown command\n", 127, False, False)

    def _file_command(self, box, argv):
        """`rm -rf -- PATH...`, `mkdir -p [--] PATH...`, `ln -s TARGET LINK`."""
        files, links = box["files"], box.setdefault("links", {})
        if argv[:3] == ["rm", "-rf", "--"]:
            for root in argv[3:]:
                for table in (files, links):
                    for path in [p for p in table if p == root or p.startswith(root + "/")]:
                        del table[path]
                box["dirs"] = {d for d in box["dirs"]
                               if not (d == root or d.startswith(root + "/"))}
            return ExecResult(b"", b"", 0, False, False)
        if argv[:2] == ["mkdir", "-p"]:
            for path in [p for p in argv[2:] if p != "--"]:
                while path not in ("/", ""):
                    box["dirs"].add(path)
                    path = posixpath.dirname(path)
            return ExecResult(b"", b"", 0, False, False)
        if len(argv) == 4 and argv[:2] == ["ln", "-s"]:
            if self._exists(box, argv[3]):
                return ExecResult(b"", b"ln: File exists\n", 1, False, False)
            links[argv[3]] = argv[2]
            return ExecResult(b"", b"", 0, False, False)
        return ExecResult(b"", b"unknown command\n", 127, False, False)

    def _conformance(self, box, command, output_limit):
        """What a conformance script does, by its directive: `mark` (remember the files),
        `corrupt <kind> <bytes>` (replace every file written since the mark, outside
        /tmp, by `kind`: empty, not_utf8, invalid_json, or `bytes` of `[` shared by the
        files: hostile), `output <bytes>` (print that many bytes), `fill <bytes> <name> ROOT...`
        (`bytes` of `x` shared by the roots that are directories and /logs/artifacts, in
        a file `name` inside each), `shim` (every image tool, `exec`, prints pass and exits 0)."""
        words = command[len(CONFORMANCE_DIRECTIVE):].split("\n", 1)[0].split()
        files = box["files"]
        if words == ["mark"]:
            box["conformance_mark"] = dict(files)
            return ExecResult(b"", b"", 0, False, False)
        if words == ["shim"]:
            box["shimmed"] = True
            return ExecResult(b"", b"", 0, False, False)
        if len(words) >= 3 and words[0] == "fill":
            roots = [root for root in words[3:] if root == "/logs/artifacts" or (
                root not in files and self._exists(box, root))]
            if not roots:
                return ExecResult(b"", b"no artifact root is a directory\n", 3, False, False)
            for root in roots:
                self._file_command(box, ["mkdir", "-p", root])
                files[f"{root}/{words[2]}"] = b"x" * (int(words[1]) // len(roots))
            return ExecResult(b"", b"", 0, False, False)
        if len(words) == 2 and words[0] == "output":
            data = b"x" * int(words[1])
            return ExecResult(data[:output_limit], b"", 0, False, len(data) > output_limit)
        if len(words) == 3 and words[0] == "corrupt":
            mark = box.get("conformance_mark")
            if mark is None:
                return ExecResult(b"", b"no mark\n", 3, False, False)
            written = sorted(p for p, data in files.items()
                             if not p.startswith("/tmp/") and mark.get(p) != data)
            if not written:
                return ExecResult(b"", b"the reference wrote no file\n", 3, False, False)
            content = {"empty": b"", "not_utf8": b"\xff\xfe\x80\xc0",
                       "invalid_json": b'{"reward": 1.0, "passed": [tru',
                       "hostile": b"[" * (int(words[2]) // len(written))}[words[1]]
            for path in written:
                files[path] = content
            return ExecResult(b"", b"", 0, False, False)
        return ExecResult(b"", b"unknown directive\n", 127, False, False)

    def read_file(self, box_id, path, max_bytes, *, user=None, timeout_s=None):
        box = self._box(box_id)
        self.helper_calls.append(("read_file", box_id, {"user": user, "timeout_s": timeout_s}))
        files = box["files"]
        if path.endswith("/"):
            raise IsADirectoryError(path)
        path = self._follow(box, path)  # like `head`: every link on the path is followed
        if path in box.get("special", set()):
            raise NotRegularFile(path)
        if path not in files:
            raise FileNotFoundError(path)
        if len(files[path]) > max_bytes:
            raise ValueError(path)
        return files[path]

    def write_file(self, box_id, path, data, *, user=None, timeout_s=None):
        box = self._box(box_id)
        self.helper_calls.append(("write_file", box_id, {"user": user, "timeout_s": timeout_s}))
        parent = posixpath.dirname(path)
        if parent not in box["dirs"] and not any(name.startswith(parent + "/") for name in box["files"]):
            raise FileNotFoundError(path)  # like `cat > path` without its parent
        box["files"][path] = bytes(data)

    def make_dirs(self, box_id, path, *, user=None, timeout_s=None):
        box = self._box(box_id)
        self.helper_calls.append(("make_dirs", box_id, {"user": user, "timeout_s": timeout_s}))
        dirs = box["dirs"]
        while path not in ("/", ""):
            dirs.add(path)
            path = posixpath.dirname(path)

    def daemon_reachable(self):
        return self.reachable

    def tar_create(self, box_id, members, max_bytes, *, user=None, timeout_s=None):
        """Members relative to `/`; `box["links"]` holds symlinks (path -> target)."""
        box = self._box(box_id)
        self.helper_calls.append(("tar_create", box_id, {"user": user, "timeout_s": timeout_s}))
        links = box.setdefault("links", {})
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            for member in members:
                root = "/" + member
                found = False
                for path in sorted(set(box["files"]) | set(links)):
                    if path != root and not path.startswith(root + "/"):
                        continue
                    found = True
                    info = tarfile.TarInfo(path.lstrip("/"))
                    if path in links:
                        info.type, info.linkname = tarfile.SYMTYPE, links[path]
                        tar.addfile(info)
                    else:
                        info.size = len(box["files"][path])
                        tar.addfile(info, io.BytesIO(box["files"][path]))
                if not found:
                    raise OSError(5, f"tar failed: {member}: No such file or directory")
        data = buffer.getvalue()
        if len(data) > max_bytes:
            raise ValueError("the archive exceeds the bound")
        return data

    def tar_extract(self, box_id, data, dest_root, *, user=None, timeout_s=None):
        box = self._box(box_id)
        self.helper_calls.append(("tar_extract", box_id, {"user": user, "timeout_s": timeout_s}))
        self.extracted.append((box_id, dest_root, bytes(data)))
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            for info in tar.getmembers():
                path = posixpath.join(dest_root, info.name)
                if info.issym():
                    box.setdefault("links", {})[path] = info.linkname
                elif info.islnk():
                    box["files"][path] = box["files"][posixpath.join(dest_root, info.linkname)]
                elif info.isfile():
                    box["files"][path] = tar.extractfile(info).read()
                elif info.isdir():
                    box["dirs"].add(path)

    def clear_root(self, box_id, root, *, user=None, timeout_s=None):
        box = self._box(box_id)
        self.helper_calls.append(("clear_root", box_id, {"user": user, "timeout_s": timeout_s}))
        links = box.setdefault("links", {})
        if root in links:
            raise InfraFault(f"The restore root {root} is a symlink in the image")
        for table in (box["files"], links):
            for path in [p for p in table if p == root or p.startswith(root + "/")]:
                del table[path]
        box["dirs"] = {d for d in box["dirs"] if not (d == root or d.startswith(root + "/"))}

    def remove_restored(self, box_id, root, *, user=None, timeout_s=None):
        box = self._box(box_id)
        self.helper_calls.append(("remove_restored", box_id, {"user": user, "timeout_s": timeout_s}))
        for table in (box["files"], box.setdefault("links", {})):
            for path in [p for p in table if p == root or p.startswith(root + "/")]:
                del table[path]
        box["dirs"] = {d for d in box["dirs"] if not (d == root or d.startswith(root + "/"))}

    def _exists(self, box, path):
        links = box.setdefault("links", {})
        return (path in box["files"] or path in links or path in box["dirs"]
                or any(name.startswith(path.rstrip("/") + "/") for name in [*box["files"], *links]))

    def _follow(self, box, path):
        """`path` with every symlink on it followed (the leaf too)."""
        links = box.setdefault("links", {})
        current, hops = "/", 0
        for part in [p for p in path.split("/") if p]:
            current = posixpath.normpath(posixpath.join(current, part))
            while current in links:
                hops += 1
                if hops > 40:
                    raise OSError(40, "too many links", path)
                current = posixpath.normpath(posixpath.join(posixpath.dirname(current),
                                                            links[current]))
        return current

    def resolve_existing(self, box_id, path, *, user=None, timeout_s=None):
        box = self._box(box_id)
        self.helper_calls.append(("resolve_existing", box_id, {"user": user, "timeout_s": timeout_s}))
        links = box.setdefault("links", {})
        while path != "/" and not self._exists(box, path):
            path = posixpath.dirname(path)
        current, hops = "/", 0
        for part in [p for p in path.split("/") if p]:
            current = posixpath.join(current, part)
            while current in links:
                hops += 1
                if hops > 40:
                    raise OSError(40, "too many links", path)
                current = posixpath.normpath(posixpath.join(posixpath.dirname(current),
                                                            links[current]))
        if current != "/" and not self._exists(box, current):
            raise OSError(2, "dangling", path)
        return current

    def resolve_links(self, box_id, paths, *, user=None, timeout_s=None):
        """Like TaskBoxRuntime.resolve_links: where each path's links lead, None
        when nothing exists there (dangling or a loop)."""
        box = self._box(box_id)
        self.helper_calls.append(("resolve_links", box_id, {"user": user, "timeout_s": timeout_s}))
        resolved = []
        for path in paths:
            try:
                target = self._follow(box, path)
            except OSError:
                target = None
            resolved.append(target if target is not None and self._exists(box, target) else None)
        return resolved

    def classify_failure(self, box_id, what, cause=None):
        return RuntimeOperationError(what)

    def cpu_ns(self, box_id):
        # Like TaskBoxRuntime.cpu_ns: a box that stopped reports its last value.
        if box_id in self.boxes and box_id in self.dead and box_id not in self.broken:
            return self.boxes[box_id]["cpu"]
        return self._box(box_id)["cpu"]

    def stop_processes(self, box_id, *, timeout_s):
        self.helper_calls.append(("stop_processes", box_id, {"timeout_s": timeout_s}))
        self._box(box_id)
        if self.fail_stop is not None:
            raise self.fail_stop


# -- the fake env ---------------------------------------------------------------------

FAKE_IMAGE = "registry.example/task@sha256:" + "a" * 64
FAKE_ENV_ENTRY = "reliquary_sandbox_service.episodes.testing:fake_sandbox_task"
TOO_LARGE = b"<state too large>"
FACTORY_CALLS: list[tuple[str, bool]] = []
"""(thread name, an event loop was running in that thread) per factory call."""


async def prepare(runtime):
    result = await runtime.run(["prepare"], {})
    if result.exit_code:
        raise RuntimeError("prepare failed")
    return result.stdout


async def extract(runtime):
    # Agent-attributable outcomes are returned, never raised.
    try:
        return await runtime.read("/work/answer.txt", max_bytes=1024)
    except FileNotFoundError:
        return b""
    except OSError as exc:
        if exc.errno == errno.EFBIG:
            return TOO_LARGE
        raise


async def grade(runtime, state):
    if state == TOO_LARGE:
        return GradeResult(reward=0.0, facts={"refused": "state_too_large"})
    await runtime.write("/work/answer.txt", state)
    result = await runtime.run(["check"], {})
    return GradeResult(reward=1.0 if result.exit_code == 0 else 0.0,
                       facts={"check_exit": result.exit_code})


def _loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def fake_sandbox_task(split: str, index: int) -> SandboxTask:
    FACTORY_CALLS.append((threading.current_thread().name, _loop_running()))
    if index < 0 or not (split in {"train", "eval"} or split.startswith("train:")):
        raise IndexError(index)
    return SandboxTask(image=FAKE_IMAGE, workdir="/work", tools=("bash", "edit"), prepare=prepare,
                       extract=extract, grade=grade, publish_state=True)


@contextmanager
def fake_gateway(workdir: Path, *, envs: Mapping[str, str] | None = None,
                 **overrides) -> Iterator[tuple]:
    """A `local_gateway` over FakeBoxes serving FAKE_IMAGE; yields (gateway, boxes)."""
    from .local_gateway import local_gateway

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    helper = workdir / "busybox"
    helper.write_bytes(b"not really busybox")
    boxes = FakeBoxes([FAKE_IMAGE])
    settings = dict(episode_allow_unquoted_disk=True, episode_helper_busybox=helper,
                    episode_helper_busybox_sha256=hashlib.sha256(helper.read_bytes()).hexdigest())
    settings.update(overrides)
    with local_gateway(workdir / "gateway", envs=dict(envs or {"fake-env": FAKE_ENV_ENTRY}),
                       images=[FAKE_IMAGE], task_runtime=boxes, **settings) as gateway:
        yield gateway, boxes
