"""The verifiers v1 `Runtime` subset env hooks call, over one of our boxes.

Members used by the bridge's hooks and the envs' rewards, including verifiers'
Harbor verifier (run, write, read with
`max_bytes`, write, name, config.workdir/type, prepare_setup,
prepare_execution, alive, process_env, env) are implemented, plus our own
`archive`, `restore_archive`, `exists` and `stop_processes`. `run_program`
is `run`. Members that need network or long-lived processes
(`prepare_uv_script`, `run_uv_script`, `open_process`, `run_background`)
raise `UnsupportedRuntimeMember`, which the grader treats as an integration
bug (`aborted`, logged).

Every exception this runtime raises because of a box, the deadline or a
bound is recorded in `faults` (also when it surfaces after the step was
cancelled: the classification runs in the worker thread), so the grader can
attribute a failure to us or to the agent.

`archive` / `restore_archive` are the safe state primitives env packages
should use instead of the image's own tar/git: the
helper busybox, as the box's numeric user, member names vetted in Python.
"""

from __future__ import annotations

import asyncio
import posixpath
import time
from types import SimpleNamespace
from typing import Any, ClassVar

from reliquary_sandbox.episode_task import ProgramResult

from ..runtime import RuntimeOperationError
from ..task_runtime import HelperBudgetTimeout, InfraFault, NotRegularFile
from .state_archive import (
    FileTooLarge,
    StateUnreadable,
    clean_roots,
    normalise_created,
    rebuild_for_restore,
    restored_link_paths,
)

__all__ = ["FileTooLarge", "LocalRuntime", "OutputOverflow", "RuntimeMisuse", "StateUnreadable",
           "StepTimeout", "UnsupportedRuntimeMember"]


class StepTimeout(TimeoutError):
    """The step's shared deadline passed (or a command ran past it)."""


class OutputOverflow(RuntimeError):
    """A command wrote more than the grader's output bound."""


class UnsupportedRuntimeMember(NotImplementedError):
    """A verifiers Runtime member signed-episode boxes do not offer."""


class RuntimeMisuse(RuntimeError):
    """Env code used the runtime against its contract (e.g. restoring into the
    same root twice): an integration bug, graded `aborted` and logged."""


LINKS_PER_HELPER = 256
"""Restored links resolved per helper run (`contained_links`)."""

AGENT_FAULTS = (StepTimeout, OutputOverflow, FileTooLarge, StateUnreadable)


class LocalRuntime:
    """Drives a box for `prepare`, `extract` and `grade`. Every call shares one
    deadline; passing it raises TimeoutError. Output beyond the configured
    bound raises instead of handing a parser a silently cut stream.

    `user` is the box's resolved numeric `uid:gid` (TaskBoxRuntime.box_user),
    so nothing is looked up inside a box the agent touched. Relative paths
    resolve against `config.workdir`. `read` without `max_bytes` is bounded
    by `read_limit` (the task's `max_state_bytes`)."""

    scripts_dir: ClassVar[str] = "/tmp/vf-scripts"
    is_local: ClassVar[bool] = True
    supports_live_processes = False

    def __init__(self, boxes: Any, box_id: str, *, workdir: str, env: dict[str, str],
                 user: str | None, deadline: float, output_limit: int,
                 read_limit: int | None = None):
        self.boxes = boxes
        self.box_id = box_id
        self.name = box_id
        self.user = user
        self.env = dict(env)
        self.config = SimpleNamespace(type="reliquary-sandbox", workdir=workdir)
        self.deadline = deadline
        self.output_limit = output_limit
        self.read_limit = output_limit if read_limit is None else read_limit
        self.faults: list[BaseException] = []
        self.stopped = False
        self.restored_roots: list[str] = []
        """Canonical roots `restore_archive` wrote; `write` stays inside them."""

    @property
    def type(self) -> str:
        return self.config.type

    # -- plumbing ------------------------------------------------------------

    def _fault(self, exc: BaseException) -> BaseException:
        self.faults.append(exc)
        return exc

    def _remaining(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise self._fault(StepTimeout("the step deadline passed"))
        return remaining

    def _left(self) -> float:
        """The time left for the next helper (raises once the deadline passed);
        read in the worker thread just before each helper starts."""
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise StepTimeout("the step deadline passed")
        return remaining

    def _path(self, path: str) -> str:
        return path if path.startswith("/") else posixpath.join(self.config.workdir, path)

    async def _call(self, work: Any) -> Any:
        """Run blocking box work in a thread. Faults are recorded there, so a
        fault that surfaces after the step was cancelled is still seen."""

        def guarded() -> Any:
            try:
                return work()
            except HelperBudgetTimeout as exc:  # the step's deadline cut a helper short
                raise self._fault(StepTimeout(str(exc))) from exc
            except (RuntimeOperationError, *AGENT_FAULTS) as exc:
                if not any(exc is fault for fault in self.faults):
                    self.faults.append(exc)
                raise

        return await asyncio.to_thread(guarded)

    # -- the verifiers Runtime surface ---------------------------------------

    def process_env(self, env: dict[str, str]) -> dict[str, str]:
        return {**self.env, **env}

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        timeout_s = self._remaining()
        argv, env = list(argv), self.process_env(env)

        def work() -> ProgramResult:
            result = self.boxes.run_program(self.box_id, argv, timeout_s=timeout_s,
                                            output_limit=self.output_limit, env=env,
                                            workdir=self.config.workdir, user=self.user)
            if result.timed_out:
                if not self.boxes.daemon_reachable():
                    raise InfraFault(f"{argv[0]} timed out and the Docker daemon is unreachable")
                raise StepTimeout(f"{argv[0]} did not finish before the step deadline")
            if result.truncated:
                raise OutputOverflow(f"{argv[0]} wrote more than {self.output_limit} bytes")
            return ProgramResult(result.exit_code, result.stdout.decode("utf-8", "replace"),
                                 result.stderr.decode("utf-8", "replace"))

        return await self._call(work)

    async def run_program(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        return await self.run(argv, env)

    def _under_restored(self, path: str) -> bool:
        """Whether `path`, as given or normalised, names a restored root or a path below one."""
        return any(item == root or item.startswith(root + "/")
                   for item in (path, posixpath.normpath(path)) for root in self.restored_roots)

    def _check_stays_inside(self, path: str, what: str, left: Any) -> None:
        """StateUnreadable unless `path` (its nearest existing ancestor, every symlink on
        it followed) resolves inside the restored roots. Run in the worker thread.
        `//` and `.` are dropped first (same meaning); `..` is left to the helper."""
        path = "/" + "/".join(part for part in path.split("/") if part not in ("", "."))
        try:
            resolved = self.boxes.resolve_existing(self.box_id, path, user=self.user,
                                                   timeout_s=left())
        except (RuntimeOperationError, StepTimeout):
            raise
        except OSError as exc:
            raise StateUnreadable(f"{path} cannot be resolved") from exc
        if not any(resolved == root or resolved.startswith(root + "/")
                   for root in self.restored_roots):
            raise StateUnreadable(f"{path} would be {what} through a symlink leaving the "
                                  "restored roots")

    async def read(self, path: str, max_bytes: int | None = None) -> bytes:
        """Read a regular file. Inside a restored root, the whole path (the file
        itself included) must resolve inside the restored roots: a restored symlink
        (kept even when it points outside, e.g. at the grader's reference files) is
        never read through (StateUnreadable)."""
        limit = self.read_limit if max_bytes is None else max_bytes
        path = self._path(path)
        self._remaining()
        left = self._left

        def work() -> bytes:
            if self._under_restored(path):
                self._check_stays_inside(path, "read", left)
            try:
                return self.boxes.read_file(self.box_id, path, limit, user=self.user,
                                            timeout_s=left())
            except NotRegularFile as exc:
                raise StateUnreadable(f"{path} is not a regular file") from exc
            except ValueError as exc:
                raise FileTooLarge(path, limit) from exc

        return await self._call(work)

    async def write(self, path: str, data: bytes) -> None:
        """Write a file, creating its parents. Inside a restored root, the
        resolved parent must stay inside the restored roots: a restored
        symlink (kept even when it points outside) is never written through."""
        path, data = self._path(path), bytes(data)
        self._remaining()
        parent = posixpath.dirname(path) or "/"
        left = self._left

        def work() -> None:
            if any(posixpath.normpath(path).startswith(root + "/") for root in self.restored_roots):
                self._check_stays_inside(parent, "written", left)
            self.boxes.make_dirs(self.box_id, parent, user=self.user, timeout_s=left())
            self.boxes.write_file(self.box_id, path, data, user=self.user, timeout_s=left())

        await self._call(work)

    async def exists(self, path: str) -> bool:
        """Whether `path` exists or is a symlink (dangling included), checked by the
        helper busybox (`file_kind`), never by the image's own `test`."""
        path = self._path(path)
        self._remaining()
        left = self._left
        return await self._call(lambda: self.boxes.file_kind(
            self.box_id, path, user=self.user, timeout_s=left()) != "absent")

    async def alive(self) -> bool:
        try:
            return (await self.run(["true"], {})).exit_code == 0
        except Exception:
            return False

    async def prepare_setup(self) -> None:
        """Boxes never have network; nothing to open."""

    async def prepare_execution(self, routes: list[str] | None = None) -> None:
        """Boxes never have network; nothing to close."""

    def host_url(self, url: str) -> str:
        return url

    async def expose(self, port: int) -> str | None:
        return None

    def _unsupported(self, member: str) -> UnsupportedRuntimeMember:
        return self._fault(UnsupportedRuntimeMember(
            f"signed-episode boxes do not offer Runtime.{member}"))  # type: ignore[return-value]

    async def prepare_uv_script(self, script: Any, env: Any = None, *, activate: bool = True):
        raise self._unsupported("prepare_uv_script")

    async def run_uv_script(self, script: Any, args: Any = None, env: Any = None):
        raise self._unsupported("run_uv_script")

    async def open_process(self, argv: list[str], env: dict[str, str]):
        raise self._unsupported("open_process")

    async def run_background(self, argv: list[str], env: dict[str, str], log: str):
        raise self._unsupported("run_background")

    async def stop_processes(self) -> None:
        """SIGKILL every process left in the box except PID 1, the entrypoint and the
        helper's own chain: the helper busybox script the grader runs before extract,
        as root from `/`, in bounded rounds. Call it when code the agent influenced
        may have left something running (the terminal env: after `test.sh`, before
        reading the reward). A failure on our side is an InfraFault (`aborted`);
        processes still respawning after the rounds are a BoxFault, the agent's
        (graded 0 `grading_box_failed` in a grading box, `box_failed` in the agent's);
        a cut by the step deadline is a StepTimeout. All are recorded in `faults`."""
        self._remaining()
        await self._call(lambda: self.boxes.stop_processes(self.box_id, timeout_s=self._left()))

    # -- safe state primitives -----------------------------------------------

    async def archive(self, paths: list[str], max_bytes: int) -> bytes:
        """The state format of `state_archive`: `paths` (absolute, or relative
        to the workdir; not `/`, no `..`, not nested) tarred by the helper
        busybox as the box's numeric user, symlinks stored and never followed
        (any target is allowed here), then rebuilt in Python. FileTooLarge past
        `max_bytes`; StateUnreadable when tar fails on the agent's tree (an
        unreadable file) or a member is refused (a FIFO, a device)."""
        roots = clean_roots(paths, self.config.workdir)
        self._remaining()

        def work() -> bytes:
            try:
                raw = self.boxes.tar_create(self.box_id, [r.lstrip("/") for r in roots],
                                            max_bytes, user=self.user, timeout_s=self._left())
            except (RuntimeOperationError, StepTimeout):
                raise
            except ValueError as exc:
                raise FileTooLarge("the archive", max_bytes) from exc
            except OSError as exc:
                raise StateUnreadable(f"the state could not be archived: {exc}") from exc
            try:
                return normalise_created(raw, roots, max_bytes)
            except ValueError as exc:
                raise StateUnreadable(f"the archive is malformed: {exc}") from exc

        return await self._call(work)

    async def restore_archive(self, data: bytes, roots: list[str], *,
                              contained_links: bool = False) -> None:
        """Restore an `archive` result: root i of the archive goes to `roots[i]`.
        The received bytes are never extracted: they are parsed and vetted in
        Python and rebuilt, each root is cleared (`rm -rf`, refusing a root that
        is a symlink in the image: InfraFault, so `aborted`), and busybox
        (`tar -xo`, numeric user) extracts the rebuilt tars only. Restore into
        a root once. See `state_archive` for what is refused.

        `contained_links` (the verifiers bridge's grading restore: image tools such
        as `test.sh` or `cat` follow links, so an inert link is not inert for them):
        every restored symlink must lead, resolved in this box from where it was
        restored, to something that exists inside the restored roots. Any other
        link (absolute or relative leaving the roots, a chain leaving them, a
        dangling link, a root restored as a link leading out) makes the state
        unreadable: the roots are cleared again and StateUnreadable is raised
        (graded 0 `state_unreadable`), the archive never stays restored."""
        roots = clean_roots(roots, self.config.workdir)
        for root in roots:
            if any(root == done or root.startswith(done + "/") or done.startswith(root + "/")
                   for done in self.restored_roots):
                raise self._fault(RuntimeMisuse(
                    f"{root} was already restored in this box: restore each root once"))
        data = bytes(data)
        self._remaining()
        left = self._left

        def work() -> None:
            try:
                rebuilt = rebuild_for_restore(data, roots, contained_links=contained_links)
            except ValueError as exc:
                raise StateUnreadable(f"the archive is malformed: {exc}") from exc
            for root, directory, clean in rebuilt:
                # Clear first, as verifiers' restore does: deleted files stay
                # deleted and image symlinks inside the root are never written through.
                self.boxes.clear_root(self.box_id, root, user=self.user, timeout_s=left())
                self.boxes.tar_extract(self.box_id, clean, directory, user=self.user,
                                       timeout_s=left())
            restored: list[str] = []
            for root in roots:
                # The parent's canonical form, never the root's own: a root that
                # was restored as a symlink must not widen what `write` allows.
                parent = posixpath.dirname(root)
                try:
                    canonical_parent = self.boxes.resolve_existing(
                        self.box_id, parent, user=self.user, timeout_s=left())
                except (RuntimeOperationError, StepTimeout):
                    raise
                except OSError as exc:
                    raise StateUnreadable(f"{parent} cannot be resolved") from exc
                canonical = posixpath.join(canonical_parent, posixpath.basename(root))
                restored.extend(item for item in dict.fromkeys([root, canonical])
                                if item not in restored)
            if contained_links:
                self._check_restored_links(restored_link_paths(rebuilt), restored, roots, left)
            for item in restored:
                if item not in self.restored_roots:
                    self.restored_roots.append(item)

        await self._call(work)

    def _check_restored_links(self, links: list[str], restored: list[str], roots: list[str],
                              left: Any) -> None:
        """StateUnreadable (after clearing `roots` again) unless every link in `links`
        leads, in this box, to something existing inside `restored`. Worker thread."""
        escaped = None
        for start in range(0, len(links), LINKS_PER_HELPER):
            chunk = links[start:start + LINKS_PER_HELPER]
            try:
                targets = self.boxes.resolve_links(self.box_id, chunk, user=self.user,
                                                   timeout_s=left())
            except (RuntimeOperationError, StepTimeout):
                raise
            except OSError:
                escaped = f"{chunk[0]} (or one after it: the links cannot be resolved)"
                break
            for link, target in zip(chunk, targets):
                if target is None or not any(target == root or target.startswith(root + "/")
                                             for root in restored):
                    escaped = link
                    break
            if escaped is not None:
                break
        if escaped is None:
            return
        for root in roots:  # never leave the refused links in the box
            try:
                self.boxes.remove_restored(self.box_id, root, user=self.user, timeout_s=left())
            except (RuntimeOperationError, StepTimeout):
                raise  # recorded: decisive, whatever the env does with the refusal
            except OSError:
                pass  # rm failed on the agent's tree: the state is refused either way
        raise StateUnreadable(f"the restored link {escaped} does not lead to a path inside "
                              "the restored roots")
