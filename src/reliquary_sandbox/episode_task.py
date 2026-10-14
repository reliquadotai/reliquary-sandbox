"""What an environment package declares to run on a signed-episode sandbox.

An env exposes `sandbox_task(split: str, index: int) -> SandboxTask`. The
sandbox calls `prepare` in the agent's box at open, `extract` in the agent's
box at finish, then `grade` in a pristine box of the same image with the
extracted state. All three receive a `TaskRuntime`: the subset of the
`verifiers` v1 `Runtime` API that the env graders already use, so the bridge's
hooks and the envs' rewards run unchanged.

Standard library only: env packages import this without the service extras.
"""

from __future__ import annotations

import math
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

TOOLS = frozenset({"bash", "edit"})
IMAGE_DIGEST = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")
VERIFIERS_ENTRY_PREFIX = "verifiers:"
VERIFIERS_ENTRY = re.compile(r"verifiers:(?P<id>[a-z][a-z0-9_-]{0,127})"
                             r"==(?P<version>[A-Za-z0-9][A-Za-z0-9_.+!-]{0,63})")
"""An `episode_envs` value naming a verifiers Taskset served through the generic bridge
(`verifiers:<taskset-id>==<version>`): the one definition the service's settings, its
registry and the bridge's loader use."""

# Hook failure contract (implemented by reliquary_sandbox_service.episodes.grader):
#   OPEN_FAILED:  an exception from `prepare` fails the open. It is our side: no record 0 is
#                 written and the session token is burnt.
#   ABORTED:      (ours, void, the task may be re-run) only for infrastructure failures: a
#                 sandbox InfraFault seen at any point of `extract` or `grade` (even one the
#                 hook caught or re-raised as another exception), every pristine-box attempt
#                 failing to start, a failure of the sandbox's own code before the hook runs,
#                 and a hook calling a Runtime member the sandbox does not offer
#                 (prepare_uv_script, run_uv_script, open_process, run_background),
#                 and an EnvInfraError raised by `extract` or `grade` (the env saying the
#                 failure is its own or ours: logged at error level and counted per env as
#                 `env_infra_error`; in `grade` it is retried like an InfraFault).
#   BOX_FAILED:   the agent's box failed under `extract` because of what the agent did
#                 (BoxFault: it killed its processes, exhausted its pids, ...). Not paid.
#                 Also, with the reason "the runtime failed to start a command in the
#                 agent's box", when the container runtime failed to start one of the
#                 sandbox's commands there (or a `run` program: never returned to the hook)
#                 in a box that still runs (RuntimeExecFailed): a machine fault, still
#                 box_failed (not a void), its reason telling it from the agent's.
#   GRADED 0.0:   everything else is the agent's outcome, graded 0.0 with a sandbox fact under
#                 the reserved "_grader" key: an exception raised by `extract` or `grade`
#                 itself (extract_failed / grading_failed: logged and counted per env, so fix
#                 real env bugs), a step past `grading_timeout_s` (extract_timeout /
#                 grading_timeout), output or state over its bound (state_too_large,
#                 grading_output_overflow), a non-regular file or a refused archive member
#                 (state_unreadable), a BoxFault in the pristine box (grading_box_failed),
#                 except that a RuntimeExecFailed there runs the grading attempt again in a
#                 fresh pristine box (up to the service's grading attempts, deadline
#                 permitting); a later attempt's result stands, and when none grades, the
#                 grading_box_failed 0.0 stays (never a void), its "_grader" fact marked
#                 "runtime_exec_failed": true for monitoring,
#                 and a `grade` result that is not a GradeResult with a finite reward in
#                 [0, 1] and canonical-JSON facts without the "_grader" key
#                 (invalid_grade_result). Facts over 64 KiB canonical are replaced by
#                 {"facts_truncated": true, "_grader": {...}}.
#   Env code should still return agent-attributable outcomes rather than raise: `extract`
#   returns a sentinel and `grade` returns GradeResult(0.0, facts) saying why. Use
#   `runtime.archive` / `runtime.restore_archive` (the sandbox's busybox, vetted and rebuilt) to move
#   file trees, never the image's own tar, or anything run as root, on state the agent
#   controlled. The box's own tools as the box user are acceptable where grading bounds the
#   impact (SWE's extract runs the agent's git to take an untrusted diff, which grading
#   applies in a pristine box with path refusals).
#   Raise EnvInfraError ONLY from steps that run before any agent-controlled state is
#   applied in the box (e.g. preparing the pristine box: stripping history, setting hidden
#   tests aside), or for env-internal invariants (decoding the state our own `extract`
#   wrote, a grading step that must have run). Never for anything the agent's state can
#   cause: that is a graded 0, or the agent could void its own failing episodes.
FINAL_STATUS_ABORTED = "aborted"


class EnvInfraError(RuntimeError):
    """Raised by an env hook for a failure that is not the agent's: the episode ends
    `aborted` (ours, void, may be re-run), never graded 0. See the failure contract above
    for where it may be raised."""


@dataclass(frozen=True)
class ProgramResult:
    exit_code: int
    stdout: str
    stderr: str


class TaskRuntime(Protocol):
    """A live box. `run` never raises for a non-zero exit; it raises
    TimeoutError when the step's deadline passes and RuntimeError when the box
    is unusable. `read` raises FileNotFoundError for a missing path and OSError
    when the file exceeds `max_bytes`."""

    env: dict[str, str]
    config: Any  # `.type` ("reliquary-sandbox") and `.workdir`, as verifiers' artifact helpers read

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult: ...

    async def read(self, path: str, max_bytes: int | None = None) -> bytes: ...

    async def write(self, path: str, data: bytes) -> None: ...

    async def alive(self) -> bool: ...

    async def prepare_setup(self) -> None: ...

    async def prepare_execution(self, routes: list[str] | None) -> None: ...

    async def archive(self, paths: list[str], max_bytes: int) -> bytes:
        """The state of `paths` (not `/`, no `..`, not nested) as a sandbox state archive:
        tarred by the sandbox's own busybox, symlinks stored and never followed, member
        `<i>/<path>` relative to root i, a manifest first. Unreadable or special files make
        it raise (graded `state_unreadable`)."""
        ...

    async def restore_archive(self, data: bytes, roots: list[str], *,
                              contained_links: bool = False) -> None:
        """Restore root i of an `archive` result to `roots[i]`. The bytes are vetted and
        rebuilt before extraction: `..` in names, members under a symlink member, FIFOs and
        devices are refused; symlink targets are resolved (against the archive's own links,
        at most 40 hops) and must end inside the roots, except absolute targets ending outside
        them, which are kept as inert links; hardlinks stay links to an earlier file of the
        same root; owners and setuid/setgid bits are dropped. Each root is cleared first
        (`rm -rf`, as verifiers' restore does; a root that is a symlink in the image aborts
        the episode). `write` refuses a path inside a restored root whose resolved parent
        leaves the restored roots. Restore each root once: a second restore into a root (or
        a root nested in it) on the same box aborts the episode as an integration bug.
        `contained_links=True` keeps no inert link: every restored symlink must lead, resolved
        in the box, to an existing path inside the roots, or the state is unreadable (graded
        0) and nothing stays restored (the verifiers bridge grades this way: image tools such
        as `test.sh` follow links)."""
        ...

    async def stop_processes(self) -> None:
        """SIGKILL every process left in this box except PID 1, the entrypoint and the
        helper's own chain (the sandbox's busybox, as root from `/`, in bounded rounds; the
        same helper that runs on the agent's box before `extract`). Call it before reading a
        result that code the agent influenced could still rewrite (e.g. after `test.sh`,
        before reading `reward.txt`). Failures follow the contract above: ours is an
        InfraFault (`aborted`); processes still respawning after the rounds are a BoxFault
        (graded 0 `grading_box_failed` in the pristine box)."""
        ...


@dataclass(frozen=True)
class GradeResult:
    reward: float
    facts: dict[str, Any] = field(default_factory=dict)
    """JSON-safe details copied into the signed final record (e.g. tests passed)."""


Prepare = Callable[[TaskRuntime], Awaitable[str]]
Extract = Callable[[TaskRuntime], Awaitable[bytes]]
Grade = Callable[[TaskRuntime, bytes], Awaitable[GradeResult]]


_LIMIT_INTEGERS = ("memory_bytes", "disk_bytes", "pids", "wall_s", "per_call_timeout_s",
                   "max_calls")
_RESOURCE_MINIMA = ("memory_bytes", "disk_bytes", "pids")


@dataclass(frozen=True)
class TaskLimits:
    """What a task needs, declared by its env. None = no requirement.

    The validator copies these into the session token's budgets. The gateway refuses a
    token whose memory, disk or pids budget is below the task's (422, before the token is
    spent: an honest task must not fail for want of memory). `wall_s`,
    `per_call_timeout_s` and `max_calls` are the validator's to choose and are never
    refused. `cpus` sizes the box, capped by the machine's `episode_nano_cpus`."""

    cpus: float | None = None
    memory_bytes: int | None = None
    disk_bytes: int | None = None
    pids: int | None = None
    wall_s: int | None = None
    per_call_timeout_s: int | None = None
    max_calls: int | None = None

    def __post_init__(self) -> None:
        cpus = self.cpus
        if cpus is not None and (isinstance(cpus, bool) or not isinstance(cpus, (int, float))
                                 or not math.isfinite(cpus) or cpus <= 0):
            raise ValueError("cpus must be a positive finite number")
        for name in _LIMIT_INTEGERS:
            value = getattr(self, name)
            if value is not None and (type(value) is not int or not 0 < value < 2**63):
                raise ValueError(f"{name} must be a positive integer below 2**63")

    def shortfall(self, budgets: Any) -> list[str]:
        """The resource budgets (memory_bytes, disk_bytes, pids) below what the task needs."""
        return [name for name in _RESOURCE_MINIMA
                if (need := getattr(self, name)) is not None and getattr(budgets, name) < need]


@dataclass(frozen=True)
class SandboxTask:
    image: str
    """`repo@sha256:<digest>`; must equal the session token's image."""
    workdir: str
    """Absolute; the cwd of every tool call and of relative `edit` paths."""
    tools: tuple[str, ...]
    prepare: Prepare
    """Runs once in the agent's box before record 0; returns its output (hashed into record 0).
    An exception fails the open (our side): no record 0, the token is burnt."""
    extract: Extract
    """Returns the final state (e.g. `git diff` bytes, or `runtime.archive([...])`). An exception
    it raises is graded 0.0 (`extract_failed`, logged as a possible env bug) unless the sandbox
    saw an infrastructure fault (then "aborted"); a BoxFault on the agent's box is "box_failed".
    For agent-attributable outcomes (state over `max_state_bytes`, refused paths) return a
    sentinel instead of raising. See the failure contract above."""
    grade: Grade
    """Runs in a pristine box from `image`; never sees the agent's box. An exception it raises is
    graded 0.0 (`grading_failed`, logged as a possible env bug) unless the sandbox saw an
    infrastructure fault (then retried, finally "aborted"). Must return a GradeResult with a
    finite reward in [0, 1]; anything else is graded 0.0 (`invalid_grade_result`).
    Agent-attributable outcomes: return GradeResult(0.0, facts) with a fact explaining why."""
    env: Mapping[str, str] = field(default_factory=dict)
    user: str | None = None
    """Exec user for tool calls and grading; None keeps the image's own user."""
    max_state_bytes: int = 32 * 1024**2
    prepare_timeout_s: float = 600.0
    grading_timeout_s: float = 600.0
    grading_workdir: str | None = None
    """Workdir of the grading box; None = `workdir`. Terminal grading runs `restore()`, which
    does `rm -rf /app` then execs there, so such tasks need `/`."""
    grading_user: str | None = None
    """Exec user of the grading box; None = `user`."""
    publish_state: bool = False
    """Return the extracted state to the miner at finish (SWE: the diff it submits)."""
    limits: TaskLimits = field(default_factory=TaskLimits)
    """What the task needs (memory, disk, pids, CPUs, suggested wall/call budgets)."""
    content_sha256: str | None = None
    """sha256 (64 hex) of what the task asks, when the declaration's other fields do not
    say it (a verifiers task's `Task.hash`). Part of the task fingerprint when set: a task
    whose content changed under the same index is not the one an episode opened."""

    def __post_init__(self) -> None:
        if not isinstance(self.limits, TaskLimits):
            raise TypeError("limits must be a TaskLimits")
        if self.content_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}",
                                                                 str(self.content_sha256)):
            raise ValueError("content_sha256 must be 64 lowercase hex digits")
        if not IMAGE_DIGEST.fullmatch(self.image):
            raise ValueError("a sandbox task image must be pinned by digest")
        if not self.workdir.startswith("/"):
            raise ValueError("the workdir must be absolute")
        if self.grading_workdir is not None and not self.grading_workdir.startswith("/"):
            raise ValueError("the grading workdir must be absolute")
        if not self.tools or not set(self.tools) <= TOOLS:
            raise ValueError(f"tools must be a non-empty subset of {sorted(TOOLS)}")
        if self.max_state_bytes <= 0 or self.prepare_timeout_s <= 0 or self.grading_timeout_s <= 0:
            raise ValueError("limits must be positive")

    @property
    def effective_grading_workdir(self) -> str:
        return self.workdir if self.grading_workdir is None else self.grading_workdir

    @property
    def effective_grading_user(self) -> str | None:
        return self.user if self.grading_user is None else self.grading_user


SandboxTaskFactory = Callable[[str, int], SandboxTask]
