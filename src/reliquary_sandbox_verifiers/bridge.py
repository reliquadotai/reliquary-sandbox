"""Serve a standard verifiers v1 Task on a signed-episode sandbox.

`task_from_verifiers(task, lock=...)` returns the `SandboxTask` the gateway runs:

* image, workdir, environment, limits and timeouts: `declaration` (the image through
  the env package's lock, refused when not pinned);
* prepare = `Task.setup` in the agent's box, run in a scoring process (below); returns
  `setup_output(task)`. Anything but a clean return fails the open (ours: `HookFailed`,
  or the runtime fault it let through);
* extract = `Task.finalize` in the agent's box, in a scoring process (every process of
  the box SIGKILLed after each of its commands: nothing it started rewrites the
  artifacts), then every artifact root that exists taken with the sandbox's safe `archive` (a required root that is
  missing is recorded and graded 0). A finalize that raises grades 0 `extract_failed`
  (`HookFailed`, counted per env), one past the step deadline `extract_timeout`; ours is
  `aborted`; a runtime fault it let through keeps its label. What verifiers' own
  `collect` may have put in `trace.state.artifacts` is ignored: it is never transported
  (a hook acts only through the box: its trace and task changes stay in its process);
* grade, in a pristine box of the same image (the raw image: `Task.setup` does not run
  there): the task's optional `grading_setup` (the env norm's hook for preparing the
  pristine box from the image: regenerating data, stripping history; in a scoring
  process killed SCORING_STARTUP_FLOOR_S + RESTORE_MARGIN_S before the step deadline, so
  the restore and the scoring still start; any failure is ours, `aborted`: nothing of
  the agent is in the box yet; a task with it needs a grading timeout of at least
  2 * (SCORING_STARTUP_FLOOR_S + RESTORE_MARGIN_S) = 70 s, else it is refused),
  absent roots removed, present ones restored with `restore_archive`, then
  `Task.score(trace, runtime)` in a scoring process started from the service's
  forkserver (started with an empty environment, the env modules preloaded), with
  CPU, memory, file size and process limits, killed at the step deadline, as the uid of
  its own slot with a minimal environment when the service runs as root (a controller
  that cannot drop privileges refuses to score: `aborted`), every process of that uid
  killed afterwards (`scoring_process`; none is started with less than
  SCORING_STARTUP_FLOOR_S left: graded 0 `grading_timeout`; one not ready within
  `scoring_ready_s` is ours); its runtime calls are served here on
  a runtime that serialises calls (verifiers runs rewards concurrently), SIGKILLs every
  process after each command and never reads through a restored link leaving the
  restored roots; then `stop_processes` (always, in a `finally`). The reward is the
  trace's weighted sum (`Trace.reward`).

Failures follow the grader's contract (reliquary_sandbox.episode_task):

* before the agent's state is applied (the state header, clearing absent roots, the
  restore) a failure keeps the grader's classification: `aborted` for ours (an
  `EnvInfraError`, an InfraFault, a runtime member we do not offer or its misuse),
  graded 0 for the agent's state (`state_unreadable`, ...);
* once it is applied, the reward code parses bytes the agent controls, so a failure of
  the env's own scoring code (any exception `Task.score` raises, a reward that did not
  run, no reward at all, a scoring process that broke its protocol) is graded 0
  `grading_failed` (`ScoringFailed`; logged and counted per env by the grader: an
  argument the pipe cannot carry and a write past its cap included), never
  `aborted`; a scoring process killed at the deadline or past its CPU or memory limit
  (or ended by the env's code after it started scoring) is graded 0 `grading_timeout`;
  one that failed because of OUR side (it could not start: the task does not pickle,
  the forkserver failed or died while it scored, it ended before it was ready for any
  reason; our code in it failed, or it could not read our replies) is `aborted`
  (`EnvInfraError`);
* the restore keeps no link leading outside the restored roots (`contained_links`):
  the env's own commands (`test.sh`, `cat`) follow links, so such a link is
  `state_unreadable` (graded 0) and never stays in the grading box;
* a runtime fault that makes the box or the result untrustworthy (InfraFault,
  BoxFault, StepTimeout, OutputOverflow, an unsupported member, a misuse) decides the
  outcome even when the env's code caught it: no reward is read after one. A
  `FileTooLarge` or `StateUnreadable` the env caught does not (a probe of an oversized
  log, a link it refuses: the env sees the error and scores on); one it let through
  keeps its label;
* a reward that is not a finite number in [0, 1] is graded 0 with `reward_out_of_range`.

A Task instance is shared by every episode of its index, possibly at the same time:
`score` and its reward functions must not keep per-episode state on `self` (the trace
and the runtime are per episode). The env conformance suite checks that two episodes
score deterministically.
"""

from __future__ import annotations

import json
import math
import pickle
import time
from typing import Any

import verifiers.v1 as vf
from verifiers.v1.state import state_cls
from verifiers.v1.trace import Trace, TraceTask

from reliquary_sandbox.episode_task import EnvInfraError, GradeResult, SandboxTask

from . import scoring_process
from .declaration import (
    DEFAULT_TOOLS,
    ArtifactRoot,
    BridgeDefaults,
    TaskDeclarationError,
    artifact_roots,
    grading_timeout_s,
    literal_runtime_env,
    prepare_timeout_s,
    setup_output,
    task_limits,
)
from .image_lock import ImageLock, ImageLockError
from .sandbox_runtime import SandboxRuntime
from .scoring_process import ScoringIdentity, ScoringOutcome, run_in_child

__all__ = ["MAX_HEADER_BYTES", "STATE_MAGIC", "HookFailed", "ScoringFailed", "decode_state",
           "encode_state", "grading_trace", "task_from_verifiers"]

STATE_MAGIC = b"reliquary-vf-bridge-state/1\n"
MAX_HEADER_BYTES = 64 * 1024
MESSAGE_MARGIN = 1024 * 1024
"""A scoring-process message may carry the state's size plus this (headers, outputs)."""
RESTORE_MARGIN_S = 5.0
"""grading_setup ends this long plus SCORING_STARTUP_FLOOR_S before the step deadline:
the clearing, the restore and the scoring process's start keep that time."""
WRITE_PAYLOAD_FACTOR = 4
"""The bytes of one `write` (or `restore_archive`) the env's scoring code may send: this
many times the state cap (`max_state_bytes`). An env stages more than the agent's
state (Harbor's `_stage_tests` writes a tests tar); past it the call raises OSError
EFBIG in the env's code (graded 0 `grading_failed` if it lets it through), the scoring
process is never killed for it."""


def grading_trace(task: Any) -> Trace:
    """A fresh trace for the task's hooks. It carries no messages: under the env norm a
    reward grades the final state, never the conversation."""
    return Trace(task=TraceTask(type=type(task).__name__, data=task.data, key=task.key,
                                hash=task.hash),
                 state=state_cls(type(task))(),
                 agent=vf.AgentInfo(config=vf.AgentConfig(), name="reliquary-sandbox",
                                    trainable=False))


def encode_state(present: list[str], missing_required: list[str], archive: bytes) -> bytes:
    """`STATE_MAGIC` + `{"missing_required": [...], "present": [...]}` + `\\n` + archive."""
    header = json.dumps({"missing_required": missing_required, "present": present},
                        sort_keys=True, separators=(",", ":")).encode()
    return STATE_MAGIC + header + b"\n" + archive


def decode_state(state: bytes, roots: tuple[ArtifactRoot, ...]
                 ) -> tuple[list[str], list[str], bytes]:
    """(present roots, missing required roots, archive). The state never leaves the
    machine between extract and grade, so a malformed one is our bug: EnvInfraError."""
    if not state.startswith(STATE_MAGIC):
        raise EnvInfraError("the bridge state has no header")
    rest = state[len(STATE_MAGIC):]
    end = rest.find(b"\n", 0, MAX_HEADER_BYTES)
    if end < 0:
        raise EnvInfraError("the bridge state header is unterminated")
    try:
        header = json.loads(rest[:end])
    except ValueError as exc:
        raise EnvInfraError("the bridge state header is not JSON") from exc
    paths = {root.path for root in roots}
    present = header.get("present") if isinstance(header, dict) else None
    missing = header.get("missing_required") if isinstance(header, dict) else None
    if (not isinstance(present, list) or not isinstance(missing, list)
            or set(header) != {"present", "missing_required"}
            or not set(present) <= paths or not set(missing) <= paths
            or set(present) & set(missing) or len(set(present)) != len(present)):
        raise EnvInfraError("the bridge state header does not match the task's artifacts")
    archive = rest[end + 1:]
    if bool(present) != bool(archive):
        raise EnvInfraError("the bridge state archive does not match its header")
    return present, missing, archive


class ScoringFailed(Exception):
    """The env's scoring code failed on the agent's restored state: graded 0
    `grading_failed` (an exception of no runtime fault, so the grader counts it as the
    env's failure). Raised outside any `except` block: it chains from nothing."""


class HookFailed(Exception):
    """The env's setup or finalize failed in its scoring process (an exception of its own
    code, or a broken result), not by a runtime fault. From setup: the open fails (ours);
    from finalize: graded 0 `extract_failed` (the agent's state can cause it)."""


def _hook_verdict(box: SandboxRuntime, outcome: ScoringOutcome, task_name: str,
                  hook: str) -> None:
    """Return when the hook ran; else raise what the grader classifies (`_verdict`'s
    rules, for a hook that returns nothing)."""
    from reliquary_sandbox_service.episodes.local_runtime import (
        FileTooLarge,
        StateUnreadable,
        StepTimeout,
    )

    faults = getattr(box.inner, "faults", None)
    recorded = list(faults) if isinstance(faults, list) else []
    decisive = [f for f in recorded if not isinstance(f, (FileTooLarge, StateUnreadable))]
    if decisive:
        raise decisive[0]
    if outcome.kind == "done":
        return
    if outcome.kind == "ours":
        raise EnvInfraError(f"{task_name} {hook}: {outcome.detail}")
    if outcome.kind == "died":
        timeout = StepTimeout(f"the {hook} process ended without a result "
                              f"({outcome.detail})")
        if isinstance(faults, list):
            faults.append(timeout)
        raise timeout
    if outcome.kind == "raised":
        for error in outcome.call_errors:
            if any(error is fault for fault in recorded):
                raise error
    raise HookFailed(f"{task_name} {hook} failed: {outcome.detail}")


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _result(summary: dict[str, Any]) -> GradeResult:
    rewards = summary["rewards"]
    if not rewards or any(score is None for score in rewards.values()):
        raise ScoringFailed("no reward, or a reward function did not run")
    facts: dict[str, Any] = {
        "rewards": {name: score for name, score in rewards.items() if _finite(score)},
        "metrics": {name: value for name, value in summary["metrics"].items()
                    if _finite(value)},
    }
    total = summary["reward"]
    if not _finite(total) or not 0.0 <= total <= 1.0:
        facts["reward_out_of_range"] = summary["reward_repr"]
        return GradeResult(0.0, facts)
    return GradeResult(float(total), facts)


def _verdict(box: SandboxRuntime, outcome: ScoringOutcome, task_name: str) -> GradeResult:
    """The grade once `Task.score` ran in its process (see the module docstring)."""
    from reliquary_sandbox_service.episodes.local_runtime import (
        FileTooLarge,
        StateUnreadable,
        StepTimeout,
    )

    faults = getattr(box.inner, "faults", None)
    recorded = list(faults) if isinstance(faults, list) else []
    decisive = [f for f in recorded if not isinstance(f, (FileTooLarge, StateUnreadable))]
    if decisive:
        raise decisive[0]  # the box or the result cannot be trusted, whatever the env did
    if outcome.kind == "ours":  # our side failed to run the scoring: void, never the agent's
        raise EnvInfraError(f"{task_name}: {outcome.detail}")
    if outcome.kind == "died":
        timeout = StepTimeout(f"the scoring process ended without a result ({outcome.detail}): "
                              "killed at the deadline or past its CPU or memory limit")
        if isinstance(faults, list):
            faults.append(timeout)
        raise timeout
    if outcome.kind == "raised":
        for error in outcome.call_errors:
            if any(error is fault for fault in recorded):
                raise error  # an agent fault the env let through keeps its label
    if outcome.kind != "scored":
        raise ScoringFailed(f"{task_name} scoring failed: {outcome.detail}")
    return _result(outcome.summary or {})


CLEAR_AND_KEEP_WORKDIR = 'rm -rf -- "$@" && mkdir -p -- "$0"'
"""Clears absent roots ($@) then recreates the workdir ($0), in one command: once the
workdir is gone, every command the box runs there fails (the OCI runtime cannot chdir)."""


def clearing_commands(absent: list[str], workdir: str) -> list[list[str]]:
    """The grading box commands that remove the absent artifact roots. When one of them
    is the workdir or above it, a single `sh` command removes them all and recreates the
    workdir empty (as a fresh box of the image has it): the restore and the scoring run
    there."""
    if not absent:
        return []
    if any(workdir == root or workdir.startswith(root.rstrip("/") + "/") for root in absent):
        return [["sh", "-c", CLEAR_AND_KEEP_WORKDIR, workdir, *absent]]
    return [["rm", "-rf", "--", root] for root in absent]


def task_from_verifiers(task: Any, *, lock: ImageLock, tools: tuple[str, ...] = DEFAULT_TOOLS,
                        defaults: BridgeDefaults = BridgeDefaults(),
                        publish_state: bool = False) -> SandboxTask:
    """The SandboxTask serving `task` (a verifiers v1 Task). Raises TaskDeclarationError
    for a task the sandbox cannot serve."""
    data = task.data
    if not data.image:
        raise TaskDeclarationError("the task names no image")
    try:
        image = lock.digest(data.image)
    except ImageLockError as exc:
        raise TaskDeclarationError(str(exc)) from exc
    workdir = data.workdir or defaults.workdir
    if not workdir.startswith("/"):
        raise TaskDeclarationError(f"the workdir {workdir!r} is not absolute")
    if not task.hooks("reward"):
        raise TaskDeclarationError("the task has no @vf.reward")
    if task.config.judges:
        raise TaskDeclarationError("LLM judges are not served")
    try:
        pickle.dumps(task)  # what the scoring process gets: refused here, not at each grade
    except Exception as exc:
        raise TaskDeclarationError(f"the task cannot be pickled for its scoring process: "
                                   f"{exc}") from exc
    roots = artifact_roots(task, workdir)
    env = literal_runtime_env(task)
    limits = task_limits(data, defaults)
    max_state_bytes = defaults.max_artifact_bytes + len(STATE_MAGIC) + MAX_HEADER_BYTES + 1
    scoring_timeout_s = grading_timeout_s(data, defaults)

    name = type(task).__name__
    prepare_s = prepare_timeout_s(data, defaults)
    has_grading_setup = callable(getattr(task, "grading_setup", None))
    floor_s = scoring_process.SCORING_STARTUP_FLOOR_S
    if has_grading_setup and scoring_timeout_s < 2 * (floor_s + RESTORE_MARGIN_S):
        raise TaskDeclarationError(
            f"grading takes up to {scoring_timeout_s:g} s; a task with grading_setup needs "
            f"at least {2 * (floor_s + RESTORE_MARGIN_S):g} s (two scoring processes, each "
            f"needing {floor_s:g} s left to start, plus {RESTORE_MARGIN_S:g} s each)")

    def deadline_of(runtime: Any, timeout_s: float) -> float:
        deadline = getattr(runtime, "deadline", None)
        return deadline if isinstance(deadline, (int, float)) else time.monotonic() + timeout_s

    async def in_child(box: SandboxRuntime, hook: str, deadline: float) -> ScoringOutcome:
        return await run_in_child(
            task, box, hook=hook, make_trace=grading_trace, deadline=deadline,
            memory_bytes=defaults.scoring_memory_bytes,
            max_message_bytes=max_state_bytes + MESSAGE_MARGIN,
            max_payload_bytes=WRITE_PAYLOAD_FACTOR * max_state_bytes,
            identity=ScoringIdentity(file_bytes=defaults.scoring_file_bytes),
            ready_s=defaults.scoring_ready_s)

    async def prepare(runtime: Any) -> str:
        box = SandboxRuntime(runtime)
        await box.prepare_setup()
        outcome = await in_child(box, "setup", deadline_of(runtime, prepare_s))
        _hook_verdict(box, outcome, name, "setup")
        await box.prepare_execution([])
        return setup_output(task)

    async def extract(runtime: Any) -> bytes:
        box = SandboxRuntime(runtime, stop_after_run=True)  # nothing finalize ran outlives it
        outcome = await in_child(box, "finalize", deadline_of(runtime, scoring_timeout_s))
        _hook_verdict(box, outcome, name, "finalize")
        present: list[str] = []
        missing: list[str] = []
        for root in roots:
            if await box.exists(root.path):
                present.append(root.path)
            elif root.required:
                missing.append(root.path)
        archive = await box.archive(present, defaults.max_artifact_bytes) if present else b""
        return encode_state(present, missing, archive)

    async def grade(runtime: Any, state: bytes) -> GradeResult:
        from reliquary_sandbox_service.task_runtime import InfraFault

        present, missing, archive = decode_state(state, roots)
        if missing:
            return GradeResult(0.0, {"missing_artifacts": missing})
        box = SandboxRuntime(runtime, stop_after_run=True)
        deadline = deadline_of(runtime, scoring_timeout_s)
        if has_grading_setup:  # stopped in time for the restore and the scoring to start
            reserved = scoring_process.SCORING_STARTUP_FLOOR_S + RESTORE_MARGIN_S
            outcome = await in_child(box, "grading_setup", deadline - reserved)
            try:
                _hook_verdict(box, outcome, name, "grading_setup")
            except InfraFault:
                raise
            except Exception as exc:
                # Nothing of the agent is in the box yet: ours, never a 0. `from None`:
                # a recorded StepTimeout or BoxFault must not be blamed on the agent.
                raise EnvInfraError(f"{name}: preparing the grading box failed before the "
                                    f"agent's state was applied: {exc}") from None
        for argv in clearing_commands([r.path for r in roots if r.path not in present],
                                      workdir):
            cleared = await box.run(argv, {})
            if cleared.exit_code:
                raise EnvInfraError(f"the absent artifacts could not be cleared: {argv}")
        if present:
            await box.restore_archive(archive, present)
        await box.prepare_execution([])
        try:
            outcome = await in_child(box, "score", deadline)
        finally:
            try:
                await box.stop_processes()  # recorded if it fails; the verdict reads it
            except Exception:
                pass
        return _verdict(box, outcome, name)

    return SandboxTask(
        image=image, workdir=workdir, tools=tuple(tools), prepare=prepare, extract=extract,
        grade=grade, env=env,
        max_state_bytes=max_state_bytes,
        prepare_timeout_s=prepare_s,
        grading_timeout_s=scoring_timeout_s,
        publish_state=publish_state, limits=limits, content_sha256=task.hash)
