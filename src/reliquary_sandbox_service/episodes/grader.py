"""Extract the agent's final state and grade it in a pristine box.

Outcome rules (the env-facing contract is in `reliquary_sandbox.episode_task`):

* `aborted` (ours, void, the task may be re-run): an `InfraFault` seen at any
  point (on the agent box during `extract`; in the pristine box on every one
  of `episode_grading_attempts` attempts), even one the env swallowed or
  re-raised as something else; a failure of our own code before the env hook
  is called (the box user was never resolved, `run_step` misuse); and an env
  calling a Runtime member we do not offer (`UnsupportedRuntimeMember`, an
  integration bug caught by the golden tests), or misusing it
  (`RuntimeMisuse`: a second `restore_archive` into a root). Logged. Also an
  `EnvInfraError` the hook raised (or one its exception chains from): the env
  saying the failure is ours (its pristine-box preparation, its own
  invariants). Logged at error level and counted per env (`env_infra_error`);
  retried in the pristine box like an InfraFault.
* `box_failed` (the agent, unpaid, not void): a `BoxFault` on the agent's box
  during `extract` (it ran `kill -9 -1`, filled its pids, ...), including an
  output lost on that box while it still runs.
* `graded` 0.0 with a `_grader` fact for everything else that goes wrong, all
  of it caused by the agent's state or exposed by it:
  `state_too_large` (extract output, read or archive over its bound, or a
  state over `max_state_bytes`), `state_unreadable` (a non-regular file, an
  archive with refused members), `extract_timeout` / `grading_timeout`,
  `grading_box_failed`, `grading_output_overflow`, and `extract_failed` /
  `grading_failed` for an exception from the env's own code (logged at error
  level with the env name and counted per env in `env_failures`, so real env
  bugs stay visible), `invalid_grade_result` for a result that is not a
  `GradeResult` with a finite reward in [0, 1] and canonical JSON facts
  (also logged and counted). Not retried.
* The runtime failing to start a command (`RuntimeExecFailed`: runsc's generic
  start failure on a box that still runs, below its pids limit, and that runs a
  helper right after). In a grading box the attempt is run again in a fresh box
  (grading a state is deterministic); when no attempt grades it, the episode gets
  the `grading_box_failed` zero it had before, marked `runtime_exec_failed` (never
  a void: the agent's code ran in that box and could have caused it). On the
  agent's box it stays `box_failed` (a machine fault), with its own reason.

The verification window: `grade(..., deadline=)` takes an absolute time (on
the grader's `clock`) by which grading must be over, so the final record is
signed inside the window the verifier accepts. Each step's timeout is the
task's, cut to the time left; a retry starts only when a full attempt still
fits (`grading_timeout_s + RETRY_MARGIN_S`). A step that times out because the
window, not the task, cut it, and a retry that cannot start for lack of time,
are `aborted` (ours: the agent did nothing wrong), never `*_timeout` graded 0.

A recorded runtime fault is blamed only when the exception the hook raised is
that fault or chains from it (`__cause__`/`__context__`); an InfraFault is
blamed wherever it was seen. Graded-0 outcomes decided during `extract` have
`state=None` and `state_sha256=None`. Grading boxes are named
`g-<sha256(episode_id)[:24]>-<attempt>` and always removed; the ids whose
removal failed are returned in `GradeOutcome.cleanup` for the reaper.

Before extract every process the agent left in its box is killed
(`stop_processes`, as root; PID 1 and the entrypoint are kept), so the state
is final and nothing tampers with extraction; a failure there is classified
like extract's. The grading box's create and first CPU reading are bounded by
`episode_grading_create_timeout_s` (ours when exceeded).

Overrun: `run_step` (asyncio.run) waits for the worker thread of the helper
running when the step is cancelled; that helper is not interrupted. Each
helper's timeout is the time the step had left when it started (archive,
restore and write hand every helper the remaining time), so the helper ends by
the step's deadline plus its own classification overhead (a settle period and
a few Docker round trips, seconds), and a lost-output re-run never outlasts
the caller's deadline.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import hashlib
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from reliquary_sandbox.attest import CanonicalError, canonical_json
from reliquary_sandbox.attest.records import STATUS_ABORTED, STATUS_BOX_FAILED, STATUS_GRADED
from reliquary_sandbox.episode_task import EnvInfraError, GradeResult, SandboxTask

from ..runtime import RuntimeOperationError
from ..task_runtime import BoxFault, BoxSpec, HelperBudgetTimeout, InfraFault, RuntimeExecFailed
from .local_runtime import (
    FileTooLarge,
    LocalRuntime,
    OutputOverflow,
    RuntimeMisuse,
    StateUnreadable,
    StepTimeout,
    UnsupportedRuntimeMember,
)
from .resources import grading_reservation

_log = logging.getLogger(__name__)

MAX_FACTS_BYTES = 64 * 1024
RETRY_MARGIN_S = 120
"""A retry starts only with `grading_timeout_s + RETRY_MARGIN_S` left."""
GRADING_CREATE_TIMEOUT_S = 90
"""Default bound on a grading box's create + first CPU read (below RETRY_MARGIN_S)."""
WINDOW_SPENT = "the grading window is spent (it would outlast the verification window)"
GRADER_KEY = "_grader"
CANARY_PROBE_TIMEOUT_S = 30.0
"""Bound on the conformance-only canary probe of a grading box."""

INFRA, UNSUPPORTED, BOX, TIMEOUT, OVERFLOW, TOO_LARGE, UNREADABLE, FAILED, ENV_INFRA = (
    "infra", "unsupported", "box", "timeout", "overflow", "too_large", "unreadable", "failed",
    "env_infra")
START = "start"
"""A RuntimeExecFailed: the runtime failed to start a command in a box that still runs."""
RUNTIME_START_FACT = "runtime_exec_failed"
"""`_grader` fact on a `grading_box_failed` zero kept after the runtime failed to start a
command on every grading attempt: the agent's verdict, as before, but marked."""
START_FAILED_REASON = "the runtime failed to start a command in the agent's box"


@dataclass(frozen=True)
class GradeOutcome:
    status: str
    """`graded`, `box_failed` or `aborted` (the signed final status)."""
    reward: float | None = None
    facts: dict[str, Any] | None = None
    state: bytes | None = None
    """The extracted state; None when extract itself decided the outcome."""
    state_sha256: str | None = None
    """sha256 of `state`; None for graded-0 outcomes decided during extract."""
    cpu_ms: int = 0
    """CPU of the pristine grading box."""
    reason: str | None = None
    """Why the episode did not end `graded`."""
    cleanup: tuple[str, ...] = field(default=())
    """Grading boxes whose removal failed: hand them to the reaper."""


def grader_facts(label: str, **extra: Any) -> dict[str, Any]:
    return {GRADER_KEY: {"agent_fault": label, **extra}}


def _graded_zero(label: str, state: bytes | None = None, cpu_ms: int = 0) -> GradeOutcome:
    return GradeOutcome(STATUS_GRADED, 0.0, grader_facts(label), state,
                        None if state is None else hashlib.sha256(state).hexdigest(), cpu_ms)


@dataclass(frozen=True)
class _StartFailed:
    """A grading attempt where the runtime failed to start a command in the grading box
    (RuntimeExecFailed): run again in a fresh box; `kept` (the `grading_box_failed` zero
    it would have been) is the episode's outcome when no attempt grades it."""
    kept: GradeOutcome


class _StepDeadline(StepTimeout):
    """`run_step`'s own timeout (not one the env raised)."""


def _require_no_running_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError("run_step must not be called from a running event loop")


def run_step(coroutine_factory, timeout_s: float):
    """Run one async env step to completion. Must be called from a thread
    with no running event loop (the manager's worker threads)."""
    _require_no_running_loop()

    async def bounded():
        try:
            async with asyncio.timeout(timeout_s) as scope:
                return await coroutine_factory()
        except TimeoutError as exc:
            if scope.expired():
                raise _StepDeadline(f"the step ran past its {timeout_s:g} s timeout") from exc
            raise

    return asyncio.run(bounded())


def _chain(exc: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    while exc is not None and not any(exc is s for s in seen):
        seen.append(exc)
        exc = exc.__cause__ or exc.__context__
    return seen


def _attribute(runtime: LocalRuntime | None, exc: BaseException | None) -> str | None:
    faults = runtime.faults if runtime is not None else []
    chain = _chain(exc) if exc is not None else []
    if any(isinstance(item, InfraFault) for item in [*faults, *chain]):
        return INFRA
    if any(isinstance(item, (UnsupportedRuntimeMember, RuntimeMisuse))
           for item in [*faults, *chain]):
        return UNSUPPORTED
    if exc is None:
        return None
    for item in chain:  # the raised exception first, then what it chains from
        if isinstance(item, _StepDeadline) or (
                any(item is fault for fault in faults) and isinstance(item, StepTimeout)):
            return TIMEOUT
        if isinstance(item, EnvInfraError):
            return ENV_INFRA
        if not any(item is fault for fault in faults):
            continue
        if isinstance(item, RuntimeExecFailed):
            return START
        if isinstance(item, BoxFault):
            return BOX
        if isinstance(item, OutputOverflow):
            return OVERFLOW
        if isinstance(item, FileTooLarge):
            return TOO_LARGE
        if isinstance(item, StateUnreadable):
            return UNREADABLE
        if isinstance(item, RuntimeOperationError):
            return INFRA  # unclassified runtime failure: never blame the agent for it
    return FAILED


class Grader:
    def __init__(self, settings: Any, boxes: Any, clock=time.time):
        self.settings = settings
        self.boxes = boxes
        self.clock = clock
        """The clock `deadline`s are read on (the manager's)."""
        self._lock = threading.Lock()
        self.canaries: dict[str, tuple[str, ...]] = {}
        """Conformance runs only (`LocalGateway.expect_absent`): episode id -> canary
        paths that must be absent from the grading box once the reward ran. Empty in
        production, where no probe ever runs."""
        self.retry_log: dict[str, dict[str, int]] = {}
        """Conformance runs only (`LocalGateway.record_retries`): episode id -> `regraded`
        (grading attempts run again in a fresh box) and `helper_retries` (read-only helpers
        run again in place). Empty in production."""
        self.env_failures: collections.Counter[tuple[str, str]] = collections.Counter()
        """(env, kind) -> count of env-code failures, for the capacity/metrics report."""
        self.pool: Any = None
        """The machine's `GradingPool`, set by the manager; None: no pool."""

    def _note_retries(self, episode_id: str, box_id: str, *, regraded: int = 0) -> None:
        log = self.retry_log.get(episode_id)
        if log is None:
            return
        # Report-only counts: never let them raise out of a `finally` that releases the box and its pool share.
        try:
            take = getattr(self.boxes, "take_helper_retries", None)
            log["regraded"] += regraded
            log["helper_retries"] += take(box_id) if take is not None else 0
        except Exception:
            _log.warning("could not count the retries of %s", box_id, exc_info=True)

    def env_failure_counts(self) -> dict[str, dict[str, int]]:
        with self._lock:
            counts: dict[str, dict[str, int]] = {}
            for (env, kind), count in self.env_failures.items():
                counts.setdefault(env, {})[kind] = count
            return counts

    def _env_failure(self, env: str, kind: str, message: str, exc: BaseException | None = None):
        with self._lock:
            self.env_failures[(env, kind)] += 1
        _log.error("env %s: %s: %s", env, kind, message, exc_info=exc)

    def runtime(self, box_id: str, task: SandboxTask, timeout_s: float, *,
                grading: bool = False) -> LocalRuntime:
        """A runtime over `box_id` as its resolved numeric user; `grading`
        selects the task's grading workdir (pristine boxes)."""
        user = self.boxes.box_user(box_id)
        if user is None:
            raise InfraFault("The box user was never resolved: restore_user() it first")
        return LocalRuntime(self.boxes, box_id,
                            workdir=task.effective_grading_workdir if grading else task.workdir,
                            env=dict(task.env), user=user, deadline=time.monotonic() + timeout_s,
                            output_limit=self.settings.episode_grader_output_bytes,
                            read_limit=task.max_state_bytes)

    @staticmethod
    def pristine_box_id(episode_id: str, attempt: int) -> str:
        return "g-" + hashlib.sha256(episode_id.encode()).hexdigest()[:24] + f"-{attempt}"

    def _window(self, timeout_s: float, deadline: float | None) -> tuple[float, bool]:
        """(the step's timeout, whether the deadline cut it below the task's)."""
        if deadline is None:
            return timeout_s, False
        left = deadline - self.clock()
        return (min(timeout_s, left), left < timeout_s)

    # -- extract -------------------------------------------------------------

    def extract(self, agent_box: str, task: SandboxTask, *, env: str = "unknown",
                deadline: float | None = None) -> bytes | GradeOutcome:
        """The state bytes, or the episode's final outcome when there is none.
        Raises only when misused (called from a running event loop)."""
        _require_no_running_loop()
        timeout_s, cut = self._window(task.grading_timeout_s, deadline)
        if timeout_s <= 0:
            return GradeOutcome(STATUS_ABORTED, reason="extract: " + WINDOW_SPENT)
        started = time.monotonic()
        stopped = self._stop_agent_processes(agent_box, timeout_s, cut)
        if stopped is not None:
            return stopped
        timeout_s -= time.monotonic() - started
        if timeout_s <= 0:
            return (GradeOutcome(STATUS_ABORTED, reason="extract: " + WINDOW_SPENT) if cut
                    else _graded_zero("extract_timeout"))
        try:
            runtime = self.runtime(agent_box, task, timeout_s)
        except Exception as exc:
            _log.error("extract could not start on %s: %s", agent_box, exc)
            return GradeOutcome(STATUS_ABORTED, reason=f"extract: {type(exc).__name__}")
        try:
            state = run_step(lambda: task.extract(runtime), timeout_s)
            cause, error = _attribute(runtime, None), None
        except Exception as exc:
            state, cause, error = None, _attribute(runtime, exc), exc
        if cause == TIMEOUT and cut:
            return GradeOutcome(STATUS_ABORTED, reason="extract: " + WINDOW_SPENT)
        if cause == INFRA:
            return GradeOutcome(STATUS_ABORTED, reason="extract: infrastructure failure")
        if cause == UNSUPPORTED:
            _log.error("env %s: extract misused the runtime", env)
            return GradeOutcome(STATUS_ABORTED, reason="extract: runtime misuse (an integration bug)")
        if cause == ENV_INFRA:
            self._env_failure(env, "env_infra_error", f"extract raised {error!r}", error)
            return GradeOutcome(STATUS_ABORTED, reason="extract: the env reported its own failure")
        if cause == BOX:
            return GradeOutcome(STATUS_BOX_FAILED, reason="the agent's box is unusable")
        if cause == START:  # a machine fault either way; named for the record
            return GradeOutcome(STATUS_BOX_FAILED, reason=START_FAILED_REASON)
        label = {TIMEOUT: "extract_timeout", OVERFLOW: "state_too_large",
                 TOO_LARGE: "state_too_large", UNREADABLE: "state_unreadable"}.get(cause)
        if label:
            return _graded_zero(label)
        if cause == FAILED:
            self._env_failure(env, "extract_failed", f"extract raised {type(error).__name__}", error)
            return _graded_zero("extract_failed")
        if not isinstance(state, (bytes, bytearray)):
            self._env_failure(env, "extract_failed",
                              f"extract returned {type(state).__name__}, not bytes")
            return _graded_zero("extract_failed")
        if len(state) > task.max_state_bytes:
            return _graded_zero("state_too_large")
        return bytes(state)

    def _stop_agent_processes(self, agent_box: str, timeout_s: float,
                              cut: bool) -> GradeOutcome | None:
        """Kill everything the agent left running in its box, so its final
        state is well defined and nothing tampers with extraction. None when
        done; otherwise the episode's outcome (classified like extract)."""
        try:
            self.boxes.stop_processes(agent_box, timeout_s=timeout_s)
        except RuntimeExecFailed:
            return GradeOutcome(STATUS_BOX_FAILED, reason=START_FAILED_REASON)
        except BoxFault:
            return GradeOutcome(STATUS_BOX_FAILED, reason="the agent's box is unusable")
        except HelperBudgetTimeout:
            return (GradeOutcome(STATUS_ABORTED, reason="extract: " + WINDOW_SPENT) if cut
                    else _graded_zero("extract_timeout"))
        except Exception as exc:
            _log.warning("the agent's processes could not be stopped in %s: %s", agent_box, exc)
            return GradeOutcome(STATUS_ABORTED, reason="extract: infrastructure failure")
        return None

    # -- grade ---------------------------------------------------------------

    def grade(self, episode_id: str, agent_box: str, task: SandboxTask, spec: BoxSpec, *,
              env: str = "unknown", deadline: float | None = None, owner: str = "") -> GradeOutcome:
        """`spec` is the agent box's; the pristine box takes the task's grading
        workdir and user. The agent box is the caller's to remove. `deadline`
        (on `self.clock`) bounds the whole grading. ``owner``: the session's hotkey,
        the grading pool's fairness key. Never raises."""
        try:
            return self._grade(episode_id, agent_box, task, spec, env, deadline, owner)
        except Exception as exc:  # our own bug: never turn it into an agent verdict
            _log.exception("grading %s failed in the grader itself", episode_id)
            return GradeOutcome(STATUS_ABORTED, reason=f"grader error: {type(exc).__name__}")

    def _grade(self, episode_id: str, agent_box: str, task: SandboxTask, spec: BoxSpec,
               env: str, deadline: float | None, owner: str = "") -> GradeOutcome:
        state = self.extract(agent_box, task, env=env, deadline=deadline)
        self._note_retries(episode_id, agent_box)
        if isinstance(state, GradeOutcome):
            return state
        spec = dataclasses.replace(spec, workdir=task.effective_grading_workdir,
                                   user=task.effective_grading_user)
        reason = "no grading attempt"
        cleanup: list[str] = []
        kept: GradeOutcome | None = None
        """The zero an attempt where the runtime failed to start a command would have
        given: what the episode gets unless a later attempt grades it (never a void)."""
        for attempt in range(self.settings.episode_grading_attempts):
            if attempt and deadline is not None and (
                    deadline - self.clock() < task.grading_timeout_s + RETRY_MARGIN_S):
                if kept is not None:
                    return dataclasses.replace(kept, cleanup=tuple(cleanup))
                return GradeOutcome(STATUS_ABORTED, state=state,
                                    reason=f"grading failed: {reason}; no time left for a retry",
                                    cleanup=tuple(cleanup))
            need = grading_reservation(spec)
            if self.pool is not None:
                # After extract: only the grade step and the retry margin must still fit.
                wait_s = None if deadline is None else (
                    deadline - self.clock() - task.grading_timeout_s - RETRY_MARGIN_S)
                refused = self.pool.acquire(need, wait_s, owner)
                if refused is not None:
                    _log.error("grading %s: %s; aborted (the machine's doing)", episode_id, refused)
                    if kept is not None:
                        return dataclasses.replace(kept, cleanup=tuple(cleanup))
                    return GradeOutcome(STATUS_ABORTED, state=state, reason=refused, cleanup=tuple(cleanup))
            box_id = self.pristine_box_id(episode_id, attempt)
            removed = True
            try:
                outcome = self._attempt(box_id, episode_id, task, spec, state, env, deadline)
            finally:
                self._note_retries(episode_id, box_id)
                try:
                    self.boxes.remove(box_id)
                except Exception:
                    _log.warning("could not remove grading box %s", box_id, exc_info=True)
                    cleanup.append(box_id)
                    removed = False
                if self.pool is not None:
                    self.pool.release(need, owner)
            if isinstance(outcome, GradeOutcome):
                return dataclasses.replace(outcome, cleanup=tuple(cleanup))
            if isinstance(outcome, _StartFailed):
                kept = kept or outcome.kept
                self._note_retries(episode_id, box_id, regraded=1)
                reason = "the runtime failed to start a command in the grading box"
                _log.warning("grading %s: %s (attempt %d)", episode_id, reason, attempt + 1)
            else:
                reason = outcome
            if not removed:
                # A grading box that could not be removed may still run: no retry beside it (its
                # pool share is released; the machine's cleanup removes the box).
                reason = f"{reason}; a grading box could not be removed"
                break
        if kept is not None:
            return dataclasses.replace(kept, cleanup=tuple(cleanup))
        return GradeOutcome(STATUS_ABORTED, state=state, reason=f"grading failed: {reason}",
                            cleanup=tuple(cleanup))

    def _attempt(self, box_id: str, episode_id: str, task: SandboxTask, spec: BoxSpec,
                 state: bytes, env: str, deadline: float | None
                 ) -> GradeOutcome | _StartFailed | str:
        """The outcome, or why this attempt failed on our side (retry). For a conformance
        episode (`canaries`), a graded outcome also records whether its canary markers
        exist in the grading box, still up here: the reward is never changed. So does
        the zero a runtime start failure would keep (`_StartFailed`): a canary that ran
        before it is the case's failure, not the host's."""
        outcome = self._attempt_box(box_id, episode_id, task, spec, state, env, deadline)
        paths = self.canaries.get(episode_id)
        if paths and isinstance(outcome, GradeOutcome) and outcome.status == STATUS_GRADED:
            outcome = self._with_canaries(outcome, box_id, task, paths)
        elif paths and isinstance(outcome, _StartFailed):
            outcome = _StartFailed(self._with_canaries(outcome.kept, box_id, task, paths))
        return outcome

    def _with_canaries(self, outcome: GradeOutcome, box_id: str, task: SandboxTask,
                       paths: tuple[str, ...]) -> GradeOutcome:
        found: list[str] = []
        failed: str | None = None
        try:
            runtime = self.runtime(box_id, task, CANARY_PROBE_TIMEOUT_S, grading=True)

            async def probe() -> None:
                for path in paths:
                    if await runtime.exists(path):
                        found.append(path)

            run_step(probe, CANARY_PROBE_TIMEOUT_S)
        except Exception as exc:  # fail closed: an unchecked canary is not an absent one
            failed = type(exc).__name__
            _log.warning("canary probe of %s failed: %s", box_id, exc)
        facts = dict(outcome.facts or {})
        label = dict(facts.get(GRADER_KEY) or {})
        label["canary_checked"] = len(paths) if failed is None else 0
        label["canary_executed"] = found
        if failed is not None:
            label["canary_probe_failed"] = failed
        facts[GRADER_KEY] = label
        return dataclasses.replace(outcome, facts=facts)

    def _attempt_box(self, box_id: str, episode_id: str, task: SandboxTask, spec: BoxSpec,
                     state: bytes, env: str, deadline: float | None
                     ) -> GradeOutcome | _StartFailed | str:
        failure = self._start_grading_box(box_id, episode_id, spec)
        if failure is not None:
            return failure
        timeout_s, cut = self._window(task.grading_timeout_s, deadline)
        if timeout_s <= 0:
            return GradeOutcome(STATUS_ABORTED, state=state, reason="grade: " + WINDOW_SPENT)
        try:
            runtime = self.runtime(box_id, task, timeout_s, grading=True)
        except Exception as exc:
            _log.warning("grading box %s could not start: %s", box_id, exc)
            return f"the grading box could not start ({type(exc).__name__})"
        try:
            result = run_step(lambda: task.grade(runtime, state), timeout_s)
            cause, error = _attribute(runtime, None), None
        except Exception as exc:
            result, cause, error = None, _attribute(runtime, exc), exc
        if cause == TIMEOUT and cut:
            return GradeOutcome(STATUS_ABORTED, state=state, reason="grade: " + WINDOW_SPENT)
        if cause == INFRA:
            return "infrastructure failure during grading"
        if cause == ENV_INFRA:
            self._env_failure(env, "env_infra_error", f"grade raised {error!r}", error)
            return "the env reported its own failure during grading"
        if cause == UNSUPPORTED:
            _log.error("env %s: grade misused the runtime (unsupported member or a "
                       "repeated restore)", env)
            return GradeOutcome(STATUS_ABORTED, state=state,
                                reason="grade: runtime misuse (an integration bug)")
        cpu_ms = self._cpu_ms(box_id)
        if cpu_ms is None:
            return "the grading box's CPU could not be read"
        if cause == FAILED:
            self._env_failure(env, "grading_failed", f"grade raised {type(error).__name__}", error)
            return _graded_zero("grading_failed", state, cpu_ms)
        if cause == START:  # re-run in a fresh box; this zero is kept if none grades
            zero = _graded_zero("grading_box_failed", state, cpu_ms)
            return _StartFailed(dataclasses.replace(zero, facts=grader_facts(
                "grading_box_failed", **{RUNTIME_START_FACT: True})))
        if cause is not None:
            label = {BOX: "grading_box_failed", TIMEOUT: "grading_timeout",
                     OVERFLOW: "grading_output_overflow", TOO_LARGE: "grading_output_overflow",
                     UNREADABLE: "state_unreadable"}[cause]
            return _graded_zero(label, state, cpu_ms)
        return self._outcome(result, state, cpu_ms, env)

    def _start_grading_box(self, box_id: str, episode_id: str, spec: BoxSpec) -> str | None:
        """Create the pristine box and take a first CPU reading (kept if the box
        dies later), within `episode_grading_create_timeout_s`. None when it
        started; otherwise why not (ours: a retry, or `aborted`). The bound keeps
        the grade step's full timeout inside the window (it is below
        RETRY_MARGIN_S), so a window-cut grade can only follow our slowness.
        A create still running past it finishes in its thread; its box is
        removed with the attempt's, or later by the reaper as an orphan."""
        limit = getattr(self.settings, "episode_grading_create_timeout_s",
                        GRADING_CREATE_TIMEOUT_S)
        result: dict[str, BaseException | None] = {}
        done = threading.Event()

        def start() -> None:
            try:
                self.boxes.create(box_id, episode_id, spec)
                self.boxes.cpu_ns(box_id)
                result["error"] = None
            except BaseException as exc:  # handed to the waiting thread
                result["error"] = exc
            finally:
                done.set()

        threading.Thread(target=start, daemon=True, name=f"start-{box_id}").start()
        if not done.wait(limit):
            _log.warning("grading box %s did not start within %g s", box_id, limit)
            return f"the grading box did not start in time ({limit:g} s)"
        error = result["error"]
        if error is not None:
            _log.warning("grading box %s could not start: %s", box_id, error)
            return f"the grading box could not start ({type(error).__name__})"
        return None

    def _cpu_ms(self, box_id: str) -> int | None:
        try:
            return self.boxes.cpu_ns(box_id) // 1_000_000
        except BoxFault:
            return 0  # died before any reading; the verdict stands
        except Exception:
            return None

    def _outcome(self, result: Any, state: bytes, cpu_ms: int, env: str) -> GradeOutcome:
        problem = self._invalid(result)
        if problem:
            self._env_failure(env, "invalid_grade_result", problem)
            return _graded_zero("invalid_grade_result", state, cpu_ms)
        facts = dict(result.facts)
        size = len(canonical_json(facts))
        if size > MAX_FACTS_BYTES:
            facts = {"facts_truncated": True, GRADER_KEY: {"facts_bytes": size}}
        return GradeOutcome(STATUS_GRADED, float(result.reward), facts, state,
                            hashlib.sha256(state).hexdigest(), cpu_ms)

    @staticmethod
    def _invalid(result: Any) -> str | None:
        try:
            if not isinstance(result, GradeResult):
                return "the grading hook must return a GradeResult"
            reward = result.reward
            if isinstance(reward, bool) or not isinstance(reward, (int, float)):
                return "the reward must be a number"
            reward = float(reward)
            if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
                return "the reward must be finite and in [0, 1]"
            if not isinstance(result.facts, dict):
                return "grading facts must be an object"
            if GRADER_KEY in result.facts:
                return f"grading facts must not use the reserved {GRADER_KEY!r} key"
            canonical_json(result.facts)
        except (CanonicalError, OverflowError, TypeError, ValueError) as exc:
            return f"grading facts must be canonical JSON ({type(exc).__name__})"
        return None
