"""What the bridge derives from a verifiers v1 Task's declarative data (env norm).

A verifiers Task declares its box in `TaskData`: `image`, `workdir`, `resources` (cpu,
memory and disk in GB), `timeout` (setup, agent, finalize, scoring, in seconds),
`artifacts`, and `Task.runtime_env()`. Everything else (pids, the per-call timeout, ...)
comes from `BridgeDefaults`. Every function here is pure and raises TaskDeclarationError
for a task the sandbox cannot serve. `network_allow` / `network_block` are ignored: our
boxes never have network.
"""

from __future__ import annotations

import json
import math
import posixpath
from dataclasses import dataclass
from typing import Any

from verifiers.v1.utils.artifacts import ARTIFACTS_DIR

from reliquary_sandbox.episode_task import TaskLimits

__all__ = [
    "BRIDGE_VERSION", "DEFAULT_TOOLS", "MAX_GRADING_TIMEOUT_S", "MAX_PREPARE_TIMEOUT_S",
    "MIN_GRADING_TIMEOUT_S", "ArtifactRoot", "BridgeDefaults",
    "TaskDeclarationError", "artifact_roots", "grading_timeout_s", "literal_runtime_env",
    "prepare_timeout_s", "setup_output", "task_limits",
]

BRIDGE_VERSION = 2
"""Bumped on any change to how the bridge prepares, extracts or grades: it is part of a
bridged env's `env_package` (`registry.bridged_env_package_of`). 2: setup, grading_setup
and finalize run in the scoring process; grading_setup runs before the restore."""
DEFAULT_TOOLS = ("bash", "edit")
MAX_GRADING_TIMEOUT_S = 810
"""The largest `grading_timeout_s` the verification window admits
(`manager.grading_fits_the_window`: 2 * 810 + 120 + 60 = 1800)."""

MIN_GRADING_TIMEOUT_S = 35.0
"""The smallest `grading_timeout_s` served: no scoring process starts with less than
`scoring_process.SCORING_STARTUP_FLOOR_S` (30 s) left after the restore, plus a 5 s
margin for the restore itself; a shorter step would grade every episode 0."""

MAX_PREPARE_TIMEOUT_S = 600.0
"""The bridge's default prepare timeout, and the most a task may ask for."""


class TaskDeclarationError(ValueError):
    """The verifiers task cannot be served on a sandbox; the message says why."""


@dataclass(frozen=True)
class BridgeDefaults:
    """What a verifiers TaskData does not declare."""

    workdir: str = "/"
    pids: int = 1024
    per_call_timeout_s: int = 180
    max_calls: int | None = None
    wall_s: int | None = None
    prepare_timeout_s: float = 600.0
    grading_timeout_s: float = 600.0
    max_artifact_bytes: int = 32 * 1024**2
    scoring_memory_bytes: int = 2 * 1024**3
    """Address space `Task.score` may add in its scoring process (RLIMIT_AS on top of
    what the process already maps); past it the episode is graded 0 `grading_timeout`."""
    scoring_ready_s: float = 120.0
    """A scoring process must be ready (the env's module imported, when the forkserver
    did not preload it) within this; past it the episode is `aborted` (ours)."""
    scoring_file_bytes: int = 64 * 1024**2
    """RLIMIT_FSIZE of the scoring process: the largest file `Task.score` may write
    locally (its TMPDIR); past it the write fails with EFBIG in the env's code. The uids
    the scoring processes run as are the service's (`RELIQUARY_SANDBOX_SCORING_UID_BASE`,
    `RELIQUARY_SANDBOX_SCORING_SLOTS`), not an env's."""


@dataclass(frozen=True)
class ArtifactRoot:
    path: str
    """Absolute and normalised."""
    required: bool


def _overlap(a: str, b: str) -> bool:
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def artifact_roots(task: Any, workdir: str) -> tuple[ArtifactRoot, ...]:
    """The task's declared artifacts (relative sources resolved against `workdir`) and,
    unless a declared root overlaps it, verifiers' implicit `/logs/artifacts` (optional).
    Exclude patterns are refused: the sandbox archives whole roots."""
    roots: list[ArtifactRoot] = []
    for artifact in task.data.artifacts:
        if artifact.exclude:
            raise TaskDeclarationError(
                f"artifact {artifact.source!r} uses exclude patterns: the sandbox archives "
                "whole roots, declare narrower ones")
        path = posixpath.join(workdir, artifact.source)
        if path.startswith("/"):
            # normpath keeps a leading "//": collapse it so `//` is `/` and `//a` is `/a`
            path = posixpath.normpath("/" + path.lstrip("/"))
        else:
            path = posixpath.normpath(path)
        if path == "/" or not path.startswith("/"):
            raise TaskDeclarationError(f"artifact {artifact.source!r} resolves to {path!r}")
        for root in roots:
            if _overlap(root.path, path):
                raise TaskDeclarationError(f"artifacts {root.path!r} and {path!r} overlap")
        roots.append(ArtifactRoot(path, bool(artifact.required)))
    if not any(_overlap(root.path, ARTIFACTS_DIR) for root in roots):
        roots.append(ArtifactRoot(ARTIFACTS_DIR, False))
    return tuple(roots)


def _string_leaves(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        return set().union(*map(_string_leaves, value)) if value else set()
    return set()


def literal_runtime_env(task: Any) -> dict[str, str]:
    """`task.runtime_env()`, refused unless every non-empty value is a literal string of
    the task's data or config: `runtime_env()` runs on the gateway, where a `${VAR}`
    template would be resolved against the gateway's own environment."""
    env = task.runtime_env()
    literals = (_string_leaves(task.data.model_dump(mode="json"))
                | _string_leaves(task.config.model_dump(mode="json")))
    out: dict[str, str] = {}
    for key, value in dict(env).items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TaskDeclarationError("runtime_env() must map strings to strings")
        if value and value not in literals:
            raise TaskDeclarationError(
                f"runtime_env()[{key!r}] is not a literal of the task's data or config: a "
                "template would be resolved against the gateway's own environment")
        out[key] = value
    return out


def _valid(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def _gib(value: float | None) -> int | None:
    return None if value is None else math.ceil(value * 1024**3)


def task_limits(data: Any, defaults: BridgeDefaults) -> TaskLimits:
    """cpus, memory and disk from `data.resources` (GB = GiB, as Harbor's MB / 1024);
    `wall_s` from `data.timeout.agent`; pids, per-call timeout and max calls from
    `defaults`. GPU tasks are refused."""
    resources, timeout = data.resources, data.timeout
    if resources.gpu:
        raise TaskDeclarationError("GPU tasks are not served")
    if timeout.agent is not None and not _valid(timeout.agent):
        raise TaskDeclarationError(f"invalid agent timeout {timeout.agent!r}")
    for name in ("cpu", "memory", "disk"):
        value = getattr(resources, name)
        if value is not None and not _valid(value):
            raise TaskDeclarationError(f"invalid resources: {name}={value!r}")
    wall_s = math.ceil(timeout.agent) if timeout.agent else defaults.wall_s
    try:
        return TaskLimits(cpus=resources.cpu, memory_bytes=_gib(resources.memory),
                          disk_bytes=_gib(resources.disk), pids=defaults.pids, wall_s=wall_s,
                          per_call_timeout_s=defaults.per_call_timeout_s,
                          max_calls=defaults.max_calls)
    except ValueError as exc:
        raise TaskDeclarationError(f"invalid resources: {exc}") from exc


def grading_timeout_s(data: Any, defaults: BridgeDefaults) -> float:
    """The larger of the declared finalize and scoring timeouts (each step, extract and
    grade, gets it), else the default; refused above MAX_GRADING_TIMEOUT_S or below
    MIN_GRADING_TIMEOUT_S."""
    declared = [t for t in (data.timeout.finalize, data.timeout.scoring) if t is not None]
    for t in declared:
        if not _valid(t):
            raise TaskDeclarationError(f"invalid grading timeout {t!r}")
    timeout_s = float(max(declared)) if declared else float(defaults.grading_timeout_s)
    if timeout_s > MAX_GRADING_TIMEOUT_S:
        raise TaskDeclarationError(
            f"grading takes up to {timeout_s:g} s; the verification window admits "
            f"{MAX_GRADING_TIMEOUT_S} s")
    if timeout_s < MIN_GRADING_TIMEOUT_S:
        raise TaskDeclarationError(
            f"grading takes up to {timeout_s:g} s; a grading step must be at least "
            f"{MIN_GRADING_TIMEOUT_S:g} s (a scoring process needs 30 s left to start, "
            "plus a 5 s margin)")
    return timeout_s


def prepare_timeout_s(data: Any, defaults: BridgeDefaults) -> float:
    """The declared setup timeout, else the default; refused above MAX_PREPARE_TIMEOUT_S
    or below MIN_GRADING_TIMEOUT_S (`Task.setup` runs in a scoring process, which needs
    30 s left to start)."""
    setup = data.timeout.setup
    if setup is None:
        return float(defaults.prepare_timeout_s)
    if not _valid(setup) or not MIN_GRADING_TIMEOUT_S <= setup <= MAX_PREPARE_TIMEOUT_S:
        raise TaskDeclarationError(
            f"invalid prepare timeout {setup!r}: must be at least {MIN_GRADING_TIMEOUT_S:g} s "
            f"(setup runs in a scoring process), at most {MAX_PREPARE_TIMEOUT_S:g} s")
    return float(setup)


def setup_output(task: Any) -> str:
    """What `prepare` returns; record 0 signs its sha256 (`setup_output_sha256`), which
    binds the bridge version and the task's content hash (`Task.hash`) into the transcript.
    A validator recomputes it from the task it loads itself."""
    return json.dumps({"bridge": BRIDGE_VERSION, "task_hash": task.hash},
                      sort_keys=True, separators=(",", ":"))
