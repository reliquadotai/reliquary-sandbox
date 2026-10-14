"""The `bash` and `edit` tools, executed in the agent's box.

Versioned by TOOLS_VERSION (copied into record 0). Semantics follow the
verifiers bash harness (`bash` returns stdout followed by stderr; `edit`
replaces a string that must occur exactly once) with two documented
differences: output beyond the cap is cut and flagged, and `edit` works on
bytes decoded as UTF-8 without newline translation.

Deliberate differences from the verifiers harness, which a miner-side runtime must match:
- a blank or whitespace-only `command` is rejected (exit 2), not executed;
- `edit` reads and writes raw bytes decoded as UTF-8: CRLF is kept, never translated to LF;
- any string argument that cannot be encoded as UTF-8 (a lone surrogate) is a bad-arguments error.
"""

from __future__ import annotations

import posixpath
import time
import uuid
from dataclasses import dataclass
from typing import Any

from reliquary_sandbox.episode_task import SandboxTask

from ..task_runtime import HelperBudgetTimeout

TOOLS_VERSION = "reliquary-tools/1"
EDIT_MIN_TIME_S = 2.0
"""An edit with less time than this left is a timed-out call, run nothing."""


@dataclass(frozen=True)
class ToolOutcome:
    output: str
    exit_code: int
    truncated: bool
    timed_out: bool


def assemble_output(stdout: bytes, stderr: bytes, cap: int, already_truncated: bool) -> tuple[str, bool]:
    combined = stdout + stderr
    return combined[:cap].decode("utf-8", "replace"), already_truncated or len(combined) > cap


def _error(message: str, exit_code: int = 2) -> ToolOutcome:
    return ToolOutcome(message, exit_code, False, False)


def _strings(arguments: dict[str, Any], names: tuple[str, ...]) -> str | None:
    if not isinstance(arguments, dict):
        return "error: arguments must be an object"
    extra = sorted(set(arguments) - set(names))
    if extra:
        return f"error: unexpected arguments: {', '.join(extra)}"
    for name in names:
        if not isinstance(arguments.get(name), str):
            return f"error: '{name}' must be a string"
        try:
            arguments[name].encode("utf-8")
        except UnicodeEncodeError:
            return f"error: '{name}' is not valid UTF-8 text"
    return None


def run_tool(boxes: Any, box_id: str, task: SandboxTask, tool: str, arguments: dict[str, Any], *,
             timeout_s: float, output_cap: int, edit_max_bytes: int) -> ToolOutcome:
    if tool not in task.tools:
        return _error(f"error: unknown tool {tool!r}", 127)
    if tool == "bash":
        problem = _strings(arguments, ("command",))
        if problem:
            return _error(problem)
        if not arguments["command"].strip() or "\x00" in arguments["command"]:
            return _error("error: 'command' must be a non-empty string without NUL bytes")
        result = boxes.bash(box_id, uuid.uuid4().hex, arguments["command"], timeout_s=timeout_s,
                            output_limit=output_cap, env=dict(task.env), workdir=task.workdir,
                            user=task.user)
        output, truncated = assemble_output(result.stdout, result.stderr, output_cap, result.truncated)
        return ToolOutcome(output, result.exit_code, truncated, result.timed_out)
    return _edit(boxes, box_id, task, arguments, edit_max_bytes, timeout_s)


class _OutOfTime(Exception):
    pass


def _edit(boxes: Any, box_id: str, task: SandboxTask, arguments: dict[str, Any], limit: int,
          timeout_s: float) -> ToolOutcome:
    """The edit's helpers share the call's timeout: each gets what is left of it.
    Running out of it (before a helper, or a helper cut by it) is a timed-out
    call, recorded like a bash timeout; the box is not blamed. A helper that
    hits its own fixed bound still raises its classified fault (BoxFault)."""
    timed_out = ToolOutcome("error: the edit ran out of time", -1, False, True)
    if timeout_s < EDIT_MIN_TIME_S:
        return timed_out
    deadline = time.monotonic() + timeout_s

    def left() -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _OutOfTime
        return remaining

    try:
        return _edit_within(boxes, box_id, task, arguments, limit, left)
    except (_OutOfTime, HelperBudgetTimeout):
        return timed_out


def _edit_within(boxes: Any, box_id: str, task: SandboxTask, arguments: dict[str, Any],
                 limit: int, left) -> ToolOutcome:
    problem = _strings(arguments, ("path", "old_str", "new_str"))
    if problem:
        return _error(problem)
    path, old, new = arguments["path"], arguments["old_str"], arguments["new_str"]
    if not path or "\x00" in path:
        return _error("error: 'path' is required")
    if not old:
        return _error("error: 'old_str' must be a non-empty string")
    target = path if path.startswith("/") else posixpath.join(task.workdir, path)
    # Plumbing runs from the controller's own busybox; a helper that cannot run
    # raises a classified fault (BoxFault: the agent's box), never a tool result.
    kind = boxes.file_kind(box_id, target, user=task.user, timeout_s=left())
    if kind == "other":
        return _error(f"error: could not read {path}: not a regular file", 1)
    if kind == "absent":
        return _error(f"error: {path} not found", 1)
    try:
        data = boxes.read_file(box_id, target, limit, user=task.user, timeout_s=left())
    except FileNotFoundError:
        return _error(f"error: {path} not found", 1)
    except IsADirectoryError:
        return _error(f"error: could not read {path}: is a directory", 1)
    except PermissionError:
        return _error(f"error: could not read {path}: permission denied", 1)
    except ValueError:
        return _error(f"error: could not read {path}: larger than {limit} bytes", 1)
    except OSError as exc:
        return _error(f"error: could not read {path}: {exc.strerror or exc}", 1)
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        return _error(f"error: could not read {path}: not UTF-8 text", 1)
    count = content.count(old)
    if count != 1:
        return _error(f"error: old_str must appear exactly once in {path} (found {count})", 1)
    try:
        boxes.write_file(box_id, target, content.replace(old, new, 1).encode("utf-8"),
                         user=task.user, timeout_s=left())
    except PermissionError:
        return _error(f"error: could not write {path}: permission denied", 1)
    except OSError as exc:  # ENOSPC: the disk is the agent's own quota
        return _error(f"error: could not write {path}: {exc.strerror or exc}", 1)
    return ToolOutcome(f"Edited {path}", 0, False, False)
