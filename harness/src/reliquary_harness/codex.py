"""OpenAI Codex CLI's harness: the template's calls to `exec_command`, `apply_patch` and `update_plan`,
results as tool messages, no submit tool (a turn without a call ends the episode).

From Codex CLI at c7234c1 (`openai/codex`, Apache License 2.0, Copyright 2025 OpenAI): the tool names
and parameters (`core/src/tools/handlers/{shell,apply_patch,plan}_spec.rs`), the base instructions of
`core/gpt-5.2-codex_prompt.md` and the patch language of `core/gpt_5_2_prompt.md` (modified as below),
the unified exec result (`Process exited with code`, `Original token count`, `Output:`, the middle cut
of a 10000-token budget at 4 bytes a token), and the apply_patch engine (`codex_patch`, a Python port of
`codex-rs/apply-patch`). Differences, all because an episode runs on a signed sandbox through canonical
tools and our model is not OpenAI's:

* the prompt does not name the model or Codex ("You are a coding agent..."), and its CLI-only sections
  (frontend design, approvals, the final answer's CLI styling) are left out;
* `apply_patch` is a function with one `input` string (the Qwen template has no freeform tool), run as
  a bash command (the Python engine, the patch base64, `--model-output`: Codex's own wrapping of a
  success, wall time included), whose first line names it;
* `exec_command` keeps `cmd` and `workdir` only (no PTY session, `write_stdin`, `yield_time_ms`,
  sandbox escalation); each command runs alone; its result has no wall time or session id (the
  sandbox signs no clock);
* `update_plan` is answered locally ("Plan updated").

Pure: no I/O, no dependency.
"""

from __future__ import annotations

import base64
import json
import math
import zlib
from collections.abc import Mapping, Sequence
from typing import Any

EXEC, PATCH, PLAN = "exec_command", "apply_patch", "update_plan"
PATCH_MARK = "# codex:apply_patch"
UNKNOWN_PREFIX, INVALID_CALL, PLAN_CALL = "?", "?invalid", "?update_plan"
MAX_OUTPUT_TOKENS = 10000
BYTES_PER_TOKEN = 4

SYSTEM = """You are a coding agent running in a command-line tool on a user's computer.

## General

- When searching for text or files, prefer using `rg` or `rg --files` respectively because `rg` is much faster than alternatives like `grep`. (If the `rg` command is not found, then use alternatives.)

## Editing constraints

- Default to ASCII when editing or creating files. Only introduce non-ASCII or other Unicode characters when there is a clear justification and the file already uses them.
- Add succinct code comments that explain what is going on if code is not self-explanatory. You should not add comments like "Assigns the value to the variable", but a brief comment might be useful ahead of a complex code block that the user would otherwise have to spend time parsing out. Usage of these comments should be rare.
- Try to use apply_patch for single file edits, but it is fine to explore other options to make the edit if it does not work well. Do not use apply_patch for changes that are auto-generated (i.e. generating package.json or running a lint or format command like gofmt) or when scripting is more efficient (such as search and replacing a string across a codebase).
- You may be in a dirty git worktree.
    * NEVER revert existing changes you did not make unless explicitly requested, since these changes were made by the user.
    * If asked to make a commit or code edits and there are unrelated changes to your work or changes that you didn't make in those files, don't revert those changes.
    * If the changes are in files you've touched recently, you should read carefully and understand how you can work with the changes rather than reverting them.
    * If the changes are in unrelated files, just ignore them and don't revert them.
- Do not amend a commit unless explicitly requested to do so.
- **NEVER** use destructive commands like `git reset --hard` or `git checkout --` unless specifically requested or approved by the user.

## Plan tool

When using the planning tool:
- Skip using the planning tool for straightforward tasks (roughly the easiest 25%).
- Do not make single-step plans.
- When you made a plan, update it after having performed one of the sub-tasks that you shared on the plan.

## apply_patch

Use the `apply_patch` tool to edit files. Your patch language is a stripped‑down, file‑oriented diff format designed to be easy to parse and safe to apply. You can think of it as a high‑level envelope:

*** Begin Patch
[ one or more file sections ]
*** End Patch

Within that envelope, you get a sequence of file operations.
You MUST include a header to specify the action you are taking.
Each operation starts with one of three headers:

*** Add File: <path> - create a new file. Every following line is a + line (the initial contents).
*** Delete File: <path> - remove an existing file. Nothing follows.
*** Update File: <path> - patch an existing file in place (optionally with a rename).

Example patch:

```
*** Begin Patch
*** Add File: hello.txt
+Hello world
*** Update File: src/app.py
*** Move to: src/main.py
@@ def greet():
-print("Hi")
+print("Hello, world!")
*** Delete File: obsolete.txt
*** End Patch
```

It is important to remember:

- You must include a header with your intended action (Add/Delete/Update)
- You must prefix new lines with `+` even when creating a new file
- Do not waste tokens by re-reading files after calling `apply_patch` on them. The tool call will fail if it didn't work.

## Commands

Each `exec_command` call runs in a fresh shell: the working directory and environment variables do not carry over between calls. Use `workdir`, or `cd /path && ...`, and absolute paths.

## Finishing

When the task is done, reply to the user with a brief explanation of the change, and do not call a tool: a reply without a tool call ends your turn and the task."""


def _tool(name: str, description: str, properties: Mapping[str, Any], required: Sequence[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": {k: dict(v) for k, v in properties.items()}, "required": list(required)}}}


TOOLS = (
    _tool(EXEC, "Runs a command, returning its output.", {
        "cmd": {"type": "string", "description": "Shell command to execute."},
        "workdir": {"type": "string", "description": "Working directory for the command."},
    }, ["cmd"]),
    _tool(PATCH, "The `apply_patch` tool can be used to edit files. Pass the whole patch, from `*** Begin Patch` "
                 "to `*** End Patch`, as `input`.", {
        "input": {"type": "string", "description": "The entire contents of the apply_patch command."},
    }, ["input"]),
    _tool(PLAN, "Updates the task plan.\nProvide an optional explanation and a list of plan items, each with a step "
                "and status.\nAt most one step can be in_progress at a time.", {
        "explanation": {"type": "string"},
        "plan": {"type": "array", "description": "The list of steps", "items": {
            "type": "object", "properties": {
                "step": {"type": "string"},
                "status": {"type": "string", "description": "One of: pending, in_progress, completed",
                           "enum": ["pending", "in_progress", "completed"]}},
            "required": ["step", "status"], "additionalProperties": False}},
    }, ["plan"]),
)


def tool_specs(names: Sequence[str] = ()) -> tuple[dict, ...]:
    return TOOLS


def system_prompt(names: Sequence[str] = ()) -> str:
    return SYSTEM


def turn_calls(texts: Sequence[str], offered: Sequence[str] = ()) -> list[tuple[str, str]]:
    raise ValueError("codex reads the template's tool calls, not text")


def _shell_quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def patch_command(patch: str) -> str:
    """The bash command that runs one `apply_patch` call in the box (the Codex engine, its patch base64)."""
    from .codex_patch import PROGRAM

    program = base64.b64encode(zlib.compress(PROGRAM.strip().encode("utf-8"), 9)).decode("ascii")
    argument = base64.b64encode(patch.encode("utf-8")).decode("ascii")
    return (f"{PATCH_MARK}\npython3 -I -c 'import base64,zlib;exec(zlib.decompress(base64.b64decode("
            f"\"{program}\")))' '{argument}' --model-output")


def _invalid(name: str, arguments: str) -> str | None:
    try:
        params = json.loads(arguments)
    except (TypeError, ValueError):
        return f"failed to parse function arguments: {arguments!r} is not JSON"
    if not isinstance(params, dict):
        return "failed to parse function arguments: not a JSON object"
    schema = next(tool for tool in TOOLS if tool["function"]["name"] == name)["function"]["parameters"]
    for key in schema["required"]:
        if key not in params:
            return f"failed to parse function arguments: missing field `{key}`"
    for key, value in params.items():
        if key not in schema["properties"]:
            return f"failed to parse function arguments: unknown field `{key}`"
        if schema["properties"][key]["type"] == "string" and not isinstance(value, str):
            return f"failed to parse function arguments: invalid type for `{key}`, expected a string"
    if name == PLAN and not isinstance(params["plan"], list):
        return "failed to parse function arguments: invalid type for `plan`, expected a sequence"
    return None


def translate(name: str, arguments: str, offered: Sequence[str] = ()) -> tuple[str, str] | None:
    """The canonical call a tool call runs as: `exec_command` on bash (in its `workdir`), `apply_patch` as the
    patch engine on bash, `update_plan` answered locally; arguments that do not fit are answered with the
    error (`?invalid`), any other name is a tool nobody offers (`?<name>`). Total: it never raises."""
    if name not in (EXEC, PATCH, PLAN):
        return UNKNOWN_PREFIX + str(name), "{}"
    error = _invalid(name, arguments)
    if error is not None:
        return INVALID_CALL, json.dumps({"message": error})
    params = json.loads(arguments)
    if name == PLAN:
        return PLAN_CALL, "{}"
    if name == PATCH:
        return "bash", json.dumps({"command": patch_command(params["input"])})
    command = params["cmd"]
    if params.get("workdir"):
        command = f"cd {_shell_quote(params['workdir'])} && {command}"
    return "bash", json.dumps({"command": command})


def truncate(output: str, max_tokens: int = MAX_OUTPUT_TOKENS) -> tuple[str, int | None]:
    """The output within `max_tokens` (4 bytes a token): half the budget from its start, half from its end,
    `…N tokens truncated…` between; and its original token count when it was cut (else None)."""
    data = output.encode("utf-8")
    budget = max_tokens * BYTES_PER_TOKEN
    if len(data) <= budget:
        return output, None
    head = data[:budget // 2].decode("utf-8", "ignore")
    tail = data[len(data) - budget // 2:].decode("utf-8", "ignore")
    removed = len(data) - len(head.encode("utf-8")) - len(tail.encode("utf-8"))
    return (f"{head}…{math.ceil(removed / BYTES_PER_TOKEN)} tokens truncated…{tail}",
            math.ceil(len(data) / BYTES_PER_TOKEN))


def observation(body: Mapping[str, Any]) -> str:
    """The tool result for a signed bash record: the patch engine's output, or the command's unified exec
    result (exit code, original token count when cut, output)."""
    output, code = body["output"], body["exit_code"]
    if not isinstance(output, str) or type(code) is not int:
        raise ValueError("a call record's output must be a string and its exit_code an integer")
    if body["arguments"]["command"].startswith(PATCH_MARK + "\n"):
        if body["timed_out"]:
            return "apply_patch timed out"
        return output
    text, original = truncate(output)
    lines = ["Process timed out at the sandbox's time limit for one command" if body["timed_out"]
             else f"Process exited with code {code}"]
    if original is not None:
        lines.append(f"Original token count: {original}")
    lines.append("Output:")
    return "\n".join(lines) + "\n" + text


def refusal(name: str, arguments: str | None) -> str | None:
    """The result of a call answered locally: the plan's acknowledgement, invalid arguments, an unknown tool."""
    if name == PLAN_CALL:
        return "Plan updated"
    if name == INVALID_CALL:
        return json.loads(arguments or "{}").get("message", "")
    if name.startswith(UNKNOWN_PREFIX):
        return f"unsupported call: {name[len(UNKNOWN_PREFIX):]}"
    return None


__all__ = ["EXEC", "INVALID_CALL", "PATCH", "PATCH_MARK", "PLAN", "PLAN_CALL", "SYSTEM", "TOOLS", "observation",
           "patch_command", "refusal", "system_prompt", "tool_specs", "translate", "truncate", "turn_calls"]
