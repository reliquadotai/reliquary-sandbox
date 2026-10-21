"""A Claude Code-style harness: the template's tool calls to `Bash`, `Read`, `Edit` and `Write`, several
calls per turn allowed, results as tool messages, and no submit tool (a turn without a call ends the episode).

The tool names and parameters follow Claude Code 2.1.306's published tool input types (`sdk-tools.d.ts` of
`@anthropic-ai/claude-agent-sdk` 0.3.306): `Bash{command, timeout?, description?}`,
`Read{file_path, offset?, limit?}`, `Edit{file_path, old_string, new_string, replace_all?}`,
`Write{file_path, content}`. Claude Code itself is not open source: the system prompt, the tool
descriptions and the result texts here are this package's own, written to behave the way the published
behaviour is documented (exact single-occurrence edit unless `replace_all`, numbered reads of 2000 lines
by default, a full-file write), never copied. Glob and Grep are left out, as Claude Code leaves them out of
its default tool set on Linux (search goes through Bash). Differences forced by the signed sandbox:

* each Bash call runs alone (no persistent shell, `run_in_background` or per-call timeout);
* Read, Edit and Write run as bash commands (a small Python program, `_FILES`, its arguments base64),
  whose first line names the tool, so the observation is rendered as that tool's;
* no read-before-edit rule (it would need the episode's history inside the box).

Pure: no I/O, no dependency.
"""

from __future__ import annotations

import base64
import json
import zlib
from collections.abc import Mapping, Sequence
from typing import Any

BASH, READ, EDIT, WRITE = "Bash", "Read", "Edit", "Write"
FILE_MARK = "# claude-code:"
MAX_OUTPUT_CHARS = 30000
NO_OUTPUT = "(No output)"

SYSTEM = """You are a coding agent working in a software repository on a Linux machine. You solve the user's task by reading code, running commands and editing files with the tools below, then you stop.

# Working style
- Be direct and concise. Explain briefly what you are about to do when it is not obvious, then do it.
- Understand before changing: find the relevant code with Bash (grep, find, git), then Read the files you need.
- Make the smallest change that solves the task. Match the style of the surrounding code. Do not add files, comments or documentation that were not asked for.
- Prefer editing existing files with Edit over rewriting them with Write. Use Write for new files.
- Verify your work: run the relevant tests or a reproduction script with Bash, and fix what fails.
- Several independent tool calls can be made in the same turn; they run in order.

# Tools
- Bash runs one shell command. Each command starts in a fresh shell: the working directory and environment variables do not carry over, so use absolute paths or `cd /path && ...`. Long outputs are truncated.
- Read shows a file with line numbers. Use offset and limit to read part of a large file.
- Edit replaces an exact string in a file. old_string must match the file exactly, whitespace and indentation included, and must occur once unless replace_all is true.
- Write writes a whole file, creating it (and its directories) or replacing its content.

# Finishing
When the task is done, reply with a short summary of what you changed and do not call any tool: a reply without a tool call ends your work."""


def _tool(name: str, description: str, properties: Mapping[str, Any], required: Sequence[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": {k: dict(v) for k, v in properties.items()}, "required": list(required)}}}


TOOLS = (
    _tool(BASH, "Run a shell command and return its output (stdout and stderr). Each call runs in a fresh shell.", {
        "command": {"type": "string", "description": "The command to run."},
        "timeout": {"type": "number", "description": "Optional timeout in milliseconds (the sandbox's own limit "
                                                     "applies)."},
        "description": {"type": "string", "description": "A short description of what the command does."},
    }, ["command"]),
    _tool(READ, "Read a text file. The result shows each line prefixed by its line number and a tab, starting at 1. "
                "By default up to 2000 lines are shown from the start of the file; lines longer than 2000 characters "
                "are cut.", {
        "file_path": {"type": "string", "description": "The absolute path to the file to read."},
        "offset": {"type": "number", "description": "The line number to start reading from. Only provide it if the "
                                                    "file is too large to read at once."},
        "limit": {"type": "number", "description": "The number of lines to read. Only provide it if the file is too "
                                                   "large to read at once."},
    }, ["file_path"]),
    _tool(EDIT, "Replace an exact string in a file. old_string must appear in the file exactly as given; it must be "
                "unique unless replace_all is true. An empty old_string creates a new file holding new_string.", {
        "file_path": {"type": "string", "description": "The absolute path to the file to modify."},
        "old_string": {"type": "string", "description": "The text to replace."},
        "new_string": {"type": "string", "description": "The text to replace it with (must be different from "
                                                        "old_string)."},
        "replace_all": {"type": "boolean", "description": "Replace all occurrences of old_string (default false)."},
    }, ["file_path", "old_string", "new_string"]),
    _tool(WRITE, "Write a file: create it, or replace its whole content. Missing parent directories are created.", {
        "file_path": {"type": "string", "description": "The absolute path to the file to write (must be absolute, "
                                                       "not relative)."},
        "content": {"type": "string", "description": "The content to write to the file."},
    }, ["file_path", "content"]),
)


def tool_specs(names: Sequence[str] = ()) -> tuple[dict, ...]:
    """The tools the model is shown: the same four for every offered set (all run through bash)."""
    return TOOLS


def system_prompt(names: Sequence[str] = ()) -> str:
    return SYSTEM


# The file tools, run with python3 in the box: their call is base64 JSON in argv[1] ({"tool", ...the
# tool's parameters}); each prints its result and exits 1 on an error (printed inside <tool_use_error>).
_FILES = r'''
import base64, json, os, sys
a = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
class E(Exception): pass
def read(p):
    if not os.path.exists(p): raise E("File does not exist.")
    if os.path.isdir(p): raise E("EISDIR: illegal operation on a directory, read")
    try:
        with open(p, encoding="utf-8", newline="") as f: return f.read()
    except UnicodeDecodeError: raise E("This file is not valid UTF-8 text and cannot be read as text.")
def write(p, s):
    d = os.path.dirname(p)
    if d: os.makedirs(d, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f: f.write(s)
def num(v, name):
    if v is None: return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v: raise E(f"{name} must be a whole number.")
    return int(v)
def run():
    t, p = a["tool"], a["file_path"]
    if t == "Read":
        s = read(p)
        if s == "": return "Warning: the file exists but the contents are empty."
        lines = s.split("\n")
        if lines[-1] == "": lines.pop()
        o, l = num(a.get("offset"), "offset"), num(a.get("limit"), "limit")
        start = max(o or 1, 1)
        count = 2000 if l is None else max(l, 0)
        if start > len(lines):
            return f"Warning: the file exists but is shorter than the provided offset ({start}). The file has {len(lines)} lines."
        shown = lines[start - 1:start - 1 + count]
        body = "\n".join(f"{start + i}\t{x if len(x) <= 2000 else x[:2000] + '... [line truncated]'}" for i, x in enumerate(shown))
        rest = len(lines) - (start - 1 + len(shown))
        if rest > 0: body += f"\n\n[Showing lines {start}-{start + len(shown) - 1} of {len(lines)}. Use offset and limit to read more.]"
        return body
    if t == "Write":
        existed = os.path.exists(p)
        if existed and os.path.isdir(p): raise E("EISDIR: illegal operation on a directory, open")
        write(p, a["content"])
        return f"The file {p} has been updated successfully." if existed else f"File created successfully at: {p}"
    old, new = a["old_string"], a["new_string"]
    if old == new: raise E("No changes to make: old_string and new_string are exactly the same.")
    if old == "":
        if os.path.exists(p) and (os.path.isdir(p) or os.path.getsize(p) > 0): raise E("Cannot create new file - file already exists.")
        write(p, new)
        return f"File created successfully at: {p}"
    s = read(p)
    n = s.count(old)
    if n == 0: raise E(f"String to replace not found in file.\nString: {old}")
    if n > 1 and not a.get("replace_all"):
        raise E(f"Found {n} matches of the string to replace, but replace_all is false. To replace all occurrences, set replace_all to true. To replace only one occurrence, please provide more context to uniquely identify the instance.\nString: {old}")
    write(p, s.replace(old, new) if a.get("replace_all") else s.replace(old, new, 1))
    return f"The file {p} has been updated successfully."
try:
    sys.stdout.write(run())
except E as e:
    sys.stdout.write(f"<tool_use_error>{e}</tool_use_error>"); sys.exit(1)
except OSError as e:
    sys.stdout.write(f"<tool_use_error>{e}</tool_use_error>"); sys.exit(1)
'''
_FILES_B64 = base64.b64encode(zlib.compress(_FILES.strip().encode("utf-8"), 9)).decode("ascii")


def file_command(tool: str, params: Mapping[str, Any]) -> str:
    """The bash command that runs one Read, Edit or Write call in the box."""
    arguments = base64.b64encode(json.dumps({"tool": tool, **dict(params)}, ensure_ascii=False).encode("utf-8"))
    return (f"{FILE_MARK}{tool}\npython3 -I -c 'import base64,zlib;exec(zlib.decompress(base64.b64decode("
            f"\"{_FILES_B64}\")))' '{arguments.decode('ascii')}'")


def turn_calls(texts: Sequence[str], offered: Sequence[str] = ()) -> list[tuple[str, str]]:
    raise ValueError("claude-code reads the template's tool calls, not text")


UNKNOWN_PREFIX, INVALID_CALL = "?", "?invalid"
_TYPES = {"string": (str,), "number": (int, float), "boolean": (bool,)}


def validation_error(name: str, arguments: str) -> str | None:
    """Why a call's arguments do not fit its tool's schema (required, allowed and typed parameters), or None."""
    try:
        params = json.loads(arguments)
    except (TypeError, ValueError):
        return f"The parameters of {name} are not a JSON object."
    if not isinstance(params, dict):
        return f"The parameters of {name} are not a JSON object."
    schema = next(tool for tool in TOOLS if tool["function"]["name"] == name)["function"]["parameters"]
    for key in schema["required"]:
        if key not in params:
            return f"The required parameter `{key}` is missing"
    for key, value in params.items():
        if key not in schema["properties"]:
            return f"An unexpected parameter `{key}` was provided"
        kind = schema["properties"][key]["type"]
        if not isinstance(value, _TYPES[kind]) or (kind == "number" and isinstance(value, bool)):
            return f"The parameter `{key}` type is expected as `{kind}` but provided as `{type(value).__name__}`"
    return None


def translate(name: str, arguments: str, offered: Sequence[str] = ()) -> tuple[str, str] | None:
    """The canonical call a tool call runs as: Bash's command on bash, the file tools as their program on
    bash. A call whose arguments do not fit is answered locally (`?invalid`, its error), and so is any
    other name (`?<name>`: a tool nobody offers, never run). Total: it never raises."""
    if name not in (BASH, READ, EDIT, WRITE):
        return UNKNOWN_PREFIX + str(name), "{}"
    error = validation_error(name, arguments)
    if error is not None:
        return INVALID_CALL, json.dumps({"tool": name, "message": error})
    params = json.loads(arguments)
    if name == BASH:
        return "bash", json.dumps({"command": params["command"]})
    return "bash", json.dumps({"command": file_command(name, params)})


def truncate(output: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(output) <= limit:
        return output
    return output[:limit] + f"\n\n... [output truncated: {len(output) - limit} more characters]"


def observation(body: Mapping[str, Any]) -> str:
    """The tool result for a signed bash record: a file tool's output, or a command's output (an exit code
    other than 0 on the first line)."""
    output, code = body["output"], body["exit_code"]
    if not isinstance(output, str) or type(code) is not int:
        raise ValueError("a call record's output must be a string and its exit_code an integer")
    command = body["arguments"]["command"]
    for tool in (READ, EDIT, WRITE):
        if command.startswith(f"{FILE_MARK}{tool}\n"):
            if body["timed_out"]:
                return f"<tool_use_error>{tool} timed out.</tool_use_error>"
            return truncate(output)
    text = truncate(output) if output.strip() else NO_OUTPUT
    if body["timed_out"]:
        return f"Command timed out at the sandbox's time limit for one command.\n{text}"
    return text if code == 0 else f"Exit code {code}\n{text}"


def refusal(name: str, arguments: str | None) -> str | None:
    """The tool result of a call answered locally: invalid parameters, or a tool that does not exist."""
    if name == INVALID_CALL:
        details = json.loads(arguments or "{}")
        return (f"<tool_use_error>InputValidationError: {details.get('tool')} failed due to the following issue:\n"
                f"{details.get('message')}</tool_use_error>")
    if name.startswith(UNKNOWN_PREFIX):
        return f"<tool_use_error>Error: No such tool available: {name[len(UNKNOWN_PREFIX):]}</tool_use_error>"
    return None


__all__ = ["BASH", "EDIT", "FILE_MARK", "READ", "SYSTEM", "TOOLS", "WRITE", "file_command", "observation", "refusal",
           "system_prompt", "tool_specs", "translate", "truncate", "turn_calls"]
