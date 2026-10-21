"""mini-swe-agent's official SWE-bench harness (`config/benchmarks/swebench.yaml` at 04d809ce, the configuration the
SWE-bench bash-only leaderboard runs): one `bash` tool through the template's tool calls (several calls a turn),
each result shown back as a tool message, mini's format error as a user message, a fixed command to submit.

The texts follow `config/benchmarks/swebench.yaml` and `models/utils/actions_toolcall.py` at 04d809ce (MIT, Copyright
(c) 2025 Kilian A. Lieret and Carlos E. Jimenez), with their Jinja templates rendered (`tests/upstream/mini/`
holds the rendered texts and the parity test renders them again). Differences, all because the episode runs on a
signed sandbox through the canonical `bash`:

* the submit command (`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT ...`) ends the episode without being run: the
  gateway grades the box's state, there is no output to read the marker from;
* a timed-out or cut command shows the gateway's flags (`<exception>`), not a Python exception;
* mini stops after 3 consecutive format errors (`RepeatedFormatError`); here the episode goes on to its limits.

Pure: no I/O, no dependency.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
FORMAT_ERROR_CALL = "?format"
LENGTH_ERROR_CALL = "?length"
LONG_OUTPUT_CHARS = 10000

SYSTEM = 'You are a helpful assistant that can interact with a computer shell to solve programming tasks.'

INSTANCE = """<pr_description>
Consider the following PR description:
{task}
</pr_description>

<instructions>
# Task Instructions

## Overview

You're a software engineer interacting continuously with a computer by submitting commands.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.
<IMPORTANT>This is an interactive process where you will think and issue AT LEAST ONE command, see the result, then think and issue your next command(s).</important>

For each response:

1. Include a THOUGHT section explaining your reasoning and what you're trying to accomplish
2. Provide one or more bash tool calls to execute

## Important Boundaries

- MODIFY: Regular source code files in /testbed (this is the working directory for all your subsequent commands)
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)

## Recommended Workflow

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust

## Command Execution Rules

You are operating in an environment where

1. You issue at least one command
2. The system executes the command(s) in a subshell
3. You see the result(s)
4. You write your next command(s)

Each response should include:

1. **Reasoning text** where you explain your analysis and plan
2. At least one tool call with your command

**CRITICAL REQUIREMENTS:**

- Your response SHOULD include reasoning text explaining what you're doing
- Your response MUST include AT LEAST ONE bash tool call. You can make MULTIPLE tool calls in a single response when the commands are independent (e.g., searching multiple files, reading different parts of the codebase).
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

Example of a CORRECT response:
<example_response>
I need to understand the Builder-related code. Let me find relevant files and check the project structure.

[Makes multiple bash tool calls: {"command": "ls -la"}, {"command": "find src -name '*.java' | grep -i builder"}, {"command": "cat README.md | head -50"}]
</example_response>

## Environment Details

- You have a full Linux shell environment
- Always use non-interactive flags (-y, -f) for commands
- Avoid interactive tools like vi, nano, or any that require user input
- You can use bash commands or invoke any tool that is available in the environment
- You can also create new tools or scripts to help you with the task
- If a tool isn't available, you can also install it

## Submission

When you've completed your work, you MUST submit your changes as a git patch.
Follow these steps IN ORDER, with SEPARATE commands:

Step 1: Create the patch file
Run `git diff -- path/to/file1 path/to/file2 > patch.txt` listing only the source files you modified.
Do NOT commit your changes.

<IMPORTANT>
The patch must only contain changes to the specific source files you modified to fix the issue.
Do not submit file creations or changes to any of the following files:

- test and reproduction files
- helper scripts, tests, or tools that you created
- installation, build, packaging, configuration, or setup scripts unless they are directly part of the issue you were fixing (you can assume that the environment is already set up for your client)
- binary or compiled files
</IMPORTANT>

Step 2: Verify your patch
Inspect patch.txt to confirm it only contains your intended changes and headers show `--- a/` and `+++ b/` paths.

Step 3: Submit (EXACT command required)
You MUST use this EXACT command to submit:

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```

If the command fails (nonzero exit status), it will not submit.

<CRITICAL>
- Creating/viewing the patch and submitting it MUST be separate commands (not combined with &&).
- If you modify patch.txt after verifying, you SHOULD verify again before submitting.
- You CANNOT continue working (reading, editing, testing) in any way on this task after submitting.
</CRITICAL>
</instructions>"""

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}

NO_TOOL_CALLS = "No tool calls found in the response. Every response MUST include at least one tool call."
FORMAT_ERROR = 'Tool call error:\n\n<error>\n' + "{error}" + '\n</error>\n\nHere is general guidance on how to submit correct toolcalls:\n\nEvery response needs to use the \'bash\' tool at least once to execute commands.\n\nCall the bash tool with your command as the argument:\n- Tool: bash\n- Arguments: {"command": "your_command_here"}\n\nIf you have completed your assignment, please consult the first message about how to\nsubmit your solution (you will not be able to continue working on this task after that).'
LENGTH_ERROR = 'Your previous response reached the output token limit (finish_reason=length) before you produced a tool call, so it was cut off. Respond more concisely and finish with exactly one bash tool call. If you need to think more, do so briefly.'

LONG_OUTPUT_WARNING = """<warning>
The output of your last command was too long.
Please try a different command that produces less output.
If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.
If you're using grep or find and it produced too much output, you can use a more selective search pattern.
If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.
</warning>"""

TIMED_OUT = "The command timed out; its process group was killed."
CUT = "The command's output exceeded the sandbox's limit and was cut."


def tool_specs(names: Sequence[str] = ()) -> tuple[dict, ...]:
    """The one tool the model is shown, whatever else the episode offers (edit is never called)."""
    return (BASH_TOOL,)


def system_prompt(names: Sequence[str] = ()) -> str:
    return SYSTEM


def turn_calls(texts: Sequence[str], offered: Sequence[str] = ()) -> list[tuple[str, str]]:
    raise ValueError("mini-swe reads the template's tool calls, not text")


def _call_error(name: str, arguments: str) -> str:
    """`parse_toolcall_actions`' error for one call, or "" when it is a valid bash call."""
    error, params = "", {}
    try:
        params = json.loads(arguments)
    except Exception as exc:
        error = f"Error parsing tool call arguments: {exc}."
    if name != "bash":
        error += f"Unknown tool '{name}'."
    if not isinstance(params, dict) or "command" not in params:
        error += "Missing 'command' argument in bash tool call."
    return error.strip()


def turn_actions(calls: Sequence[tuple[str, str]], *, ended: bool) -> list[tuple[str, str]] | None:
    """The turn as `parse_toolcall_actions` reads it: no call is mini's format error (`?length` when the turn was
    cut by its token cap, `ended` False); any call that is not a bash call with a command is the format error of
    the whole turn, none of its calls run; a submit command ends the episode (None); else the calls."""
    if not calls:
        return [(LENGTH_ERROR_CALL, "{}")] if not ended else [(FORMAT_ERROR_CALL, json.dumps({"error": NO_TOOL_CALLS}))]
    for name, arguments in calls:
        error = _call_error(name, arguments)
        if error:
            return [(FORMAT_ERROR_CALL, json.dumps({"error": error}))]
    for _, arguments in calls:
        command = json.loads(arguments)["command"]
        if isinstance(command, str) and command.lstrip().startswith(SUBMIT):
            return None
    return list(calls)


def translate(name: str, arguments: str, offered: Sequence[str] = ()) -> tuple[str, str]:
    """A read call as the canonical call: bash with its command (a local `?` call passes as is)."""
    if name.startswith("?"):
        return name, arguments
    return "bash", json.dumps({"command": json.loads(arguments)["command"]})


def refusal(name: str, arguments: str | None) -> str | None:
    """mini's text for a local error call, or None."""
    if name == LENGTH_ERROR_CALL:
        return LENGTH_ERROR
    if name == FORMAT_ERROR_CALL:
        try:
            error = str(json.loads(arguments or "{}").get("error") or NO_TOOL_CALLS)
        except (ValueError, TypeError, AttributeError):
            error = NO_TOOL_CALLS
        return FORMAT_ERROR.replace("{error}", error, 1)
    return None


def user_message(text: str) -> bool:
    """Whether a local answer goes back as a user message (mini's format errors are; results are tool messages)."""
    return text == LENGTH_ERROR or text.startswith(FORMAT_ERROR.split("{error}")[0])


def observation(body: Mapping[str, Any]) -> str:
    """mini's `observation_template` of a signed `bash` record: exception (timed out, cut), return code, then
    the output as is, or its head and tail with a warning from 10000 characters."""
    code, output = body["exit_code"], body["output"]
    if type(code) is not int or not isinstance(output, str):
        raise ValueError("a call record's exit_code must be an integer and its output a string")
    flags = [text for flag, text in ((body.get("timed_out"), TIMED_OUT), (body.get("truncated"), CUT)) if flag is True]
    head = "".join(f"<exception>{text}</exception>\n" for text in flags) + f"<returncode>{code}</returncode>\n"
    if len(output) < LONG_OUTPUT_CHARS:
        return f"{head}<output>\n{output}</output>"
    elided = len(output) - LONG_OUTPUT_CHARS
    return (f"{head}{LONG_OUTPUT_WARNING}<output_head>\n{output[:5000]}\n</output_head>\n<elided_chars>\n"
            f"{elided} characters elided\n</elided_chars>\n<output_tail>\n{output[-5000:]}\n</output_tail>")


__all__ = ["BASH_TOOL", "FORMAT_ERROR", "FORMAT_ERROR_CALL", "INSTANCE", "LENGTH_ERROR", "LENGTH_ERROR_CALL",
           "NO_TOOL_CALLS", "SUBMIT", "SYSTEM", "observation", "refusal", "system_prompt", "tool_specs", "translate",
           "turn_actions", "turn_calls", "user_message"]
