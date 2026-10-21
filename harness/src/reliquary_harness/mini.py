"""mini-swe-agent's text-based harness: one bash command per turn in a fenced block, its result
shown back as a user message, a fixed command to submit.

The texts follow mini-swe-agent's `config/mini_textbased.yaml` and
`models/utils/actions_text.py` at 04d809ce (MIT, Copyright (c) 2025 Kilian A. Lieret and
Carlos E. Jimenez), with the Jinja expressions resolved: the system information line names a
Linux box, the format error and the long-output form are written out here. Two differences, both
because the episode runs on a signed sandbox rather than in mini's own environment:

* the submit command ends the episode without being run (the gateway grades the box's state,
  there is no output to read the marker from);
* a timed-out or cut command shows the gateway's flags (`<exception>`), not a Python exception.

Pure: no I/O, no dependency.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

ACTION_REGEX = r"```mswea_bash_command\s*\n(.*?)\n```"
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
FORMAT_ERROR_CALL = "?format"
LENGTH_ERROR_CALL = "?length"
LONG_OUTPUT_CHARS = 10000

SYSTEM = """You are a helpful assistant that can interact with a computer.

Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
Your reasoning and analysis here. Explain why you want to perform the action.

```mswea_bash_command
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected."""

INSTANCE = """Please solve this issue: {task}

You can execute bash commands and edit files to implement the necessary changes.

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your changes and any possible problems.

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust
6. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
   Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

## Important Rules

1. Every response must contain exactly one action
2. The action must be enclosed in triple backticks
3. Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
   However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

<system_information>
Linux x86_64
</system_information>

## Formatting your response

Here is an example of a correct response:

<example_response>
THOUGHT: I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.

```mswea_bash_command
ls -la
```
</example_response>

## Useful command examples

### Create a new file:

```mswea_bash_command
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed:

```mswea_bash_command
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.py

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.py
```

### View file content:

```mswea_bash_command
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run

```mswea_bash_command
anything
```"""

FORMAT_ERROR = """Format error:

<error>
Expected exactly 1 action, found {found}.
</error>

Here is general guidance on how to format your response:

Please always provide EXACTLY ONE action in triple backticks, found {found} actions.

Please format your action in triple backticks as shown in <response_example>.

<response_example>
Here are some thoughts about why you want to perform the action.

```mswea_bash_command
<action>
```
</response_example>

If you have completed your assignment, please consult the first message about how to
submit your solution (you will not be able to continue working on this task after that)."""

LENGTH_ERROR = ("Your previous response reached the output token limit (finish_reason=length) before you produced "
                "a complete action, so it was cut off. Respond more concisely and provide exactly one action in the "
                "required format. If you need to think more, do so briefly.")

LONG_OUTPUT_WARNING = """<warning>
The output of your last command was too long.
Please try a different command that produces less output.
If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.
If you're using grep or find and it produced too much output, you can use a more selective search pattern.
If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.
</warning>"""

TIMED_OUT = "The command timed out; its process group was killed."
CUT = "The command's output exceeded the sandbox's limit and was cut."


def actions(content: str) -> list[str]:
    """The fenced commands of a turn's answer (its reasoning excluded), stripped."""
    return [found.strip() for found in re.findall(ACTION_REGEX, content or "", re.DOTALL)]


def turn_calls(content: str, *, ended: bool) -> list[tuple[str, str]]:
    """What a turn asks for, as canonical calls: `[("bash", {"command"})]` for exactly one
    command; `[]` for the submit command (the episode ends); otherwise one call to a tool
    nobody offers, answered locally with mini's error (`refusal`): `?format` (zero or several
    blocks), `?length` when the turn was cut by its token cap (`ended` False) with no command."""
    found = actions(content)
    if len(found) == 1:
        command = found[0]
        if command.splitlines()[0].strip().startswith(SUBMIT):
            return []
        return [("bash", json.dumps({"command": command}))]
    if not found and not ended:
        return [(LENGTH_ERROR_CALL, "{}")]
    return [(FORMAT_ERROR_CALL, json.dumps({"found": len(found)}))]


def refusal(name: str, arguments: str | None) -> str | None:
    """mini's text for a local error call, or None (another refusal keeps the gateway's text)."""
    if name == LENGTH_ERROR_CALL:
        return LENGTH_ERROR
    if name == FORMAT_ERROR_CALL:
        try:
            found = int(json.loads(arguments or "{}").get("found", 0))
        except (ValueError, TypeError, AttributeError):
            found = 0
        return FORMAT_ERROR.format(found=found)
    return None


def observation(body: Mapping[str, Any]) -> str:
    """mini's observation of a signed `bash` record: exception (timed out, cut), return code,
    then the output, or its head and tail with a warning past 10000 characters."""
    code, output = body["exit_code"], body["output"]
    if type(code) is not int or not isinstance(output, str):
        raise ValueError("a call record's exit_code must be an integer and its output a string")
    flags = [text for flag, text in ((body.get("timed_out"), TIMED_OUT), (body.get("truncated"), CUT)) if flag is True]
    head = "".join(f"<exception>{text}</exception>\n" for text in flags) + f"<returncode>{code}</returncode>\n"
    if len(output) < LONG_OUTPUT_CHARS:
        return f"{head}<output>\n{output.rstrip()}</output>"
    elided = len(output) - LONG_OUTPUT_CHARS
    return (f"{head}{LONG_OUTPUT_WARNING}<output_head>\n{output[:5000]}\n</output_head>\n<elided_chars>\n"
            f"{elided} characters elided\n</elided_chars>\n<output_tail>\n{output[-5000:]}\n</output_tail>")


__all__ = ["ACTION_REGEX", "FORMAT_ERROR", "FORMAT_ERROR_CALL", "INSTANCE", "LENGTH_ERROR", "LENGTH_ERROR_CALL", "SUBMIT",
           "SYSTEM", "actions", "observation", "refusal", "turn_calls"]
