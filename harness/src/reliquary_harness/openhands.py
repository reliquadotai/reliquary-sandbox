"""OpenHands CodeAct's text (non-function-calling) harness: its system prompt and tools described
in the prompt, one `<function=NAME><parameter=K>V</parameter></function>` call per turn, results
shown back as user messages (`EXECUTION RESULT of [tool]:`).

The texts follow OpenHands 0.59.0 (d39f7ae, MIT, Copyright (c) 2023 All Hands AI):
`agenthub/codeact_agent/prompts/system_prompt.j2`, `tools/{bash,think,finish,str_replace_editor}.py`,
`llm/fn_call_converter.py` (the function description, the suffix, the call regexes, the
parameter checks, `_fix_stopword`), `events/serialization/event.py` (the middle truncation),
`evaluation/utils/shared.py` (`codeact_user_response`), and the editor's messages of
`software-agent-sdk` 0996c33 (`openhands-tools/.../file_editor/editor.py`). The differences, all
because an episode runs on a signed sandbox through two canonical tools:

* the shell is not persistent: each `execute_bash` runs alone (no `is_input`, soft timeout or
  `timeout` parameter); the description says so;
* `str_replace_editor` runs as a bash command (a small Python editor, `_EDITOR`, its arguments
  base64), without `undo_edit` or the Markdown view of binary files; its first line names it,
  so the observation is rendered as the editor's;
* no `security_risk` parameter, no in-context example, no workspace path or base commit in the
  task (the sandbox's task text is the issue);
* a turn without a function call is answered with OpenHands' evaluation reply
  (`codeact_user_response`), and the episode goes on; `finish` ends it (the gateway grades the box).

Pure: no I/O, no dependency.
"""

from __future__ import annotations

import base64
import json
import re
import zlib
from collections.abc import Mapping, Sequence
from typing import Any

EXECUTE_BASH, THINK, FINISH, EDITOR = "execute_bash", "think", "finish", "str_replace_editor"
SUBMIT_TOOL = FINISH
CONTINUE_CALL, ERROR_CALL, THINK_CALL = "?continue", "?error", "?think"
EDITOR_MARK = "# openhands:str_replace_editor"
MAX_MESSAGE_CHARS = 30000

SYSTEM_BASE = """You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.

<ROLE>
Your primary role is to assist users by executing commands, modifying code, and solving technical problems effectively. You should be thorough, methodical, and prioritize quality over speed.
* If the user asks a question, like "why is X happening", don't try to fix the problem. Just give an answer to the question.
</ROLE>

<EFFICIENCY>
* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.
* When exploring the codebase, use efficient tools like find, grep, and git commands with appropriate filters to minimize unnecessary operations.
</EFFICIENCY>

<FILE_SYSTEM_GUIDELINES>
* When a user provides a file path, do NOT assume it's relative to the current working directory. First explore the file system to locate the file before working on it.
* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.
* For global search-and-replace operations, consider using `sed` instead of opening file editors multiple times.
* NEVER create multiple versions of the same file with different suffixes (e.g., file_test.py, file_fix.py, file_simple.py). Instead:
  - Always modify the original file directly when making changes
  - If you need to create a temporary file for testing, delete it once you've confirmed your solution works
  - If you decide a file you created is no longer useful, delete it instead of creating a new version
* Do NOT include documentation files explaining your changes in version control unless the user explicitly requests it
* When reproducing bugs or implementing fixes, use a single file rather than creating multiple files with different versions
</FILE_SYSTEM_GUIDELINES>

<CODE_QUALITY>
* Write clean, efficient code with minimal comments. Avoid redundancy in comments: Do not repeat information that can be easily inferred from the code itself.
* When implementing solutions, focus on making the minimal changes needed to solve the problem.
* Before implementing any changes, first thoroughly understand the codebase through exploration.
* If you are adding a lot of code to a function or file, consider splitting the function or file into smaller pieces when appropriate.
* Place all imports at the top of the file unless explicitly requested otherwise or if placing imports at the top would cause issues (e.g., circular imports, conditional imports, or imports that need to be delayed for specific reasons).
* If working in a git repo, before you commit code create a .gitignore file if one doesn't exist. And if there are existing files that should not be included then update the .gitignore file as appropriate.
</CODE_QUALITY>

<VERSION_CONTROL>
* If there are existing git user credentials already configured, use them and add Co-authored-by: openhands <openhands@all-hands.dev> to any commits messages you make. if a git config doesn't exist use "openhands" as the user.name and "openhands@all-hands.dev" as the user.email by default, unless explicitly instructed otherwise.
* Exercise caution with git operations. Do NOT make potentially dangerous changes (e.g., pushing to main, deleting repositories) unless explicitly asked to do so.
* When committing changes, use `git status` to see all modified files, and stage all files necessary for the commit. Use `git commit -a` whenever possible.
* Do NOT commit files that typically shouldn't go into version control (e.g., node_modules/, .env files, build directories, cache files, large binaries) unless explicitly instructed by the user.
* If unsure about committing certain files, check for the presence of .gitignore files or ask the user for clarification.
</VERSION_CONTROL>

<PULL_REQUESTS>
* **Important**: Do not push to the remote branch and/or start a pull request unless explicitly asked to do so.
* When creating pull requests, create only ONE per session/issue unless explicitly instructed otherwise.
* When working with an existing PR, update it with new commits rather than creating additional PRs for the same issue.
* When updating a PR, preserve the original PR title and purpose, updating description only when necessary.
</PULL_REQUESTS>

<PROBLEM_SOLVING_WORKFLOW>
1. EXPLORATION: Thoroughly explore relevant files and understand the context before proposing solutions
2. ANALYSIS: Consider multiple approaches and select the most promising one
3. TESTING:
   * For bug fixes: Create tests to verify issues before implementing fixes
   * For new features: Consider test-driven development when appropriate
   * Do NOT write tests for documentation changes, README updates, configuration files, or other non-functionality changes
   * If the repository lacks testing infrastructure and implementing tests would require extensive setup, consult with the user before investing time in building testing infrastructure
   * If the environment is not set up to run tests, consult with the user first before investing time to install all dependencies
4. IMPLEMENTATION:
   * Make focused, minimal changes to address the problem
   * Always modify existing files directly rather than creating new versions with different suffixes
   * If you create temporary files for testing, delete them after confirming your solution works
5. VERIFICATION: If the environment is set up to run tests, test your implementation thoroughly, including edge cases. If the environment is not set up to run tests, consult with the user first before investing time to run tests.
</PROBLEM_SOLVING_WORKFLOW>

<SECURITY>
* Only use GITHUB_TOKEN and other credentials in ways the user has explicitly requested and would expect.
* Use APIs to work with GitHub or other platforms, unless the user asks otherwise or your task requires browsing.
</SECURITY>

<EXTERNAL_SERVICES>
* When interacting with external services like GitHub, GitLab, or Bitbucket, use their respective APIs instead of browser-based interactions whenever possible.
* Only resort to browser-based interactions with these services if specifically requested by the user or if the required operation cannot be performed via API.
</EXTERNAL_SERVICES>

<ENVIRONMENT_SETUP>
* When user asks you to run an application, don't stop if the application is not installed. Instead, please install the application and run the command again.
* If you encounter missing dependencies:
  1. First, look around in the repository for existing dependency files (requirements.txt, pyproject.toml, package.json, Gemfile, etc.)
  2. If dependency files exist, use them to install all dependencies at once (e.g., `pip install -r requirements.txt`, `npm install`, etc.)
  3. Only install individual packages directly if no dependency files are found or if only specific packages are needed
* Similarly, if you encounter missing dependencies for essential tools requested by the user, install them when possible.
</ENVIRONMENT_SETUP>

<TROUBLESHOOTING>
* If you've made repeated attempts to solve a problem but tests still fail or the user reports it's still broken:
  1. Step back and reflect on 5-7 different possible sources of the problem
  2. Assess the likelihood of each possible cause
  3. Methodically address the most likely causes, starting with the highest probability
  4. Document your reasoning process
* When you run into any major issue while executing a plan from the user, please don't try to directly work around it. Instead, propose a new plan and confirm with the user before proceeding.
</TROUBLESHOOTING>

<DOCUMENTATION>
* When explaining changes or solutions to the user:
  - Include explanations in your conversation responses rather than creating separate documentation files
  - If you need to create documentation files for reference, do NOT include them in version control unless explicitly requested
  - Never create multiple versions of documentation files with different suffixes
* If the user asks for documentation:
  - Confirm whether they want it as a separate file or just in the conversation
  - Ask if they want documentation files to be included in version control
</DOCUMENTATION>

<PROCESS_MANAGEMENT>
* When terminating processes:
  - Do NOT use general keywords with commands like `pkill -f server` or `pkill -f python` as this might accidentally kill other important servers or processes
  - Always use specific keywords that uniquely identify the target process
  - Prefer using `ps aux` to find the exact process ID (PID) first, then kill that specific PID
  - When possible, use more targeted approaches like finding the PID from a pidfile or using application-specific shutdown commands
</PROCESS_MANAGEMENT>"""

BASH_DESCRIPTION = """Execute a bash command in the terminal.


### Command Execution
* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.
* Fresh shell: Each command runs in a new shell. Environment variables, virtual environments and the working directory do NOT persist between commands: prefix a command with `cd /path &&` or export what it needs.
* Shell options: Do NOT use `set -e`, `set -eu`, or `set -euo pipefail` in shell scripts or commands in this environment. If you want to run multi-line bash commands, write the commands to a file and then run it, instead.

### Long-running Commands
* Every command has a time limit. For commands that may run indefinitely, run them in the background and redirect output to a file, e.g. `python3 app.py > server.log 2>&1 &`.

### Best Practices
* Directory verification: Before creating new directories or files, first verify the parent directory exists and is the correct location.
* Directory management: Try to maintain working directory by using absolute paths and avoiding excessive use of `cd`.

### Output Handling
* Output truncation: If the output exceeds a maximum length, it will be truncated before being returned.
"""

THINK_DESCRIPTION = """Use the tool to think about something. It will not obtain new information or make any changes to the repository, but just log the thought. Use it when complex reasoning or brainstorming is needed.

Common use cases:
1. When exploring a repository and discovering the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be simplest and most effective.
2. After receiving test results, use this tool to brainstorm ways to fix failing tests.
3. When planning a complex refactoring, use this tool to outline different approaches and their tradeoffs.
4. When designing a new feature, use this tool to think through architecture decisions and implementation details.
5. When debugging a complex issue, use this tool to organize your thoughts and hypotheses.

The tool simply logs your thought process for better transparency and does not execute any code or make changes."""

FINISH_DESCRIPTION = """Signals the completion of the current task or conversation.

Use this tool when:
- You have successfully completed the user's requested task
- You cannot proceed further due to technical limitations or missing information

The message should include:
- A clear summary of actions taken and their results
- Any next steps for the user
- Explanation if you're unable to complete the task
- Any follow-up questions if more information is needed
"""

EDITOR_DESCRIPTION = """Custom editing tool for viewing, creating and editing files in plain-text format
* State is persistent across command calls and discussions with the user
* If `path` is a text file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* This tool can be used for creating and editing files in plain-text format.


Before using this tool:
1. Use the view tool to understand the file's contents and context
2. Verify the directory path is correct (only applicable when creating new files):
   - Use the view tool to verify the parent directory exists and is the correct location

When making edits:
   - Ensure the edit results in idiomatic, correct code
   - Do not leave the code in a broken state
   - Always use absolute file paths (starting with /)

CRITICAL REQUIREMENTS FOR USING THIS TOOL:

1. EXACT MATCHING: The `old_str` parameter must match EXACTLY one or more consecutive lines from the file, including all whitespace and indentation. The tool will fail if `old_str` matches multiple locations or doesn't match exactly with the file content.

2. UNIQUENESS: The `old_str` must uniquely identify a single instance in the file:
   - Include sufficient context before and after the change point (3-5 lines recommended)
   - If not unique, the replacement will not be performed

3. REPLACEMENT: The `new_str` parameter should contain the edited lines that replace the `old_str`. Both strings must be different.

Remember: when making multiple file edits in a row to the same file, you should prefer to send all edits in a single message with multiple calls to this tool, rather than multiple messages with a single call each.
"""


def _function(name: str, description: str, properties: Mapping[str, Any], required: Sequence[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": {k: dict(v) for k, v in properties.items()}, "required": list(required)}}}


TOOLS = (
    _function(EXECUTE_BASH, BASH_DESCRIPTION, {"command": {
        "type": "string", "description": "The bash command to execute."}}, ["command"]),
    _function(THINK, THINK_DESCRIPTION, {"thought": {"type": "string", "description": "The thought to log."}},
              ["thought"]),
    _function(FINISH, FINISH_DESCRIPTION, {"message": {
        "type": "string", "description": "Final message to send to the user"}}, ["message"]),
    _function(EDITOR, EDITOR_DESCRIPTION, {
        "command": {"description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, "
                                   "`insert`.", "enum": ["view", "create", "str_replace", "insert"], "type": "string"},
        "path": {"description": "Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`.",
                 "type": "string"},
        "file_text": {"description": "Required parameter of `create` command, with the content of the file to be "
                                     "created.", "type": "string"},
        "old_str": {"description": "Required parameter of `str_replace` command containing the string in `path` to "
                                   "replace.", "type": "string"},
        "new_str": {"description": "Optional parameter of `str_replace` command containing the new string (if not "
                                   "given, no string will be added). Required parameter of `insert` command "
                                   "containing the string to insert.", "type": "string"},
        "insert_line": {"description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER "
                                       "the line `insert_line` of `path`.", "type": "integer"},
        "view_range": {"description": "Optional parameter of `view` command when `path` points to a file. If none is "
                                      "given, the full file is shown. If provided, the file will be shown in the "
                                      "indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing "
                                      "at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to "
                                      "the end of the file.", "items": {"type": "integer"}, "type": "array"},
    }, ["command", "path"]),
)
TOOLS_BY_NAME = {tool["function"]["name"]: tool for tool in TOOLS}

SUFFIX = """
You have access to the following functions:

{description}

If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>
"""


def tools_description(tools: Sequence[Mapping[str, Any]]) -> str:
    """`fn_call_converter.convert_tools_to_description`."""
    out = ""
    for i, tool in enumerate(tools):
        fn = tool["function"]
        if i > 0:
            out += "\n"
        out += f"---- BEGIN FUNCTION #{i + 1}: {fn['name']} ----\n"
        out += f"Description: {fn['description']}\n"
        out += "Parameters:\n"
        properties = fn["parameters"].get("properties", {})
        required = set(fn["parameters"].get("required", []))
        for j, (name, info) in enumerate(properties.items()):
            desc = info.get("description", "No description provided")
            if "enum" in info:
                desc += "\nAllowed values: [" + ", ".join(f"`{v}`" for v in info["enum"]) + "]"
            status = "required" if name in required else "optional"
            out += f"  ({j + 1}) {name} ({info.get('type', 'string')}, {status}): {desc}\n"
        out += f"---- END FUNCTION #{i + 1} ----\n"
    return out


SYSTEM = SYSTEM_BASE + SUFFIX.format(description=tools_description(TOOLS))

INSTANCE = """<issue_description>
{task}
</issue_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?
I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Also the development Python environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.
Your task is to make the minimal changes to non-test files in the repository to ensure the <issue_description> is satisfied.

Follow these phases to resolve the issue:

Phase 1. READING: read the problem and reword it in clearer terms
   1.1 If there are code or config snippets. Express in words any best practices or conventions in them.
   1.2 Hightlight message errors, method names, variables, file names, stack traces, and technical details.
   1.3 Explain the problem in clear terms.
   1.4 Enumerate the steps to reproduce the problem.
   1.5 Hightlight any best practices to take into account when testing and fixing the issue

Phase 2. RUNNING: install and run the tests on the repository
   2.1 Follow the readme
   2.2 Install the environment and anything needed
   2.2 Iterate and figure out how to run the tests

Phase 3. EXPLORATION: find the files that are related to the problem and possible solutions
   3.1 Use `grep` to search for relevant methods, classes, keywords and error messages.
   3.2 Identify all files related to the problem statement.
   3.3 Propose the methods and files to fix the issue and explain why.
   3.4 From the possible file locations, select the most likely location to fix the issue.

Phase 4. TEST CREATION: before implementing any fix, create a script to reproduce and verify the issue.
   4.1 Look at existing test files in the repository to understand the test format/structure.
   4.2 Create a minimal reproduction script that reproduces the located issue.
   4.3 Run the reproduction script to confirm you are reproducing the issue.
   4.4 Adjust the reproduction script as necessary.

Phase 5. FIX ANALYSIS: state clearly the problem and how to fix it
   5.1 State clearly what the problem is.
   5.2 State clearly where the problem is located.
   5.3 State clearly how the test reproduces the issue.
   5.4 State clearly the best practices to take into account in the fix.
   5.5 State clearly how to fix the problem.

Phase 6. FIX IMPLEMENTATION: Edit the source code to implement your chosen solution.
   6.1 Make minimal, focused changes to fix the issue.

Phase 7. VERIFICATION: Test your implementation thoroughly.
   7.1 Run your reproduction script to verify the fix works.
   7.2 Add edge cases to your test script to ensure comprehensive coverage.
   7.3 Run existing tests related to the modified code to ensure you haven't broken anything.

8. FINAL REVIEW: Carefully re-read the problem description and compare your changes with the original code (`git diff`).
   8.1 Ensure you've fully addressed all requirements.
   8.2 Run any tests in the repository related to:
     8.2.1 The issue you are fixing
     8.2.2 The files you modified
     8.2.3 The functions you changed
   8.3 If any tests fail, revise your implementation until all tests pass

Be thorough in your exploration, testing, and reasoning. It's fine if your thinking process is lengthy - quality and completeness are more important than brevity."""

USER_RESPONSE = ("Please continue working on the task on whatever approach you think is suitable.\n"
                 "When you think you have solved the question, please use the finish tool and include your final "
                 "answer in the message parameter of the finish tool.\n"
                 "IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n")
THOUGHT_LOGGED = "Your thought has been logged."

FN_REGEX = re.compile(r"<function=([^>]+)>\n(.*?)</function>", re.DOTALL)
PARAM_REGEX = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)
_MALFORMED_PARAM = re.compile(r"<parameter=([a-zA-Z0-9_]+)=([^<]*)</parameter>")


class _Invalid(ValueError):
    pass


def _fix_stopword(content: str) -> str:
    if "<function=" in content and content.count("<function=") == 1:
        if content.endswith("</"):
            content = content.rstrip() + "function>"
        else:
            content = content + "\n</function>"
    return content


def _params(tool: Mapping[str, Any], body: str, name: str) -> dict[str, Any]:
    """`_extract_and_validate_params`: allowed names, integer and array coercion, enums, required."""
    schema = tool["function"]["parameters"]
    properties, required = schema["properties"], set(schema.get("required", []))
    params: dict[str, Any] = {}
    for match in PARAM_REGEX.finditer(_MALFORMED_PARAM.sub(r"<parameter=\1>\2</parameter>", body)):
        key, value = match.group(1), match.group(2)
        if key not in properties:
            raise _Invalid(f"Parameter '{key}' is not allowed for function '{name}'. "
                           f"Allowed parameters: {set(properties)}")
        kind = properties[key].get("type", "string")
        if kind == "integer":
            try:
                value = int(value)
            except ValueError:
                raise _Invalid(f"Parameter '{key}' is expected to be an integer.") from None
        elif kind == "array":
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise _Invalid(f"Parameter '{key}' is expected to be an array.") from None
        if "enum" in properties[key] and value not in properties[key]["enum"]:
            raise _Invalid(f"Parameter '{key}' is expected to be one of {properties[key]['enum']}.")
        params[key] = value
    missing = required - set(params)
    if missing:
        raise _Invalid(f"Missing required parameters for function '{name}': {missing}")
    return params


def system_prompt(names: Sequence[str] = ()) -> str:
    """The system prompt for an offered tool set: the same for every set (all four functions run
    through bash)."""
    return SYSTEM


def turn_calls(texts: Sequence[str], offered: Sequence[str] = ()) -> list[tuple[str, str]]:
    """The turn's one action as `(surface name, arguments JSON)`: the first function call in `texts`
    (the answer, then the raw text of any call the template parser took out of it: a call the model
    wrapped in `<tool_call>`), else `?continue`; a call OpenHands would refuse is `?error` with its
    message."""
    match = None
    for text in texts:
        match = FN_REGEX.search(_fix_stopword(text or ""))
        if match is not None:
            break
    if match is None:
        return [(CONTINUE_CALL, "{}")]
    name, body = match.group(1), match.group(2)
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return [(ERROR_CALL, json.dumps({"message": f"Function '{name}' not found in available tools: "
                                                    f"{list(TOOLS_BY_NAME)}"}))]
    try:
        return [(name, json.dumps(_params(tool, body, name)))]
    except _Invalid as invalid:
        return [(ERROR_CALL, json.dumps({"message": str(invalid)}))]


# The editor, run with python3 in the box: its arguments are base64 JSON in argv[1]; it prints the
# observation OpenHands' editor would return and exits 1 on a ToolError (printed as `ERROR:\n...`).
_EDITOR = r'''
import base64, json, os, re, sys
a = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
W, N = 4, 16000
TEXT = ("<response clipped><NOTE>Due to the max output limit, only part of this file has been shown to you. "
        "You should retry this tool after you have searched inside the file with `grep -n` in order to find "
        "the line numbers of what you are looking for.</NOTE>")
DIR = ("<response clipped><NOTE>Due to the max output limit, only part of this directory has been shown to "
       "you. You should use `ls -la` instead to view large directories incrementally.</NOTE>")
class E(Exception): pass
def bad(p, v, h): raise E(f"Invalid `{p}` parameter: {v}. {h}")
def cut(s, notice): return s if len(s) <= N else s[:N] + notice
def read(p):
    try:
        with open(p, encoding="utf-8", newline="") as f: return f.read()
    except UnicodeDecodeError: raise
    except OSError as e: raise E(f"Ran into {e} while trying to read {p}") from None
def write(p, s):
    try:
        with open(p, "w", encoding="utf-8", newline="") as f: f.write(s)
    except OSError as e: raise E(f"Ran into {e} while trying to write to {p}") from None
def lines_of(s):
    ls = s.split("\n")
    if ls and ls[-1] == "": ls.pop()
    return ls
def kept(s):
    parts = s.split("\n")
    return [l + "\n" for l in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
def out(snippet, desc, start):
    ls = lines_of(cut(snippet, TEXT))
    body = "\n".join(f"{i + start:6}\t{l}" for i, l in enumerate(ls))
    return f"Here's the result of running `cat -n` on {desc}:\n" + (body + "\n" if body else "")
def segment(s, a1, b1):
    ls = s.split("\n")
    return "\n".join(ls[a1 - 1:b1])
def run():
    c, p = a["command"], a["path"]
    if not os.path.isabs(p): bad("path", p, "The path should be an absolute path.")
    if c == "create" and os.path.exists(p):
        bad("path", p, f"File already exists at: {p}. Cannot overwrite files using command `create`.")
    if c != "create" and not os.path.exists(p):
        bad("path", p, f"The path {p} does not exist. Please provide a valid path.")
    if c != "view" and os.path.isdir(p):
        bad("path", p, f"The path {p} is a directory and only the `view` command can be used on directories.")
    if c == "view":
        r = a.get("view_range")
        if os.path.isdir(p):
            if r: bad("view_range", str(r), "The `view_range` parameter is not allowed when `path` points to a directory.")
            names = sorted(os.listdir(p))
            hidden = sum(1 for n in names if n.startswith("."))
            shown = [p]
            for n in names:
                if n.startswith("."): continue
                q = os.path.join(p, n)
                shown.append(q + "/" if os.path.isdir(q) else q)
                if os.path.isdir(q):
                    try: kids = sorted(os.listdir(q))
                    except OSError: kids = []
                    for k in kids:
                        if k.startswith("."): continue
                        x = os.path.join(q, k)
                        shown.append(x + "/" if os.path.isdir(x) else x)
            shown[0] = p.rstrip("/") + "/"
            msg = (f"Here's the files and directories up to 2 levels deep in {p}, excluding hidden items:\n"
                   + "\n".join(shown))
            if hidden: msg += f"\n\n{hidden} hidden files/directories in this directory are excluded. You can use 'ls -la {p}' to see them."
            return cut(msg, DIR)
        try: s = read(p)
        except UnicodeDecodeError as e:
            raise E(f"Cannot view {p}: file contains binary content that cannot be decoded as text. Error: {e}")
        if not r: return out(s, p, 1)
        n = len(kept(s))
        if len(r) != 2 or not all(isinstance(i, int) for i in r): bad("view_range", str(r), "It should be a list of two integers.")
        lo, hi = r
        if lo < 1 or lo > n: bad("view_range", str(r), f"Its first element `{lo}` should be within the range of lines of the file: {[1, n]}.")
        note = None
        if hi == -1: hi = n
        elif hi > n:
            note = f"We only show up to {n} since there're only {n} lines in this file."
            hi = n
        if hi < lo: bad("view_range", str(r), f"Its second element `{hi}` should be greater than or equal to the first element `{lo}`.")
        o = out(segment(s, lo, hi), p, lo)
        return f"NOTE: {note}\n{o}" if note else o
    if c == "create":
        if "file_text" not in a: raise E("Parameter `file_text` is required for command: create.")
        write(p, a["file_text"])
        return f"File created successfully at: {p}"
    s = read(p)
    if c == "str_replace":
        if "old_str" not in a: raise E("Parameter `old_str` is required for command: str_replace.")
        old, new = a["old_str"], a.get("new_str") or ""
        found = [(s.count("\n", 0, m.start()) + 1, m.group(), m.start()) for m in re.finditer(re.escape(old), s)]
        if not found:
            old = old.strip()
            found = [(s.count("\n", 0, m.start()) + 1, m.group(), m.start()) for m in re.finditer(re.escape(old), s)]
            if not found: raise E(f"No replacement was performed, old_str `{old}` did not appear verbatim in {p}.")
        if len(found) > 1:
            raise E(f"No replacement was performed. Multiple occurrences of old_str `{old}` in lines {sorted(set(l for l, _, _ in found))}. Please ensure it is unique.")
        line, text, i = found[0]
        t = s[:i] + new + s[i + len(text):]
        write(p, t)
        lo = max(0, line - W)
        hi = line + W + new.count("\n")
        return (f"The file {p} has been edited. " + out(segment(t, lo + 1, hi), f"a snippet of {p}", lo + 1)
                + "Review the changes and make sure they are as expected. Edit the file again if necessary.")
    if c == "insert":
        if "insert_line" not in a: raise E("Parameter `insert_line` is required for command: insert.")
        if "new_str" not in a: raise E("Parameter `new_str` is required for command: insert.")
        k, new = a["insert_line"], a["new_str"]
        old = kept(s)
        n = len(old)
        if k < 0 or k > n: bad("insert_line", str(k), f"It should be within the range of allowed values: {[0, n]}")
        add = new.split("\n")
        t = "".join(old[:k]) + "".join(l + "\n" for l in add) + "".join(old[k:])
        write(p, t)
        lo = max(0, k - W)
        hi = min(n + len(add), k + W + len(add))
        return ("The file " + p + " has been edited. " + out(segment(t, lo + 1, hi), "a snippet of the edited file", max(1, k - W + 1))
                + "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary.")
    raise E(f"Unrecognized command {c}. The allowed commands for the str_replace_editor tool are: view, create, str_replace, insert")
try:
    sys.stdout.write(run())
except E as e:
    sys.stdout.write("ERROR:\n" + str(e)); sys.exit(1)
except (OSError, UnicodeDecodeError) as e:
    sys.stdout.write("ERROR:\n" + f"Ran into {e} while trying to read {a.get('path')}"); sys.exit(1)
'''
_EDITOR_B64 = base64.b64encode(zlib.compress(_EDITOR.strip().encode("utf-8"), 9)).decode("ascii")


def editor_command(params: Mapping[str, Any]) -> str:
    """The bash command that runs one `str_replace_editor` call in the box."""
    arguments = base64.b64encode(json.dumps(dict(params), ensure_ascii=False).encode("utf-8")).decode("ascii")
    return (f"{EDITOR_MARK}\npython3 -I -c 'import base64,zlib;exec(zlib.decompress(base64.b64decode("
            f"\"{_EDITOR_B64}\")))' '{arguments}'")


def translate(name: str, arguments: str, offered: Sequence[str] = ()) -> tuple[str, str] | None:
    """The canonical call a surface call runs as: `execute_bash` and `str_replace_editor` on bash,
    `think` and the turn's errors answered locally, None for `finish` (the episode ends)."""
    if name == FINISH:
        return None
    if name == THINK:
        return THINK_CALL, "{}"
    if name == EXECUTE_BASH:
        return "bash", json.dumps({"command": json.loads(arguments)["command"]})
    if name == EDITOR:
        return "bash", json.dumps({"command": editor_command(json.loads(arguments))})
    return name, arguments


def truncate(content: str, limit: int = MAX_MESSAGE_CHARS) -> str:
    """`truncate_content`: the middle of an observation longer than `limit`."""
    if len(content) <= limit:
        return content
    half = limit // 2
    return content[:half] + "\n[... Observation truncated due to length ...]\n" + content[-half:]


def observation(body: Mapping[str, Any]) -> str:
    """What the model is shown for a signed bash record: the editor's output, or the command's output
    (middle-truncated) and how it finished, as OpenHands' tool result."""
    output, code = body["output"], body["exit_code"]
    if not isinstance(output, str) or type(code) is not int:
        raise ValueError("a call record's output must be a string and its exit_code an integer")
    if body["arguments"]["command"].startswith(EDITOR_MARK + "\n"):
        if body["timed_out"]:
            output = "ERROR:\nThe editor command timed out."
        return f"EXECUTION RESULT of [{EDITOR}]:\n{truncate(output)}"
    if body["timed_out"]:
        suffix = "\n[The command timed out at the sandbox's time limit for one command and was killed.]"
    else:
        suffix = f"\n[Command finished with exit code {code}]"
    return f"EXECUTION RESULT of [{EXECUTE_BASH}]:\n{truncate(output)}{suffix}"


def refusal(name: str, arguments: str | None) -> str | None:
    """OpenHands' text for a call answered locally, or None."""
    if name == THINK_CALL:
        return f"EXECUTION RESULT of [{THINK}]:\n{THOUGHT_LOGGED}"
    if name == CONTINUE_CALL:
        return USER_RESPONSE
    if name == ERROR_CALL:
        return json.loads(arguments or "{}").get("message", "")
    return None


__all__ = ["CONTINUE_CALL", "EDITOR_MARK", "ERROR_CALL", "INSTANCE", "SUBMIT_TOOL", "SYSTEM", "THINK_CALL", "TOOLS",
           "editor_command", "observation", "refusal", "system_prompt", "tools_description", "translate", "truncate",
           "turn_calls"]
