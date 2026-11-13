"""Harbor's Terminus 2 (the Terminal-Bench 2.x reference agent), JSON form: the task inside Terminus' prompt as the
first user message, one JSON answer per turn (`analysis`, `plan`, `commands` of `keystrokes`, `task_complete`),
the terminal's new output shown back as a user message, and a task marked complete twice in a row ends the episode.

The texts and rules follow harbor 06d0e86a (Apache-2.0, `src/harbor/agents/terminus_2/`; `HARBOR_LICENSE.md`):
`templates/terminus-json-plain.txt` (verbatim), `terminus_json_plain_parser.py` (ported: its errors, warnings and
auto-fixes), and `terminus_2.py` (`_handle_llm_interaction`'s feedback, the parsing-error and warning messages,
`_get_completion_confirmation_message`, `_limit_output_length`, the output-length error, the double confirmation).
`tests/upstream/terminus/` holds the upstream files and the parity tests run them against this module.

The differences, all because an episode runs on a signed sandbox through the canonical `bash` (each call a fresh
non-interactive process with a per-call time limit):

* no tmux: a turn's keystrokes, concatenated, are typed into a fresh interactive `bash -i` reading them from a file
  (one canonical call per turn). It echoes each line after its prompt, as a terminal shows it, with stderr merged.
  A shell session is emulated across turns: the shell's variables, functions, aliases, options and directory are
  saved under `STATE_DIR` before each prompt and restored by the next turn's shell; the first turn sources the
  login profiles once (`bash --login`). Jobs do not persist (background processes keep running, but a timed-out
  call's are killed with it); `exit` ends only that turn's shell (upstream: the session, so the episode);
* text after the turn's last newline is not run: it stays typed on the prompt line and the next turn's keystrokes
  continue it, as in tmux;
* no waiting: `duration` is read (its warnings are upstream's) but never slept on; the call returns when the
  commands finish, or the sandbox's per-call limit kills them (a note says so; nothing keeps running for a later
  turn to poll). No terminal: interactive programs (vim, less, REPLs, password prompts) cannot be driven; a program
  that reads stdin reads the turn's following keystrokes;
* tmux key names as a whole keystroke: Enter (`Enter`, `C-m`, `C-j`, `KPEnter`, `^M`, `^J`) is a newline, `Space` a
  space, `C-c` clears the line typed so far (nothing runs while keys are typed), any other key (`C-d`, `Escape`,
  `Up`, `Tab`, ...) is not sent; a note in the observation says so;
* the first prompt's terminal state is an empty `Current Terminal Screen:` (the box is not known when the prompt
  is rendered); later outputs are always `New Terminal Output:` (the shell's echo is never empty), the call's
  whole output rather than a scrollback diff, escape sequences dropped and a carriage return overwriting its line
  (what tmux's text capture shows); a gateway cut or timeout is noted after it;
* no system message, as upstream: the chat template's own default system message (if it has one) stands;
* a turn's keystrokes must fit the gateway's argument limit (120 KiB with the shell setup, base64): a larger turn
  is refused with the gateway's text, nothing runs;
* no summarization when the context fills (the episode stops at `context_length`), no turn cap of its own (the
  job's `max_turns` applies), no API retries;
* the second `task_complete` ends the episode without running that turn's commands (the loop sends nothing on a
  turn that ends it); upstream runs them, then stops;
* the pending confirmation is read back from the transcript: the latest user message that is not a parsing or
  output-length error is the confirmation question (`turn_actions(..., earlier=...)`), so the reader needs the
  turn's prompt; upstream keeps it in the agent object;
* the turn cut at its token cap gets upstream's output-length error with "the maximum output length" (upstream's
  wording when the model's limit is unknown);
* unknown command fields are listed in their order in the answer (upstream: set order, random per process), and
  JSON decode errors are worded as CPython 3.12 words them on every Python (3.13 words a trailing comma
  differently): miner and validator must render the same text.

Pure: no I/O, no dependency.
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

HARBOR_COMMIT = "06d0e86a8cfbac213548e0caec533afc5d81ff31"
RESPONSE_CALL, KEYS_CALL = "?response", "keystrokes"
PARSE_ERROR_CALL, LENGTH_ERROR_CALL = "?parse", "?length"
MAX_OUTPUT_BYTES = 10000
STATE_DIR = "/tmp/.terminus-2"
MARK = "# terminus-2 "
READS_EARLIER_TURNS = True
"""`turn_actions` needs the turn's earlier user messages (`earlier`): the renderer passes them."""

TEMPLATE = r"""You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

Format your response as JSON with the following structure:

{{
  "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?",
  "plan": "Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.",
  "commands": [
    {{
      "keystrokes": "ls -la\n",
      "duration": 0.1
    }},
    {{
      "keystrokes": "cd project\n",
      "duration": 0.1
    }}
  ],
  "task_complete": true
}}

Required fields:
- "analysis": Your analysis of the current situation
- "plan": Your plan for the next steps
- "commands": Array of command objects to execute

Optional fields:
- "task_complete": Boolean indicating if the task is complete (defaults to false if not present)

Command object structure:
- "keystrokes": String containing the exact keystrokes to send to the terminal (required)
- "duration": Number of seconds to wait for the command to complete before the next command will be executed (defaults to 1.0 if not present)

IMPORTANT: The text inside "keystrokes" will be used completely verbatim as keystrokes. Write commands exactly as you want them sent to the terminal:
- You must end every command with a newline (\n) or it will not execute.
- For special key sequences, use tmux-style escape sequences:
  - C-c for Ctrl+C
  - C-d for Ctrl+D

The "duration" attribute specifies the number of seconds to wait for the command to complete (default: 1.0) before the next command will be executed. On immediate tasks (e.g., cd, ls, echo, cat) set a duration of 0.1 seconds. On commands (e.g., gcc, find, rustc) set a duration of 1.0 seconds. On slow commands (e.g., make, python3 [long running script], wget [file]) set an appropriate duration as you determine necessary.

It is better to set a smaller duration than a longer duration. It is always possible to wait again if the prior output has not finished, by running {{"keystrokes": "", "duration": 10.0}} on subsequent requests to wait longer. Never wait longer than 60 seconds; prefer to poll to see intermediate result status.

Important notes:
- Each command's keystrokes are sent exactly as written to the terminal
- Do not include extra whitespace before or after the keystrokes unless it's part of the intended command
- Extra text before or after the JSON will generate warnings but be tolerated
- The JSON must be valid - use proper escaping for quotes and special characters within strings
- Commands array can be empty if you want to wait without taking action

Task Description:
{instruction}

Current terminal state:
{terminal_state}
"""

INITIAL_SCREEN = "Current Terminal Screen:\n"
INSTANCE = TEMPLATE.format(instruction="{task}", terminal_state=INITIAL_SCREEN)
"""The first user message, `{task}` standing for the instruction."""

PARSE_ERROR_PREFIX = "Previous response had parsing errors:\n"
PARSE_ERROR = PARSE_ERROR_PREFIX + "{feedback}\n\nPlease fix these issues and provide a proper JSON response."
WARNINGS_PREFIX = "Previous response had warnings:\n"
CONFIRM_PREFIX = "Current terminal state:\n"
CONFIRM_SUFFIX = ("\n\nAre you sure you want to mark the task as complete? This will trigger your solution to be "
                  "graded and you won't be able to make any further corrections. If so, include "
                  '"task_complete": true in your JSON response again.')
LENGTH_LIMIT = "the maximum output length"
LENGTH_ERROR = (f"ERROR!! NONE of the actions you just requested were performed because you exceeded {LENGTH_LIMIT}. "
                f"Your outputs must be less than {LENGTH_LIMIT}. Re-issue this request, breaking it into chunks each "
                f"of which is less than {LENGTH_LIMIT}.")
LENGTH_WARNINGS = "\n\nParser warnings from your truncated response:\n"
NEW_OUTPUT = "New Terminal Output:\n"

TIMED_OUT = ("[The commands did not finish within the sandbox's time limit for one call and were killed; the "
             "shell is back at its prompt. Run long jobs in the background with their output redirected to a file.]")
CUT = "[The output exceeded the sandbox's limit and was cut.]"
CLEARED = "[C-c: no command was running; the line typed so far was cleared.]"
NOT_SENT = "[{key}: not sent; this terminal takes typed text, Enter and C-c only.]"

ENTER_KEYS = frozenset({"Enter", "C-m", "KPEnter", "C-j", "^M", "^J"})
INTERRUPT_KEYS = frozenset({"C-c", "^C"})
_KEY_NAME = re.compile(r"(?:[CMS]-)+\S+|\^[A-Za-z@\[\]\\^_?]|Escape|Tab|BTab|BSpace|Up|Down|Left|Right|Home|End|"
                       r"PageUp|PageDown|PgUp|PgDn|PPage|NPage|IC|DC|Insert|Delete|F[1-9]|F1[0-2]|KP[A-Za-z0-9]+")


# -- the parser (terminus_json_plain_parser.py) ------------------------------------------------------------------

def _result(commands, complete, error, warnings) -> dict[str, Any]:
    return {"commands": commands, "complete": complete, "error": error,
            "warning": "- " + "\n- ".join(warnings) if warnings else ""}


def _decode_error(exc: json.JSONDecodeError) -> str:
    """`str(exc)` as CPython 3.12 words it (3.13 names a trailing comma at the comma)."""
    message, doc, pos = exc.msg, exc.doc, exc.pos
    trailing = {"Illegal trailing comma before end of object": "Expecting property name enclosed in double quotes",
                "Illegal trailing comma before end of array": "Expecting value"}
    if message in trailing:
        message, pos = trailing[message], pos + 1
        while pos < len(doc) and doc[pos] in " \t\n\r":
            pos += 1
    line = doc.count("\n", 0, pos) + 1
    column = pos - doc.rfind("\n", 0, pos)
    return f"{message}: line {line} column {column} (char {pos})"


def _extract_json(response: str) -> tuple[str, list[str]]:
    start, end, depth, in_string, escaped = -1, -1, 0, False, False
    for i, char in enumerate(response):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"' and not escaped:
            in_string = not in_string
            continue
        if not in_string:
            if char == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    end = i + 1
                    break
    if start == -1 or end == -1:
        return "", ["No valid JSON object found"]
    warnings = []
    if response[:start].strip():
        warnings.append("Extra text detected before JSON object")
    if response[end:].strip():
        warnings.append("Extra text detected after JSON object")
    return response[start:end], warnings


def _check_order(content: str, warnings: list[str]) -> None:
    expected = ["analysis", "plan", "commands"]
    positions = {}
    for field in expected:
        match = re.search(f'"({field})"\\s*:', content)
        if match:
            positions[field] = match.start()
    if len(positions) < 2:
        return
    present = [(field, positions[field]) for field in expected if field in positions]
    actual = [field for field, _ in sorted(present, key=lambda x: x[1])]
    wanted = [field for field in expected if field in positions]
    if actual != wanted:
        warnings.append(f"Fields appear in wrong order. Found: {' → '.join(actual)}, expected: {' → '.join(wanted)}")


def _validate(data: Any, content: str, warnings: list[str]) -> str:
    if not isinstance(data, dict):
        return "Response must be a JSON object"
    missing = [field for field in ("analysis", "plan", "commands") if field not in data]
    if missing:
        return f"Missing required fields: {', '.join(missing)}"
    if not isinstance(data.get("analysis", ""), str):
        warnings.append("Field 'analysis' should be a string")
    if not isinstance(data.get("plan", ""), str):
        warnings.append("Field 'plan' should be a string")
    if not isinstance(data.get("commands", []), list):
        return "Field 'commands' must be an array"
    _check_order(content, warnings)
    complete = data.get("task_complete")
    if complete is not None and not isinstance(complete, (bool, str)):
        warnings.append("Field 'task_complete' should be a boolean or string")
    return ""


def _commands(items: list, warnings: list[str]) -> tuple[list[str], str]:
    keystrokes = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            return [], f"Command {i + 1} must be an object"
        if "keystrokes" not in item:
            return [], f"Command {i + 1} missing required 'keystrokes' field"
        keys = item["keystrokes"]
        if not isinstance(keys, str):
            return [], f"Command {i + 1} 'keystrokes' must be a string"
        if "duration" in item:
            if not isinstance(item["duration"], (int, float)):
                warnings.append(f"Command {i + 1}: Invalid duration value, using default 1.0")
        else:
            warnings.append(f"Command {i + 1}: Missing duration field, using default 1.0")
        unknown = [key for key in item if key not in ("keystrokes", "duration")]
        if unknown:
            warnings.append(f"Command {i + 1}: Unknown fields: {', '.join(unknown)}")
        if i < len(items) - 1 and keys and not keys.endswith("\n"):
            warnings.append(f"Command {i + 1} should end with newline when followed by another command. Otherwise "
                            "the two commands will be concatenated together on the same line.")
        keystrokes.append(keys)
    return keystrokes, ""


def _try_parse(response: str) -> dict[str, Any]:
    content, warnings = _extract_json(response)
    if not content:
        return _result([], False, "No valid JSON found in response", warnings)
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        error = f"Invalid JSON: {_decode_error(exc)}"
        error += f" | Content: {content!r}" if len(content) < 200 else f" | Content preview: {content[:100]!r}..."
        return _result([], False, error, warnings)
    error = _validate(data, content, warnings)
    if error:
        return _result([], False, error, warnings)
    complete = data.get("task_complete", False)
    if isinstance(complete, str):
        complete = complete.lower() in ("true", "1", "yes")
    commands, error = _commands(data.get("commands", []), warnings)
    if error:
        if complete:
            warnings.append(error)
            return _result([], True, "", warnings)
        return _result([], False, error, warnings)
    return _result(commands, bool(complete), "", warnings)


def _fix_incomplete(response: str, error: str) -> tuple[str, bool]:
    if "Invalid JSON" in error or "Expecting" in error or "Unterminated" in error or "No valid JSON found" in error:
        missing = response.count("{") - response.count("}")
        if missing > 0:
            return response + "}" * missing, True
    return response, False


def _fix_mixed(response: str, error: str) -> tuple[str, bool]:
    for match in re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL):
        try:
            json.loads(match)
            return match, True
        except json.JSONDecodeError:
            continue
    return response, False


def parse_response(response: str) -> dict[str, Any]:
    """`TerminusJSONPlainParser.parse_response`: `{commands (keystrokes), complete, error, warning}`."""
    result = _try_parse(response)
    if result["error"]:
        for name, fix in (("Fixed incomplete JSON by adding missing closing brace", _fix_incomplete),
                          ("Extracted JSON from mixed content", _fix_mixed)):
            corrected, fixed = fix(response, result["error"])
            if fixed:
                again = _try_parse(corrected)
                if again["error"] == "":
                    note = f"AUTO-CORRECTED: {name} - please fix this in future responses"
                    again["warning"] = f"- {note}\n{again['warning']}" if again["warning"] else f"- {note}"
                    return again
    return result


def feedback(result: Mapping[str, Any]) -> str:
    """`_handle_llm_interaction`'s feedback: the error, then the warnings."""
    if result["error"]:
        text = f"ERROR: {result['error']}"
        return text + f"\nWARNINGS: {result['warning']}" if result["warning"] else text
    return f"WARNINGS: {result['warning']}" if result["warning"] else ""


# -- the harness interface ------------------------------------------------------------------------------------

def tool_specs(names: Sequence[str] = ()) -> tuple[dict, ...]:
    """None: the model writes its commands in its answer."""
    return ()


def system_prompt(names: Sequence[str] = ()) -> str:
    """No system message (the renderer then sends none: the template's own default stands)."""
    return ""


def turn_calls(texts: Sequence[str], offered: Sequence[str] = ()) -> list[tuple[str, str]]:
    """The turn's answer parsed: one `?response` carrying the parse result (`turn_actions` reads it). `texts` are
    the answer and the raw text of any call the template parser took out of it, joined as the model wrote them."""
    response = "\n".join(text for text in texts if text)
    return [(RESPONSE_CALL, json.dumps(parse_response(response)))]


def pending(earlier: Iterable[str]) -> bool:
    """Whether the previous turn marked the task complete: the latest user message (newest first in `earlier`)
    that is not a parsing or output-length error is the confirmation question. Every message it reads begins with
    a fixed text of this module, never with the model's or the box's."""
    for text in earlier:
        text = text.strip()
        if text.startswith(PARSE_ERROR_PREFIX) or text.startswith(LENGTH_ERROR):
            continue
        return text.startswith(CONFIRM_PREFIX) and text.endswith(CONFIRM_SUFFIX)
    return False


def turn_actions(calls: Sequence[tuple[str, str]], *, ended: bool, earlier: Iterable[str]) -> list[tuple[str, str]] | None:
    """The turn as Terminus plays it: cut at its token cap, the output-length error; a parsing error, its message
    (nothing runs); `task_complete` with the confirmation pending, the end (None); else the keystrokes."""
    (name, arguments), = calls
    result = json.loads(arguments)
    if not ended:
        return [(LENGTH_ERROR_CALL, json.dumps({"warning": result["warning"]}))]
    text = feedback(result)
    if "ERROR:" in text:
        return [(PARSE_ERROR_CALL, json.dumps({"feedback": text}))]
    if result["complete"] and pending(earlier):
        return None
    return [(KEYS_CALL, json.dumps({"keystrokes": result["commands"], "feedback": text,
                                    "complete": result["complete"]}))]


def typed(keystrokes: Sequence[str]) -> tuple[str, bool, list[str]]:
    """What the keystrokes type: the text, whether the line left typed by an earlier turn is cleared first (a
    C-c before any newline), and the notes on keys that are not text."""
    text, clear, notes = "", False, []
    for keys in keystrokes:
        keys = keys.replace("\x00", "")
        if keys in ENTER_KEYS:
            text += "\n"
        elif keys == "Space":
            text += " "
        elif keys in INTERRUPT_KEYS:
            if "\n" in text:
                text = text[:text.rfind("\n") + 1]
            else:
                text, clear = "", True
            notes.append(CLEARED)
        elif _KEY_NAME.fullmatch(keys):
            notes.append(NOT_SENT.format(key=keys))
        else:
            text += keys.replace("\r\n", "\n").replace("\r", "\n")
    return text, clear, list(dict.fromkeys(notes))


# The shell each turn types into: `bash -i` with this rcfile, its input the turn's text. Restores the session
# (else sources the login profiles once), saves it before every prompt, and once the input is all read prints the
# last prompt and the line left typed itself, then hides bash's own (and its "exit" at end of input).
RCFILE = r"""exec 1>&3 2>&1 3>&-
export -n __T2 __T2_SIZE __T2_LINE
if [ -f "$__T2/state" ]; then
  # Functions are read back as bash-completion's need (extglob), without aliases or history expansion; the
  # state's last lines restore the shell's own options.
  set +H; shopt -u expand_aliases; shopt -s extglob
  . "$__T2/state" 2>/dev/null
else
  { [ -f /etc/profile ] && . /etc/profile
    for __t2_f in ~/.bash_profile ~/.bash_login ~/.profile; do [ -f "$__t2_f" ] && { . "$__t2_f"; break; }; done
  } </dev/null >/dev/null 2>&1
  unset __t2_f
fi
__t2_save() {
  local __t2_v
  {
    printf 'cd -- %q\n' "$PWD"
    for __t2_v in $(compgen -v); do
      case $__t2_v in
        __t2*|__T2*|BASH|BASH_*|BASHOPTS|BASHPID|COLUMNS|COMP_*|DIRSTACK|EPOCHREALTIME|EPOCHSECONDS|EUID|FUNCNAME) ;;
        GROUPS|HISTCMD|LINENO|LINES|OPTARG|OPTIND|PIPESTATUS|PPID|PROMPT_COMMAND|PWD|RANDOM|SECONDS|SHELLOPTS) ;;
        SHLVL|SRANDOM|UID|_) ;;
        *) declare -p "$__t2_v" ;;
      esac
    done
    declare -f
    alias -p
    shopt -p
    set +o
  } > "$__T2/state.new" 2>/dev/null && mv -f "$__T2/state.new" "$__T2/state"
}
__t2_prompt() {
  local __t2_i __t2_p
  __t2_save
  __t2_i=$(cat "/proc/$$/fdinfo/0" 2>/dev/null)
  __t2_i=${__t2_i#pos:}
  [ "$(( ${__t2_i%%$'\n'*} + 0 ))" -ge "$__T2_SIZE" ] 2>/dev/null || return 0
  eval '__t2_p=${PS1@P}' 2>/dev/null || __t2_p='$ '
  printf '%s%s' "${__t2_p//[$'\001\002']/}" "$__T2_LINE"
  exec 2>/dev/null
}
PROMPT_COMMAND=__t2_prompt
"""


def script(keystrokes: Sequence[str], header: Mapping[str, Any]) -> str:
    """The canonical bash command of one turn: its first line `MARK` and the header JSON (what `observation`
    needs), then the rcfile and the keystrokes (base64) written under `STATE_DIR`, and the turn's shell."""
    text, clear, _ = typed(keystrokes)
    data = base64.b64encode(text.encode("utf-8", "surrogatepass")).decode("ascii")
    return "\n".join([
        MARK + json.dumps(dict(header), sort_keys=True, ensure_ascii=True),
        f"__T2={STATE_DIR}",
        'mkdir -p "$__T2"',
        "cat > \"$__T2/rc\" <<'TERMINUS_2_RC'",
        RCFILE + "TERMINUS_2_RC",
        *([': > "$__T2/line"'] if clear else []),
        f"printf '%s' '{data}' | base64 -d > \"$__T2/typed\"",
        '__t2_s=$(cat "$__T2/line" 2>/dev/null; cat "$__T2/typed"; printf x); __t2_s=${__t2_s%x}',
        "__T2_LINE=${__t2_s##*$'\\n'}",
        "printf '%s' \"${__t2_s%\"$__T2_LINE\"}\" > \"$__T2/run\"",
        "printf '%s' \"$__T2_LINE\" > \"$__T2/line\"",
        "__T2_SIZE=$(( $(wc -c < \"$__T2/run\") ))",
        "export __T2 __T2_SIZE __T2_LINE",
        'exec /bin/bash --rcfile "$__T2/rc" -i < "$__T2/run" 3>&1 >/dev/null 2>&1',
    ])


def translate(name: str, arguments: str, offered: Sequence[str] = ()) -> tuple[str, str]:
    """The turn's keystrokes as the canonical bash call; the local errors pass as they are."""
    if name != KEYS_CALL:
        return name, arguments
    turn = json.loads(arguments)
    _, _, notes = typed(turn["keystrokes"])
    header = {"complete": turn["complete"], "feedback": turn["feedback"], "notes": notes}
    return "bash", json.dumps({"command": script(turn["keystrokes"], header)})


def limit_output(output: str, max_bytes: int = MAX_OUTPUT_BYTES) -> str:
    """`_limit_output_length`: the first and last `max_bytes // 2` bytes around a marker."""
    data = output.encode("utf-8")
    if len(data) <= max_bytes:
        return output
    half = max_bytes // 2
    first = data[:half].decode("utf-8", errors="ignore")
    last = data[-half:].decode("utf-8", errors="ignore")
    omitted = len(data) - len(first.encode("utf-8")) - len(last.encode("utf-8"))
    return f"{first}\n[... output limited to {max_bytes} bytes; {omitted} interior bytes omitted ...]\n{last}"


_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)?|[ -/]*[0-~])")


def screen_text(output: str) -> str:
    """The output as a terminal's text capture shows it: escape sequences dropped, a carriage return overwriting
    its line from the start (only what follows the line's last one is kept)."""
    lines = _ESCAPE.sub("", output).replace("\r\n", "\n").split("\n")
    return "\n".join(line[line.rfind("\r") + 1:] for line in lines)


def confirmation(terminal_output: str) -> str:
    """`_get_completion_confirmation_message` (json)."""
    return CONFIRM_PREFIX + terminal_output + CONFIRM_SUFFIX


def header_of(command: str) -> dict[str, Any]:
    first = command.split("\n", 1)[0]
    if not first.startswith(MARK):
        raise ValueError("not a terminus-2 call")
    return json.loads(first[len(MARK):])


def observation(body: Mapping[str, Any]) -> str:
    """What Terminus shows after a turn's commands ran: the terminal's new output (with the sandbox's flags and the
    key notes), limited to 10000 bytes; the confirmation question when the turn marked the task complete, else
    the parser's warnings first when there were any."""
    output, code = body["output"], body["exit_code"]
    if not isinstance(output, str) or type(code) is not int:
        raise ValueError("a call record's output must be a string and its exit_code an integer")
    output = screen_text(output)
    header = header_of(body["arguments"]["command"])
    flags = [text for flag, text in ((body.get("truncated"), CUT), (body.get("timed_out"), TIMED_OUT)) if flag is True]
    if flags:
        output += ("" if output.endswith("\n") or not output else "\n") + "\n".join(flags)
    notes = "".join(note + "\n" for note in header["notes"])
    terminal = limit_output(NEW_OUTPUT + notes + output)
    if header["complete"]:
        return confirmation(terminal)
    if "WARNINGS:" in header["feedback"]:
        return f"{WARNINGS_PREFIX}{header['feedback']}\n\n{limit_output(terminal)}"
    return terminal


def refusal(name: str, arguments: str | None) -> str | None:
    """Terminus' message for a turn answered locally (a parsing or output-length error), or None."""
    if name == PARSE_ERROR_CALL:
        return PARSE_ERROR.replace("{feedback}", json.loads(arguments or "{}")["feedback"], 1)
    if name == LENGTH_ERROR_CALL:
        warning = json.loads(arguments or "{}").get("warning") or ""
        return LENGTH_ERROR + (LENGTH_WARNINGS + warning if warning else "")
    return None


def user_message(text: str) -> bool:
    """Every answer is a user message."""
    return True


__all__ = ["CONFIRM_PREFIX", "CONFIRM_SUFFIX", "HARBOR_COMMIT", "INSTANCE", "KEYS_CALL", "LENGTH_ERROR",
           "LENGTH_ERROR_CALL", "PARSE_ERROR", "PARSE_ERROR_CALL", "RESPONSE_CALL", "STATE_DIR", "TEMPLATE",
           "confirmation", "feedback", "limit_output", "observation", "parse_response", "pending", "refusal",
           "script", "system_prompt", "tool_specs", "translate", "turn_actions", "turn_calls", "typed", "user_message"]
