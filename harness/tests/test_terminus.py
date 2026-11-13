"""Harbor's Terminus 2 (`terminus-2@1`): its texts and parser against the vendored upstream files (never skipped),
the turn rules (parsing errors, output-length error, double confirmation), the keystrokes run by a real bash, the
observation, and a frozen conversation (its sha256)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest
import terminus_upstream as upstream
from reliquary_harness import terminus as t
from reliquary_harness.spec import SPECS, TERMINUS_2_V1, digest_refusal, get_spec

try:
    from test_qwen_goldens import TOKENIZERS
except pytest.skip.Exception:            # no `renderers` here: only the conversation tests skip
    TOKENIZERS = {}

RENDERED = json.loads((upstream.UPSTREAM / "rendered.json").read_text(encoding="utf-8"))
TASK = "Write hello to /app/out.txt."


def _response(commands=(("ls -la\n", 0.1),), complete=None, **extra):
    data = {"analysis": "a", "plan": "p", "commands": [{"keystrokes": k, "duration": d} for k, d in commands], **extra}
    if complete is not None:
        data["task_complete"] = complete
    return json.dumps(data)


def _actions(text, *, ended=True, earlier=()):
    return t.turn_actions(t.turn_calls([text]), ended=ended, earlier=iter(earlier))


def _body(command, output, code=0, **flags):
    return {"tool": "bash", "arguments": {"command": command}, "output": output, "exit_code": code,
            "truncated": False, "timed_out": False, **flags}


# -- upstream parity ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("relative", sorted(TERMINUS_2_V1.upstream["files"]))
def test_the_vendored_files_are_the_pinned_ones(relative):
    data = (upstream.UPSTREAM / relative.rsplit("/", 1)[-1]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == TERMINUS_2_V1.upstream["files"][relative]
    assert TERMINUS_2_V1.upstream["harbor"] == t.HARBOR_COMMIT


def test_the_prompt_is_the_template_rendered():
    assert t.TEMPLATE == upstream.source("terminus-json-plain.txt")
    assert t.INSTANCE.replace("{task}", "<<TASK>>") == RENDERED["instance_marker"]
    assert TERMINUS_2_V1.user_prompt(TASK) == RENDERED["instance_marker"].replace("<<TASK>>", TASK)
    assert upstream.rendered() == RENDERED


def test_the_messages_are_terminus_2s():
    up = upstream.agent()
    assert t.confirmation("New Terminal Output:\nroot@box:/app# ") == RENDERED["confirmation"]
    assert t.confirmation("x") == up._get_completion_confirmation_message("x")
    loop = upstream.method("_run_agent_loop")
    assert 'f"Previous response had parsing errors:\\n{feedback}\\n\\n"' in loop
    assert 'f"Please fix these issues and provide a proper "' in loop and up._get_error_response_type() == "JSON response"
    assert t.PARSE_ERROR == "Previous response had parsing errors:\n{feedback}\n\nPlease fix these issues and " \
                            "provide a proper JSON response."
    assert 'f"Previous response had warnings:\\n{feedback}\\n\\n"' in loop and 'if feedback and "ERROR:" in feedback:' in loop
    assert "if self._pending_completion:\n                    observation = terminal_output" in loop
    interaction = upstream.method("_handle_llm_interaction")
    for piece in ('feedback += f"ERROR: {result.error}"', 'feedback += f"\\nWARNINGS: {result.warning}"',
                  'feedback += f"WARNINGS: {result.warning}"'):
        assert piece in interaction, piece
    query = upstream.method("_query_llm")
    assert '"ERROR!! NONE of the actions you just requested were performed "' in query
    assert 'limit_str = "the maximum output length"' in query
    assert 'f"\\n\\nParser warnings from your truncated response:\\n"' in query
    assert t.LENGTH_ERROR == ("ERROR!! NONE of the actions you just requested were performed because you exceeded "
                              "the maximum output length. Your outputs must be less than the maximum output length. "
                              "Re-issue this request, breaking it into chunks each of which is less than the maximum "
                              "output length.")
    tmux = upstream.source("tmux_session.py")
    assert '_ENTER_KEYS = {"Enter", "C-m", "KPEnter", "C-j", "^M", "^J"}' in tmux
    assert t.ENTER_KEYS == {"Enter", "C-m", "KPEnter", "C-j", "^M", "^J"}
    assert 'return f"New Terminal Output:\\n{new_content}"' in tmux and t.NEW_OUTPUT == "New Terminal Output:\n"


@pytest.mark.parametrize("output", ["", "short\n", "x" * 10000, "x" * 10001, "é" * 6000 + "x" * 3,
                                    "a" * 4999 + "€" * 3000 + "z" * 4999, "line\n" * 5000])
def test_the_output_limit_is_upstreams(output):
    assert t.limit_output(output) == upstream.agent()._limit_output_length(output)
    assert t.limit_output("é" * 6000 + "x" * 3) == RENDERED["limited"]


def test_the_output_limit_marker():
    limited = t.limit_output("a" * 6000 + "b" * 6000)
    assert limited == "a" * 5000 + "\n[... output limited to 10000 bytes; 2000 interior bytes omitted ...]\n" + "b" * 5000


RESPONSES = [
    _response(),
    _response(complete=True),
    _response(complete="yes"),
    _response(complete="no"),
    _response(complete=1),
    _response((("ls\n", 0.1), ("cd x\n", 1)), complete=False),
    _response((("echo a", 0.1), ("echo b\n", 0.1))),                       # missing newline before the next one
    _response(()),
    "Sure, here it is:\n```json\n" + _response() + "\n```\nDone.",
    '{"plan": "p", "analysis": "a", "commands": []}',                       # wrong order
    '{"analysis": "a", "plan": "p"}',
    '{"analysis": "a", "plan": "p", "commands": "ls"}',
    '{"analysis": 1, "plan": ["p"], "commands": [], "task_complete": 3}',
    '{"analysis": "a", "plan": "p", "commands": [{"keystrokes": 5, "duration": 1}]}',
    '{"analysis": "a", "plan": "p", "commands": [{"keystrokes": 5}], "task_complete": true}',
    '{"analysis": "a", "plan": "p", "commands": ["ls"]}',
    '{"analysis": "a", "plan": "p", "commands": [{"duration": 1}]}',
    '{"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\\n", "duration": "fast"}]}',
    '{"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\\n"}]}',
    '{"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\\n", "duration": 1, "wait": true}]}',
    '{"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\\n", "duration": 1}]',   # unclosed
    '{"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\\n", "duration": 1}],',
    '{"analysis": "a" "plan": "p", "commands": []}',
    '{"analysis": "a\\q", "plan": "p", "commands": []}',
    '[1, 2]',
    'no json at all',
    '',
    'prefix {"x": 1} {"analysis": "a", "plan": "p", "commands": []}',
    '{"analysis": "say \\"}\\" here", "plan": "p", "commands": []} trailing',
    '{' + '"analysis": "' + "x" * 300 + '", oops}',
    '{"analysis": "a", "plan": "p", "commands": [],}',
    '{"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\\n", "duration": 1},]}',
]


@pytest.mark.parametrize("response", RESPONSES)
def test_the_parser_is_upstreams(response):
    expected = upstream.parser().parse_response(response)
    got = t.parse_response(response)
    if "Illegal trailing comma" in expected.error:                      # CPython >= 3.13: worded as 3.12 here
        assert sys.version_info >= (3, 13) and got["error"].startswith("Invalid JSON: Expecting ")
        return
    assert got == {"commands": [c.keystrokes for c in expected.commands], "complete": bool(expected.is_task_complete),
                   "error": expected.error, "warning": expected.warning}


def test_a_json_error_is_worded_as_python_3_12_words_it():
    for doc, pos, message, expected in [
            ('{"a": 1,}', 7, "Illegal trailing comma before end of object",
             "Expecting property name enclosed in double quotes: line 1 column 9 (char 8)"),
            ('[1,\n ]', 2, "Illegal trailing comma before end of array", "Expecting value: line 2 column 2 (char 5)"),
            ('{"a" 1}', 5, "Expecting ':' delimiter", "Expecting ':' delimiter: line 1 column 6 (char 5)")]:
        assert t._decode_error(json.JSONDecodeError(message, doc, pos)) == expected
        if sys.version_info[:2] == (3, 12) and doc != '{"a" 1}':
            with pytest.raises(json.JSONDecodeError) as raised:
                json.loads(doc)
            assert str(raised.value) == expected


# -- the turn ----------------------------------------------------------------------------------------------------

def test_a_turn_runs_its_keystrokes_as_one_bash_call():
    (name, arguments), = _actions(_response((("ls\n", 0.1), ("pwd\n", 0.1))))
    assert name == t.KEYS_CALL
    assert json.loads(arguments) == {"keystrokes": ["ls\n", "pwd\n"], "feedback": "", "complete": False}
    tool, call = t.translate(name, arguments)
    command = json.loads(call)["command"]
    assert tool == "bash" and command.startswith(t.MARK) and t.header_of(command) == {
        "complete": False, "feedback": "", "notes": []}
    assert t.translate(t.PARSE_ERROR_CALL, "{}") == (t.PARSE_ERROR_CALL, "{}")


def test_a_parsing_error_is_answered_locally_with_upstreams_message():
    (name, arguments), = _actions("I will run ls.")
    assert name == t.PARSE_ERROR_CALL
    assert t.refusal(name, arguments) == ("Previous response had parsing errors:\nERROR: No valid JSON found in "
                                          "response\nWARNINGS: - No valid JSON object found\n\nPlease fix these "
                                          "issues and provide a proper JSON response.")
    (name, arguments), = _actions('{"analysis": "a", "plan": "p"}')
    assert t.refusal(name, arguments).startswith(
        "Previous response had parsing errors:\nERROR: Missing required fields: commands\n\n")


def test_a_turn_cut_at_its_cap_gets_the_output_length_error():
    (name, arguments), = _actions('{"analysis": "long', ended=False)
    assert name == t.LENGTH_ERROR_CALL
    assert t.refusal(name, arguments) == t.LENGTH_ERROR + "\n\nParser warnings from your truncated response:\n" \
                                                          "- No valid JSON object found"
    assert t.refusal(*_actions(_response(), ended=False)[0]) == t.LENGTH_ERROR


def test_task_complete_twice_in_a_row_ends_the_episode():
    instance = t.INSTANCE.replace("{task}", TASK)
    asked = t.confirmation("New Terminal Output:\n$ ")
    # First true: its commands still run, the observation asks to confirm.
    (name, arguments), = _actions(_response(complete=True), earlier=[instance])
    assert name == t.KEYS_CALL and json.loads(arguments)["complete"] is True
    command = json.loads(t.translate(name, arguments)[1])["command"]
    assert t.observation(_body(command, "$ ")) == asked
    # The confirmation: the episode ends, nothing runs.
    assert _actions(_response(complete=True), earlier=[asked, instance]) is None
    assert _actions(_response((), complete="true"), earlier=["  " + asked + "\n"]) is None   # the template strips
    # Parsing and output-length errors in between keep it pending, as upstream (they `continue` before the reset).
    parse = t.refusal(*_actions("oops")[0])
    length = t.refusal(*_actions("{", ended=False)[0])
    assert _actions(_response(complete=True), earlier=[parse, length, asked, instance]) is None
    # A turn without it resets it.
    plain = t.observation(_body(command.replace('"complete": true', '"complete": false'), "$ "))
    assert _actions(_response(complete=True), earlier=[plain, asked, instance])[0][0] == t.KEYS_CALL
    # Output that only mimics the question is not it.
    fake = t.observation(_body(command.replace('"complete": true', '"complete": false'), asked))
    assert fake.startswith(t.NEW_OUTPUT) and _actions(_response(complete=True), earlier=[fake])[0][0] == t.KEYS_CALL
    # Parse errors in a complete turn's commands are warnings (upstream): complete, nothing to run.
    first = _actions('{"analysis": "a", "plan": "p", "commands": [5], "task_complete": true}', earlier=[instance])
    assert json.loads(first[0][1])["keystrokes"] == [] and json.loads(first[0][1])["complete"] is True


def test_keystrokes_are_typed_text_enter_and_c_c():
    assert t.typed(["ls -la\n", "Enter", "cd x", "C-m", "a\r\nb\rc"]) == ("ls -la\n\ncd x\na\nb\nc", False, [])
    assert t.typed(["echo half", "C-c", "ls\n"]) == ("ls\n", True, [t.CLEARED])
    assert t.typed(["ls\n", "echo half", "C-c"]) == ("ls\n", False, [t.CLEARED])
    text, clear, notes = t.typed(["C-d", "Escape", "Up", "M-x", "q", "Space", "C-cat\n", "C-c"])
    assert text == "q C-cat\n" and not clear
    assert notes == [t.NOT_SENT.format(key=k) for k in ("C-d", "Escape", "Up", "M-x")] + [t.CLEARED]


# -- the box -----------------------------------------------------------------------------------------------------

needs_bash = pytest.mark.skipif(not (os.path.exists("/bin/bash") and shutil.which("base64")), reason="no bash")


class _Box:
    """Runs each turn's canonical call as the gateway does: `/bin/bash -c` in the workdir, stdin /dev/null."""

    def __init__(self, tmp_path):
        self.state, self.work = tmp_path / "state", tmp_path / "work"
        self.work.mkdir()
        self.env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(self.work)}

    def turn(self, keystrokes, complete=False, timeout=None):
        _, arguments = t.translate(t.KEYS_CALL, json.dumps({"keystrokes": keystrokes, "feedback": "",
                                                              "complete": complete}))
        command = json.loads(arguments)["command"]
        argv = ["/bin/bash", "-c", command.replace(t.STATE_DIR, str(self.state))]
        if timeout:
            argv = ["timeout", "-s", "KILL", str(timeout), "setsid", "--wait", *argv]
        done = subprocess.run(argv, cwd=self.work, env=self.env, stdin=subprocess.DEVNULL, capture_output=True)
        output = (done.stdout + done.stderr).decode("utf-8", "replace")
        return t.observation(_body(command, output, done.returncode, timed_out=done.returncode == -9))


def _lines(observation):
    return observation.split("\n")


@needs_bash
def test_the_shell_session_persists_across_turns(tmp_path):
    box = _Box(tmp_path)
    first = box.turn(["mkdir -p sub && cd sub\n", "export FOO='a b'; X=1\n", "f() { echo fn$1; }\n",
                      "alias hi='echo hi'\n", "echo err >&2\n", "for i in 1 2; do\n", "echo $i; done\n"])
    assert first.startswith(t.NEW_OUTPUT) and "err\n" in first and "\n1\n2\n" in first
    assert "> echo $i; done\n" in first                                            # continuation prompt, as typed
    second = box.turn(["pwd; echo \"$FOO|$X\"; f 2; hi\n", "echo -n partial"])
    assert f"{tmp_path}/work/sub\na b|1\nfn2\nhi\n" in second
    assert second.endswith("echo -n partial")                                     # typed, not run
    third = box.turn([" more\n"])
    assert "partialmore" not in third and "partial more\n" in third


@needs_bash
def test_keystrokes_are_never_interpolated(tmp_path):
    box = _Box(tmp_path)
    tricky = "it's $HOME `id` $(echo no) \\ \"q\" TERMINUS_2_RC EOF"
    seen = box.turn([f"cat > f.txt <<'EOF'\n{tricky}\nEOF\n", "cat f.txt\n"])
    assert (tmp_path / "work" / "f.txt").read_text() == tricky + "\n"
    assert seen.count(tricky) >= 2
    assert "\x1b" not in box.turn(["printf '\\033[31mred\\033[0m\\n'\n"])


@needs_bash
def test_c_c_clears_the_line_left_typed(tmp_path):
    box = _Box(tmp_path)
    box.turn(["rm -rf /nonexistent-dir-x"])                                        # no Enter: nothing runs
    seen = box.turn(["C-c", "echo fresh\n"])
    assert seen.startswith(t.NEW_OUTPUT + t.CLEARED + "\n") and "\nfresh\n" in seen and "nonexistent" not in seen


@needs_bash
def test_an_empty_turn_shows_the_prompt(tmp_path):
    seen = _Box(tmp_path).turn([])
    assert seen.startswith(t.NEW_OUTPUT) and "exit" not in seen


@needs_bash
@pytest.mark.skipif(not shutil.which("setsid") or not shutil.which("timeout"), reason="no setsid/timeout")
def test_a_timed_out_turn_keeps_the_shell_state(tmp_path):
    box = _Box(tmp_path)
    seen = box.turn(["cd /\n", "export Y=2\n", "sleep 30\n", "echo never\n"], timeout=2)
    assert seen.endswith(t.TIMED_OUT) and "never\n" not in seen
    assert "\n/\nY=2\n" in box.turn(["pwd; echo Y=$Y\n"])


# -- the observation ---------------------------------------------------------------------------------------------

def test_the_observation_is_the_new_terminal_output():
    command = t.script(["ls\n"], {"complete": False, "feedback": "", "notes": []})
    assert t.observation(_body(command, "$ ls\na\n$ ")) == "New Terminal Output:\n$ ls\na\n$ "
    long = t.observation(_body(command, "x" * 20000))
    assert long == t.limit_output("New Terminal Output:\n" + "x" * 20000)
    assert "\n[... output limited to 10000 bytes; 10021 interior bytes omitted ...]\n" in long
    warned = t.script(["ls\n"], {"complete": False, "feedback": "WARNINGS: - Extra text detected before JSON object",
                                 "notes": ["[note]"]})
    assert t.observation(_body(warned, "out\n", truncated=True, timed_out=True)) == (
        "Previous response had warnings:\nWARNINGS: - Extra text detected before JSON object\n\n"
        f"New Terminal Output:\n[note]\nout\n{t.CUT}\n{t.TIMED_OUT}")
    assert t.observation(_body(command, "a\x1b[1mb\x1b[0m\r\nc 10%\rc 100%\n")) == "New Terminal Output:\nab\nc 100%\n"
    with pytest.raises(ValueError):
        t.observation(_body("ls", "x"))


def test_every_answer_is_a_user_message():
    assert t.user_message("anything") and t.tool_specs(("bash", "edit")) == () and t.system_prompt() == ""
    assert t.refusal("?other", "{}") is None


# -- the spec ----------------------------------------------------------------------------------------------------

def test_the_spec_is_registered_and_pinned():
    assert get_spec("terminus-2", "1") is TERMINUS_2_V1 and "terminus-2@1" in SPECS
    assert TERMINUS_2_V1.text_actions and TERMINUS_2_V1.translated and TERMINUS_2_V1.translator() is t
    assert TERMINUS_2_V1.tool_specs(("bash",)) == () and TERMINUS_2_V1.system_prompt_for(("bash", "edit")) == ""
    assert TERMINUS_2_V1.observation_format() == "terminus"
    assert TERMINUS_2_V1.stops["after_turn"] == {"task_complete_confirmed": "agent_completed"}
    assert TERMINUS_2_V1.stops["max_tokens_per_turn"] == "job"
    assert digest_refusal("terminus-2", "1", TERMINUS_2_V1.digest) is None
    assert "the contract pins" in digest_refusal("terminus-2", "1", "0" * 64)


# -- the conversation in the pinned template -----------------------------------------------------------------------

needs_tokenizer = pytest.mark.skipif(not TOKENIZERS, reason="set RELIQUARY_QWEN38_TOKENIZER")
TURNS = [
    "Look.\n</think>\n\n" + _response((("ls -la\n", 0.1), ("cat README\n", 0.5))),
    "Hmm.\n</think>\n\nI will look around first.",
    "Done.\n</think>\n\n" + _response((("cat out.txt\n", 0.1),), complete=True),
    "Sure.\n</think>\n\n" + _response((), complete=True),
]
FROZEN = {"422fa18269c6e298": "aeb330f72e3436b6b44c2ec4f0df5bd234d008c80795d2e0cbde078c60832f90",
          "cd79c4a4fa1a9867": "aeb330f72e3436b6b44c2ec4f0df5bd234d008c80795d2e0cbde078c60832f90"}
"""sha256 of each tokenizer's conversation (canonical JSON of `_conversation`)."""


def _renderer(directory):
    from reliquary_harness.qwen import load_turn_renderer

    return load_turn_renderer(directory, ("bash", "edit"), spec=TERMINUS_2_V1)


def _conversation(directory):
    r = _renderer(directory)
    prompt = r.initial_ids(TASK)
    rows = []
    for text in TURNS:
        completion = [int(i) for i in r._tokenizer.encode(text + "<|im_end|>", add_special_tokens=False)]
        calls = r.tool_calls(completion, prompt)
        row = {"completion_ids": completion, "tool_calls": [list(c) for c in calls],
               "assistant_message": r.assistant_message(completion)}
        if calls:
            name, arguments = calls[0]
            seen = (r.refusal_observation(name, arguments) if name.startswith("?")
                    else r.observation(_body(json.loads(arguments)["command"], "$ ls -la\na.txt\n$ "), "unused"))
            prompt = r.next_prompt(prompt, completion, [seen])
            row.update(observation=seen, next_prompt_ids=prompt)
        rows.append(row)
    return {"spec": TERMINUS_2_V1.key, "initial_ids": r.initial_ids(TASK), "turns": rows}


@needs_tokenizer
def test_a_terminus_conversation_renders_and_reads_back():
    directory = next(iter(TOKENIZERS.values()))
    r = _renderer(directory)
    text = r._tokenizer.decode(r.initial_ids(TASK), skip_special_tokens=False)
    assert "<tools>" not in text and t.INSTANCE.replace("{task}", TASK).strip() in text
    assert text.count("<|im_start|>system") <= 1 and "You are an AI assistant tasked" not in text.split("<|im_start|>user")[0]
    golden = _conversation(directory)
    run, words, complete, confirmed = golden["turns"]
    assert run["tool_calls"][0][0] == "bash" and run["observation"].startswith("New Terminal Output:\n$ ls -la")
    assert words["tool_calls"][0][0] == t.PARSE_ERROR_CALL and words["observation"].startswith(t.PARSE_ERROR_PREFIX)
    assert complete["tool_calls"][0][0] == "bash" and complete["observation"] == t.confirmation(
        "New Terminal Output:\n$ ls -la\na.txt\n$ ")
    assert confirmed["tool_calls"] == []                                           # confirmed: the episode ends
    bridged = r._tokenizer.decode(run["next_prompt_ids"][len(golden["initial_ids"]):], skip_special_tokens=False)
    assert "<|im_start|>user\nNew Terminal Output:" in bridged and "<tool_response>" not in bridged
    with pytest.raises(ValueError, match="reads a turn with the prompt"):
        r.tool_calls(confirmed["completion_ids"])
    # Without the confirmation in its prompt, the same answer is a first `task_complete`.
    assert r.tool_calls(confirmed["completion_ids"], golden["initial_ids"])[0][0] == "bash"


@needs_tokenizer
def test_terminus_renders_its_frozen_conversation():
    for digest, directory in TOKENIZERS.items():
        got = hashlib.sha256(json.dumps(_conversation(directory), sort_keys=True).encode()).hexdigest()
        if os.environ.get("HARNESS_WRITE_GOLDENS") == "1":
            print(f"{digest}: {got}")
            continue
        assert got == FROZEN[digest], digest
