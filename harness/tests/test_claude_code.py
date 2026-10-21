"""The Claude Code-style harness (`claude-code@1`): the template's calls to Bash, Read, Edit and Write, each
run through bash (the file tools as a small Python program), results as tool messages, a turn without a call
ending the episode; a call to a tool it does not have, or with arguments that do not fit, answered locally."""

from __future__ import annotations

import json
import os
import subprocess

import pytest
from reliquary_harness import claude_code as cc
from reliquary_harness.spec import CLAUDE_CODE_V1, digest_refusal
from test_qwen_goldens import GOLDENS, TOKENIZERS


def _run(tool, **params):
    done = subprocess.run(["bash", "-c", cc.file_command(tool, params)], capture_output=True, text=True)
    return done.returncode, done.stdout


def test_each_tool_runs_as_a_canonical_call_and_the_rest_is_answered_locally():
    assert cc.translate("Bash", json.dumps({"command": "ls", "description": "list"})) == \
        ("bash", json.dumps({"command": "ls"}))
    name, arguments = cc.translate("Read", json.dumps({"file_path": "/x"}))
    assert name == "bash" and json.loads(arguments)["command"].startswith("# claude-code:Read\npython3 -I -c ")
    assert cc.translate("Grep", "{}") == ("?Grep", "{}")
    assert cc.translate("bash", json.dumps({"command": "rm -rf /"})) == ("?bash", "{}")     # never the canonical tool
    for tool, arguments, why in [("Edit", {"file_path": "/x", "old_string": "a"}, "`new_string` is missing"),
                                 ("Read", {"file_path": 3}, "expected as `string`"),
                                 ("Bash", {"command": "ls", "cwd": "/"}, "unexpected parameter `cwd`"),
                                 ("Read", {"file_path": "/x", "offset": True}, "expected as `number`")]:
        name, details = cc.translate(tool, json.dumps(arguments))
        assert name == cc.INVALID_CALL and why in json.loads(details)["message"]
    assert cc.translate("Bash", "not json")[0] == cc.INVALID_CALL
    assert cc.refusal("?Grep", "{}") == "<tool_use_error>Error: No such tool available: Grep</tool_use_error>"
    assert cc.refusal(cc.INVALID_CALL, json.dumps({"tool": "Edit", "message": "m"})).startswith(
        "<tool_use_error>InputValidationError: Edit failed due to the following issue:\nm")


def test_the_file_tools_read_edit_and_write(tmp_path):
    f = tmp_path / "a.py"
    assert _run("Read", file_path=str(f)) == (1, "<tool_use_error>File does not exist.</tool_use_error>")
    assert _run("Write", file_path=str(tmp_path / "d" / "a.py"), content="x\n") == \
        (0, f"File created successfully at: {tmp_path / 'd' / 'a.py'}")
    f.write_text("one\ntwo\ntwo\n")
    assert _run("Read", file_path=str(f)) == (0, "1\tone\n2\ttwo\n3\ttwo")
    assert _run("Read", file_path=str(f), offset=2, limit=1) == \
        (0, "2\ttwo\n\n[Showing lines 2-2 of 3. Use offset and limit to read more.]")
    assert "shorter than the provided offset (9)" in _run("Read", file_path=str(f), offset=9)[1]
    assert "EISDIR" in _run("Read", file_path=str(tmp_path))[1]
    code, text = _run("Edit", file_path=str(f), old_string="two", new_string="2")
    assert code == 1 and text.startswith("<tool_use_error>Found 2 matches of the string to replace, but replace_all")
    assert _run("Edit", file_path=str(f), old_string="two", new_string="2", replace_all=True) == \
        (0, f"The file {f} has been updated successfully.")
    assert f.read_text() == "one\n2\n2\n"
    assert "String to replace not found in file.\nString: zzz" in _run("Edit", file_path=str(f), old_string="zzz",
                                                                          new_string="y")[1]
    assert "No changes to make" in _run("Edit", file_path=str(f), old_string="one", new_string="one")[1]
    assert _run("Edit", file_path=str(f), old_string="one", new_string="1") == (0, f"The file {f} has been updated "
                                                                                     "successfully.")
    assert "Cannot create new file - file already exists." in _run("Edit", file_path=str(f), old_string="",
                                                                   new_string="z")[1]
    assert _run("Edit", file_path=str(tmp_path / "new.py"), old_string="", new_string="z\n")[0] == 0
    (tmp_path / "empty").write_text("")
    assert _run("Read", file_path=str(tmp_path / "empty"))[1] == "Warning: the file exists but the contents are empty."
    (tmp_path / "long").write_text("x" * 2500 + "\n")
    assert _run("Read", file_path=str(tmp_path / "long"))[1].endswith("... [line truncated]")


def _body(command, output="", exit_code=0, timed_out=False):
    return {"tool": "bash", "arguments": {"command": command}, "output": output, "exit_code": exit_code,
            "truncated": False, "timed_out": timed_out}


def test_a_result_is_the_output_with_its_exit_code_when_it_failed():
    assert cc.observation(_body("ls", "a.py\n")) == "a.py\n"
    assert cc.observation(_body("false", "", 1)) == "Exit code 1\n(No output)"
    assert cc.observation(_body("true", "")) == "(No output)"
    assert cc.observation(_body("big", "y" * 40000)).endswith("[output truncated: 10000 more characters]")
    assert cc.observation(_body("sleep 999", "", -1, timed_out=True)).startswith("Command timed out")
    command = cc.file_command("Read", {"file_path": "/x"})
    assert cc.observation(_body(command, "<tool_use_error>File does not exist.</tool_use_error>", 1)) == \
        "<tool_use_error>File does not exist.</tool_use_error>"


def test_the_spec_shows_its_four_tools():
    assert [t["function"]["name"] for t in CLAUDE_CODE_V1.tool_specs(("bash", "edit"))] == ["Bash", "Read", "Edit",
                                                                                            "Write"]
    assert CLAUDE_CODE_V1.tool_specs(("bash",)) == CLAUDE_CODE_V1.tool_specs(("bash", "edit"))
    assert CLAUDE_CODE_V1.system_prompt_for(("bash",)) == cc.SYSTEM
    assert CLAUDE_CODE_V1.user_prompt("Fix it.") == "Fix it."
    assert digest_refusal(CLAUDE_CODE_V1.id, CLAUDE_CODE_V1.version, CLAUDE_CODE_V1.digest) is None


needs_tokenizer = pytest.mark.skipif(not TOKENIZERS, reason="set RELIQUARY_QWEN38_TOKENIZER")
TASK = "Fix the failing test in tests/test_dates.py."


def _call(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


TURNS = [
    "Look.\n</think>\n\nI will read the test and run it.\n\n" + _call("Read", file_path="/testbed/tests/test_dates.py")
    + "\n" + _call("Bash", command="cd /testbed && python -m pytest -q tests/test_dates.py"),
    "Fix.\n</think>\n\n" + _call("Edit", file_path="/testbed/dates.py", old_string="return 1", new_string="return 2"),
    "Search.\n</think>\n\n" + _call("Grep", pattern="dates"),
    "Done.\n</think>\n\nI changed dates.py so the test passes.",
]


def _renderer(directory, tools=("bash", "edit")):
    from reliquary_harness.qwen import load_turn_renderer

    return load_turn_renderer(directory, tools, spec=CLAUDE_CODE_V1)


def _golden(directory):
    r = _renderer(directory)
    prompt = r.initial_ids(TASK)
    rows = []
    for text in TURNS:
        completion = [int(t) for t in r._tokenizer.encode(text + "<|im_end|>", add_special_tokens=False)]
        calls = r.tool_calls(completion)
        row = {"completion_ids": completion, "tool_calls": [list(c) for c in calls],
               "assistant_message": r.assistant_message(completion),
               "canonical": r.span_is_canonical(prompt, completion)}
        if calls:
            seen = [r.refusal_observation(name, arguments) if name.startswith("?")
                    else r.observation(_body(json.loads(arguments)["command"], "ok\n"), "ok\n")
                    for name, arguments in calls]
            prompt = r.next_prompt(prompt, completion, seen)
            row.update(observations=seen, next_prompt_ids=prompt)
        rows.append(row)
    return {"spec": CLAUDE_CODE_V1.key, "initial_ids": r.initial_ids(TASK), "turns": rows}


@needs_tokenizer
def test_a_claude_code_conversation_renders_and_reads_back():
    directory = next(iter(TOKENIZERS.values()))
    golden = _golden(directory)
    first, second, third, last = golden["turns"]
    assert [name for name, _ in first["tool_calls"]] == ["bash", "bash"] and first["canonical"]
    assert [c["function"]["name"] for c in first["assistant_message"]["tool_calls"]] == ["Read", "Bash"]
    assert second["tool_calls"][0][0] == "bash" and "# claude-code:Edit" in second["tool_calls"][0][1]
    assert third["observations"] == ["<tool_use_error>Error: No such tool available: Grep</tool_use_error>"]
    assert last["tool_calls"] == [] and last["canonical"]
    r = _renderer(directory)
    bridged = r._tokenizer.decode(first["next_prompt_ids"][len(golden["initial_ids"]):], skip_special_tokens=False)
    assert bridged.count("<tool_response>\nok\n</tool_response>") == 2


@needs_tokenizer
def test_claude_code_renders_its_frozen_golden():
    for digest, directory in TOKENIZERS.items():
        path = GOLDENS / f"claude-code.{digest}.json"
        got = json.loads(json.dumps(_golden(directory)))
        if os.environ.get("HARNESS_WRITE_GOLDENS") == "1":
            path.write_text(json.dumps(got, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        assert got == json.loads(path.read_text(encoding="utf-8"))


def test_the_file_program_compiles_for_old_pythons():
    compile(cc._FILES, "<files>", "exec")
    assert ":=" not in cc._FILES
