"""OpenHands CodeAct's text harness (`openhands-xml@1`): one `<function=...>` call per turn read from the
answer, its four functions run through bash (the editor as a small Python program), results shown back as
OpenHands' tool results in user messages, `finish` ending the episode."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys

import pytest
from reliquary_harness import openhands as oh
from reliquary_harness.spec import OPENHANDS_XML_V1, digest_refusal
from test_qwen_goldens import GOLDENS, TOKENIZERS


def _fn(name, **params):
    body = "".join(f"<parameter={k}>{v}</parameter>\n" for k, v in params.items())
    return f"<function={name}>\n{body}</function>"


def _one(*texts):
    (call,) = oh.turn_calls(list(texts))
    return call[0], json.loads(call[1])


def test_a_turn_names_one_function_and_openhands_refuses_what_it_refuses():
    assert _one("I list.\n" + _fn("execute_bash", command="ls -la")) == ("execute_bash", {"command": "ls -la"})
    assert _one("", "\n" + _fn("execute_bash", command="pwd") + "\n") == ("execute_bash", {"command": "pwd"})
    # The first function only (OpenHands stops at `</function`); values are raw.
    assert _one(_fn("execute_bash", command="a") + _fn("execute_bash", command="b"))[1] == {"command": "a"}
    assert _one("<function=execute_bash>\n<parameter=command>\nls\n</parameter>\n</function>")[1] == \
        {"command": "\nls\n"}
    # A missing `</function>` is closed (`_fix_stopword`); a malformed tag is normalized.
    assert _one("<function=execute_bash>\n<parameter=command>ls</parameter>")[1] == {"command": "ls"}
    assert _one("<function=execute_bash>\n<parameter=command=ls</parameter>\n</function>")[1] == {"command": "ls"}
    assert _one(_fn("str_replace_editor", command="view", path="/x", view_range="[1, 3]"))[1]["view_range"] == [1, 3]
    assert _one(_fn("str_replace_editor", command="insert", path="/x", insert_line="2", new_str="y"))[1][
        "insert_line"] == 2
    assert oh.turn_calls(["No call, just words."]) == [(oh.CONTINUE_CALL, "{}")]
    for text, why in [(_fn("browse", url="x"), "Function 'browse' not found in available tools"),
                      (_fn("str_replace_editor", command="undo_edit", path="/x"), "is expected to be one of"),
                      (_fn("str_replace_editor", path="/x"), "Missing required parameters for function"),
                      (_fn("execute_bash", command="ls", is_input="true"), "Parameter 'is_input' is not allowed"),
                      (_fn("str_replace_editor", command="insert", path="/x", insert_line="two", new_str="y"),
                       "expected to be an integer"),
                      (_fn("str_replace_editor", command="view", path="/x", view_range="1,2"),
                       "expected to be an array")]:
        name, arguments = _one(text)
        assert name == oh.ERROR_CALL and why in arguments["message"], (text, arguments)


def test_each_function_runs_as_a_canonical_call():
    assert oh.translate("execute_bash", json.dumps({"command": "ls"})) == ("bash", json.dumps({"command": "ls"}))
    name, arguments = oh.translate("str_replace_editor", json.dumps({"command": "view", "path": "/x"}))
    assert name == "bash" and json.loads(arguments)["command"].startswith(oh.EDITOR_MARK + "\npython3 -I -c ")
    assert oh.translate("think", json.dumps({"thought": "hmm"})) == (oh.THINK_CALL, "{}")
    assert oh.translate("finish", json.dumps({"message": "done"})) is None
    assert oh.translate(oh.CONTINUE_CALL, "{}") == (oh.CONTINUE_CALL, "{}")


def _edit(tmp_path, **params):
    command = oh.editor_command(params)
    done = subprocess.run(["bash", "-c", command], capture_output=True, text=True, cwd=tmp_path)
    return done.returncode, done.stdout


def test_the_editor_does_what_openhands_editor_does(tmp_path):
    f = tmp_path / "a.py"
    assert _edit(tmp_path, command="create", path=str(f), file_text="one\ntwo\nthree\n") == \
        (0, f"File created successfully at: {f}")
    code, text = _edit(tmp_path, command="create", path=str(f), file_text="x")
    assert code == 1 and text == (f"ERROR:\nInvalid `path` parameter: {f}. File already exists at: {f}. "
                                  "Cannot overwrite files using command `create`.")
    assert _edit(tmp_path, command="view", path=str(f)) == \
        (0, f"Here's the result of running `cat -n` on {f}:\n     1\tone\n     2\ttwo\n     3\tthree\n")
    assert _edit(tmp_path, command="view", path=str(f), view_range=[2, -1])[1].endswith("     2\ttwo\n     3\tthree\n")
    assert _edit(tmp_path, command="view", path=str(f), view_range=[2, 9])[1].startswith(
        "NOTE: We only show up to 3 since there're only 3 lines in this file.\n")
    assert "Its first element `5` should be within the range of lines of the file: [1, 3]." in \
        _edit(tmp_path, command="view", path=str(f), view_range=[5, 6])[1]
    code, text = _edit(tmp_path, command="str_replace", path=str(f), old_str="two", new_str="TWO")
    assert code == 0 and text == (f"The file {f} has been edited. Here's the result of running `cat -n` on a snippet "
                                  f"of {f}:\n     1\tone\n     2\tTWO\n     3\tthree\nReview the changes and make sure "
                                  "they are as expected. Edit the file again if necessary.")
    assert f.read_text() == "one\nTWO\nthree\n"
    assert _edit(tmp_path, command="str_replace", path=str(f), old_str="  three  ", new_str="3")[0] == 0   # stripped retry
    assert "did not appear verbatim" in _edit(tmp_path, command="str_replace", path=str(f), old_str="zzz")[1]
    (tmp_path / "b.py").write_text("x\nx\n")
    assert "Multiple occurrences of old_str `x` in lines [1, 2]" in \
        _edit(tmp_path, command="str_replace", path=str(tmp_path / "b.py"), old_str="x", new_str="y")[1]
    code, text = _edit(tmp_path, command="insert", path=str(f), insert_line=1, new_str="inserted")
    assert code == 0 and "     2\tinserted" in text and f.read_text() == "one\ninserted\nTWO\n3\n"
    assert "It should be within the range of allowed values: [0, 4]" in \
        _edit(tmp_path, command="insert", path=str(f), insert_line=9, new_str="y")[1]
    assert "The path should be an absolute path." in _edit(tmp_path, command="view", path="a.py")[1]
    assert "does not exist" in _edit(tmp_path, command="view", path=str(tmp_path / "nope"))[1]
    assert "only the `view` command can be used on directories" in \
        _edit(tmp_path, command="str_replace", path=str(tmp_path), old_str="x")[1]
    (tmp_path / ".hidden").write_text("")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("")
    code, text = _edit(tmp_path, command="view", path=str(tmp_path))
    assert text == (f"Here's the files and directories up to 2 levels deep in {tmp_path}, excluding hidden items:\n"
                    f"{tmp_path}/\n{tmp_path}/a.py\n{tmp_path}/b.py\n{tmp_path}/sub/\n{tmp_path}/sub/c.py\n\n"
                    f"1 hidden files/directories in this directory are excluded. You can use 'ls -la {tmp_path}' "
                    "to see them.")


def _body(command, output="", exit_code=0, timed_out=False):
    return {"tool": "bash", "arguments": {"command": command}, "output": output, "exit_code": exit_code,
            "truncated": False, "timed_out": timed_out}


def test_an_observation_is_openhands_tool_result():
    assert oh.observation(_body("ls", "a.py\n")) == \
        "EXECUTION RESULT of [execute_bash]:\na.py\n\n[Command finished with exit code 0]"
    long = oh.observation(_body("cat big", "x" * 40000, 1))
    assert "\n[... Observation truncated due to length ...]\n" in long and long.endswith("[Command finished with exit code 1]")
    assert len(long) < 30200
    assert "timed out" in oh.observation(_body("sleep 999", "", -1, timed_out=True))
    editor = oh.editor_command({"command": "view", "path": "/x"})
    assert oh.observation(_body(editor, "ERROR:\nboom", 1)) == "EXECUTION RESULT of [str_replace_editor]:\nERROR:\nboom"
    assert oh.refusal(oh.THINK_CALL, "{}") == "EXECUTION RESULT of [think]:\nYour thought has been logged."
    assert oh.refusal(oh.CONTINUE_CALL, "{}").startswith("Please continue working on the task")
    assert oh.refusal(oh.ERROR_CALL, json.dumps({"message": "bad"})) == "bad"
    assert oh.refusal("search", "{}") is None


def test_the_spec_shows_no_template_tool_and_describes_the_functions():
    assert OPENHANDS_XML_V1.tool_specs(("bash", "edit")) == ()
    prompt = OPENHANDS_XML_V1.system_prompt_for(("bash",))
    assert prompt.startswith("You are OpenHands agent") and "---- BEGIN FUNCTION #4: str_replace_editor ----" in prompt
    assert "SECURITY_RISK" not in prompt and "security_risk" not in prompt and "undo_edit" not in prompt
    assert OPENHANDS_XML_V1.user_prompt("Fix it.").startswith("<issue_description>\nFix it.\n</issue_description>")
    assert digest_refusal(OPENHANDS_XML_V1.id, OPENHANDS_XML_V1.version, OPENHANDS_XML_V1.digest) is None


needs_tokenizer = pytest.mark.skipif(not TOKENIZERS, reason="set RELIQUARY_QWEN38_TOKENIZER")
TASK = "Fix the failing test in tests/test_dates.py."
TURNS = [
    "Look first.\n</think>\n\nI will list the files.\n" + _fn("execute_bash", command="ls"),
    "Edit it.\n</think>\n\n<tool_call>\n" + _fn("str_replace_editor", command="view", path="/testbed") + "\n</tool_call>",
    "Thinking aloud.\n</think>\n\nI believe it works.",
    "Done.\n</think>\n\n" + _fn("finish", message="Fixed."),
]


def _renderer(directory, tools=("bash", "edit")):
    from reliquary_harness.qwen import load_turn_renderer

    return load_turn_renderer(directory, tools, spec=OPENHANDS_XML_V1)


def _golden(directory):
    r = _renderer(directory)
    prompt = r.initial_ids(TASK)
    rows = []
    for text in TURNS:
        completion = [int(t) for t in r._tokenizer.encode(text + "<|im_end|>", add_special_tokens=False)]
        calls = r.tool_calls(completion)
        row = {"completion_ids": completion, "tool_calls": [list(c) for c in calls],
               "assistant_message": r.assistant_message(completion)}
        if calls:
            name, arguments = calls[0]
            seen = (r.refusal_observation(name, arguments) if name.startswith("?")
                    else r.observation(_body(json.loads(arguments)["command"], "a.py\n"), "a.py\n"))
            prompt = r.next_prompt(prompt, completion, [seen])
            row.update(observation=seen, next_prompt_ids=prompt)
        rows.append(row)
    return {"spec": OPENHANDS_XML_V1.key, "initial_ids": r.initial_ids(TASK), "turns": rows}


@needs_tokenizer
def test_an_openhands_conversation_renders_and_reads_back():
    directory = next(iter(TOKENIZERS.values()))
    r = _renderer(directory)
    text = r._tokenizer.decode(r.initial_ids(TASK), skip_special_tokens=False)
    assert "# Tools" not in text and "<tools>" not in text and "You are OpenHands agent" in text
    golden = _golden(directory)
    first, second, third, last = golden["turns"]
    assert first["tool_calls"] == [["bash", json.dumps({"command": "ls"})]]
    assert second["tool_calls"][0][0] == "bash" and oh.EDITOR_MARK in second["tool_calls"][0][1]
    assert third["tool_calls"] == [[oh.CONTINUE_CALL, "{}"]] and third["observation"].startswith("Please continue")
    assert last["tool_calls"] == []                                       # finish: the episode ends
    bridged = r._tokenizer.decode(first["next_prompt_ids"][len(golden["initial_ids"]):], skip_special_tokens=False)
    assert "<|im_start|>user\nEXECUTION RESULT of [execute_bash]:\na.py\n" in bridged and "<tool_response>" not in bridged


@needs_tokenizer
def test_openhands_renders_its_frozen_golden():
    for digest, directory in TOKENIZERS.items():
        path = GOLDENS / f"openhands-xml.{digest}.json"
        got = json.loads(json.dumps(_golden(directory)))
        if os.environ.get("HARNESS_WRITE_GOLDENS") == "1":
            path.write_text(json.dumps(got, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        assert got == json.loads(path.read_text(encoding="utf-8"))


def test_the_editor_program_is_pure_python3():
    """The editor is decoded and run by the box's python3; it must compile under the oldest Python a SWE
    image may carry (3.6 at least: no walrus, no match)."""
    source = oh._EDITOR
    compile(source, "<editor>", "exec")
    assert ":=" not in source and "match " not in source.replace("re.finditer", "")
    assert base64.b64decode(oh._EDITOR_B64)
    assert sys.version_info >= (3, 8)
