"""Codex CLI's harness (`codex@1`): the template's calls to exec_command, apply_patch and update_plan, the first
two run through bash (the patch by the ported Codex engine, `codex_patch`), results as Codex's unified exec
output, a turn without a call ending the episode."""

from __future__ import annotations

import json
import os
import subprocess

import pytest
from reliquary_harness import codex
from reliquary_harness.spec import CODEX_V1, digest_refusal
from test_qwen_goldens import GOLDENS, TOKENIZERS


def test_each_tool_runs_as_a_canonical_call_and_the_rest_is_answered_locally():
    assert codex.translate("exec_command", json.dumps({"cmd": "ls"})) == ("bash", json.dumps({"command": "ls"}))
    assert codex.translate("exec_command", json.dumps({"cmd": "ls", "workdir": "/test bed"})) == \
        ("bash", json.dumps({"command": "cd '/test bed' && ls"}))
    name, arguments = codex.translate("apply_patch", json.dumps({"input": "*** Begin Patch\n*** End Patch"}))
    assert name == "bash" and json.loads(arguments)["command"].startswith(codex.PATCH_MARK + "\npython3 -I -c ")
    assert json.loads(arguments)["command"].endswith(" --model-output")
    assert codex.translate("update_plan", json.dumps({"plan": [{"step": "a", "status": "pending"}]})) == \
        (codex.PLAN_CALL, "{}")
    assert codex.translate("shell", json.dumps({"command": ["ls"]})) == ("?shell", "{}")
    assert codex.translate("bash", json.dumps({"command": "ls"})) == ("?bash", "{}")
    for name, arguments, why in [("exec_command", {}, "missing field `cmd`"),
                                 ("exec_command", {"cmd": "ls", "tty": True}, "unknown field `tty`"),
                                 ("apply_patch", {"input": 3}, "expected a string"),
                                 ("update_plan", {"plan": "x"}, "expected a sequence")]:
        local, details = codex.translate(name, json.dumps(arguments))
        assert local == codex.INVALID_CALL and why in json.loads(details)["message"]
    assert codex.refusal(codex.PLAN_CALL, "{}") == "Plan updated"
    assert codex.refusal("?shell", "{}") == "unsupported call: shell"


def _patch(tmp_path, patch):
    done = subprocess.run(["bash", "-c", codex.patch_command(patch)], capture_output=True, text=True, cwd=tmp_path)
    return done.returncode, done.stdout


def test_a_patch_is_applied_by_codex_engine_in_the_box(tmp_path):
    (tmp_path / "app.py").write_text("def greet():\n    print('Hi')\n")
    code, text = _patch(tmp_path, "*** Begin Patch\n*** Add File: hello.txt\n+Hello world\n"
                                  "*** Update File: app.py\n@@ def greet():\n-    print('Hi')\n+    print('Hello')\n"
                                  "*** End Patch")
    assert code == 0 and text.startswith("Exit code: 0\nWall time: ")
    assert text.endswith("Output:\nSuccess. Updated the following files:\nA hello.txt\nM app.py\n")
    assert (tmp_path / "hello.txt").read_text() == "Hello world\n"
    assert (tmp_path / "app.py").read_text() == "def greet():\n    print('Hello')\n"
    code, text = _patch(tmp_path, "*** Begin Patch\n*** Update File: app.py\n@@\n-nothing here\n+x\n*** End Patch")
    assert code == 1 and text.startswith("apply_patch verification failed: Failed to find expected lines in")
    assert (tmp_path / "app.py").read_text() == "def greet():\n    print('Hello')\n"     # nothing written
    assert _patch(tmp_path, "not a patch")[1].startswith("apply_patch verification failed: invalid patch:")


def _body(command, output="", exit_code=0, timed_out=False):
    return {"tool": "bash", "arguments": {"command": command}, "output": output, "exit_code": exit_code,
            "truncated": False, "timed_out": timed_out}


def test_a_result_is_codex_unified_exec_output():
    assert codex.observation(_body("ls", "a.py\n")) == "Process exited with code 0\nOutput:\na.py\n"
    assert codex.observation(_body("false", "", 1)) == "Process exited with code 1\nOutput:\n"
    long = codex.observation(_body("big", "y" * 100000))
    assert long.startswith("Process exited with code 0\nOriginal token count: 25000\nOutput:\n")
    assert "…15000 tokens truncated…" in long and len(long) < 40200
    patch = codex.patch_command("*** Begin Patch\n*** End Patch")
    assert codex.observation(_body(patch, "patch rejected: empty patch", 1)) == "patch rejected: empty patch"
    assert codex.observation(_body("sleep 9", "", -1, timed_out=True)).startswith("Process timed out")


def test_the_spec_shows_codex_tools():
    assert [t["function"]["name"] for t in CODEX_V1.tool_specs(("bash",))] == ["exec_command", "apply_patch",
                                                                               "update_plan"]
    prompt = CODEX_V1.system_prompt_for(("bash", "edit"))
    assert prompt == codex.SYSTEM and "Codex" not in prompt and "GPT" not in prompt and "*** Begin Patch" in prompt
    assert CODEX_V1.user_prompt("Fix it.") == "Fix it."
    assert digest_refusal(CODEX_V1.id, CODEX_V1.version, CODEX_V1.digest) is None


needs_tokenizer = pytest.mark.skipif(not TOKENIZERS, reason="set RELIQUARY_QWEN38_TOKENIZER")
TASK = "Fix the failing test in tests/test_dates.py."


def _call(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


TURNS = [
    "Look.\n</think>\n\n" + _call("exec_command", cmd="rg -n dates tests", workdir="/testbed")
    + "\n" + _call("update_plan", plan='[{"step": "fix", "status": "in_progress"}]'),
    "Fix.\n</think>\n\n" + _call("apply_patch", input="*** Begin Patch\n*** Update File: dates.py\n@@\n-return 1\n"
                                                         "+return 2\n*** End Patch"),
    "Done.\n</think>\n\nFixed `dates.py`: it now returns 2.",
]


def _renderer(directory, tools=("bash", "edit")):
    from reliquary_harness.qwen import load_turn_renderer

    return load_turn_renderer(directory, tools, spec=CODEX_V1)


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
    return {"spec": CODEX_V1.key, "initial_ids": r.initial_ids(TASK), "turns": rows}


@needs_tokenizer
def test_a_codex_conversation_renders_and_reads_back():
    golden = _golden(next(iter(TOKENIZERS.values())))
    first, second, last = golden["turns"]
    assert [name for name, _ in first["tool_calls"]] == ["bash", codex.PLAN_CALL] and first["canonical"]
    assert first["observations"][1] == "Plan updated"
    assert codex.PATCH_MARK in second["tool_calls"][0][1]
    assert last["tool_calls"] == [] and last["canonical"]


@needs_tokenizer
def test_codex_renders_its_frozen_golden():
    for digest, directory in TOKENIZERS.items():
        path = GOLDENS / f"codex.{digest}.json"
        got = json.loads(json.dumps(_golden(directory)))
        if os.environ.get("HARNESS_WRITE_GOLDENS") == "1":
            path.write_text(json.dumps(got, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        assert got == json.loads(path.read_text(encoding="utf-8"))
