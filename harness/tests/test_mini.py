"""mini-swe-agent's official SWE-bench harness (`mini-swe@2`): its texts against the upstream configuration (never
skipped), one bash tool through the template's tool calls, mini's turn rules (no call or a bad call is the turn's
format error, a user message; the submit command ends the episode), its observation, and a frozen golden."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import pytest
from reliquary_harness import mini
from reliquary_harness.spec import MINI_SWE_V2, SPECS, digest_refusal
from test_qwen_goldens import GOLDENS, TOKENIZERS

UPSTREAM = Path(__file__).resolve().parent / "upstream" / "mini-swe-agent-04d809ce"
PINNED = {"swebench.yaml": "9a9c86ac10428b86b932c972b15fefc2f7b6e92230bac5ebdc47e83232a8315e",
          "actions_toolcall.py": "47b666412e4f838508b50009cf34c7f63fe15e9e40affc257a17c0829896526f"}
RENDERED = json.loads((UPSTREAM / "rendered.json").read_text(encoding="utf-8"))
TASK = "Fix the failing test in tests/test_dates.py."


def _body(output, code=0, **flags):
    return {"output": output, "exit_code": code, "truncated": False, "timed_out": False, **flags}


def _bash(command):
    return "bash", json.dumps({"command": command})


def test_the_vendored_upstream_files_are_the_pinned_ones():
    for name, digest in PINNED.items():
        assert hashlib.sha256((UPSTREAM / name).read_bytes()).hexdigest() == digest
    assert MINI_SWE_V2.upstream["mini-swe-agent"].startswith("04d809ce")


def test_the_texts_are_the_upstream_templates_rendered():
    assert mini.SYSTEM == RENDERED["system"]
    assert mini.INSTANCE.replace("{task}", "<<TASK>>") == RENDERED["instance_marker"]
    assert mini.refusal(mini.FORMAT_ERROR_CALL, json.dumps({"error": mini.NO_TOOL_CALLS})) == RENDERED["format_error_none"]
    assert mini.refusal(mini.FORMAT_ERROR_CALL, json.dumps({"error": "Unknown tool 'cat'."})) == RENDERED["format_error_unknown"]
    assert mini.refusal(mini.LENGTH_ERROR_CALL, "{}") == RENDERED["format_error_length"]
    assert mini.observation(_body("a.py\nb.py\n")) == RENDERED["observation_short"]
    assert mini.observation(_body("x" * 12000, 1)) == RENDERED["observation_long"]
    tool = next(node for node in ast.parse((UPSTREAM / "actions_toolcall.py").read_text()).body
                if isinstance(node, ast.Assign) and node.targets[0].id == "BASH_TOOL")
    assert mini.BASH_TOOL == ast.literal_eval(tool.value)


def test_the_rendered_texts_are_what_the_upstream_yaml_renders():
    """The rendered texts came from the vendored YAML (re-rendered here when Jinja and YAML are installed)."""
    jinja2 = pytest.importorskip("jinja2")
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load((UPSTREAM / "swebench.yaml").read_text())
    render = lambda template, **kw: jinja2.Template(template, undefined=jinja2.StrictUndefined).render(**kw)  # noqa: E731
    assert render(config["agent"]["system_template"]) == RENDERED["system"]
    assert render(config["agent"]["instance_template"], task="<<TASK>>") == RENDERED["instance_marker"]
    assert render(config["model"]["format_error_template"], error=mini.NO_TOOL_CALLS, actions=[],
                  has_tool_calls=False) == RENDERED["format_error_none"]
    assert render(config["model"]["observation_template"],
                  output={"output": "a.py\nb.py\n", "returncode": 0, "exception_info": ""}) == RENDERED["observation_short"]


def test_a_turn_is_read_as_mini_reads_it():
    assert mini.turn_actions([_bash("ls"), _bash("pwd")], ended=True) == [_bash("ls"), _bash("pwd")]
    no_call = mini.turn_actions([], ended=True)
    assert no_call == [(mini.FORMAT_ERROR_CALL, json.dumps({"error": mini.NO_TOOL_CALLS}))]
    assert mini.turn_actions([], ended=False) == [(mini.LENGTH_ERROR_CALL, "{}")]
    bad = mini.turn_actions([_bash("ls"), ("cat", '{"path": "a"}')], ended=True)
    assert bad == [(mini.FORMAT_ERROR_CALL, json.dumps(
        {"error": "Unknown tool 'cat'.Missing 'command' argument in bash tool call."}))]          # none of the turn runs
    assert "Error parsing tool call arguments" in json.loads(mini.turn_actions([("bash", "{x")], ended=True)[0][1])["error"]
    assert mini.turn_actions([_bash(mini.SUBMIT + " && cat patch.txt")], ended=True) is None
    assert mini.translate(*_bash("ls -la")) == _bash("ls -la")
    assert mini.user_message(mini.refusal(*no_call[0])) and not mini.user_message(mini.observation(_body("")))


def test_an_observation_shows_the_gateways_flags_and_the_output_as_is():
    assert mini.observation(_body("", -1, timed_out=True)) == (
        "<exception>The command timed out; its process group was killed.</exception>\n"
        "<returncode>-1</returncode>\n<output>\n</output>")
    assert mini.observation(_body("ok\n", truncated=True)).startswith("<exception>The command's output exceeded")


def test_it_is_the_one_mini_swe_served():
    assert [key for key in SPECS if key.startswith("mini-swe")] == ["mini-swe@2"]
    assert MINI_SWE_V2.tool_specs(("bash", "edit")) == (mini.BASH_TOOL,) == MINI_SWE_V2.tool_specs(("bash",))
    assert MINI_SWE_V2.system_prompt_for(("bash", "edit")) == mini.SYSTEM
    assert MINI_SWE_V2.user_prompt(TASK).startswith(f"<pr_description>\nConsider the following PR description:\n{TASK}\n")
    assert digest_refusal(MINI_SWE_V2.id, MINI_SWE_V2.version, MINI_SWE_V2.digest) is None


needs_tokenizer = pytest.mark.skipif(not TOKENIZERS, reason="set RELIQUARY_QWEN38_TOKENIZER")


def _renderer(directory, tools=("bash", "edit")):
    from reliquary_harness.qwen import load_turn_renderer

    return load_turn_renderer(directory, tools, spec=MINI_SWE_V2)


def _call(command):
    return f"<tool_call>\n<function=bash>\n<parameter=command>\n{command}\n</parameter>\n</function>\n</tool_call>"


def _turn(r, text):
    return [int(t) for t in r._tokenizer.encode(text + "<|im_end|>", add_special_tokens=False)]


TURNS = [
    "Look around.\n</think>\n\nTHOUGHT: list the files and the README.\n\n" + _call("ls -la") + "\n" + _call("head README.md"),
    "Hmm.\n</think>\n\nI think I know what to do.",
    "Read it.\n</think>\n\n<tool_call>\n<function=cat>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>",
    "Done.\n</think>\n\nTHOUGHT: submit.\n\n" + _call(mini.SUBMIT + " && cat patch.txt"),
]


def _golden(directory):
    r = _renderer(directory)
    prompt = r.initial_ids(TASK)
    rows = []
    for text in TURNS:
        completion = _turn(r, text)
        calls = r.tool_calls(completion)
        row = {"completion_ids": completion, "tool_calls": [list(c) for c in calls],
               "assistant_message": r.assistant_message(completion)}
        if calls:
            seen = [r.refusal_observation(name, arguments) if name.startswith("?")
                    else r.observation(_body("a.py\n"), "a.py\n") for name, arguments in calls]
            prompt = r.next_prompt(prompt, completion, seen)
            row.update(observations=seen, next_prompt_ids=prompt)
        rows.append(row)
    return {"spec": MINI_SWE_V2.key, "initial_ids": r.initial_ids(TASK), "turns": rows}


@needs_tokenizer
def test_a_mini_conversation_renders_and_reads_back():
    r = _renderer(next(iter(TOKENIZERS.values())))
    text = r._tokenizer.decode(r.initial_ids(TASK), skip_special_tokens=False)
    assert '"name": "bash"' in text and '"name": "edit"' not in text and mini.SYSTEM in text
    golden = _golden(next(iter(TOKENIZERS.values())))
    two, none, unknown, submit = golden["turns"]
    assert two["tool_calls"] == [list(_bash("ls -la")), list(_bash("head README.md"))]
    assert none["tool_calls"] == [[mini.FORMAT_ERROR_CALL, json.dumps({"error": mini.NO_TOOL_CALLS})]]
    assert unknown["tool_calls"][0][0] == mini.FORMAT_ERROR_CALL and "Unknown tool 'cat'" in unknown["observations"][0]
    assert submit["tool_calls"] == []
    bridged = r._tokenizer.decode(none["next_prompt_ids"][len(two["next_prompt_ids"]):], skip_special_tokens=False)
    assert "<|im_start|>user\nTool call error:" in bridged and "<tool_response>" not in bridged      # a user message
    first = r._tokenizer.decode(two["next_prompt_ids"][len(golden["initial_ids"]):], skip_special_tokens=False)
    assert first.count("<tool_response>") == 2 and "<returncode>0</returncode>" in first


@needs_tokenizer
def test_mini_renders_its_frozen_golden():
    for digest, directory in TOKENIZERS.items():
        path = GOLDENS / f"mini-swe.{digest}.json"
        got = json.loads(json.dumps(_golden(directory)))
        if os.environ.get("HARNESS_WRITE_GOLDENS") == "1":
            path.write_text(json.dumps(got, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        assert got == json.loads(path.read_text(encoding="utf-8"))


def test_another_renderers_release_is_refused(monkeypatch):
    import importlib.metadata

    from reliquary_harness import qwen

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.1.12")
    with pytest.raises(ValueError, match=r"renderers==0\.1\.11, not the installed 0\.1\.12"):
        qwen.build_renderer("/nonexistent", MINI_SWE_V2)
