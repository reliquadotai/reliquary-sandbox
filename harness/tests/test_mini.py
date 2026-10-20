"""mini-swe-agent's text-based harness (`mini-swe@1`): one fenced bash block per turn, mini's
observation and error texts, the submit command ending the episode."""

from __future__ import annotations

import json
import os

import pytest
from reliquary_harness import mini
from reliquary_harness.spec import MINI_SWE_V1, digest_refusal
from test_qwen_goldens import GOLDENS, TOKENIZERS

TASK = "Fix the failing test in tests/test_dates.py."


def _block(command):
    return f"```mswea_bash_command\n{command}\n```"


def test_one_block_is_one_bash_call_and_the_submit_command_ends_the_episode():
    assert mini.turn_calls(f"THOUGHT: look.\n\n{_block('ls -la')}", ended=True) == \
        [("bash", json.dumps({"command": "ls -la"}))]
    assert mini.turn_calls(_block("cd /testbed && \\\n  pytest -q"), ended=True) == \
        [("bash", json.dumps({"command": "cd /testbed && \\\n  pytest -q"}))]
    assert mini.turn_calls(_block(mini.SUBMIT), ended=True) == []
    assert mini.turn_calls(_block(mini.SUBMIT + " && cat patch.txt"), ended=True) == []


def test_zero_or_several_blocks_are_answered_with_minis_errors():
    assert mini.turn_calls("I am done thinking.", ended=True) == [(mini.FORMAT_ERROR_CALL, '{"found": 0}')]
    assert mini.turn_calls(_block("ls") + "\n" + _block("pwd"), ended=True) == \
        [(mini.FORMAT_ERROR_CALL, '{"found": 2}')]
    assert mini.turn_calls("I was still writing the comm", ended=False) == [(mini.LENGTH_ERROR_CALL, "{}")]
    assert "found 2 actions" in mini.refusal(mini.FORMAT_ERROR_CALL, '{"found": 2}')
    assert mini.refusal(mini.LENGTH_ERROR_CALL, "{}") == mini.LENGTH_ERROR
    assert mini.refusal("search", "{}") is None


def test_an_observation_is_minis_return_code_and_output():
    assert mini.observation({"exit_code": 0, "output": "a.py\nb.py\n", "truncated": False, "timed_out": False}) == \
        "<returncode>0</returncode>\n<output>\na.py\nb.py</output>"
    long = mini.observation({"exit_code": 1, "output": "x" * 12000, "truncated": True, "timed_out": False})
    assert long.startswith("<exception>The command's output exceeded the sandbox's limit and was cut.</exception>\n"
                           "<returncode>1</returncode>\n<warning>")
    assert "<elided_chars>\n2000 characters elided\n</elided_chars>" in long
    timed = mini.observation({"exit_code": -1, "output": "", "truncated": False, "timed_out": True})
    assert timed == "<exception>The command timed out; its process group was killed.</exception>\n" \
                    "<returncode>-1</returncode>\n<output>\n</output>"


def test_the_spec_shows_no_tool_and_wraps_the_task():
    assert MINI_SWE_V1.tool_specs(("bash", "edit")) == () and MINI_SWE_V1.tool_specs(("bash",)) == ()
    assert MINI_SWE_V1.system_prompt_for(("bash",)) == mini.SYSTEM
    assert MINI_SWE_V1.user_prompt(TASK).startswith(f"Please solve this issue: {TASK}\n")
    assert digest_refusal(MINI_SWE_V1.id, MINI_SWE_V1.version, MINI_SWE_V1.digest) is None


needs_tokenizer = pytest.mark.skipif(not TOKENIZERS, reason="set RELIQUARY_QWEN38_TOKENIZER")


def _renderer(directory, tools=("bash", "edit")):
    from reliquary_harness.qwen import load_turn_renderer

    return load_turn_renderer(directory, tools, spec=MINI_SWE_V1)


def _turn(r, text):
    """`text` as a completion after a prompt-opened <think> (Qwen3.5's generation prompt)."""
    return [int(t) for t in r._tokenizer.encode(text + "<|im_end|>", add_special_tokens=False)]


TURNS = [
    "I should look around first.\n</think>\n\nTHOUGHT: list the files.\n\n" + _block("ls -la"),
    "Two at once.\n</think>\n\n" + _block("ls") + "\n" + _block("pwd"),
    "Done.\n</think>\n\nTHOUGHT: submit.\n\n" + _block(mini.SUBMIT),
]


def _golden(directory):
    r = _renderer(directory)
    prompt = r.initial_ids(TASK)
    rows = []
    for text in TURNS:
        completion = _turn(r, text)
        calls = r.tool_calls(completion)
        row = {"completion_ids": completion, "tool_calls": [list(c) for c in calls],
               "assistant_message": r.assistant_message(completion),
               "canonical": r.span_is_canonical(prompt, completion)}
        if calls:
            name, arguments = calls[0]
            text_seen = (r.refusal_observation(name, arguments) if name.startswith("?")
                         else r.observation({"exit_code": 0, "output": "a.py\n", "truncated": False,
                                             "timed_out": False}, "a.py\n"))
            prompt = r.next_prompt(prompt, completion, [text_seen])
            row.update(observation=text_seen, next_prompt_ids=prompt)
        rows.append(row)
    return {"spec": MINI_SWE_V1.key, "initial_ids": r.initial_ids(TASK), "turns": rows}


@needs_tokenizer
def test_a_mini_conversation_renders_and_reads_back():
    r = _renderer(next(iter(TOKENIZERS.values())))
    text = r._tokenizer.decode(r.initial_ids(TASK), skip_special_tokens=False)
    assert "# Tools" not in text and "<tools>" not in text
    assert mini.SYSTEM in text and f"Please solve this issue: {TASK}" in text
    golden = _golden(next(iter(TOKENIZERS.values())))
    first, second, last = golden["turns"]
    assert first["tool_calls"] == [["bash", json.dumps({"command": "ls -la"})]] and first["canonical"]
    assert "tool_calls" not in first["assistant_message"]
    assert second["tool_calls"] == [[mini.FORMAT_ERROR_CALL, '{"found": 2}']] and "found 2 actions" in second["observation"]
    assert last["tool_calls"] == [] and last["canonical"]
    bridged = r._tokenizer.decode(second["next_prompt_ids"][len(first["next_prompt_ids"]):],
                                  skip_special_tokens=False)
    assert "<|im_start|>user\nFormat error:" in bridged and "<tool_response>" not in bridged


@needs_tokenizer
def test_mini_renders_its_frozen_golden():
    for digest, directory in TOKENIZERS.items():
        path = GOLDENS / f"mini-swe.{digest}.json"
        got = json.loads(json.dumps(_golden(directory)))
        if os.environ.get("HARNESS_WRITE_GOLDENS") == "1":
            path.write_text(json.dumps(got, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        assert got == json.loads(path.read_text(encoding="utf-8"))
