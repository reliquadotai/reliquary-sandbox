"""`QwenTurnRenderer` over `BASH_VF_B2E4E81` renders the goldens Catalyst froze from the
code this package replaced (feat/rl-async a5df544f, `goldens/`), id for id: initial
prompts, parsed calls and messages, canonical and reasoning checks, observations read
back, bridged prompts, export renderings, and real models' turns.

A golden whose tokenizer is not on this host is skipped by name: set
RELIQUARY_QWEN38_TOKENIZER and/or RELIQUARY_HARNESS_TOKENIZERS (paths joined with
os.pathsep)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("renderers")

from reliquary_harness.qwen import load_turn_renderer  # noqa: E402
from reliquary_harness.spec import BASH_VF_B2E4E81  # noqa: E402

GOLDENS = Path(__file__).resolve().parent / "goldens"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
CASES = sorted(p for p in GOLDENS.glob("*.*.json") if not p.name.startswith(("miner-loop.", "surface-variants.", "mini-swe.")))


def tokenizer_digest(directory) -> str:
    digest = hashlib.sha256()
    for name in TOKENIZER_FILES:
        data = (Path(directory) / name).read_bytes()
        digest.update(name.encode() + b"\0" + hashlib.sha256(data).digest())
    return digest.hexdigest()[:16]


def tokenizers() -> dict[str, str]:
    paths = [os.environ.get("RELIQUARY_QWEN38_TOKENIZER", "")]
    paths += os.environ.get("RELIQUARY_HARNESS_TOKENIZERS", "").split(os.pathsep)
    return {tokenizer_digest(p): p for p in filter(None, paths)
            if all((Path(p) / name).is_file() for name in TOKENIZER_FILES)}


TOKENIZERS = tokenizers()


def _renderer(path: Path, tools):
    tokenizer = path.name.split(".")[1]
    if tokenizer not in TOKENIZERS:
        pytest.skip(f"tokenizer {tokenizer} is not on this host")
    return load_turn_renderer(TOKENIZERS[tokenizer], tools=tuple(tools))


def _ids(renderer, text):
    return [int(t) for t in renderer._tokenizer.encode(text, add_special_tokens=False)]


def _read_back(renderer, prompt, completion):
    return {"tool_calls": [list(call) for call in renderer.tool_calls(completion)],
            "assistant_message": renderer.assistant_message(completion),
            "canonical": renderer.span_is_canonical(prompt, completion),
            "reasoning_unclosed": renderer.reasoning_unclosed(prompt, completion)}


@pytest.mark.parametrize("path", [p for p in CASES if not p.name.startswith("real-turns.")], ids=lambda p: p.name)
def test_a_conversation_renders_its_golden(path):
    golden = json.loads(path.read_text(encoding="utf-8"))
    r = _renderer(path, golden["tools"])
    assert r.system_prompt == golden["system_prompt"] == BASH_VF_B2E4E81.system_prompt_for(golden["tools"])
    assert r.tools == golden["tool_specs"]
    assert (sorted(r.stop_ids), r.terminator_id, sorted(r.turn_markup_ids)) == \
        (golden["stop_ids"], golden["terminator_id"], golden["turn_markup_ids"])
    prompt = r.initial_ids(golden["prompt"])
    assert prompt == golden["initial_ids"]
    for turn in golden["turns"]:
        completion = _ids(r, turn["text"])
        assert completion == turn["completion_ids"]
        assert _read_back(r, prompt, completion) == {key: turn[key] for key in
                                                      ("tool_calls", "assistant_message", "canonical",
                                                       "reasoning_unclosed")}
        if "observations" in turn:
            following = r.next_prompt(prompt, completion, turn["observations"])
            assert following == turn["next_prompt_ids"]
            assert r.observations(following[len(prompt) + len(completion):]) == turn["observations_read_back"]
            prompt = following
    for row in golden["parse_only"]:
        completion = _ids(r, row["text"])
        assert [list(c) for c in r.tool_calls(completion)] == row["tool_calls"]
        assert r.assistant_message(completion) == row["assistant_message"]


def test_the_export_rendering_of_each_conversation_is_its_golden():
    for path in [p for p in CASES if not p.name.startswith("real-turns.")]:
        golden = json.loads(path.read_text(encoding="utf-8"))
        if path.name.split(".")[1] not in TOKENIZERS:
            continue
        r = _renderer(path, golden["tools"])
        messages = [{"role": "system", "content": golden["system_prompt"]},
                    {"role": "user", "content": golden["prompt"]}]
        for turn in golden["turns"]:
            messages.append(turn["assistant_message"])
            messages += r.tool_messages(turn.get("observations", ()))
        assert r.render_messages(messages) == golden["export_ids"], path.name


@pytest.mark.parametrize("path", [p for p in CASES if p.name.startswith("real-turns.")], ids=lambda p: p.name)
def test_real_turns_read_back_as_frozen(path):
    golden = json.loads(path.read_text(encoding="utf-8"))
    texts = json.loads((GOLDENS / "real_turns.json").read_text(encoding="utf-8"))
    r = _renderer(path, ("bash", "edit"))
    # The prompt is swe-signed-bash-edit's (its initial ids are checked there).
    for text, row in zip(texts, golden["turns"], strict=True):
        completion = _ids(r, text)
        assert completion == row["completion_ids"]
        assert _read_back(r, golden["prompt_ids"], completion) == {key: row[key] for key in row
                                                                  if key != "completion_ids"}
