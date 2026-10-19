"""Surface variants: what the model is shown changes, what the gateway runs does not.

`goldens/surface-variants.<tokenizer>.json` freezes every variant's initial prompt and one
round trip. HARNESS_WRITE_GOLDENS=1 writes it (only when a variant is added on purpose)."""

from __future__ import annotations

import dataclasses
import json
import os

import pytest

pytest.importorskip("renderers")

from reliquary_harness.qwen import HIDDEN_PREFIX, load_turn_renderer  # noqa: E402
from reliquary_harness.spec import (  # noqa: E402
    BASH_SURFACE_V1,
    BASH_VF_B2E4E81,
    decorate_observation,
)
from reliquary_harness.variant import variant_count  # noqa: E402
from test_qwen_goldens import GOLDENS, TOKENIZERS  # noqa: E402

needs_tokenizer = pytest.mark.skipif(not TOKENIZERS, reason="set RELIQUARY_QWEN38_TOKENIZER")
PROMPT = "Fix the failing test in tests/test_dates.py."
TOOL_SETS = (("bash",), ("bash", "edit"))


def _directory():
    return next(iter(TOKENIZERS.values()))


def _call_turn(renderer, calls):
    """The span a model writes for `calls` (surface names), as the template renders it."""
    message = {"role": "assistant", "content": "",
               "reasoning_content": "Look first.",
               "tool_calls": [{"id": f"call_{i}", "type": "function",
                               "function": {"name": name, "arguments": json.dumps(args)}}
                              for i, (name, args) in enumerate(calls)]}
    rendered = renderer.render_messages([{"role": "user", "content": "x"}, message])
    tail = renderer._tail(rendered)
    tail = tail[:max(k for k, t in enumerate(tail) if t == renderer.terminator_id) + 1]
    prompt_tail = renderer._tail(renderer.initial_ids(PROMPT))
    assert tail[:len(prompt_tail)] == prompt_tail
    return tail[len(prompt_tail):]


def test_the_surface_spec_has_four_variants_and_variant_0_is_the_compatibility_surface():
    assert variant_count(BASH_SURFACE_V1) == 4 and variant_count(BASH_VF_B2E4E81) == 1
    for tools in TOOL_SETS:
        assert BASH_SURFACE_V1.system_prompt_for(tools, {"system_prompt": BASH_SURFACE_V1.variants["axes"][1]
                                                         ["values"][0]}) == BASH_VF_B2E4E81.system_prompt_for(tools)
    assert BASH_SURFACE_V1.digest != BASH_VF_B2E4E81.digest


def test_an_exit_code_observation_puts_the_signed_code_first():
    body = {"exit_code": 2, "output": "boom"}
    assert decorate_observation("plain", body, "boom") == "boom"
    assert decorate_observation("exit-code", body, "boom") == "[exit code: 2]\nboom"
    with pytest.raises(ValueError):
        decorate_observation("exit-code", {"exit_code": True}, "x")
    with pytest.raises(ValueError):
        decorate_observation("json", body, "x")


@needs_tokenizer
def test_variant_0_renders_the_compatibility_harness_token_for_token():
    for tools in TOOL_SETS:
        vf = load_turn_renderer(_directory(), tools, spec=BASH_VF_B2E4E81)
        surface = load_turn_renderer(_directory(), tools, spec=BASH_SURFACE_V1, variant=0)
        assert surface.initial_ids(PROMPT) == vf.initial_ids(PROMPT)
        first = vf.initial_ids(PROMPT)
        span = _call_turn(vf, [("bash", {"command": "ls"})])
        assert surface.next_prompt(first, span, ["a.py"]) == vf.next_prompt(first, span, ["a.py"])


@needs_tokenizer
@pytest.mark.parametrize("index", range(4))
def test_every_variant_reads_its_surface_calls_back_as_canonical_calls(index):
    r = load_turn_renderer(_directory(), ("bash", "edit"), spec=BASH_SURFACE_V1, variant=index)
    names = BASH_SURFACE_V1.surface_names(r.variant.values)
    assert [tool["function"]["name"] for tool in r.tools] == [names["bash"], names["edit"]]
    assert names["bash"] in r.system_prompt and names["edit"] in r.system_prompt
    prompt = r.initial_ids(PROMPT)
    span = _call_turn(r, [(names["edit"], {"path": "a.py", "old_str": "x", "new_str": "y"}),
                          (names["bash"], {"command": "pytest -q"})])
    assert r.tool_calls(span) == [("edit", json.dumps({"path": "a.py", "old_str": "x", "new_str": "y"})),
                                  ("bash", json.dumps({"command": "pytest -q"}))]
    assert [c["function"]["name"] for c in r.assistant_message(span)["tool_calls"]] == [names["edit"], names["bash"]]
    assert r.span_is_canonical(prompt, span)


@needs_tokenizer
def test_a_canonical_name_a_variant_hides_is_no_offered_tool():
    r = load_turn_renderer(_directory(), ("bash", "edit"), spec=BASH_SURFACE_V1, variant=2)
    assert BASH_SURFACE_V1.surface_names(r.variant.values)["bash"] == "shell"
    span = _call_turn(r, [("bash", {"command": "ls"}), ("python", {"code": "1"})])
    assert [name for name, _ in r.tool_calls(span)] == [HIDDEN_PREFIX + "bash", "python"]


@needs_tokenizer
def test_an_observation_axis_decorates_what_the_model_reads():
    axes = {"axes": [{"name": "observation", "values": ["plain", "exit-code"]}]}
    spec = dataclasses.replace(BASH_SURFACE_V1, version="test", variants=axes)
    plain = load_turn_renderer(_directory(), ("bash",), spec=spec, variant=0)
    coded = load_turn_renderer(_directory(), ("bash",), spec=spec, variant=1)
    body = {"exit_code": 1, "output": "no"}
    assert plain.observation(body, "no") == "no"
    assert coded.observation(body, "no") == "[exit code: 1]\nno"
    assert plain.initial_ids(PROMPT) == coded.initial_ids(PROMPT)


def _variants_golden(directory) -> dict:
    rows = {}
    for tools in TOOL_SETS:
        for index in range(variant_count(BASH_SURFACE_V1)):
            r = load_turn_renderer(directory, tools, spec=BASH_SURFACE_V1, variant=index)
            names = BASH_SURFACE_V1.surface_names(r.variant.values)
            prompt = r.initial_ids(PROMPT)
            span = _call_turn(r, [(names["bash"], {"command": "ls"})])
            rows[f"{'+'.join(tools)}/{index}"] = {
                "values": r.variant.values, "system_prompt": r.system_prompt, "tools": r.tools,
                "initial_ids": prompt, "span": span, "tool_calls": [list(c) for c in r.tool_calls(span)],
                "next_prompt_ids": r.next_prompt(prompt, span, ["a.py\nb.py"])}
    return {"spec": BASH_SURFACE_V1.key, "rows": rows}


@needs_tokenizer
def test_every_variant_renders_its_frozen_golden():
    for digest, directory in TOKENIZERS.items():
        path = GOLDENS / f"surface-variants.{digest}.json"
        got = json.loads(json.dumps(_variants_golden(directory)))
        if os.environ.get("HARNESS_WRITE_GOLDENS") == "1":
            path.write_text(json.dumps(got, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        frozen = json.loads(path.read_text(encoding="utf-8"))
        assert got == frozen
        initial = {key: tuple(row["initial_ids"]) for key, row in got["rows"].items()}
        for tools in TOOL_SETS:
            prefix = "+".join(tools) + "/"
            assert len({ids for key, ids in initial.items() if key.startswith(prefix)}) == 4
