"""`BASH_VF_B2E4E81` against verifiers b2e4e81, the code it reproduces. Never skipped.

The upstream files are vendored under `upstream/verifiers-b2e4e81/` (MIT, see its
LICENSE) and pinned by sha256 in the spec; the spec's constants are read out of them
here. When verifiers is installed (every miner, the sandbox bridge), the installed files
must be the pinned ones too: a verifiers that moved is a new harness, not this one."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path

import pytest
from reliquary_harness.spec import BASH_SENTENCE, CAPABILITY_NOTICE, EDIT_SENTENCE
from reliquary_harness.spec import BASH_VF_B2E4E81 as SPEC

UPSTREAM = Path(__file__).resolve().parent / "upstream" / "verifiers-b2e4e81"


def _source(relative: str) -> str:
    return (UPSTREAM / relative).read_text(encoding="utf-8")


def _constant(relative: str, name: str):
    for node in ast.parse(_source(relative)).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not a literal of {relative}")


def _function(relative: str, name: str) -> str:
    source = _source(relative)
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"no function {name} in {relative}")


@pytest.mark.parametrize("relative", sorted(SPEC.upstream["files"]))
def test_the_vendored_files_are_the_pinned_ones(relative):
    data = (UPSTREAM / relative).read_bytes()
    assert hashlib.sha256(data).hexdigest() == SPEC.upstream["files"][relative]


def test_the_installed_verifiers_is_the_pinned_one_when_there_is_one():
    spec = importlib.util.find_spec("verifiers")
    if spec is None:
        return                      # nothing installed to drift; the vendored files hold
    root = Path(spec.origin).resolve().parent
    for relative, sha in SPEC.upstream["files"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == sha, relative


def test_the_system_prompt_is_the_bash_harness_fragments_joined_by_a_space():
    assert BASH_SENTENCE == _constant("v1/harnesses/bash/harness.py", "BASH_SYSTEM_PROMPT")
    assert EDIT_SENTENCE == _constant("v1/harnesses/bash/harness.py", "EDIT_SYSTEM_PROMPT")
    launch = _function("v1/harnesses/bash/harness.py", "launch")
    assert '" ".join(fragments)' in launch and "fragments.append(EDIT_SYSTEM_PROMPT)" in launch
    assert SPEC.system_prompt_for(("bash",)) == BASH_SENTENCE
    assert SPEC.system_prompt_for(("bash", "edit")) == BASH_SENTENCE + " " + EDIT_SENTENCE


def test_the_tools_are_the_programs_in_its_order():
    assert SPEC.tools == (_constant("v1/harnesses/bash/program.py", "BASH_TOOL"),
                          _constant("v1/harnesses/bash/program.py", "EDIT_TOOL"))
    main = _function("v1/harnesses/bash/program.py", "main")
    assert "tools = [BASH_TOOL]" in main and "tools.append(EDIT_TOOL)" in main
    # The loop ends on a turn without a call; each call is answered in order.
    assert "if not message.tool_calls:\n            break" in main
    assert "for call in message.tool_calls:" in main


def test_the_network_notice_is_the_dialects():
    assert CAPABILITY_NOTICE == _constant("v1/dialects/base.py", "CAPABILITY_NOTICE")
    assert 'f"{content}\\n\\n{CAPABILITY_NOTICE}" if content else CAPABILITY_NOTICE' in \
        _function("v1/dialects/base.py", "append_user_notice")
    assert SPEC.with_network_notice("x") == "x\n\n" + CAPABILITY_NOTICE
    assert SPEC.with_network_notice("") == CAPABILITY_NOTICE


def test_the_call_filter_is_response_from_generates():
    source = _function("v1/clients/train.py", "response_from_generate")
    for piece in ('id=tc.id or f"call_{i}"', "if isinstance(tc.arguments, str)",
                  "else json.dumps(tc.arguments or {})", 'if getattr(tc, "name", None)',
                  "and getattr(tc, \"status\", None) != ToolCallParseStatus.UNKNOWN_TOOL"):
        assert piece in source, piece


def test_the_turn_request_is_the_train_clients():
    source = _function("v1/clients/train.py", "get_response")
    for piece in ("renderer.bridge_to_next_turn(", "previous_prompt_ids,", "previous_completion_ids,",
                  'sampling_params["routed_experts_prompt_start"] = max(',
                  "len(previous_prompt_ids) + len(previous_completion_ids) - 1,",
                  "wire_messages, tools=wire_tools, add_generation_prompt=True",
                  "extra_headers={SESSION_ID_HEADER: session_id}"):
        assert piece in source, piece


def test_max_turns_is_checked_before_each_model_call():
    source = _function("v1/session.py", "reached")
    assert 'if self.max_turns is not None and trace.num_turns >= self.max_turns:\n            return "max_turns"' \
        in source
    refused = _function("v1/session.py", "refused")
    assert refused.index("self.limits.reached") < refused.index("self.trace_stops")
