"""What the model is shown for a signed call record, and which calls are never sent."""

import ast
import json
from pathlib import Path

import pytest

from reliquary_sandbox import observation as ob
from reliquary_sandbox.attest import canonical_json


def body(output="", tool="bash", truncated=False, timed_out=False, exit_code=0):
    return {"tool": tool, "output": output, "truncated": truncated, "timed_out": timed_out,
            "exit_code": exit_code}


def test_plain_output_is_shown_as_is():
    assert ob.render_observation(body("a\nb\n")) == "a\nb\n"


def test_an_empty_output_is_never_an_empty_observation():
    # An empty tool message would make verifiers' hook answer "allow" and run the command
    # on the miner's host (see the module docstring).
    assert ob.render_observation(body("")) == ob.NO_OUTPUT
    assert ob.render_observation(body("", tool="edit")) == ob.NO_OUTPUT
    assert ob.NO_OUTPUT


@pytest.mark.parametrize("output", [" ", "\n", " \n", "\t\r\n", "\n\n"])
def test_whitespace_only_output_renders_as_no_output(output):
    # Qwen's chat template strips tool content: "\n" would reach the model as an empty
    # tool response.
    assert ob.render_observation(body(output)) == ob.NO_OUTPUT
    assert ob.render_observation(body(output, tool="edit")) == ob.NO_OUTPUT


def test_output_with_text_keeps_its_surrounding_whitespace():
    assert ob.render_observation(body("  x \n\n")) == "  x \n\n"


def test_truncation_and_timeout_notes_follow_the_output_on_their_own_line():
    assert ob.render_observation(body("partial", truncated=True)) == "partial\n" + ob.TRUNCATED_NOTE
    assert ob.render_observation(body("done\n", timed_out=True)) == "done\n" + ob.TIMED_OUT_NOTE
    assert (ob.render_observation(body("x", truncated=True, timed_out=True))
            == "x\n" + ob.TRUNCATED_NOTE + "\n" + ob.TIMED_OUT_NOTE)


@pytest.mark.parametrize("output", ["", " \n"])
def test_an_empty_output_with_notes_is_no_output_then_the_notes(output):
    # The whitespace rule applies to the output part, before the notes are added.
    assert (ob.render_observation(body(output, truncated=True, timed_out=True))
            == ob.NO_OUTPUT + "\n" + ob.TRUNCATED_NOTE + "\n" + ob.TIMED_OUT_NOTE)
    assert ob.render_observation(body(output, timed_out=True)) == ob.NO_OUTPUT + "\n" + ob.TIMED_OUT_NOTE


def test_an_output_cut_inside_a_character_renders_its_replacement_then_the_note():
    from reliquary_sandbox_service.episodes.tools import assemble_output

    output, truncated = assemble_output("ab\u00e9".encode(), b"", 3, False)
    assert (output, truncated) == ("ab\ufffd", True)
    rendered = ob.render_observation(body(output, truncated=truncated))
    assert rendered == "ab\ufffd\n" + ob.TRUNCATED_NOTE


def test_edit_flags_add_no_notes_and_exit_codes_are_not_shown():
    text = "error: the edit ran out of time"
    assert ob.render_observation(body(text, tool="edit", timed_out=True)) == text
    assert ob.render_observation(body("x\n", exit_code=2)) == "x\n"


def test_scrubbed_runtime_text_is_shown_as_recorded():
    # The gateway replaces container-runtime lines before signing; the renderer shows the
    # signed text unchanged, so miner and validator agree on it.
    scrubbed = "a\n[runtime output removed]\nb"
    assert ob.render_observation(body(scrubbed)) == scrubbed
    assert (ob.render_observation(body(scrubbed, truncated=True))
            == scrubbed + "\n" + ob.TRUNCATED_NOTE)


@pytest.mark.parametrize("bad", [
    dict(output=None), dict(output=b"x"), dict(truncated="false"), dict(timed_out=0),
    dict(tool=None),
])
def test_a_body_that_is_not_a_call_record_is_refused(bad):
    with pytest.raises(ValueError):
        ob.render_observation({**body("x"), **bad})


@pytest.mark.parametrize("text, reason", [
    ("{", "invalid_json"),
    ("[1]", "not_an_object"),
    ('"x"', "not_an_object"),
    ('{"command": "\\ud800"}', "not_canonical"),
    ('{"n": NaN}', "not_canonical"),
    ('{"n": 9223372036854775808}', "not_canonical"),
])
def test_arguments_without_a_canonical_object_are_answered_locally(text, reason):
    plan = ob.plan_call("bash", text)
    assert isinstance(plan, ob.Refuse) and plan.reason == reason and plan.observation


def test_arguments_nested_past_the_canonical_depth_are_answered_locally():
    text = '{"a": ' + "[" * 70 + "]" * 70 + "}"
    assert ob.plan_call("bash", text) == ob.Refuse("not_canonical", ob.NOT_CANONICAL)
    assert "nest" in ob.NOT_CANONICAL


@pytest.mark.parametrize("arguments", [b"{}", {}, 1, ["{}"]])
def test_arguments_that_are_not_text_are_answered_locally(arguments):
    assert ob.plan_call("bash", arguments) == ob.Refuse("not_text", ob.NOT_TEXT)


def test_no_argument_text_is_an_empty_object():
    assert ob.plan_call("bash", None) == ob.Send("bash", {})


@pytest.mark.parametrize("tool", ["", "x" * 65, "\ud800"])
def test_a_tool_name_the_gateway_would_refuse_is_answered_locally(tool):
    assert ob.plan_call(tool, "{}") == ob.Refuse("bad_tool_name", ob.BAD_TOOL_NAME)


def test_empty_argument_text_is_an_empty_object_as_in_the_bash_harness():
    assert ob.plan_call("bash", "") == ob.Send("bash", {})


def test_arguments_are_forwarded_exactly_as_parsed():
    text = '{"path": "a.py", "old_str": "    x = 1\\r\\n", "new_str": "    y = \\u00e9\\r\\n"}'
    assert ob.plan_call("edit", text) == ob.Send(
        "edit", {"path": "a.py", "old_str": "    x = 1\r\n", "new_str": "    y = é\r\n"})


def test_the_argument_size_limit_is_inclusive():
    # canonical {"command":"<s>"} is len(s) + 14 bytes
    at_limit = json.dumps({"command": "x" * (ob.MAX_ARGUMENT_BYTES - 14)})
    assert len(canonical_json(json.loads(at_limit))) == ob.MAX_ARGUMENT_BYTES
    assert isinstance(ob.plan_call("bash", at_limit), ob.Send)
    over = json.dumps({"command": "x" * (ob.MAX_ARGUMENT_BYTES - 13)})
    assert ob.plan_call("bash", over) == ob.Refuse("too_large", ob.TOO_LARGE)


def test_refused_calls_take_no_position_in_the_turn():
    plans = [ob.Send("bash", {}), ob.Refuse("invalid_json", ob.INVALID_JSON), ob.Send("edit", {})]
    assert ob.sent_positions(plans) == [0, None, 1]


def test_the_tools_version_is_checked():
    ob.require_tools_version("reliquary-tools/1")
    with pytest.raises(ob.UnsupportedToolsVersion):
        ob.require_tools_version("reliquary-tools/2")


def test_the_gateway_agrees_with_the_renderer_on_its_limits_and_version():
    from reliquary_sandbox_service.episodes import manager, routes
    from reliquary_sandbox_service.episodes.tools import TOOLS_VERSION
    from reliquary_sandbox_service.settings import Settings

    assert TOOLS_VERSION in ob.TOOLS_VERSIONS
    # One constant, not a per-machine knob: the gateway imports the renderer's.
    assert "episode_max_argument_bytes" not in Settings.model_fields
    assert manager.MAX_ARGUMENT_BYTES is ob.MAX_ARGUMENT_BYTES
    assert routes.MAX_ARGUMENT_BYTES is ob.MAX_ARGUMENT_BYTES


def test_the_module_imports_only_the_standard_library_and_attest():
    tree = ast.parse(Path(ob.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
    assert imported <= {"__future__", "json", "collections.abc", "dataclasses", "typing",
                        ".attest.canonical"}


def test_a_tool_outside_the_offered_set_is_answered_locally():
    refused = ob.plan_call("edit", '{"path": "x"}', offered=("bash",))
    assert refused == ob.Refuse("not_offered", ob.NOT_OFFERED)
    assert ob.NOT_OFFERED.strip()
    assert ob.plan_call("bash", '{"command": "ls"}', offered=("bash",)) == ob.Send(
        "bash", {"command": "ls"})
    # Without an offered set, every well-formed name is sent (the gateway records it).
    assert isinstance(ob.plan_call("edit", "{}"), ob.Send)


def test_the_offered_check_follows_the_name_check_and_precedes_the_arguments():
    assert ob.plan_call("", "{}", offered=("bash",)).reason == "bad_tool_name"
    assert ob.plan_call("edit", "{bad", offered=("bash",)).reason == "not_offered"
    assert ob.plan_call("bash", "{bad", offered=("bash",)).reason == "invalid_json"
