import errno
import os

import pytest
from episode_fixtures import IMAGE
from fake_boxes import FakeBoxes
from fake_episode_env import sandbox_task

from reliquary_sandbox_service.episodes import tools as tools_module
from reliquary_sandbox_service.episodes.tools import TOOLS_VERSION, assemble_output, run_tool
from reliquary_sandbox_service.runtime import RuntimeOperationError
from reliquary_sandbox_service.task_runtime import HelperBudgetTimeout


@pytest.fixture
def box():
    boxes = FakeBoxes([IMAGE])
    boxes.create("s-1", "s-1", None)
    return boxes


def call(boxes, tool, arguments, cap=64):
    return run_tool(boxes, "s-1", sandbox_task("train", 0), tool, arguments, timeout_s=5,
                    output_cap=cap, edit_max_bytes=1024)


def test_tools_version_is_pinned():
    assert TOOLS_VERSION == "reliquary-tools/1"


def test_bash_returns_stdout_then_stderr(box):
    outcome = call(box, "bash", {"command": "fail"})
    assert (outcome.output, outcome.exit_code) == ("out\nboom\n", 2)


def test_output_beyond_the_cap_is_cut_and_flagged(box):
    outcome = call(box, "bash", {"command": "big 100"}, cap=10)
    assert outcome.output == "y" * 10 and outcome.truncated


def test_assembled_output_cuts_combined_streams_and_replaces_split_utf8():
    assert assemble_output(b"ab", b"cd", 3, False) == ("abc", True)
    assert assemble_output("é".encode(), b"", 1, False) == ("�", True)
    assert assemble_output(b"ab", b"", 3, False) == ("ab", False)


def test_timeout_is_reported_not_fatal(box):
    outcome = call(box, "bash", {"command": "sleep 60"})
    assert outcome.timed_out and outcome.exit_code == -1
    assert call(box, "bash", {"command": "echo still here"}).output == "still here\n"


@pytest.mark.parametrize("arguments", [{}, {"command": 3}, {"command": "ls", "extra": 1}, {"command": "  "}])
def test_bad_bash_arguments_are_errors_not_executions(box, arguments):
    outcome = call(box, "bash", arguments)
    assert outcome.exit_code == 2 and outcome.output.startswith("error:")


def test_unknown_tool_is_a_signed_error(box):
    outcome = call(box, "browser", {"url": "x"})
    assert outcome.exit_code == 127 and "unknown tool" in outcome.output


def test_edit_replaces_exactly_one_occurrence_relative_to_workdir(box):
    box.write_file("s-1", "/work/a.py", b"x = 1\ny = 2\n")
    assert call(box, "edit", {"path": "a.py", "old_str": "x = 1", "new_str": "x = 3"}).output == "Edited a.py"
    assert box.read_file("s-1", "/work/a.py", 100) == b"x = 3\ny = 2\n"


@pytest.mark.parametrize("content,arguments,message", [
    (None, {"path": "missing.py", "old_str": "a", "new_str": "b"}, "error: missing.py not found"),
    (b"a a", {"path": "a.py", "old_str": "a", "new_str": "b"}, "found 2"),
    (b"abc", {"path": "a.py", "old_str": "", "new_str": "b"}, "non-empty"),
    (b"\xff\xfe", {"path": "a.py", "old_str": "a", "new_str": "b"}, "not UTF-8"),
    (b"x" * 2000, {"path": "a.py", "old_str": "x", "new_str": "b"}, "larger than 1024 bytes"),
])
def test_edit_failures_are_reported(box, content, arguments, message):
    if content is not None:
        box.write_file("s-1", "/work/a.py", content)
    outcome = call(box, "edit", arguments)
    assert outcome.exit_code in (1, 2) and message in outcome.output


@pytest.mark.parametrize("tool,arguments", [
    ("bash", {"command": "\ud800"}),
    ("edit", {"path": "\ud800", "old_str": "a", "new_str": "b"}),
    ("edit", {"path": "a.py", "old_str": "\ud800", "new_str": "b"}),
    ("edit", {"path": "a.py", "old_str": "a", "new_str": "\ud800"}),
])
def test_lone_surrogates_are_bad_arguments(box, tool, arguments):
    outcome = call(box, tool, arguments)
    assert outcome.exit_code == 2 and outcome.output.startswith("error:")


def test_non_object_arguments_are_bad_arguments(box):
    assert call(box, "bash", "ls").exit_code == 2
    assert call(box, "edit", None).exit_code == 2


def test_disabled_tool_is_refused_and_nothing_runs(box):
    from reliquary_sandbox.episode_task import SandboxTask
    task = sandbox_task("train", 0)
    only_bash = SandboxTask(image=task.image, workdir=task.workdir, tools=("bash",),
                            prepare=task.prepare, extract=task.extract, grade=task.grade,
                            publish_state=True)
    outcome = run_tool(box, "s-1", only_bash, "edit", {"path": "a", "old_str": "a", "new_str": "b"},
                       timeout_s=5, output_cap=64, edit_max_bytes=1024)
    assert outcome.exit_code == 127 and "unknown tool" in outcome.output


def test_edit_on_a_non_regular_file_is_an_edit_error(box):
    box.boxes["s-1"]["special"] = {"/work/fifo"}
    outcome = call(box, "edit", {"path": "fifo", "old_str": "a", "new_str": "b"})
    assert outcome.exit_code == 1 and "not a regular file" in outcome.output


@pytest.mark.parametrize("method", ["read_file", "write_file"])
def test_infrastructure_errors_from_read_or_write_propagate(box, monkeypatch, method):
    box.write_file("s-1", "/work/a.py", b"x")

    def fail(*args, **kwargs):
        raise RuntimeOperationError("The box is gone")
    monkeypatch.setattr(box, method, fail)
    with pytest.raises(RuntimeOperationError):
        call(box, "edit", {"path": "a.py", "old_str": "x", "new_str": "y"})


def test_edit_helpers_get_the_remaining_call_time(box):
    box.write_file("s-1", "/work/a.py", b"x")
    box.helper_calls.clear()
    outcome = run_tool(box, "s-1", sandbox_task("train", 0), "edit",
                       {"path": "a.py", "old_str": "x", "new_str": "y"}, timeout_s=2.5,
                       output_cap=64, edit_max_bytes=1024)
    assert outcome.exit_code == 0
    names = [name for name, _, _ in box.helper_calls]
    assert names == ["file_kind", "read_file", "write_file"]
    assert all(0 < kwargs["timeout_s"] <= 2.5 for _, _, kwargs in box.helper_calls)


def test_an_edit_that_runs_out_of_time_is_a_timed_out_call(box, monkeypatch):
    box.write_file("s-1", "/work/a.py", b"x")
    clock = iter([100.0, 100.0, 103.0, 103.0, 103.0])
    monkeypatch.setattr(tools_module.time, "monotonic", lambda: next(clock))
    outcome = run_tool(box, "s-1", sandbox_task("train", 0), "edit",
                       {"path": "a.py", "old_str": "x", "new_str": "y"}, timeout_s=2.5,
                       output_cap=64, edit_max_bytes=1024)
    assert outcome.timed_out and outcome.exit_code == -1
    assert box.boxes["s-1"]["files"]["/work/a.py"] == b"x"


def test_a_helper_fault_from_the_type_check_propagates(box, monkeypatch):
    box.write_file("s-1", "/work/a.py", b"x")
    sentinel = RuntimeOperationError("classified")

    def fail(*args, **kwargs):
        raise sentinel
    monkeypatch.setattr(box, "file_kind", fail)
    with pytest.raises(RuntimeOperationError) as caught:
        call(box, "edit", {"path": "a.py", "old_str": "x", "new_str": "y"})
    assert caught.value is sentinel


@pytest.mark.parametrize("method,code", [("write_file", errno.ENOSPC), ("read_file", errno.EIO)])
def test_os_errors_in_edit_are_signed_tool_errors_never_exceptions(box, monkeypatch, method, code):
    box.write_file("s-1", "/work/a.py", b"x")

    def fail(*args, **kwargs):
        raise OSError(code, os.strerror(code), "/work/a.py")
    monkeypatch.setattr(box, method, fail)
    outcome = call(box, "edit", {"path": "a.py", "old_str": "x", "new_str": "y"})
    assert outcome.exit_code == 1 and os.strerror(code) in outcome.output


def edit_with(box, timeout_s):
    return run_tool(box, "s-1", sandbox_task("train", 0), "edit",
                    {"path": "a.py", "old_str": "x", "new_str": "y"}, timeout_s=timeout_s,
                    output_cap=64, edit_max_bytes=1024)


def test_an_edit_with_under_two_seconds_left_times_out_without_running_a_helper(box):
    box.write_file("s-1", "/work/a.py", b"x")
    box.helper_calls.clear()
    outcome = edit_with(box, 0.5)
    assert outcome.timed_out and outcome.exit_code == -1
    assert box.helper_calls == [] and box.boxes["s-1"]["files"]["/work/a.py"] == b"x"


@pytest.mark.parametrize("method", ["file_kind", "read_file", "write_file"])
def test_a_helper_cut_by_the_call_budget_is_a_timed_out_edit(box, monkeypatch, method):
    box.write_file("s-1", "/work/a.py", b"x")

    def out_of_time(*args, **kwargs):
        raise HelperBudgetTimeout("ran out of the caller's time")

    monkeypatch.setattr(box, method, out_of_time)
    outcome = edit_with(box, 5)
    assert outcome.timed_out and outcome.exit_code == -1
