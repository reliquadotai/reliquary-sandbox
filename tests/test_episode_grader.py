import asyncio
import dataclasses
import hashlib
import logging
import time
from types import SimpleNamespace

import fake_episode_env
import pytest
from episode_fixtures import IMAGE
from fake_boxes import FakeBoxes
from fake_episode_env import sandbox_task

from reliquary_sandbox.episode_task import GradeResult
from reliquary_sandbox_service.episodes.grader import Grader, run_step
from reliquary_sandbox_service.episodes.local_runtime import (
    FileTooLarge,
    LocalRuntime,
    OutputOverflow,
    StateUnreadable,
    StepTimeout,
    UnsupportedRuntimeMember,
)
from reliquary_sandbox_service.settings import Settings
from reliquary_sandbox_service.task_runtime import BoxFault, BoxSpec, InfraFault

SETTINGS = SimpleNamespace(episode_grader_output_bytes=1024, episode_grading_attempts=2)
SPEC = BoxSpec(image=IMAGE, workdir="/work", env={}, user=None, memory_bytes=1, nano_cpus=1,
               pids=16, disk_bytes=1, lifetime_s=60)
G0, G1 = Grader.pristine_box_id("s-1", 0), Grader.pristine_box_id("s-1", 1)


@pytest.fixture
def boxes():
    boxes = FakeBoxes([IMAGE])
    boxes.create("s-1", "s-1", SPEC)
    return boxes


def task_with(**changes):
    return dataclasses.replace(sandbox_task("train", 0), **changes)


def grade(boxes, task=None, settings=SETTINGS, grader=None):
    grader = grader or Grader(settings, boxes)
    return grader.grade("s-1", "s-1", task or sandbox_task("train", 0), SPEC, env="fake-env")


def fault(label):
    return {"_grader": {"agent_fault": label}}


def make_runtime(boxes, box_id="s-1", *, deadline=10, output_limit=1024, **kwargs):
    return LocalRuntime(boxes, box_id, workdir="/work", env=kwargs.pop("env", {}),
                        user=kwargs.pop("user", "0:0"), deadline=time.monotonic() + deadline,
                        output_limit=output_limit, **kwargs)


# -- the runtime surface ----------------------------------------------------

def test_local_runtime_offers_the_verifiers_runtime_surface(boxes):
    runtime = make_runtime(boxes, env={"A": "1"})

    async def scenario():
        await runtime.prepare_setup()
        await runtime.prepare_execution([])
        await runtime.write("/work/answer.txt", b"42\n")
        assert await runtime.read("answer.txt") == b"42\n"  # relative to the workdir
        with pytest.raises(FileNotFoundError):
            await runtime.read("/work/missing")
        with pytest.raises(OSError):
            await runtime.read("/work/answer.txt", max_bytes=1)
        assert (await runtime.run(["check"], {"B": "2"})).exit_code == 0
        assert (await runtime.run_program(["check"], {})).exit_code == 0
        assert await runtime.alive()

    asyncio.run(scenario())
    assert runtime.config.type == runtime.type == "reliquary-sandbox"
    assert runtime.config.workdir == "/work" and runtime.scripts_dir == "/tmp/vf-scripts"
    assert runtime.box_id == runtime.name == "s-1"
    assert runtime.process_env({"B": "2"}) == {"A": "1", "B": "2"}
    assert ("s-1", ["check"], "/work", "0:0") in boxes.exec_log


def test_write_creates_missing_parents_through_the_helper(boxes):
    runtime = make_runtime(boxes)
    asyncio.run(runtime.write("deep/er/file.txt", b"x"))
    assert boxes.boxes["s-1"]["files"]["/work/deep/er/file.txt"] == b"x"
    assert [c[0] for c in boxes.helper_calls[-2:]] == ["make_dirs", "write_file"]


def test_read_defaults_to_the_read_limit(boxes):
    boxes.write_file("s-1", "/work/big", b"x" * 2000)
    runtime = make_runtime(boxes, output_limit=1024, read_limit=4096)
    assert asyncio.run(runtime.read("/work/big")) == b"x" * 2000
    small = make_runtime(boxes, output_limit=1024, read_limit=100)
    with pytest.raises(FileTooLarge):
        asyncio.run(small.read("/work/big"))
    assert isinstance(small.faults[-1], FileTooLarge)


def test_helper_calls_get_the_remaining_deadline(boxes):
    runtime = make_runtime(boxes, deadline=5)
    asyncio.run(runtime.write("/work/a", b"1"))
    asyncio.run(runtime.read("/work/a"))
    for name, _, kwargs in boxes.helper_calls:
        assert 0 < kwargs["timeout_s"] <= 5, name


def test_non_regular_files_are_state_unreadable(boxes):
    boxes.boxes["s-1"]["special"] = {"/work/fifo"}
    runtime = make_runtime(boxes)
    with pytest.raises(StateUnreadable):
        asyncio.run(runtime.read("/work/fifo"))


def test_local_runtime_refuses_cut_output_and_late_calls(boxes):
    runtime = make_runtime(boxes, user=None, output_limit=8)
    with pytest.raises(OutputOverflow):
        asyncio.run(runtime.run(["flood"], {}))
    assert isinstance(runtime.faults[-1], OutputOverflow)
    late = make_runtime(boxes, user=None, deadline=-1, output_limit=8)
    with pytest.raises(TimeoutError):
        asyncio.run(late.run(["true"], {}))


def test_a_run_timeout_with_the_daemon_unreachable_is_ours(boxes):
    runtime = make_runtime(boxes)
    with pytest.raises(TimeoutError):
        asyncio.run(runtime.run(["slow"], {}))
    boxes.reachable = False
    with pytest.raises(InfraFault):
        asyncio.run(runtime.run(["slow"], {}))


def test_local_runtime_records_box_faults_it_raises(boxes):
    runtime = make_runtime(boxes, output_limit=8)
    boxes.dead.add("s-1")
    with pytest.raises(BoxFault):
        asyncio.run(runtime.read("/work/answer.txt"))
    assert isinstance(runtime.faults[-1], BoxFault)
    assert not asyncio.run(runtime.alive())


def test_unsupported_members_raise_not_implemented(boxes):
    runtime = make_runtime(boxes)
    for call in (runtime.run_uv_script("x"), runtime.prepare_uv_script("x"),
                 runtime.open_process(["x"], {}), runtime.run_background(["x"], {}, "log")):
        with pytest.raises(NotImplementedError):
            asyncio.run(call)
    assert all(isinstance(f, UnsupportedRuntimeMember) for f in runtime.faults)


def test_run_step_refuses_a_running_event_loop():
    async def inside():
        with pytest.raises(RuntimeError):
            run_step(lambda: asyncio.sleep(0), 1)

    asyncio.run(inside())


# -- grading -----------------------------------------------------------------

def test_grading_happens_in_a_pristine_box_with_only_the_state(boxes):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")
    boxes.write_file("s-1", "/work/scratch", b"agent junk")
    outcome = grade(boxes)
    assert outcome.status == "graded" and outcome.reward == 1.0
    assert outcome.facts == {"check_exit": 0} and outcome.cleanup == ()
    assert outcome.state == b"42\n" and outcome.state_sha256 == hashlib.sha256(b"42\n").hexdigest()
    assert G0 in boxes.removed and G0 not in boxes.boxes
    assert "s-1" in boxes.boxes  # the agent box is the caller's to remove
    assert boxes.specs[G0].image == IMAGE


def test_pristine_box_ids_are_short_hashes_of_the_episode():
    long_id = "e" * 200
    assert Grader.pristine_box_id(long_id, 1) == (
        "g-" + hashlib.sha256(long_id.encode()).hexdigest()[:24] + "-1")


def test_wrong_or_missing_answer_grades_zero(boxes):
    assert grade(boxes).reward == 0.0  # nothing written: the env extracts an empty state
    boxes.write_file("s-1", "/work/answer.txt", b"41\n")
    outcome = grade(boxes)
    assert outcome.status == "graded" and outcome.reward == 0.0


def test_pristine_box_uses_the_grading_workdir_and_resolved_grading_user(boxes):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")
    outcome = grade(boxes, task_with(user="agent", grading_workdir="/", grading_user="root"))
    assert outcome.reward == 1.0
    spec = boxes.specs[G0]
    assert (spec.workdir, spec.user) == ("/", "root")
    assert [(w, u) for b, argv, w, u in boxes.exec_log if b == G0] == [("/", "0:0")]


def test_extract_runs_as_the_agent_boxs_resolved_user(boxes):
    boxes.restore_user("s-1", "1000:1000")
    seen = []

    async def extract(runtime):
        seen.append((runtime.user, runtime.config.workdir))
        return b"42\n"

    grade(boxes, task_with(extract=extract, user="agent", grading_workdir="/"))
    assert seen == [("1000:1000", "/work")]


def test_an_agent_box_the_agent_killed_is_box_failed_not_aborted(boxes):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")
    boxes.dead.add("s-1")  # e.g. `kill -9 -1` then finish
    outcome = grade(boxes)
    assert outcome.status == "box_failed" and outcome.reward is None
    assert G0 not in boxes.specs


def test_an_agent_box_gone_under_us_is_aborted(boxes):
    boxes.broken.add("s-1")
    assert grade(boxes).status == "aborted"


def test_oversized_state_is_graded_zero_as_the_agents_fault(boxes):
    boxes.write_file("s-1", "/work/answer.txt", b"4" * 2000)  # the env returns its sentinel
    outcome = grade(boxes)
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"refused": "state_too_large"}
    boxes.specs.clear()

    async def huge(runtime):
        return b"x" * 33

    outcome = grade(boxes, task_with(extract=huge, max_state_bytes=32))
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == fault("state_too_large")
    assert outcome.state is None and outcome.state_sha256 is None
    assert G0 not in boxes.specs


def test_extract_output_beyond_the_grader_bound_is_state_too_large(boxes):
    async def flood(runtime):
        return (await runtime.run(["flood"], {})).stdout.encode()

    outcome = grade(boxes, task_with(extract=flood))
    assert (outcome.status, outcome.reward, outcome.facts) == ("graded", 0.0,
                                                               fault("state_too_large"))


def test_extract_of_a_non_regular_file_is_state_unreadable(boxes):
    boxes.boxes["s-1"]["special"] = {"/work/answer.txt"}
    outcome = grade(boxes)
    assert (outcome.status, outcome.facts) == ("graded", fault("state_unreadable"))


def test_extract_past_its_timeout_is_graded_zero(boxes):
    async def slow(runtime):
        await asyncio.sleep(5)

    outcome = grade(boxes, task_with(extract=slow, grading_timeout_s=0.1))
    assert (outcome.status, outcome.reward, outcome.facts) == ("graded", 0.0,
                                                               fault("extract_timeout"))


@pytest.mark.parametrize("hook,label", [("extract", "extract_failed"), ("grade", "grading_failed")])
def test_an_exception_from_the_env_code_is_graded_zero_logged_and_counted(boxes, hook, label,
                                                                          caplog):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")

    async def raises(*args):
        # e.g. verifiers' artifacts.collect on a missing artifact, a harbor hook timeout
        raise RuntimeError("declared artifact '/app' does not exist in the runtime")

    grader = Grader(SETTINGS, boxes)
    with caplog.at_level(logging.ERROR):
        outcome = grade(boxes, task_with(**{hook: raises}), grader=grader)
    assert (outcome.status, outcome.reward, outcome.facts) == ("graded", 0.0, fault(label))
    assert grader.env_failure_counts() == {"fake-env": {label: 1}}
    assert any("fake-env" in r.getMessage() and r.levelno == logging.ERROR for r in caplog.records)
    assert all(box == "s-1" or box in boxes.removed for box in boxes.specs)


def test_an_env_exception_unrelated_to_an_earlier_recorded_fault_is_not_blamed_on_it(boxes):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")

    async def extract(runtime):
        try:
            await runtime.run(["flood"], {})
        except OutputOverflow:
            pass
        raise KeyError("unrelated env bug")

    outcome = grade(boxes, task_with(extract=extract))
    assert outcome.facts == fault("extract_failed")


def test_an_env_exception_chained_from_our_fault_is_attributed_to_it(boxes):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")

    async def extract(runtime):
        try:
            await runtime.run(["flood"], {})
        except OutputOverflow as exc:
            raise RuntimeError("collect failed") from exc

    assert grade(boxes, task_with(extract=extract)).facts == fault("state_too_large")


def test_extract_returning_non_bytes_is_graded_zero(boxes):
    async def text(runtime):
        return "42\n"

    outcome = grade(boxes, task_with(extract=text))
    assert (outcome.status, outcome.facts) == ("graded", fault("extract_failed"))


def test_an_unsupported_runtime_member_is_aborted(boxes):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")

    async def uses_uv(runtime, state):
        await runtime.run_uv_script("print(1)")

    assert grade(boxes, task_with(grade=uses_uv)).status == "aborted"


def test_pristine_box_creation_failure_retries_then_aborts(boxes):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")
    boxes.fail_create = True
    outcome = grade(boxes)
    assert outcome.status == "aborted" and outcome.reward is None


def test_grading_infrastructure_failure_retries_then_aborts(boxes):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")
    calls = []

    async def flaky(runtime, state):
        calls.append(runtime.box_id)
        boxes.broken.add(runtime.box_id)  # the box vanished under us
        await runtime.run(["check"], {})

    outcome = grade(boxes, task_with(grade=flaky))
    assert outcome.status == "aborted"
    assert calls == [G0, G1]
    assert {G0, G1} <= set(boxes.removed)


def test_grading_infrastructure_failure_then_success(boxes):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")

    async def once_flaky(runtime, state):
        if runtime.box_id == G0:
            boxes.broken.add(runtime.box_id)
        return await fake_episode_env.grade(runtime, state)

    outcome = grade(boxes, task_with(grade=once_flaky))
    assert (outcome.status, outcome.reward) == ("graded", 1.0)


def test_an_infra_fault_the_env_swallowed_is_still_ours(boxes):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")

    async def swallow(runtime, state):
        boxes.broken.add(runtime.box_id)
        try:
            await runtime.run(["check"], {})
        except InfraFault:
            pass
        return GradeResult(reward=0.0)

    assert grade(boxes, task_with(grade=swallow)).status == "aborted"


def test_failed_removals_are_returned_for_the_reaper(boxes):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")
    boxes.fail_remove = True
    outcome = grade(boxes)
    assert outcome.reward == 1.0 and outcome.cleanup == (G0,)


@pytest.mark.parametrize("hook_name,label", [
    ("kills_its_box", "grading_box_failed"), ("floods", "grading_output_overflow"),
    ("hangs", "grading_timeout")])
def test_what_the_agents_state_does_to_the_pristine_box_grades_zero(boxes, hook_name, label):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")

    async def kills_its_box(runtime, state):
        boxes.dead.add(runtime.box_id)
        await runtime.run(["check"], {})

    async def floods(runtime, state):
        await runtime.run(["flood"], {})

    async def hangs(runtime, state):
        await asyncio.sleep(5)

    hooks = {"kills_its_box": kills_its_box, "floods": floods, "hangs": hangs}
    extra = {"grading_timeout_s": 0.1} if hook_name == "hangs" else {}
    outcome = grade(boxes, task_with(grade=hooks[hook_name], **extra))
    assert (outcome.status, outcome.reward, outcome.facts) == ("graded", 0.0, fault(label))
    assert outcome.state == b"42\n" and outcome.state_sha256 is not None
    assert G1 not in boxes.specs  # never retried


@pytest.mark.parametrize("result", [
    GradeResult(reward=float("nan")),
    GradeResult(reward=float("inf")),
    GradeResult(reward=True),
    GradeResult(reward=1.5),
    GradeResult(reward=-0.1),
    GradeResult(reward="1"),
    GradeResult(reward=10**400),
    GradeResult(reward=1.0, facts={"bad": {1, 2}}),
    GradeResult(reward=1.0, facts=["not", "a", "dict"]),
    GradeResult(reward=1.0, facts={"_grader": "spoofed"}),
    "not a result",
])
def test_invalid_hook_results_grade_zero_and_are_not_retried(boxes, result):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")

    async def hook(runtime, state):
        return result

    grader = Grader(SETTINGS, boxes)
    outcome = grade(boxes, task_with(grade=hook), grader=grader)
    assert (outcome.status, outcome.reward, outcome.facts) == ("graded", 0.0,
                                                               fault("invalid_grade_result"))
    assert G1 not in boxes.specs and G0 in boxes.removed
    assert grader.env_failure_counts() == {"fake-env": {"invalid_grade_result": 1}}


def test_an_integer_reward_in_range_is_accepted(boxes):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")

    async def hook(runtime, state):
        return GradeResult(reward=1)

    outcome = grade(boxes, task_with(grade=hook))
    assert outcome.reward == 1.0 and type(outcome.reward) is float


def test_oversized_facts_are_replaced(boxes):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")

    async def hook(runtime, state):
        return GradeResult(reward=1.0, facts={"log": "x" * 70_000})

    outcome = grade(boxes, task_with(grade=hook))
    assert outcome.reward == 1.0 and outcome.facts["facts_truncated"] is True
    assert outcome.facts["_grader"]["facts_bytes"] > 64 * 1024


def test_runtime_for_a_box_whose_user_was_never_resolved_is_ours(boxes):
    boxes.users.pop("s-1")
    with pytest.raises(InfraFault):
        Grader(SETTINGS, boxes).runtime("s-1", sandbox_task("train", 0), 10)
    assert grade(boxes).status == "aborted"


def test_grade_never_raises_on_its_own_bugs(boxes, monkeypatch):
    grader = Grader(SETTINGS, boxes)
    monkeypatch.setattr(grader, "extract", lambda *a, **k: 1 / 0)
    assert grade(boxes, grader=grader).status == "aborted"


def test_the_grader_output_bound_defaults_to_32_mib():
    assert Settings.model_fields["episode_grader_output_bytes"].default == 32 * 1024**2


def test_a_helper_cut_by_the_step_deadline_is_an_extract_timeout_not_box_failed(boxes, monkeypatch):
    from reliquary_sandbox_service.task_runtime import HelperBudgetTimeout

    def out_of_time(*args, **kwargs):
        raise HelperBudgetTimeout("Reading a file: the caller's time ran out")

    boxes.boxes["s-1"]["files"]["/work/answer.txt"] = b"42\n"
    monkeypatch.setattr(boxes, "read_file", out_of_time)
    outcome = grade(boxes)
    assert (outcome.status, outcome.reward, outcome.facts) == ("graded", 0.0,
                                                               fault("extract_timeout"))


# -- the verification window (an absolute grading deadline) -----------------


class FakeClock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def test_a_step_cut_by_the_grading_deadline_is_aborted_not_graded_zero(boxes):
    async def slow(runtime):
        await asyncio.sleep(5)

    grader = Grader(SETTINGS, boxes)
    outcome = grader.grade("s-1", "s-1", task_with(extract=slow), SPEC, env="fake-env",
                           deadline=time.time() + 0.1)
    assert outcome.status == "aborted" and outcome.reward is None
    assert "window" in outcome.reason


def test_a_grading_step_cut_by_the_deadline_is_aborted(boxes):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")

    async def slow(runtime, state):
        await asyncio.sleep(5)

    grader = Grader(SETTINGS, boxes)
    outcome = grader.grade("s-1", "s-1", task_with(grade=slow), SPEC, env="fake-env",
                           deadline=time.time() + 0.2)
    assert outcome.status == "aborted" and "window" in outcome.reason
    assert G1 not in boxes.removed  # no retry: the window is spent


def test_the_task_timeout_still_grades_zero_when_the_window_is_not_the_limit(boxes):
    async def slow(runtime):
        await asyncio.sleep(5)

    grader = Grader(SETTINGS, boxes)
    outcome = grader.grade("s-1", "s-1", task_with(extract=slow, grading_timeout_s=0.1), SPEC,
                           env="fake-env", deadline=time.time() + 3600)
    assert (outcome.status, outcome.facts) == ("graded", fault("extract_timeout"))


def test_step_timeouts_are_cut_to_the_grading_deadline(boxes):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")
    clock = FakeClock()
    left = []

    async def measured(runtime, state):
        left.append(runtime.deadline - time.monotonic())
        return await fake_episode_env.grade(runtime, state)

    grader = Grader(SETTINGS, boxes, clock=clock)
    outcome = grader.grade("s-1", "s-1", task_with(grade=measured), SPEC, env="fake-env",
                           deadline=clock.now + 5)
    assert outcome.status == "graded" and 0 < left[0] <= 5


def test_no_retry_starts_without_time_for_a_full_attempt(boxes):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")
    clock = FakeClock()
    calls = []

    async def late_infra_fault(runtime, state):
        calls.append(runtime.box_id)
        clock.now += 650
        raise InfraFault("docker went away")

    grader = Grader(SETTINGS, boxes, clock=clock)
    outcome = grader.grade("s-1", "s-1", task_with(grade=late_infra_fault), SPEC,
                           env="fake-env", deadline=clock.now + 1300)
    assert outcome.status == "aborted" and "no time left" in outcome.reason
    assert calls == [G0]


def test_a_retry_starts_when_a_full_attempt_still_fits(boxes):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")
    clock = FakeClock()

    async def once_late(runtime, state):
        if runtime.box_id == G0:
            clock.now += 100
            raise InfraFault("docker went away")
        return await fake_episode_env.grade(runtime, state)

    grader = Grader(SETTINGS, boxes, clock=clock)
    outcome = grader.grade("s-1", "s-1", task_with(grade=once_late), SPEC, env="fake-env",
                           deadline=clock.now + 600 + 120 + 101)
    assert (outcome.status, outcome.reward) == ("graded", 1.0)


def test_no_grading_starts_once_the_deadline_passed(boxes):
    clock = FakeClock()
    grader = Grader(SETTINGS, boxes, clock=clock)
    outcome = grader.grade("s-1", "s-1", sandbox_task("train", 0), SPEC, env="fake-env",
                           deadline=clock.now - 1)
    assert outcome.status == "aborted" and boxes.helper_calls == []


def test_env_programs_run_through_the_framed_program_runner(boxes):
    """Env commands go through `run_program`, which frames their stdout and
    detects a lost output."""
    runtime = make_runtime(boxes, env={"A": "1"})
    assert asyncio.run(runtime.run(["check"], {"B": "2"})).exit_code == 1
    calls = [c for c in boxes.helper_calls if c[0] == "run_program"]
    assert calls and calls[-1][2]["env"] == {"A": "1", "B": "2"}
    assert calls[-1][2]["workdir"] == "/work" and calls[-1][2]["user"] == "0:0"


# -- the agent box is stilled before extract -----------------------------------


def test_extract_first_stops_every_agent_process(boxes):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")
    boxes.helper_calls.clear()
    assert grade(boxes).reward == 1.0
    assert boxes.helper_calls[0][:2] == ("stop_processes", "s-1")
    assert 0 < boxes.helper_calls[0][2]["timeout_s"] <= 600


@pytest.mark.parametrize("error,status", [(BoxFault("dead"), "box_failed"),
                                          (InfraFault("docker down"), "aborted")])
def test_a_failed_stop_is_classified_like_extract(boxes, error, status):
    boxes.fail_stop = error
    outcome = grade(boxes)
    assert outcome.status == status and outcome.reward is None
    assert G0 not in boxes.specs


def test_a_lost_output_on_the_live_agent_box_is_box_failed(boxes, monkeypatch):
    """TaskBoxRuntime raises BoxFault for an output lost on a box that still
    runs (the agent's doing); extract maps it to box_failed."""
    def lost(*args, **kwargs):
        raise BoxFault("Reading a file: the output was lost; the box is running")

    monkeypatch.setattr(boxes, "read_file", lost)
    assert grade(boxes).status == "box_failed"


# -- the grading box's start is bounded ----------------------------------------


def test_a_grading_box_that_starts_too_slowly_is_ours(boxes):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")
    boxes.create_delay = 0.6
    settings = SimpleNamespace(episode_grader_output_bytes=1024, episode_grading_attempts=2,
                               episode_grading_create_timeout_s=0.1)
    started = time.monotonic()
    outcome = grade(boxes, settings=settings)
    assert outcome.status == "aborted" and "in time" in outcome.reason
    assert time.monotonic() - started < 1.2  # two attempts, each cut at 0.1 s


@pytest.mark.parametrize("operation", ["archive", "restore", "write_restored"])
def test_a_deadline_inside_a_helper_is_a_timeout_not_unreadable_state(boxes, monkeypatch, operation):
    runtime = make_runtime(boxes)

    def spent():
        raise StepTimeout("the step deadline passed")

    async def scenario():
        await runtime.write("/work/a", b"1")
        data = await runtime.archive(["/work"], 1 << 20)
        if operation == "write_restored":
            await runtime.restore_archive(data, ["/work"])
        monkeypatch.setattr(runtime, "_left", spent)
        if operation == "archive":
            await runtime.archive(["/work"], 1 << 20)
        elif operation == "restore":
            await runtime.restore_archive(data, ["/work"])
        else:
            await runtime.write("/work/x", b"1")

    with pytest.raises(StepTimeout):
        asyncio.run(scenario())


# -- EnvInfraError: the env says the failure is ours ------------------------

def test_an_env_infra_error_from_grade_is_aborted_retried_logged_and_counted(boxes, caplog):
    from reliquary_sandbox.episode_task import EnvInfraError

    boxes.write_file("s-1", "/work/answer.txt", b"42\n")
    calls = []

    async def prep_fails(runtime, state):
        calls.append(runtime.box_id)
        raise EnvInfraError("could not strip history in the pristine box")

    grader = Grader(SETTINGS, boxes)
    with caplog.at_level(logging.ERROR):
        outcome = grade(boxes, task_with(grade=prep_fails), grader=grader)
    assert (outcome.status, outcome.reward) == ("aborted", None)
    assert calls == [G0, G1]
    assert grader.env_failure_counts() == {"fake-env": {"env_infra_error": 2}}
    assert any("fake-env" in r.getMessage() and r.levelno == logging.ERROR for r in caplog.records)


def test_an_env_infra_error_from_extract_is_aborted_and_counted(boxes):
    from reliquary_sandbox.episode_task import EnvInfraError

    async def extract(runtime):
        raise EnvInfraError("our own state is undecodable")

    grader = Grader(SETTINGS, boxes)
    outcome = grade(boxes, task_with(extract=extract), grader=grader)
    assert (outcome.status, outcome.reward) == ("aborted", None)
    assert grader.env_failure_counts() == {"fake-env": {"env_infra_error": 1}}


def test_an_env_infra_error_then_success_grades(boxes):
    from reliquary_sandbox.episode_task import EnvInfraError

    boxes.write_file("s-1", "/work/answer.txt", b"42\n")

    async def once(runtime, state):
        if runtime.box_id == G0:
            raise EnvInfraError("transient")
        return await fake_episode_env.grade(runtime, state)

    outcome = grade(boxes, task_with(grade=once))
    assert (outcome.status, outcome.reward) == ("graded", 1.0)


def test_a_plain_runtime_error_still_grades_zero_not_aborted(boxes):
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")

    async def raises(runtime, state):
        raise RuntimeError("an env bug")

    assert grade(boxes, task_with(grade=raises)).facts == fault("grading_failed")
