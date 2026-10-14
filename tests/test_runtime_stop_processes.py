"""`LocalRuntime.stop_processes`: env hooks stop stray processes in their box (the terminal
env, after `test.sh` and before it reads the reward), with the helper the grader uses
before extract, and its failures attributed like every other runtime fault."""

import asyncio
import dataclasses
import time
from types import SimpleNamespace

import pytest
from episode_fixtures import IMAGE
from fake_boxes import FakeBoxes
from fake_episode_env import sandbox_task

from reliquary_sandbox.episode_task import GradeResult
from reliquary_sandbox_service.episodes.grader import BOX, INFRA, Grader, _attribute
from reliquary_sandbox_service.episodes.local_runtime import LocalRuntime, StepTimeout
from reliquary_sandbox_service.task_runtime import (
    BoxFault,
    BoxSpec,
    HelperBudgetTimeout,
    InfraFault,
)

SETTINGS = SimpleNamespace(episode_grader_output_bytes=1024, episode_grading_attempts=2)
SPEC = BoxSpec(image=IMAGE, workdir="/work", env={}, user=None, memory_bytes=1, nano_cpus=1,
               pids=16, disk_bytes=1, lifetime_s=60)


class GradingBoxFails(FakeBoxes):
    """stop_processes fails in grading boxes only (the agent box's stop before extract
    works)."""

    def __init__(self, error):
        super().__init__([IMAGE])
        self.grading_error = error

    def stop_processes(self, box_id, *, timeout_s):
        super().stop_processes(box_id, timeout_s=timeout_s)
        if box_id.startswith("g-") and self.grading_error is not None:
            raise self.grading_error


@pytest.fixture
def boxes():
    boxes = FakeBoxes([IMAGE])
    boxes.create("s-1", "s-1", SPEC)
    return boxes


def runtime_for(boxes, deadline=10):
    return LocalRuntime(boxes, "s-1", workdir="/work", env={}, user="1000:1000",
                        deadline=time.monotonic() + deadline, output_limit=1024)


def test_the_runtime_stops_the_box_processes_with_the_helper(boxes):
    runtime = runtime_for(boxes, deadline=5)
    assert asyncio.run(runtime.stop_processes()) is None
    [(name, box, kwargs)] = boxes.helper_calls
    assert (name, box) == ("stop_processes", "s-1")
    assert 0 < kwargs["timeout_s"] <= 5
    assert runtime.faults == []


@pytest.mark.parametrize("error, cause", [
    (InfraFault("Docker is unreachable"), INFRA),
    (BoxFault("Stopping the agent's processes: the box is running"), BOX),
])
def test_stop_failures_are_recorded_and_attributed(boxes, error, cause):
    boxes.fail_stop = error
    runtime = runtime_for(boxes)
    with pytest.raises(type(error)) as raised:
        asyncio.run(runtime.stop_processes())
    assert runtime.faults == [error]
    assert _attribute(runtime, raised.value) == cause


def test_a_stop_cut_by_the_step_deadline_is_a_step_timeout(boxes):
    boxes.fail_stop = HelperBudgetTimeout("cut by the budget")
    runtime = runtime_for(boxes)
    with pytest.raises(StepTimeout):
        asyncio.run(runtime.stop_processes())
    assert isinstance(runtime.faults[-1], StepTimeout)


def test_a_stop_after_the_deadline_is_refused_before_the_helper(boxes):
    runtime = runtime_for(boxes, deadline=-1)
    with pytest.raises(StepTimeout):
        asyncio.run(runtime.stop_processes())
    assert boxes.helper_calls == []


def grade_with_a_stop(boxes):
    task = sandbox_task("train", 0)

    async def grade(runtime, state):
        await runtime.stop_processes()
        return GradeResult(reward=1.0, facts={})

    boxes.create("s-1", "s-1", SPEC)
    return Grader(SETTINGS, boxes).grade("s-1", "s-1", dataclasses.replace(task, grade=grade),
                                          SPEC, env="fake-env")


def test_respawning_processes_in_the_grading_box_grade_zero():
    boxes = GradingBoxFails(BoxFault("Stopping the agent's processes: the box is running"))
    outcome = grade_with_a_stop(boxes)
    assert outcome.status == "graded" and outcome.reward == 0.0
    assert outcome.facts == {"_grader": {"agent_fault": "grading_box_failed"}}


def test_our_failure_stopping_grading_box_processes_is_aborted():
    outcome = grade_with_a_stop(GradingBoxFails(InfraFault("Docker is unreachable")))
    assert outcome.status == "aborted"


def test_a_clean_stop_in_the_grading_box_keeps_the_grade():
    boxes = GradingBoxFails(None)
    outcome = grade_with_a_stop(boxes)
    assert outcome.status == "graded" and outcome.reward == 1.0
    stops = [box for name, box, _ in boxes.helper_calls if name == "stop_processes"]
    assert stops == ["s-1", Grader.pristine_box_id("s-1", 0)]  # before extract, then ours
