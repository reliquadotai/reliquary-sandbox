"""Per-task limits: validated, compared with the token's budgets, sizing the box's CPU."""

import pytest
from episode_fixtures import IMAGE, budgets, claims, episode_settings
from fake_boxes import FakeBoxes
from fastapi.testclient import TestClient
from test_service import FakeRuntime

from reliquary_sandbox.attest import issue_session_token
from reliquary_sandbox.episode_task import SandboxTask, TaskLimits
from reliquary_sandbox_service.app import create_app

GIB = 1024**3


def test_limits_refuse_non_positive_and_non_integer_values():
    for bad in (dict(cpus=0), dict(cpus=float("nan")), dict(cpus=True), dict(memory_bytes=0),
                dict(memory_bytes=1.5), dict(pids=True), dict(wall_s=-1),
                dict(cpus=float("inf")), dict(disk_bytes=2**63), dict(max_calls=2**64)):
        with pytest.raises(ValueError):
            TaskLimits(**bad)
    assert TaskLimits(cpus=2, memory_bytes=GIB).cpus == 2
    assert TaskLimits(disk_bytes=2**63 - 1).disk_bytes == 2**63 - 1


def test_shortfall_names_only_resource_budgets_below_the_task():
    limits = TaskLimits(memory_bytes=4 * GIB, disk_bytes=GIB, pids=512, wall_s=10**6,
                        per_call_timeout_s=10**5, max_calls=10**4)
    short = limits.shortfall(budgets(memory_bytes=2 * GIB, disk_bytes=GIB, pids=256))
    assert short == ["memory_bytes", "pids"]
    assert TaskLimits().shortfall(budgets()) == []


def test_sandbox_task_refuses_limits_of_another_type():
    from fake_episode_env import extract, grade, prepare

    with pytest.raises(TypeError):
        SandboxTask(image=IMAGE, workdir="/work", tools=("bash",), prepare=prepare,
                    extract=extract, grade=grade, limits={"cpus": 1})


@pytest.fixture
def api(tmp_path):
    settings, validator = episode_settings(
        tmp_path, episode_envs={"fake-env": "limited_episode_env:sandbox_task"},
        episode_nano_cpus=2_000_000_000)
    boxes = FakeBoxes([IMAGE])
    with TestClient(create_app(settings, FakeRuntime(), task_runtime=boxes)) as client:
        yield client, validator, boxes


def test_a_token_below_the_tasks_memory_is_refused_before_it_is_spent(api):
    client, validator, boxes = api
    low = issue_session_token(claims("s-1", budgets=budgets(memory_bytes=2 * GIB)), validator)
    refused = client.post("/episodes", json={"token": low})
    assert refused.status_code == 422
    assert "memory_bytes" in refused.json()["detail"]
    assert boxes.specs == {}
    enough = issue_session_token(claims("s-2", budgets=budgets(memory_bytes=3 * GIB)), validator)
    assert client.post("/episodes", json={"token": enough}).status_code == 200


def test_the_task_cpus_size_the_box(api):
    client, validator, boxes = api
    token = issue_session_token(claims("s-3", budgets=budgets(memory_bytes=3 * GIB)), validator)
    assert client.post("/episodes", json={"token": token}).status_code == 200
    assert boxes.specs["s-3"].nano_cpus == 500_000_000


def test_the_machine_caps_the_task_cpus(api):
    client, validator, boxes = api
    token = issue_session_token(
        claims("s-4", index=8, budgets=budgets(memory_bytes=3 * GIB)), validator)
    assert client.post("/episodes", json={"token": token}).status_code == 200
    assert boxes.specs["s-4"].nano_cpus == 2_000_000_000


def test_the_limits_are_part_of_the_task_fingerprint():
    from limited_episode_env import sandbox_task

    from reliquary_sandbox_service.episodes.registry import ResolvedTask, task_fingerprint

    def fingerprint(task):
        return task_fingerprint(ResolvedTask(task, "fake==1"))

    assert fingerprint(sandbox_task("train", 7)) != fingerprint(sandbox_task("train", 8))


def test_a_huge_cpu_request_is_capped_by_the_machine(api):
    client, validator, boxes = api
    token = issue_session_token(
        claims("s-5", index=9, budgets=budgets(memory_bytes=3 * GIB)), validator)
    assert client.post("/episodes", json={"token": token}).status_code == 200
    assert boxes.specs["s-5"].nano_cpus == 2_000_000_000


def test_integer_and_float_cpus_have_one_fingerprint():
    from dataclasses import replace

    from limited_episode_env import sandbox_task

    from reliquary_sandbox_service.episodes.registry import ResolvedTask, task_fingerprint

    task = sandbox_task("train", 7)
    as_int = replace(task, limits=replace(task.limits, cpus=2))
    as_float = replace(task, limits=replace(task.limits, cpus=2.0))
    assert (task_fingerprint(ResolvedTask(as_int, "fake==1"))
            == task_fingerprint(ResolvedTask(as_float, "fake==1")))
