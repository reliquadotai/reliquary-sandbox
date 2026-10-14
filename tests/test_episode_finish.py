import base64
import dataclasses
import hashlib

import fake_episode_env
import pytest
from episode_fixtures import budgets
from episode_manager_fixtures import build, restarted
from fastapi import HTTPException

from reliquary_sandbox.attest import GRADING_GRACE_S, Reason
from reliquary_sandbox_service.episodes.grader import GradeOutcome, Grader
from reliquary_sandbox_service.episodes.manager import BOX_GRACE_S
from reliquary_sandbox_service.episodes.store import CLOSED, RUNNING
from reliquary_sandbox_service.task_runtime import InfraFault


@pytest.fixture
def h(tmp_path):
    return build(tmp_path)


def solve(h, answer="42"):
    key = h.open()["episode_key"]
    h.manager.call("s-1", key, 0, 0, "bash", {"command": f"write /work/answer.txt {answer}"})
    h.manager.call("s-1", key, 1, 0, "bash", {"command": "burn 30"})
    return key


def with_task(monkeypatch, **changes):
    original = fake_episode_env.sandbox_task
    monkeypatch.setattr(fake_episode_env, "sandbox_task",
                        lambda split, index: dataclasses.replace(original(split, index), **changes))


def test_finish_grades_in_a_pristine_box_and_signs_the_reward(h):
    key = solve(h)
    response = h.manager.finish("s-1", key)
    body = response["record"]["body"]
    assert body["status"] == "graded" and body["reward"] == 1.0
    assert body["grading"] == {"check_exit": 0}
    assert body["state_sha256"] == hashlib.sha256(b"42\n").hexdigest()
    assert body["cpu_total_ms"] == 30 and body["reason"] is None
    assert base64.b64decode(response["state"]) == b"42\n"
    assert h.boxes.boxes == {}
    assert {"s-1", Grader.pristine_box_id("s-1", 0)} <= set(h.boxes.removed)
    result = h.verify("s-1")
    assert result.ok, result.reasons
    assert result.final.reward == 1.0
    assert h.manager.active_count() == 0


def test_wrong_answer_is_graded_zero(h):
    key = solve(h, "41")
    assert h.manager.finish("s-1", key)["record"]["body"]["reward"] == 0.0


def test_finish_is_idempotent(h):
    key = solve(h)
    first = h.manager.finish("s-1", key)
    assert h.manager.finish("s-1", key) == first
    assert h.boxes.removed.count(Grader.pristine_box_id("s-1", 0)) == 1  # graded once


def test_finish_is_idempotent_across_a_restart_and_keeps_the_state(h):
    key = solve(h)
    first = h.manager.finish("s-1", key)
    h2 = restarted(h)
    assert h2.manager.finish("s-1", key) == first
    assert base64.b64decode(first["state"]) == b"42\n"


def test_the_state_is_kept_outside_the_episode_row(h):
    key = solve(h)
    h.manager.finish("s-1", key)
    episode = h.manager.store.get("s-1")
    assert "state" not in episode and "state_b64" not in episode
    assert h.manager.store.state("s-1") == b"42\n"


def test_the_state_is_not_returned_unless_the_task_publishes_it(tmp_path, monkeypatch):
    with_task(monkeypatch, publish_state=False)
    h = build(tmp_path)
    key = solve(h)
    response = h.manager.finish("s-1", key)
    assert response["state"] is None and response["record"]["body"]["reward"] == 1.0
    assert h.manager.finish("s-1", key)["state"] is None


def test_calls_after_finish_are_refused_with_the_final_record(h):
    key = solve(h)
    final = h.manager.finish("s-1", key)["record"]
    with pytest.raises(HTTPException) as error:
        h.manager.call("s-1", key, 5, 0, "bash", {"command": "echo late"})
    assert error.value.detail["final"] == final


def test_finish_with_a_wrong_key_is_refused(h):
    solve(h)
    with pytest.raises(HTTPException) as error:
        h.manager.finish("s-1", "not-the-key")
    assert error.value.status_code == 401
    assert h.manager.store.get("s-1")["status"] == RUNNING


def test_an_env_grading_exception_is_graded_zero_not_aborted(tmp_path, monkeypatch):
    """An exception from the env's own grade hook (no infra
    fault seen) is graded 0.0 with a fact, never a void re-roll."""
    async def broken(runtime, state):
        raise RuntimeError("grader crashed")

    with_task(monkeypatch, grade=broken)
    h = build(tmp_path)
    key = solve(h)
    body = h.manager.finish("s-1", key)["record"]["body"]
    assert body["status"] == "graded" and body["reward"] == 0.0
    assert body["grading"] == {"_grader": {"agent_fault": "grading_failed"}}


def test_a_grading_box_that_never_starts_ends_aborted_without_reward(h):
    key = solve(h)
    h.boxes.fail_create = True
    response = h.manager.finish("s-1", key)
    body = response["record"]["body"]
    assert body["status"] == "aborted" and body["reward"] is None
    assert body["reason"].startswith("grading failed") and response["state"] is None
    assert h.verify("s-1", require_graded=False).ok


def test_finish_after_the_deadline_is_expired(tmp_path):
    h = build(tmp_path)
    key = h.manager.open(h.token(budgets=budgets(wall_s=60)))["episode_key"]
    h.clock.now += 120
    with pytest.raises(HTTPException) as error:
        h.manager.finish("s-1", key)
    assert error.value.detail["final"]["body"]["status"] == "expired"
    assert h.verify("s-1").reasons == (Reason.NOT_GRADED,)
    assert h.manager.finish("s-1", key)["record"] == error.value.detail["final"]


def test_missing_state_is_graded_zero(h):
    """The fake env returns an empty state when the answer file is missing."""
    key = h.open()["episode_key"]
    body = h.manager.finish("s-1", key)["record"]["body"]
    assert body["status"] == "graded" and body["reward"] == 0.0
    assert body["state_sha256"] == hashlib.sha256(b"").hexdigest()


def test_finish_with_a_pending_call_is_aborted(h):
    key = solve(h)
    episode = h.manager.store.get("s-1")
    episode["pending"] = {"turn": 2, "k": 0}
    h.manager.store.commit(episode)
    graded = []
    h.manager.grader.grade = lambda *args, **kwargs: graded.append(args)
    response = h.manager.finish("s-1", key)
    body = response["record"]["body"]
    assert body["status"] == "aborted" and response["state"] is None and graded == []
    assert h.manager.finish("s-1", key) == response
    assert h.verify("s-1", require_graded=False).ok


SHA = "c" * 64


@pytest.mark.parametrize("outcome, expected", [
    (GradeOutcome("graded", 0.5, {"tests": 3}, b"s", SHA, cpu_ms=7),
     dict(status="graded", reward=0.5, grading={"tests": 3}, state_sha256=SHA, reason=None,
          cpu_total_ms=37)),
    (GradeOutcome("graded", 0.0, {"_grader": {"agent_fault": "extract_timeout"}}),
     dict(status="graded", reward=0.0, grading={"_grader": {"agent_fault": "extract_timeout"}},
          state_sha256=None, reason=None, cpu_total_ms=30)),
    (GradeOutcome("box_failed", reason="the agent's box is unusable"),
     dict(status="box_failed", reward=None, grading=None, state_sha256=None,
          reason="the agent's box is unusable", cpu_total_ms=30)),
    (GradeOutcome("aborted", state=b"s", reason="grading failed: infra", cpu_ms=3),
     dict(status="aborted", reward=None, grading=None, state_sha256=None,
          reason="grading failed: infra", cpu_total_ms=33)),
    (GradeOutcome("weird", reason="?"),
     dict(status="aborted", reward=None, grading=None, state_sha256=None, cpu_total_ms=30)),
], ids=["graded", "graded-zero-from-extract", "box_failed", "aborted", "unknown-status"])
def test_every_grade_outcome_becomes_the_right_signed_final(h, outcome, expected):
    key = solve(h)
    seen = {}

    def grade(episode_id, agent_box, task, spec, *, env, deadline, owner=""):
        seen.update(episode_id=episode_id, agent_box=agent_box, env=env, spec=spec,
                    deadline=deadline)
        return outcome

    h.manager.grader.grade = grade
    response = h.manager.finish("s-1", key)
    body = response["record"]["body"]
    assert {name: body[name] for name in expected} == expected
    assert seen["env"] == "fake-env" and seen["agent_box"] == "s-1"
    expires_at = h.manager.store.get("s-1")["claims"]["expires_at"]
    assert seen["deadline"] == expires_at + GRADING_GRACE_S - 60
    assert seen["spec"].lifetime_s >= 600 + BOX_GRACE_S
    publishes = expected["status"] == "graded" and outcome.state is not None
    assert response["state"] == (base64.b64encode(b"s").decode() if publishes else None)
    assert h.verify("s-1", require_graded=False).ok
    assert h.manager.store.get("s-1")["status"] == CLOSED


def test_grading_leftovers_go_to_the_cleanup_list(h):
    key = solve(h)
    real = h.manager.grader.grade

    def grade(*args, **kwargs):
        return dataclasses.replace(real(*args, **kwargs), cleanup=("g-left-0",))

    remove = h.boxes.remove

    def stuck(box_id):
        if box_id == "g-left-0":
            raise OSError("still busy")
        remove(box_id)

    h.manager.grader.grade = grade
    h.boxes.remove = stuck
    h.manager.finish("s-1", key)
    assert h.manager.store.get("s-1")["cleanup"] == ["g-left-0"]
    assert "s-1" not in h.boxes.boxes


def test_an_unexpected_error_while_finishing_aborts_and_frees_the_slot(h, caplog):
    key = solve(h)

    def grade(*args, **kwargs):
        raise KeyError("our bug")

    h.manager.grader.grade = grade
    body = h.manager.finish("s-1", key)["record"]["body"]
    assert body["status"] == "aborted" and h.manager.active_count() == 0
    assert "unexpected error" in caplog.text


def test_the_episode_is_finishing_while_it_is_graded(h):
    key = solve(h)
    real = h.manager.grader.grade
    seen = []

    def grade(*args, **kwargs):
        seen.append(h.manager.store.get("s-1")["status"])
        return real(*args, **kwargs)

    h.manager.grader.grade = grade
    h.manager.finish("s-1", key)
    assert seen == ["FINISHING"]


def test_the_agent_box_outlives_a_long_extract(tmp_path, monkeypatch):
    """Extract runs on the agent's box after `finish`, which may arrive just
    before the deadline: the box must live for the whole grading timeout."""
    with_task(monkeypatch, grading_timeout_s=800.0)  # the longest that fits the window
    h = build(tmp_path)
    h.open()
    episode = h.manager.store.get("s-1")
    lifetime = h.boxes.specs["s-1"].lifetime_s
    assert lifetime >= episode["deadline"] - h.clock.now + 800


def test_transcript_returns_the_status_and_the_records(h):
    key = solve(h)
    view = h.manager.transcript("s-1", key)
    assert view["status"] == RUNNING and view["final_status"] is None
    assert [r["body"]["i"] for r in view["transcript"]["records"]] == [0, 1, 2]
    h.manager.finish("s-1", key)
    view = h.manager.transcript("s-1", key)
    assert view["status"] == CLOSED and view["final_status"] == "graded"
    assert view["transcript"]["token"] == h.manager.store.get("s-1")["token"]
    with pytest.raises(HTTPException):
        h.manager.transcript("s-1", "wrong")


# -- the verification window ---------------------------------------------------


def test_grading_that_would_outlast_the_verification_window_is_aborted(tmp_path, monkeypatch):
    """The reviewer's case: finish one second before the token expires, extract
    takes 600 s, attempt 0 hits an InfraFault late. A retry would sign the
    final past `expires_at + GRADING_GRACE_S`: the honest 1.0 would not verify."""
    holder = {}

    async def slow_extract(runtime):
        holder["h"].clock.now += 600
        return b"42\n"

    async def late_then_fine(runtime, state):
        holder["h"].clock.now += 650
        if runtime.box_id == Grader.pristine_box_id("s-1", 0):
            raise InfraFault("docker went away")
        return await fake_episode_env.grade(runtime, state)

    with_task(monkeypatch, extract=slow_extract, grade=late_then_fine)
    h = holder["h"] = build(tmp_path)
    key = h.open()["episode_key"]
    expires_at = h.manager.store.get("s-1")["claims"]["expires_at"]
    h.clock.now = expires_at - 1
    body = h.manager.finish("s-1", key)["record"]["body"]
    assert body["status"] == "aborted" and "no time left" in body["reason"]
    assert body["at"] <= expires_at + GRADING_GRACE_S
    result = h.verify("s-1", require_graded=False)
    assert result.ok, result.reasons


def test_a_final_that_would_be_signed_past_the_window_is_aborted(h):
    key = solve(h)
    expires_at = h.manager.store.get("s-1")["claims"]["expires_at"]
    real = h.manager.grader.grade

    def overrun(*args, **kwargs):
        outcome = real(*args, **kwargs)
        h.clock.now = expires_at + GRADING_GRACE_S + 5
        return outcome

    h.manager.grader.grade = overrun
    body = h.manager.finish("s-1", key)["record"]["body"]
    assert body["status"] == "aborted" and body["reward"] is None
    assert "verification window" in body["reason"]


def test_open_refuses_a_task_whose_grading_cannot_fit_the_window(tmp_path, monkeypatch):
    with_task(monkeypatch, grading_timeout_s=900.0)  # 2 * 900 + 180 > 1800
    h = build(tmp_path)
    with pytest.raises(HTTPException) as error:
        h.open()
    assert error.value.status_code == 422
    # The validator chose the task: the refusal is signed and the token spent, no box.
    assert h.manager.store.get("s-1")["final_status"] == "refused" and h.boxes.boxes == {}


# -- CPU after the last call ---------------------------------------------------


def test_cpu_burnt_after_the_last_call_is_charged_at_finish(h):
    key = solve(h)
    h.boxes.boxes["s-1"]["cpu"] += 12_000_000  # a background job, after the last call
    body = h.manager.finish("s-1", key)["record"]["body"]
    assert body["status"] == "graded" and body["cpu_total_ms"] == 42


def test_a_background_job_over_the_cpu_budget_ends_budget_exhausted_at_finish(tmp_path):
    h = build(tmp_path)
    key = h.manager.open(h.token(budgets=budgets(cpu_s=1)))["episode_key"]
    h.manager.call("s-1", key, 0, 0, "bash", {"command": "burn 500"})
    h.boxes.boxes["s-1"]["cpu"] += 600_000_000
    body = h.manager.finish("s-1", key)["record"]["body"]
    assert (body["status"], body["reason"], body["cpu_total_ms"]) == (
        "budget_exhausted", "cpu_s", 1100)
    assert Grader.pristine_box_id("s-1", 0) not in h.boxes.removed  # never graded
    assert h.verify("s-1", require_graded=False).ok


def test_a_box_the_agent_killed_before_finish_is_box_failed(h):
    """Its CPU reads the last value (as the real runtime does), so finish
    reaches extract, which fails on the dead box: box_failed."""
    key = solve(h)
    h.boxes.dead.add("s-1")
    body = h.manager.finish("s-1", key)["record"]["body"]
    assert body["status"] == "box_failed"
    assert [c[0] for c in h.boxes.helper_calls if c[1] == "s-1"][-1] == "stop_processes"


def test_a_call_on_an_episode_already_over_its_cpu_budget_closes_it(tmp_path):
    """A crash between a call's commit and its budget close leaves the episode
    over budget and RUNNING: the next call closes it instead of running."""
    h = build(tmp_path)
    key = h.manager.open(h.token(budgets=budgets(cpu_s=1)))["episode_key"]
    episode = h.manager.store.get("s-1")
    episode["cpu_calls_ms"] = 1500
    h.manager.store.commit(episode)
    with pytest.raises(HTTPException) as error:
        h.manager.call("s-1", key, 0, 0, "bash", {"command": "echo a"})
    assert error.value.detail["final"]["body"]["status"] == "budget_exhausted"
    assert h.manager.store.records("s-1")[-1]["body"]["kind"] == "final"
