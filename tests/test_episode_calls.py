import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from episode_fixtures import budgets, directory_for, make_signer
from episode_manager_fixtures import EXPECTED, build, restarted
from fastapi import HTTPException

from reliquary_sandbox.attest import (
    Ed25519TokenVerifier,
    Expected,
    Reason,
    chain_hash,
    verify_transcript,
)
from reliquary_sandbox.attest.records import record_signature_valid
from reliquary_sandbox_service.episodes import manager as manager_module
from reliquary_sandbox_service.episodes.store import CLOSED


@pytest.fixture
def h(tmp_path):
    return build(tmp_path)


def bash(h, key, turn, k, command, session="s-1"):
    return h.manager.call(session, key, turn, k, "bash", {"command": command})


def closed_with(error, status):
    final = error.value.detail["final"]["body"]
    assert final["kind"] == "final" and final["status"] == status
    return final


def test_calls_are_signed_chained_and_verifiable(h):
    key = h.open()["episode_key"]
    first = bash(h, key, 0, 0, "echo one")
    second = bash(h, key, 0, 1, "fail")
    third = bash(h, key, 2, 0, "echo two")
    assert [r["record"]["body"]["output"] for r in (first, second, third)] == [
        "one\n", "out\nboom\n", "two\n"]
    assert second["record"]["body"]["exit_code"] == 2
    result = h.verify("s-1", require_graded=False)
    assert result.reasons == (Reason.NO_FINAL,)
    assert [(c.turn, c.k) for c in result.calls] == [(0, 0), (0, 1), (2, 0)]


def test_every_record_is_signed_by_the_machine_key_and_chained(h):
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "echo a")
    bash(h, key, 1, 0, "echo b")
    with pytest.raises(HTTPException):
        h.clock.now += 7200
        bash(h, key, 2, 0, "echo late")
    records = h.manager.store.records("s-1")
    assert [r["body"]["kind"] for r in records] == ["open", "call", "call", "final"]
    public = h.machine.private_key.public_key()
    previous = None
    for i, record in enumerate(records):
        assert record["key_id"] == h.machine.key_id and record_signature_valid(public, record)
        assert record["body"]["i"] == i and record["body"]["prev"] == previous
        previous = chain_hash(record["body"])


def test_record_times_never_go_backwards_even_if_the_clock_does(h):
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "echo a")
    h.clock.now -= 20  # NTP steps the host clock back
    second = bash(h, key, 1, 0, "echo b")["record"]["body"]
    ats = [r["body"]["at"] for r in h.manager.store.records("s-1")]
    assert ats == sorted(ats) and second["at"] == ats[1]
    assert Reason.TIME_NOT_MONOTONIC not in h.verify("s-1", require_graded=False).reasons


def test_resending_a_call_returns_the_stored_record_without_rerunning(h):
    key = h.open()["episode_key"]
    first = bash(h, key, 0, 0, "burn 50")
    again = bash(h, key, 0, 0, "burn 50")
    assert again["record"] == first["record"]
    assert h.boxes.boxes["s-1"]["cpu"] == 50_000_000
    with pytest.raises(HTTPException) as error:
        bash(h, key, 0, 0, "echo different")
    assert error.value.status_code == 409
    with pytest.raises(HTTPException) as error:
        h.manager.call("s-1", key, 0, 0, "edit", {"command": "burn 50"})
    assert error.value.status_code == 409
    assert len(h.manager.store.records("s-1")) == 2


def test_a_retry_after_the_episode_closed_returns_the_record_and_the_final(tmp_path):
    h = build(tmp_path)
    key = h.manager.open(h.token(budgets=budgets(cpu_s=1)))["episode_key"]
    first = bash(h, key, 0, 0, "burn 1500")
    again = bash(h, key, 0, 0, "burn 1500")
    assert again == first and again["final"]["body"]["status"] == "budget_exhausted"


def test_a_retry_after_a_controller_restart_does_not_rerun(h):
    key = h.open()["episode_key"]
    first = bash(h, key, 0, 0, "burn 40")["record"]
    h2 = restarted(h)
    assert bash(h2, key, 0, 0, "burn 40")["record"] == first
    assert h.boxes.boxes["s-1"]["cpu"] == 40_000_000


@pytest.mark.parametrize("turn,k", [(0, 1), (1, 1)])
def test_the_first_call_must_be_k_zero(h, turn, k):
    key = h.open()["episode_key"]
    with pytest.raises(HTTPException) as error:
        bash(h, key, turn, k, "echo x")
    assert error.value.status_code == 409
    assert len(h.manager.store.records("s-1")) == 1


@pytest.mark.parametrize("turn,k", [(1, 2), (1, 0), (0, 0), (0, 2), (2, 1)])
def test_a_call_that_skips_or_goes_backwards_is_refused(h, turn, k):
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "echo a")
    bash(h, key, 1, 0, "echo b")
    if (turn, k) in {(1, 0), (0, 0)}:  # recorded slots: only a retry of the same call is accepted
        with pytest.raises(HTTPException) as error:
            bash(h, key, turn, k, "echo other")
    else:
        with pytest.raises(HTTPException) as error:
            bash(h, key, turn, k, "echo x")
    assert error.value.status_code == 409
    assert len(h.manager.store.records("s-1")) == 3
    assert bash(h, key, 1, 1, "echo c")["record"]["body"]["i"] == 3


def test_concurrent_calls_on_one_episode_get_busy(h):
    key = h.open()["episode_key"]
    started = threading.Thread(target=bash, args=(h, key, 0, 0, "hang"))
    started.start()
    try:
        for _ in range(100):
            lock = h.manager.episode_locks.get("s-1")
            if lock is not None and lock.locked():
                break
            threading.Event().wait(0.01)
        with pytest.raises(HTTPException) as error:
            bash(h, key, 0, 1, "echo x")
        assert error.value.status_code == 409
    finally:
        h.boxes.block.set()
        started.join()


def test_calls_fired_concurrently_run_one_at_a_time(h):
    key = h.open()["episode_key"]
    running, overlaps = [0], []
    guard = threading.Lock()
    original = h.boxes.bash

    def tracked(*args, **kwargs):
        with guard:
            running[0] += 1
            overlaps.append(running[0])
        try:
            threading.Event().wait(0.05)
            return original(*args, **kwargs)
        finally:
            with guard:
                running[0] -= 1

    h.boxes.bash = tracked

    def attempt(n):
        try:
            return bash(h, key, 0, 0, "burn 10")
        except HTTPException as error:
            return error.status_code

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(attempt, range(8)))
    assert max(overlaps) == 1
    records = [r for r in results if isinstance(r, dict)]
    assert records and all(r["record"] == records[0]["record"] for r in records)
    assert all(r == 409 for r in results if not isinstance(r, dict))
    assert h.boxes.boxes["s-1"]["cpu"] == 10_000_000  # executed exactly once
    assert len(h.manager.store.records("s-1")) == 2


def test_wrong_episode_key_is_refused(h):
    h.open()
    with pytest.raises(HTTPException) as error:
        bash(h, "not-the-key", 0, 0, "echo x")
    assert error.value.status_code == 401
    with pytest.raises(HTTPException) as error:
        bash(h, None, 0, 0, "echo x")
    assert error.value.status_code == 401
    assert len(h.manager.store.records("s-1")) == 1


def test_the_key_of_another_episode_is_refused(h):
    h.open("s-1")
    other = h.open("s-2")["episode_key"]
    with pytest.raises(HTTPException) as error:
        bash(h, other, 0, 0, "echo x", session="s-1")
    assert error.value.status_code == 401


def test_the_episode_key_is_compared_in_constant_time(h, monkeypatch):
    key = h.open()["episode_key"]
    compared = []
    real = manager_module.hmac.compare_digest

    def spy(a, b):
        compared.append((a, b))
        return real(a, b)

    monkeypatch.setattr(manager_module.hmac, "compare_digest", spy)
    bash(h, key, 0, 0, "echo x")
    with pytest.raises(HTTPException):
        bash(h, "wrong", 1, 0, "echo x")
    assert len(compared) == 2


def test_an_unknown_episode_and_a_wrong_key_look_the_same(h):
    h.open()
    with pytest.raises(HTTPException) as unknown:
        bash(h, "anything", 0, 0, "echo x", session="nope")
    with pytest.raises(HTTPException) as wrong:
        bash(h, "anything", 0, 0, "echo x")
    assert unknown.value.status_code == wrong.value.status_code == 401
    assert unknown.value.detail == wrong.value.detail


@pytest.mark.parametrize("turn,k", [(-1, 0), (0, -1), (2**63, 0), (0, 2**63)])
def test_turn_and_k_outside_the_signed_range_are_refused(h, turn, k):
    key = h.open()["episode_key"]
    with pytest.raises(HTTPException) as error:
        bash(h, key, turn, k, "echo x")
    assert error.value.status_code == 422
    assert len(h.manager.store.records("s-1")) == 1


def test_a_call_executed_but_never_recorded_aborts_the_episode(h):
    """A crash between execution and the record leaves `pending`: what ran is
    unknown, so nothing more runs and the episode is void (ours)."""
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "echo a")
    episode = h.manager.store.get("s-1")
    episode["pending"] = {"turn": 1, "k": 0}
    h.manager.store.commit(episode)
    h.boxes.boxes["s-1"]["cpu"] = 0
    h2 = restarted(h)
    with pytest.raises(HTTPException) as error:
        bash(h2, key, 1, 0, "burn 30")  # even the same (turn, k): it may have run already
    assert error.value.status_code == 503
    final = closed_with(error, "aborted")
    assert final["i"] == 2 and h.boxes.removed == ["s-1"]
    assert all(entry != "burn 30" for entry in h.boxes.exec_log)
    assert h2.verify("s-1", require_graded=False).final.status == "aborted"


def test_a_record_that_could_not_be_committed_aborts_and_nothing_reruns(h):
    key = h.open()["episode_key"]
    store = h.manager.store
    real = store.commit
    failed = []

    def flaky(episode, record=None, state=None):
        if record is not None and record["body"]["kind"] == "call" and not failed:
            failed.append(record)
            raise OSError("disk I/O error")
        return real(episode, record, state)

    store.commit = flaky
    with pytest.raises(HTTPException) as error:
        bash(h, key, 0, 0, "burn 30")
    assert error.value.status_code == 503
    closed_with(error, "aborted")
    with pytest.raises(HTTPException) as error:
        bash(h, key, 0, 0, "burn 70")  # a retry with different arguments
    assert error.value.status_code == 409
    closed_with(error, "aborted")
    assert h.boxes.removed == ["s-1"]
    records = store.records("s-1")
    assert [r["body"]["kind"] for r in records] == ["open", "final"]
    assert h.verify("s-1").reasons == (Reason.NOT_GRADED,)


def test_a_failure_signing_the_call_aborts_the_episode(h, monkeypatch):
    key = h.open()["episode_key"]
    real = h.manager._sign

    def sign(body):
        if body.kind == "call":
            raise ValueError("signer failed")
        return real(body)

    monkeypatch.setattr(h.manager, "_sign", sign)
    with pytest.raises(HTTPException) as error:
        bash(h, key, 0, 0, "echo x")
    assert error.value.status_code == 503
    closed_with(error, "aborted")


def test_timeout_is_recorded_and_the_episode_continues(h):
    key = h.open()["episode_key"]
    timed_out = bash(h, key, 0, 0, "sleep 100")["record"]["body"]
    assert timed_out["timed_out"] and timed_out["exit_code"] == -1
    assert bash(h, key, 1, 0, "echo alive")["record"]["body"]["output"] == "alive\n"


def test_the_call_timeout_is_cut_to_the_deadline(tmp_path):
    h = build(tmp_path)
    key = h.manager.open(h.token(budgets=budgets(wall_s=60, per_call_timeout_s=30)))["episode_key"]
    h.clock.now += 50  # 10 s left: a 20 s command times out
    body = bash(h, key, 0, 0, "sleep 20")["record"]["body"]
    assert body["timed_out"]


def test_truncation_is_recorded(tmp_path):
    h = build(tmp_path, episode_output_cap_bytes=1024)
    key = h.open()["episode_key"]
    body = bash(h, key, 0, 0, "big 5000")["record"]["body"]
    assert body["truncated"] and len(body["output"]) == 1024


def test_cpu_is_accounted_per_call(h):
    key = h.open()["episode_key"]
    assert bash(h, key, 0, 0, "burn 120")["record"]["body"]["cpu_ms"] == 120
    assert bash(h, key, 1, 0, "echo x")["record"]["body"]["cpu_ms"] == 0


def test_cpu_burnt_between_calls_is_charged_to_the_next_call(h):
    key = h.open()["episode_key"]
    h.boxes.boxes["s-1"]["cpu"] += 70_400_000  # a background job the agent left running
    assert bash(h, key, 0, 0, "echo x")["record"]["body"]["cpu_ms"] == 70
    # the 0.4 ms remainder is carried, not lost
    h.boxes.boxes["s-1"]["cpu"] += 600_000
    assert bash(h, key, 1, 0, "echo y")["record"]["body"]["cpu_ms"] == 1


def test_a_cpu_reading_that_goes_down_never_decreases_the_total(h):
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "burn 100")
    h.boxes.boxes["s-1"]["cpu"] -= 60_000_000  # a counter reset or a stale reading
    body = bash(h, key, 1, 0, "echo x")["record"]["body"]
    assert body["cpu_ms"] == 0
    assert h.manager.store.get("s-1")["cpu_calls_ms"] == 100
    h.boxes.boxes["s-1"]["cpu"] += 80_000_000  # back above the mark: only the excess is new
    assert bash(h, key, 2, 0, "echo y")["record"]["body"]["cpu_ms"] == 20


def test_call_budget_exhaustion_ends_the_episode_with_a_signed_record(tmp_path):
    h = build(tmp_path)
    key = h.manager.open(h.token(budgets=budgets(max_calls=1)))["episode_key"]
    bash(h, key, 0, 0, "echo ok")
    with pytest.raises(HTTPException) as error:
        bash(h, key, 1, 0, "echo too many")
    assert error.value.status_code == 409
    final = closed_with(error, "budget_exhausted")
    assert final["reward"] is None and final["reason"] == "max_calls"
    assert "s-1" not in h.boxes.boxes
    episode = h.manager.store.get("s-1")
    assert episode["status"] == CLOSED and episode["closed_at"] == h.clock.now
    result = h.verify("s-1")
    assert result.reasons == (Reason.NOT_GRADED,)
    with pytest.raises(HTTPException) as error:
        bash(h, key, 2, 0, "echo after")
    assert error.value.status_code == 409 and error.value.detail["final"] is not None


def test_cpu_budget_exhaustion_closes_after_the_call(tmp_path):
    h = build(tmp_path)
    key = h.manager.open(h.token(budgets=budgets(cpu_s=1)))["episode_key"]
    response = bash(h, key, 0, 0, "burn 1500")
    assert response["record"]["body"]["cpu_ms"] == 1500
    assert response["final"]["body"]["status"] == "budget_exhausted"
    assert response["final"]["body"]["cpu_total_ms"] == 1500
    assert h.verify("s-1").reasons == (Reason.NOT_GRADED,)


def test_deadline_expiry_closes_the_episode(tmp_path):
    h = build(tmp_path)
    key = h.manager.open(h.token(budgets=budgets(wall_s=60)))["episode_key"]
    h.clock.now += 61
    with pytest.raises(HTTPException) as error:
        bash(h, key, 0, 0, "echo late")
    assert error.value.status_code == 409
    closed_with(error, "expired")
    assert "s-1" not in h.boxes.boxes
    assert h.verify("s-1").reasons == (Reason.NOT_GRADED,)


def test_a_call_that_ends_past_the_deadline_closes_the_episode(tmp_path):
    h = build(tmp_path)
    key = h.manager.open(h.token(budgets=budgets(wall_s=60)))["episode_key"]
    original = h.boxes.bash

    def slow(*args, **kwargs):
        h.clock.now += 61
        return original(*args, **kwargs)

    h.boxes.bash = slow
    response = bash(h, key, 0, 0, "echo slow")
    assert response["record"]["body"]["output"] == "slow\n"
    assert response["final"]["body"]["status"] == "expired"


def test_a_box_the_agent_broke_ends_the_episode_box_failed(h):
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "echo ok")
    h.boxes.dead.add("s-1")  # the agent ran kill -9 -1
    with pytest.raises(HTTPException) as error:
        bash(h, key, 1, 0, "echo x")
    assert error.value.status_code == 409
    final = closed_with(error, "box_failed")
    assert final["reward"] is None
    assert h.manager.store.get("s-1")["pending"] is None
    assert h.verify("s-1").reasons == (Reason.NOT_GRADED,)


def test_lost_box_aborts_the_episode(h):
    key = h.open()["episode_key"]
    h.boxes.broken.add("s-1")
    with pytest.raises(HTTPException) as error:
        bash(h, key, 0, 0, "echo x")
    assert error.value.status_code == 503
    closed_with(error, "aborted")
    assert h.verify("s-1", require_graded=False).final.status == "aborted"


@pytest.mark.parametrize("error_type", [OSError, KeyError, ValueError])
def test_an_unexpected_error_aborts_the_episode(h, error_type, caplog):
    key = h.open()["episode_key"]

    def broken(*args, **kwargs):
        raise error_type("our bug")

    h.boxes.bash = broken
    with pytest.raises(HTTPException) as error:
        bash(h, key, 0, 0, "echo x")
    assert error.value.status_code == 503
    closed_with(error, "aborted")
    assert "unexpected error" in caplog.text


def test_a_box_that_cannot_be_removed_is_left_for_the_reaper(tmp_path):
    h = build(tmp_path)
    key = h.manager.open(h.token(budgets=budgets(max_calls=1)))["episode_key"]
    bash(h, key, 0, 0, "echo ok")
    h.boxes.fail_remove = True
    with pytest.raises(HTTPException):
        bash(h, key, 1, 0, "echo x")
    assert h.manager.store.get("s-1")["cleanup"] == ["s-1"]


def test_a_pending_marker_is_persisted_before_execution(h):
    key = h.open()["episode_key"]
    seen = []
    original = h.boxes.bash

    def observing(*args, **kwargs):
        seen.append(h.manager.store.get("s-1")["pending"])
        return original(*args, **kwargs)

    h.boxes.bash = observing
    bash(h, key, 0, 0, "echo x")
    assert seen == [{"turn": 0, "k": 0}]
    assert h.manager.store.get("s-1")["pending"] is None


@pytest.mark.parametrize("arguments,status", [
    ({"command": "\ud800"}, 422), ({"command": "x" * 300_000}, 413), ("echo x", 422)])
def test_arguments_without_canonical_form_or_too_large_are_refused(h, arguments, status):
    key = h.open()["episode_key"]
    with pytest.raises(HTTPException) as error:
        h.manager.call("s-1", key, 0, 0, "bash", arguments)
    assert error.value.status_code == status
    assert len(h.manager.store.records("s-1")) == 1


def test_unknown_tool_is_recorded_not_executed(h):
    key = h.open()["episode_key"]
    body = h.manager.call("s-1", key, 0, 0, "python", {"code": "1"})["record"]["body"]
    assert body["exit_code"] == 127 and body["tool"] == "python"


def test_the_edit_tool_is_recorded(h):
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "write /work/a.txt 41")
    body = h.manager.call("s-1", key, 1, 0, "edit",
                          {"path": "a.txt", "old_str": "41", "new_str": "42"})["record"]["body"]
    assert body["exit_code"] == 0 and h.boxes.boxes["s-1"]["files"]["/work/a.txt"] == b"42\n"


def test_non_utf8_output_is_replaced_and_still_verifies(h):
    key = h.open()["episode_key"]
    body = bash(h, key, 0, 0, "binary")["record"]["body"]
    assert body["output"] == "��ok\n"
    assert h.verify("s-1", require_graded=False).reasons == (Reason.NO_FINAL,)


def test_a_transcript_signed_by_another_key_does_not_verify(h, tmp_path):
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "echo x")
    other, _ = make_signer(tmp_path, "other-machine", h.machine.key_id)
    result = verify_transcript(
        h.transcript("s-1"), directory_for(other),
        Ed25519TokenVerifier({h.validator.key_id: h.validator.public_key_b64}),
        Expected(**EXPECTED, require_graded=False))
    assert Reason.BAD_SIGNATURE in result.reasons


def test_an_edit_with_half_a_second_left_is_a_timed_out_record(tmp_path):
    h = build(tmp_path)
    key = h.manager.open(h.token(budgets=budgets(wall_s=60)))["episode_key"]
    h.clock.now += 59.5
    response = h.manager.call("s-1", key, 0, 0, "edit",
                              {"path": "a.txt", "old_str": "1", "new_str": "2"})
    body = response["record"]["body"]
    assert body["timed_out"] and body["exit_code"] == -1
    assert h.manager.store.records("s-1")[-1]["body"]["kind"] == "call"


def test_a_helper_cut_by_the_call_budget_is_recorded_not_box_failed(h, monkeypatch):
    from reliquary_sandbox_service.task_runtime import HelperBudgetTimeout

    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "write /work/a.txt 41")

    def out_of_time(*args, **kwargs):
        raise HelperBudgetTimeout("ran out of the caller's time")

    monkeypatch.setattr(h.boxes, "read_file", out_of_time)
    response = h.manager.call("s-1", key, 1, 0, "edit",
                              {"path": "a.txt", "old_str": "41", "new_str": "42"})
    assert response["record"]["body"]["timed_out"] and response["final"] is None
    assert h.manager.store.get("s-1")["status"] == "RUNNING"
