import dataclasses
import threading

import pytest
from episode_fixtures import budgets
from episode_manager_fixtures import build, restarted
from fastapi import HTTPException

from reliquary_sandbox_service.episodes.grader import Grader
from reliquary_sandbox_service.episodes.store import CLOSED, FINISHING, OPENING, RUNNING

RETENTION_S = 7 * 86400


def bash(h, key, turn, k, command, session="s-1"):
    return h.manager.call(session, key, turn, k, "bash", {"command": command})


def set_fields(h, session, **fields):
    episode = h.manager.store.get(session)
    episode.update(fields)
    h.manager.store.commit(episode)


@pytest.fixture
def h_default(tmp_path):
    return build(tmp_path)


# -- the reaper (periodic, no restart) -------------------------------------------


def test_reaper_expires_idle_episodes_and_frees_their_slot(tmp_path):
    h = build(tmp_path, episode_capacity=1)
    key = h.manager.open(h.token(budgets=budgets(wall_s=60)))["episode_key"]
    with pytest.raises(HTTPException) as error:
        h.open("s-2")
    assert error.value.status_code == 503
    h.clock.now += 61
    h.manager.reap()
    transcript = h.manager.transcript("s-1", key)
    assert transcript["status"] == CLOSED and transcript["final_status"] == "expired"
    assert h.manager.active_count() == 0
    assert "s-1" not in h.boxes.boxes
    assert h.verify("s-1", require_graded=False).ok
    h.open("s-3")  # the slot is free again without a restart (s-2's token was spent by its refusal)


def test_reaper_leaves_episodes_before_their_deadline_alone(h_default):
    h = h_default
    h.open()
    h.manager.reap()
    assert h.manager.store.get("s-1")["status"] == RUNNING and "s-1" in h.boxes.boxes


def test_reaper_aborts_an_expired_episode_with_an_unrecorded_call(tmp_path):
    h = build(tmp_path)
    key = h.manager.open(h.token(budgets=budgets(wall_s=60)))["episode_key"]
    set_fields(h, "s-1", pending={"turn": 0, "k": 0})
    h.clock.now += 61
    h.manager.reap()
    assert h.manager.transcript("s-1", key)["final_status"] == "aborted"


def test_reaper_skips_an_episode_whose_call_holds_the_lock(tmp_path):
    h = build(tmp_path)
    key = h.manager.open(h.token(budgets=budgets(wall_s=60)))["episode_key"]
    h.boxes.block.clear()
    worker = threading.Thread(target=bash, args=(h, key, 0, 0, "hang"))
    worker.start()
    try:
        for _ in range(200):
            if h.manager.store.get("s-1").get("pending"):
                break
            threading.Event().wait(0.01)
        h.clock.now += 61
        h.manager.reap()  # the call is in flight: it clips itself to the deadline
        assert h.manager.store.get("s-1")["status"] == RUNNING
    finally:
        h.boxes.block.set()
        worker.join()
    final = h.manager.transcript("s-1", key)
    assert final["final_status"] == "expired"  # closed by the call itself, after its record
    assert [r["body"]["kind"] for r in final["transcript"]["records"]] == ["open", "call", "final"]


def test_reaper_retries_box_removals_that_failed(tmp_path):
    h = build(tmp_path)
    key = h.manager.open(h.token(budgets=budgets(max_calls=1)))["episode_key"]
    bash(h, key, 0, 0, "echo a")
    h.boxes.fail_remove = True
    h.manager.finish("s-1", key)
    left = h.manager.store.get("s-1")["cleanup"]
    assert "s-1" in left and Grader.pristine_box_id("s-1", 0) in left
    h.manager.reap()  # still failing: kept for the next sweep
    assert h.manager.store.get("s-1")["cleanup"] == left
    h.boxes.fail_remove = False
    h.manager.reap()
    assert h.manager.store.get("s-1")["cleanup"] == []
    assert h.boxes.boxes == {}


def test_reaper_removes_leftover_boxes_no_live_episode_owns(h_default):
    h = h_default
    h.open()
    h.boxes.create("stray-box", "long-gone", None)
    h.boxes.create(Grader.pristine_box_id("s-1", 0), "s-1", None)  # its episode is live
    h.manager.reap()
    assert "stray-box" not in h.boxes.boxes
    assert {"s-1", Grader.pristine_box_id("s-1", 0)} <= set(h.boxes.boxes)


def test_reaper_keeps_the_grading_boxes_of_an_episode_being_graded(h_default):
    h = h_default
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "write /work/answer.txt 42")
    grade = h.manager.grader.grade
    seen = []

    def grade_with_a_sweep(*args, **kwargs):
        h.boxes.create(Grader.pristine_box_id("s-1", 0), "s-1", None)
        h.manager.reap()
        seen.append(set(h.boxes.boxes))
        h.boxes.remove(Grader.pristine_box_id("s-1", 0))
        return grade(*args, **kwargs)

    h.manager.grader.grade = grade_with_a_sweep
    assert h.manager.finish("s-1", key)["record"]["body"]["reward"] == 1.0
    assert {"s-1", Grader.pristine_box_id("s-1", 0)} <= seen[0]


def test_reaper_prunes_closed_episodes_past_retention(tmp_path, monkeypatch):
    h = build(tmp_path)
    key = h.manager.open(h.token(budgets=budgets(wall_s=60)))["episode_key"]
    h.clock.now += 61
    h.manager.reap()
    calls = []
    prune = h.manager.store.prune
    monkeypatch.setattr(h.manager.store, "prune",
                        lambda now, retention_s: calls.append((now, retention_s))
                        or prune(now, retention_s))
    h.manager.reap()
    assert calls == [(h.clock.now, RETENTION_S)]
    assert h.manager.store.get("s-1") is not None  # token still valid, within retention
    h.clock.now += 3600 + RETENTION_S
    h.manager.reap()
    assert h.manager.store.get("s-1") is None
    with pytest.raises(HTTPException):
        h.manager.transcript("s-1", key)


def test_reaper_prunes_the_locks_of_closed_episodes(tmp_path):
    h = build(tmp_path)
    key1 = h.open("s-1")["episode_key"]
    key2 = h.open("s-2")["episode_key"]
    bash(h, key1, 0, 0, "write /work/answer.txt 42")
    bash(h, key2, 0, 0, "echo a", session="s-2")
    h.manager.finish("s-1", key1)
    assert {"s-1", "s-2"} <= set(h.manager.episode_locks)
    h.manager.reap()
    assert "s-1" not in h.manager.episode_locks and "s-2" in h.manager.episode_locks


def test_a_held_lock_is_never_pruned(tmp_path):
    h = build(tmp_path)
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "write /work/answer.txt 42")
    h.manager.finish("s-1", key)
    with h.manager._exclusive("s-1"):
        h.manager.reap()
        assert "s-1" in h.manager.episode_locks


def test_reaper_cannot_expire_an_episode_whose_finish_holds_the_lock(tmp_path):
    h = build(tmp_path)
    key = h.manager.open(h.token(budgets=budgets(wall_s=60)))["episode_key"]
    bash(h, key, 0, 0, "write /work/answer.txt 42")
    with h.manager._exclusive("s-1"):
        h.clock.now += 61
        h.manager.reap()
        assert h.manager.store.get("s-1")["status"] == RUNNING


def test_deadline_passing_during_grading_does_not_expire_the_episode(tmp_path):
    h = build(tmp_path)
    key = h.manager.open(h.token(budgets=budgets(wall_s=60)))["episode_key"]
    bash(h, key, 0, 0, "write /work/answer.txt 42")
    grade = h.manager.grader.grade

    def slow_grade(*args, **kwargs):
        h.clock.now += 120
        h.manager.reap()
        return grade(*args, **kwargs)

    h.manager.grader.grade = slow_grade
    final = h.manager.finish("s-1", key)["record"]["body"]
    assert final["status"] == "graded" and final["reward"] == 1.0
    assert h.verify("s-1").ok


# -- restart recovery --------------------------------------------------------


def test_restart_keeps_running_episodes_with_live_boxes(h_default):
    h = h_default
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "echo a")
    h2 = restarted(h)
    h2.manager.reconcile()
    assert bash(h2, key, 0, 1, "echo b")["record"]["body"]["i"] == 2


def test_restart_restores_the_box_user_before_any_call(h_default):
    h = h_default
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "echo a")
    persisted = h.manager.store.get("s-1")
    h2 = restarted(h)
    assert h.boxes.users == {}  # the new runtime knows no box user yet
    h2.manager.reconcile()
    assert h.boxes.restored == [("s-1", persisted["box_user"], persisted["box_user_name"])]
    assert h.boxes.users["s-1"] == persisted["box_user"] == "0:0"
    bash(h2, key, 0, 1, "write /work/answer.txt 42")
    assert h2.manager.finish("s-1", key)["record"]["body"]["reward"] == 1.0


def test_without_reconcile_a_restarted_controller_cannot_run_calls(h_default):
    """What reconcile prevents: the runtime refuses a box whose user it never resolved."""
    h = h_default
    key = h.open()["episode_key"]
    h2 = restarted(h)
    with pytest.raises(HTTPException) as error:
        bash(h2, key, 0, 0, "echo a")
    assert error.value.detail["final"]["body"]["status"] == "aborted"


def test_restart_aborts_an_episode_whose_box_user_was_never_persisted(h_default):
    h = h_default
    key = h.open()["episode_key"]
    set_fields(h, "s-1", box_user=None)
    h2 = restarted(h)
    h2.manager.reconcile()
    assert h2.manager.transcript("s-1", key)["final_status"] == "aborted"
    assert h2.verify("s-1", require_graded=False).ok


def test_restart_aborts_episodes_interrupted_mid_call_or_with_lost_boxes(tmp_path):
    h = build(tmp_path)
    key1 = h.open("s-1")["episode_key"]
    key2 = h.open("s-2")["episode_key"]
    set_fields(h, "s-1", pending={"turn": 0, "k": 0})
    h.boxes.boxes.pop("s-2")
    h.boxes.create("orphan-g0", "gone", None)
    h2 = restarted(h)
    h2.manager.reconcile()
    t1, t2 = h2.manager.transcript("s-1", key1), h2.manager.transcript("s-2", key2)
    assert t1["final_status"] == "aborted" and t2["final_status"] == "aborted"
    assert "during a call" in t1["transcript"]["records"][-1]["body"]["reason"]
    assert "box lost" in t2["transcript"]["records"][-1]["body"]["reason"]
    assert h2.verify("s-1", require_graded=False).ok
    assert h2.verify("s-2", require_graded=False).ok
    assert "orphan-g0" not in h.boxes.boxes and "s-1" not in h.boxes.boxes
    assert h2.manager.active_count() == 0


def test_restart_aborts_a_box_the_agent_killed(h_default):
    """An exited box at restart is `aborted`: after a restart the agent killing
    it cannot be told from a host or daemon restart stopping it."""
    h = h_default
    key = h.open()["episode_key"]
    h.boxes.dead.add("s-1")  # exited, like the real runtime reports it
    h2 = restarted(h)
    h2.manager.reconcile()
    final = h2.manager.transcript("s-1", key)["transcript"]["records"][-1]["body"]
    assert (final["status"], final["reason"]) == ("aborted", "task box lost")


def test_restart_aborts_grading_in_progress_and_removes_its_boxes(h_default):
    h = h_default
    key = h.open()["episode_key"]
    set_fields(h, "s-1", status=FINISHING)
    h.boxes.create(Grader.pristine_box_id("s-1", 0), "s-1", None)
    h2 = restarted(h)
    h2.manager.reconcile()
    final = h2.manager.transcript("s-1", key)["transcript"]["records"][-1]["body"]
    assert final["status"] == "aborted" and "grading" in final["reason"]
    assert h.boxes.boxes == {}
    assert h2.manager.finish("s-1", key)["record"]["body"] == final


def test_restart_settles_an_interrupted_open_without_records(h_default):
    h = h_default
    h.open()
    set_fields(h, "s-1", status=OPENING)
    h2 = restarted(h)
    h2.manager.reconcile()
    episode = h2.manager.store.get("s-1")
    assert episode["status"] == CLOSED and episode["final_status"] == "open_failed"
    assert episode["closed_at"] is not None
    assert "s-1" not in h.boxes.boxes and h2.manager.active_count() == 0
    with pytest.raises(HTTPException) as error:  # the token stays spent
        h2.manager.open(h.token())
    assert error.value.status_code == 409


def test_restart_expires_running_episodes_past_their_deadline(tmp_path):
    h = build(tmp_path)
    key = h.manager.open(h.token(budgets=budgets(wall_s=60)))["episode_key"]
    h.clock.now += 61
    h2 = restarted(h)
    h2.manager.reconcile()
    assert h2.manager.transcript("s-1", key)["final_status"] == "expired"
    assert h.boxes.restored == []  # nothing to serve: no user restored


def test_restart_retries_pending_box_removals(h_default):
    h = h_default
    key = h.open()["episode_key"]
    h.boxes.fail_remove = True
    h.manager.finish("s-1", key)
    h.boxes.fail_remove = False
    h2 = restarted(h)
    h2.manager.reconcile()
    assert h2.manager.store.get("s-1")["cleanup"] == [] and h.boxes.boxes == {}


def test_a_call_retried_after_a_restart_returns_the_stored_record(h_default):
    h = h_default
    key = h.open()["episode_key"]
    first = bash(h, key, 0, 0, "burn 40")["record"]
    h2 = restarted(h)
    h2.manager.reconcile()
    again = bash(h2, key, 0, 0, "burn 40")["record"]
    assert again == first
    assert h.boxes.boxes["s-1"]["cpu"] == 40_000_000


def test_cpu_accounting_continues_across_a_restart(h_default):
    h = h_default
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "burn 40")
    h.boxes.boxes["s-1"]["cpu"] += 5_000_000  # a background process, while we restart
    h2 = restarted(h)
    h2.manager.reconcile()
    assert bash(h2, key, 1, 0, "burn 10")["record"]["body"]["cpu_ms"] == 15


def test_an_honest_episode_interrupted_by_a_restart_still_verifies(h_default):
    h = h_default
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "burn 20")
    h2 = restarted(h)
    h2.manager.reconcile()
    bash(h2, key, 0, 1, "write /work/answer.txt 42")
    h3 = restarted(h2)
    h3.manager.reconcile()
    bash(h3, key, 1, 0, "cat /work/answer.txt")
    final = h3.manager.finish("s-1", key)["record"]["body"]
    assert final["status"] == "graded" and final["reward"] == 1.0 and final["cpu_total_ms"] == 20
    result = h3.verify("s-1")
    assert result.ok, result.reasons
    assert [c.output for c in result.calls][-1] == "42\n"


def test_an_honest_episode_without_a_restart_verifies(h_default):
    h = h_default
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "write /work/answer.txt 42")
    bash(h, key, 0, 1, "cat /work/answer.txt")
    h.manager.finish("s-1", key)
    result = h.verify("s-1")
    assert result.ok, result.reasons
    assert len(result.calls) == 2 and result.final.reward == 1.0


def test_a_failing_cleanup_step_does_not_stop_the_others_and_is_reported(tmp_path, monkeypatch):
    h = build(tmp_path)
    key = h.manager.open(h.token(budgets=budgets(wall_s=60)))["episode_key"]
    h.boxes.create("stray-box", "long-gone", None)

    def broken(now, retention_s):
        raise OSError("database is locked")

    monkeypatch.setattr(h.manager.store, "prune", broken)
    h.clock.now += 61
    with pytest.raises(RuntimeError, match="incomplete"):
        h.manager.reap()
    assert h.manager.transcript("s-1", key)["final_status"] == "expired"
    assert "stray-box" not in h.boxes.boxes


def test_a_recovery_that_raises_aborts_that_episode_only(tmp_path, monkeypatch):
    h = build(tmp_path)
    key1 = h.open("s-1")["episode_key"]
    key2 = h.open("s-2")["episode_key"]
    set_fields(h, "s-1", box_user="not-a-uid")
    h2 = restarted(h)
    restore = h.boxes.restore_user

    def strict(box_id, uid_gid, spec_user=None):
        if uid_gid == "not-a-uid":
            raise ValueError("uid:gid expected")
        restore(box_id, uid_gid, spec_user)

    monkeypatch.setattr(h.boxes, "restore_user", strict)
    h2.manager.reconcile()
    assert h2.manager.transcript("s-1", key1)["final_status"] == "aborted"
    assert h2.manager.transcript("s-2", key2)["status"] == RUNNING
    assert h2.verify("s-1", require_graded=False).ok


def test_a_finish_whose_close_failed_is_settled_by_the_next_reap(tmp_path):
    """The store fails while finish signs its final (twice: the close and the
    abort after it); the episode stays FINISHING with nobody grading it. The
    next sweep sees its lock free, so it is orphaned, and closes it aborted."""
    h = build(tmp_path, episode_capacity=1)
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "write /work/answer.txt 42")
    store = h.manager.store
    real = store.commit
    failures = []

    def flaky(episode, record=None, state=None):
        if record is not None and record["body"]["kind"] == "final" and len(failures) < 2:
            failures.append(record)
            raise OSError("disk I/O error")
        return real(episode, record, state)

    store.commit = flaky
    with pytest.raises(OSError):
        h.manager.finish("s-1", key)
    assert store.get("s-1")["status"] == FINISHING and len(failures) == 2
    h.manager.reap()
    final = h.manager.transcript("s-1", key)
    assert final["final_status"] == "aborted"
    assert "grading interrupted" in final["transcript"]["records"][-1]["body"]["reason"]
    assert h.manager.active_count() == 0
    assert h.verify("s-1", require_graded=False).ok
    h.open("s-2")  # the slot is free


def test_reaper_leaves_a_finish_in_progress_alone(h_default):
    h = h_default
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "write /work/answer.txt 42")
    grade = h.manager.grader.grade
    seen = []

    def grade_with_a_sweep(*args, **kwargs):
        h.manager.reap()
        seen.append(h.manager.store.get("s-1")["status"])
        return grade(*args, **kwargs)

    h.manager.grader.grade = grade_with_a_sweep
    assert h.manager.finish("s-1", key)["record"]["body"]["status"] == "graded"
    assert seen == [FINISHING]


def test_an_episode_left_over_its_cpu_budget_by_a_crash_ends_budget_exhausted(tmp_path):
    """A crash between a call's commit and its budget close: after the restart
    the episode is over budget; it must not be graded."""
    h = build(tmp_path)
    key = h.manager.open(h.token(budgets=budgets(cpu_s=1)))["episode_key"]
    set_fields(h, "s-1", cpu_calls_ms=1500)
    h2 = restarted(h)
    h2.manager.reconcile()
    body = h2.manager.finish("s-1", key)["record"]["body"]
    assert (body["status"], body["reason"]) == ("budget_exhausted", "cpu_s")
    assert h2.verify("s-1", require_graded=False).ok


def test_a_lock_being_taken_is_never_pruned_from_under_it(h_default):
    """`_exclusive` takes the lock while holding the table lock, so the pruner
    never deletes a lock between its lookup and its acquisition."""
    h = h_default
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "write /work/answer.txt 42")
    h.manager.finish("s-1", key)
    with h.manager._exclusive("s-1"):
        h.manager._prune_locks()
        assert "s-1" in h.manager.episode_locks
        with pytest.raises(HTTPException) as error:
            with h.manager._exclusive("s-1"):
                pass
        assert error.value.status_code == 409
    h.manager._prune_locks()
    assert "s-1" not in h.manager.episode_locks


# -- machine key rotation ------------------------------------------------------


def rotated(h, tmp_path):
    """The same machine restarted with a new signing key (a new key id)."""
    from episode_fixtures import make_signer

    new_signer, _ = make_signer(tmp_path, "machine-2", "m-2")
    h2 = restarted(h)
    h2.manager.signer = new_signer
    return dataclasses.replace(h2, machine=new_signer)


def test_a_key_rotated_under_a_live_episode_voids_it_unsigned(tmp_path, caplog):
    h = build(tmp_path)
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "echo a")
    records = h.manager.store.records("s-1")
    h2 = rotated(h, tmp_path)
    with caplog.at_level("ERROR"):
        h2.manager.reconcile()
    episode = h2.manager.store.get("s-1")
    assert episode["status"] == CLOSED and episode["final_status"] == "aborted_unsigned"
    assert h2.manager.store.records("s-1") == records  # nothing signed with the new key
    assert h2.manager.active_count() == 0 and "s-1" not in h.boxes.boxes
    assert any("key" in r.getMessage() and r.levelname == "ERROR" for r in caplog.records)
    with pytest.raises(HTTPException) as error:
        h2.manager.finish("s-1", key)
    assert error.value.status_code == 409 and error.value.detail["final"] is None


def test_the_key_id_of_record_zero_is_persisted(tmp_path):
    h = build(tmp_path)
    h.open()
    assert h.manager.store.get("s-1")["key_id"] == h.machine.key_id


def test_a_call_after_a_key_change_is_never_signed(tmp_path):
    """Defence in depth if recovery missed it: a call or a close checks too."""
    h = build(tmp_path)
    key = h.open()["episode_key"]
    h2 = rotated(h, tmp_path)  # no reconcile
    h2.manager.boxes.restore_user("s-1", "0:0")
    with pytest.raises(HTTPException) as error:
        bash(h2, key, 0, 0, "echo a")
    assert error.value.status_code == 409 and error.value.detail["final"] is None
    assert [r["body"]["kind"] for r in h2.manager.store.records("s-1")] == ["open"]
    assert h2.manager.store.get("s-1")["final_status"] == "aborted_unsigned"
