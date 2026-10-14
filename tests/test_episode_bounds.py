"""Final-fix bounds: the task pinned across restarts, transcript size and
reads, record 0's time, and opens stuck without their opener."""

import json
import threading
import time

import pytest
from episode_fixtures import budgets
from episode_manager_fixtures import build, restarted
from fastapi import HTTPException

from reliquary_sandbox.episode_client import encode_body
from reliquary_sandbox_service.episodes.store import CLOSED, OPENING, RUNNING


def bash(h, key, turn, k, command, session="s-1"):
    return h.manager.call(session, key, turn, k, "bash", {"command": command})


def pin_package(h, version):
    h.manager.registry._packages["fake-env"] = f"fake-env=={version}"


# -- the task is pinned at open ------------------------------------------------


def test_the_task_fingerprint_is_persisted_at_open(tmp_path):
    h = build(tmp_path)
    h.open()
    assert len(h.manager.store.get("s-1")["task_fingerprint"]) == 64


@pytest.mark.parametrize("ending", ["call", "finish"])
def test_an_env_upgraded_across_a_restart_aborts_the_episode(tmp_path, ending):
    h = build(tmp_path)
    pin_package(h, "1")
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "write /work/answer.txt 42")
    h2 = restarted(h)
    pin_package(h2, "2")
    h2.manager.reconcile()
    if ending == "call":
        with pytest.raises(HTTPException) as error:
            bash(h2, key, 1, 0, "echo again")
        final = error.value.detail["final"]["body"]
    else:
        final = h2.manager.finish("s-1", key)["record"]["body"]
    assert (final["status"], final["reason"]) == ("aborted", "task_changed")
    assert h2.verify("s-1", require_graded=False).ok


def test_the_same_env_after_a_restart_goes_on(tmp_path):
    h = build(tmp_path)
    pin_package(h, "1")
    key = h.open()["episode_key"]
    h2 = restarted(h)
    pin_package(h2, "1")
    h2.manager.reconcile()
    bash(h2, key, 0, 0, "write /work/answer.txt 42")
    assert h2.manager.finish("s-1", key)["record"]["body"]["reward"] == 1.0


def test_a_changed_task_declaration_is_task_changed_too(tmp_path, monkeypatch):
    import dataclasses

    import fake_episode_env

    h = build(tmp_path)
    key = h.open()["episode_key"]
    h2 = restarted(h)
    original = fake_episode_env.sandbox_task
    monkeypatch.setitem(h2.manager.registry._factories, "fake-env",
                        lambda split, index: dataclasses.replace(original(split, index),
                                                                 grading_timeout_s=900))
    h2.manager.reconcile()
    final = h2.manager.finish("s-1", key)["record"]["body"]
    assert (final["status"], final["reason"]) == ("aborted", "task_changed")


# -- transcript bounds ---------------------------------------------------------


def test_records_are_stored_as_utf8_and_their_bytes_counted(tmp_path):
    h = build(tmp_path)
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "binary")  # U+FFFD in the output
    rows = h.manager.store.db.execute(
        "SELECT record FROM episode_records WHERE episode_id='s-1' ORDER BY i").fetchall()
    assert "�" in rows[1][0] and "\\ufffd" not in rows[1][0]
    assert h.manager.store.get("s-1")["transcript_bytes"] == compact_size(h, key)


def compact_size(h, key):
    """The transcript as a validator receives it: compact UTF-8 JSON (ensure_ascii=False)."""
    return len(encode_body(h.manager.transcript("s-1", key)["transcript"]))


def test_the_counted_bytes_are_the_compact_transcript_after_every_call(tmp_path):
    h = build(tmp_path)
    key = h.open()["episode_key"]
    assert h.manager.store.get("s-1")["transcript_bytes"] == compact_size(h, key)
    bash(h, key, 0, 0, "echo é \" \\ \t")
    bash(h, key, 1, 0, "echo two")
    assert h.manager.store.get("s-1")["transcript_bytes"] == compact_size(h, key)


def test_the_transcript_cap_ends_the_episode_while_the_final_still_fits(tmp_path):
    cap = 1024**2
    h = build(tmp_path, episode_max_transcript_bytes=cap)
    key = h.manager.open(h.token(budgets=budgets(max_calls=64)))["episode_key"]
    k = 0
    while True:
        response = bash(h, key, 0, k, "big 70000")
        if response["final"] is not None:
            break
        k += 1
    final = response["final"]["body"]
    assert (final["status"], final["reason"]) == ("budget_exhausted", "transcript_bytes")
    assert response["record"]["body"]["k"] == k and k >= 3
    # The closing call left no room for a worst-case call and final; the whole signed
    # transcript, final included, still fits the cap a validator enforces.
    assert compact_size(h, key) <= cap
    assert h.verify("s-1", require_graded=False).ok


def test_the_final_is_read_without_the_whole_transcript(tmp_path, monkeypatch):
    h = build(tmp_path)
    key = h.open()["episode_key"]
    final = h.manager.finish("s-1", key)["record"]

    def everything(*args):
        raise AssertionError("the whole transcript was read")

    monkeypatch.setattr(h.manager.store, "records", everything)
    assert h.manager.finish("s-1", key)["record"] == final
    with pytest.raises(HTTPException) as error:
        bash(h, key, 0, 0, "echo late")
    assert error.value.detail["final"] == final


def test_a_transcript_read_does_not_hold_the_store_while_it_pages(tmp_path, monkeypatch):
    h = build(tmp_path)
    key1 = h.manager.open(h.token(budgets=budgets(max_calls=64)))["episode_key"]
    for k in range(40):
        bash(h, key1, 0, k, "echo x")
    key2 = h.open("s-2")["episode_key"]
    store, paused, release = h.manager.store, threading.Event(), threading.Event()
    page = store._record_page

    def slow_page(*args):
        rows = page(*args)
        paused.set()
        assert release.wait(10)
        return rows

    monkeypatch.setattr(store, "_record_page", slow_page)
    reader = threading.Thread(target=h.manager.transcript, args=("s-1", key1))
    reader.start()
    try:
        assert paused.wait(5)
        started = time.monotonic()
        bash(h, key2, 0, 0, "echo other", session="s-2")  # another episode's call is not blocked
        assert time.monotonic() - started < 1
        with pytest.raises(HTTPException) as error:  # one read at a time per episode
            h.manager.transcript("s-1", key1)
        assert error.value.status_code == 409
    finally:
        release.set()
        reader.join()
    monkeypatch.setattr(store, "_record_page", page)
    records = h.manager.transcript("s-1", key1)["transcript"]["records"]
    assert [r["body"]["i"] for r in records] == list(range(41))


def test_the_transcript_is_a_consistent_snapshot_of_committed_records(tmp_path):
    h = build(tmp_path)
    key = h.open()["episode_key"]
    bash(h, key, 0, 0, "echo a")
    data = json.loads(h.manager.transcript_json("s-1", key))
    assert data["status"] == RUNNING and len(data["transcript"]["records"]) == 2
    assert data == h.manager.transcript("s-1", key)


# -- item 2: record 0 carries the time preparation ended -----------------------


def test_record_zero_uses_the_prepared_time(tmp_path):
    class Ticking:
        def __init__(self):
            self.now = 1_800_000_000.0

        def __call__(self):
            self.now += 1
            return self.now

    h = build(tmp_path, clock=Ticking())
    body = h.manager.open(h.token(budgets=budgets(wall_s=120)))["record"]["body"]
    episode = h.manager.store.get("s-1")
    assert body["at"] == episode["deadline"] - 120  # the same `prepared` reading


# -- item 8: an open left OPENING without its opener ---------------------------


def stuck_open(h, age_s):
    h.open()
    episode = h.manager.store.get("s-1")
    episode.update(status=OPENING, open_by=h.clock.now - age_s)
    h.manager.store.commit(episode)


def test_the_reaper_settles_an_open_nobody_is_finishing(tmp_path):
    h = build(tmp_path, episode_capacity=1)
    stuck_open(h, age_s=1)
    h.manager.reap()
    episode = h.manager.store.get("s-1")
    assert episode["status"] == CLOSED and episode["final_status"] == "open_failed"
    assert h.manager.active_count() == 0 and "s-1" not in h.boxes.boxes
    h.open("s-2")  # the slot is free


def test_the_reaper_leaves_an_open_still_in_time_alone(tmp_path):
    h = build(tmp_path)
    stuck_open(h, age_s=-60)
    h.manager.reap()
    assert h.manager.store.get("s-1")["status"] == OPENING


def test_the_reaper_leaves_an_open_whose_opener_holds_the_lock(tmp_path):
    h = build(tmp_path)
    stuck_open(h, age_s=1)
    with h.manager._exclusive("s-1"):
        h.manager.reap()
    assert h.manager.store.get("s-1")["status"] == OPENING


def test_an_open_holds_its_episode_lock_while_preparing(tmp_path):
    h = build(tmp_path)
    h.boxes.create_delay = 0.5
    opener = threading.Thread(target=h.open)
    opener.start()
    try:
        for _ in range(200):
            if h.manager.store.get("s-1") is not None:
                break
            time.sleep(0.005)
        with pytest.raises(HTTPException) as error:
            with h.manager._exclusive("s-1"):
                pass
        assert error.value.status_code == 409
    finally:
        opener.join()
    assert h.manager.store.get("s-1")["status"] == RUNNING
