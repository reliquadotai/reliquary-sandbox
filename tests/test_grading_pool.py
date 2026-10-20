"""The grading pool: grading boxes run inside it, entered after
extract and served fairly by hotkey; one that cannot start in time is aborted by the machine, with a reason that
says whether the miner's own gradings held the pool."""

import dataclasses
import threading
import time
from types import SimpleNamespace

import pytest
from episode_fixtures import IMAGE, budgets
from episode_manager_fixtures import build
from fake_boxes import FakeBoxes
from fake_episode_env import sandbox_task
from fastapi import HTTPException

from reliquary_sandbox.episode_client import decode_open_refusal
from reliquary_sandbox_service.episodes.capacity import capacity_report
from reliquary_sandbox_service.episodes.grader import Grader
from reliquary_sandbox_service.episodes.resources import (
    GRADING_POOL_FULL,
    GRADING_POOL_OWN,
    GradingPool,
    Reservation,
)
from reliquary_sandbox_service.task_runtime import BoxSpec

GIB = 1024**3
LIMITED = {"fake-env": "limited_episode_env:sandbox_task"}
SHARE = Reservation(600, GIB)


def until(predicate, seconds=5.0):
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "condition never met"
        time.sleep(0.005)


def test_the_pool_admits_what_fits_and_waits_for_the_rest():
    pool = GradingPool(Reservation(1_000, 4 * GIB))
    assert pool.acquire(SHARE, 0, "a") is None
    assert pool.acquire(SHARE, 0, "b") == GRADING_POOL_FULL               # 1200 > 1000; b holds nothing
    released = threading.Timer(0.05, pool.release, args=(SHARE, "a"))
    released.start()
    assert pool.acquire(SHARE, 2.0, "b") is None                            # room after the release
    snapshot = pool.snapshot()
    # two acquires waited (the zero-timeout one and the one the release served), one timed out
    assert (snapshot["used_cpu_millis"], snapshot["waits"], snapshot["timeouts"]) == (600, 2, 1)


def test_a_box_larger_than_the_pool_never_waits():
    pool = GradingPool(Reservation(1_000, GIB))
    started = time.monotonic()
    assert pool.acquire(Reservation(2_000, GIB), 5.0, "a") == GRADING_POOL_FULL
    assert time.monotonic() - started < 1.0 and pool.snapshot()["timeouts"] == 1


def test_restoring_the_same_cleanup_hold_twice_counts_it_once():
    grader = Grader(SimpleNamespace(), FakeBoxes([IMAGE]))
    grader.pool = GradingPool(Reservation(1_000, 4 * GIB))
    for _ in range(2):
        grader.retain_cleanup("leftover", SHARE, "owner", restore=True)
    assert grader.pool.snapshot()["used_cpu_millis"] == SHARE.cpu_millis
    assert grader.pool.acquire(SHARE, 0, "other") == GRADING_POOL_FULL
    for _ in range(2):
        grader.release_cleanup("leftover")
    assert grader.pool.snapshot()["used_cpu_millis"] == 0


def test_a_timeout_says_whose_gradings_held_the_pool():
    pool = GradingPool(Reservation(1_000, 4 * GIB))
    half = Reservation(500, GIB)
    assert pool.acquire(half, 0, "burst") is None and pool.acquire(half, 0, "burst") is None
    assert pool.acquire(half, 0, "burst") == GRADING_POOL_OWN               # its own gradings fill the pool
    assert pool.acquire(half, 0, "victim") == GRADING_POOL_FULL             # someone else's fill it


def test_holding_exactly_half_the_pool_is_not_own():
    pool = GradingPool(Reservation(1_000, 4 * GIB))
    half = Reservation(500, GIB)
    assert pool.acquire(half, 0, "heavy") is None and pool.acquire(half, 0, "hanging") is None
    assert pool.acquire(half, 0, "heavy") == GRADING_POOL_FULL               # half its own, half another's


def test_a_burst_of_one_hotkey_queues_behind_the_other_hotkeys():
    pool = GradingPool(Reservation(1_000, 4 * GIB))
    small = Reservation(400, GIB)
    assert pool.acquire(small, 0, "burst") is None and pool.acquire(small, 0, "burst") is None
    results = {}

    def wait(name, owner):
        results[name] = pool.acquire(small, 5.0, owner)

    burst = threading.Thread(target=wait, args=("burst-3", "burst"))
    burst.start()
    until(lambda: pool.snapshot()["waiting"] == 1)
    other = threading.Thread(target=wait, args=("other", "other"))
    other.start()
    until(lambda: pool.snapshot()["waiting"] == 2)
    pool.release(small, "burst")      # burst still holds 400, other 0: other goes first, although it came second
    other.join(5.0)
    assert results == {"other": None}
    pool.release(small, "burst")
    burst.join(5.0)
    assert results == {"other": None, "burst-3": None}


def harness(tmp_path, **overrides):
    values = dict(episode_envs=LIMITED, episode_nano_cpus=2_000_000_000, episode_capacity=8,
                  episode_cpu_millis_total=4_000, episode_memory_bytes_total=16 * GIB,
                  episode_grading_cpu_millis_total=2_000, episode_grading_memory_bytes_total=8 * GIB,
                  episode_present_window_s=120)
    values.update(overrides)
    return build(tmp_path, **values)


def test_grading_runs_in_the_pool_and_gives_its_room_back(tmp_path):
    h = harness(tmp_path)
    key = h.open("s-1", index=7, budgets=budgets(memory_bytes=3 * GIB))["episode_key"]
    final = h.manager.finish("s-1", key)["record"]["body"]
    assert final["status"] == "graded"
    assert h.manager.grading_pool.snapshot()["used_cpu_millis"] == 0


def test_the_pool_is_taken_after_extract_by_the_sessions_hotkey(tmp_path):
    h = harness(tmp_path)
    order, owners = [], []
    real_extract, real_acquire = h.manager.grader.extract, h.manager.grading_pool.acquire

    def extract(*args, **kwargs):
        order.append("extract")
        return real_extract(*args, **kwargs)

    def acquire(need, timeout_s, owner=""):
        order.append("acquire")
        owners.append(owner)
        return real_acquire(need, timeout_s, owner)

    h.manager.grader.extract = extract
    h.manager.grading_pool.acquire = acquire
    key = h.open("s-1", index=7, budgets=budgets(memory_bytes=3 * GIB))["episode_key"]
    h.manager.finish("s-1", key)
    assert order[:2] == ["extract", "acquire"]
    assert owners[0] == h.manager.store.get("s-1")["claims"]["hotkey"]


def test_a_full_pool_aborts_the_grading_as_the_machines_doing(tmp_path):
    h = harness(tmp_path)
    key = h.open("s-1", index=7, budgets=budgets(memory_bytes=3 * GIB))["episode_key"]
    # Already-open episodes keep their established finish behavior if the pool changes.
    h.manager.grading_pool.total = Reservation(100, 8 * GIB)
    final = h.manager.finish("s-1", key)["record"]["body"]
    assert final["status"] == "aborted" and final["reason"] == GRADING_POOL_FULL


@pytest.mark.parametrize("limits", [dict(episode_grading_cpu_millis_total=100),
                                   dict(episode_grading_memory_bytes_total=2 * GIB)])
def test_a_task_that_cannot_ever_fit_the_grading_pool_is_refused_before_work(tmp_path, limits):
    h = harness(tmp_path, **limits)
    token = h.token("s-1", index=7, budgets=budgets(memory_bytes=3 * GIB))
    with pytest.raises(HTTPException) as error:
        h.manager.open(token)
    assert error.value.status_code == 422 and "grading pool" in error.value.detail
    receipt = decode_open_refusal(error.value.headers["X-Reliquary-Open-Refusal"])
    assert receipt["document"]["reason"] == "task_refused"
    assert h.boxes.boxes == {} and h.manager.active_count() == 0
    with pytest.raises(HTTPException) as repeated:
        h.manager.open(token)
    assert repeated.value.status_code == 409
    assert repeated.value.headers == error.value.headers


def test_the_report_shows_the_pool(tmp_path):
    h = harness(tmp_path)
    h.manager.host_facts = h.boxes.preflight()
    block = capacity_report(h.manager)["document"]["grading_pool"]
    assert (block["cpu_millis"], block["memory_bytes"], block["used_cpu_millis"]) == (2_000, 8 * GIB, 0)


def test_without_a_pool_grading_is_unchanged(tmp_path):
    h = build(tmp_path, episode_envs=LIMITED, episode_capacity=2)
    assert h.manager.grading_pool is None and h.manager.grader.pool is None
    key = h.open("s-1", index=7, budgets=budgets(memory_bytes=3 * GIB))["episode_key"]
    assert h.manager.finish("s-1", key)["record"]["body"]["status"] == "graded"


def test_a_grading_box_that_could_not_be_removed_stops_the_retries():
    spec = BoxSpec(image=IMAGE, workdir="/work", env={}, user=None, memory_bytes=GIB, nano_cpus=500_000_000,
                   pids=16, disk_bytes=1, lifetime_s=60)
    boxes = FakeBoxes([IMAGE])
    boxes.create("s-1", "s-1", spec)
    boxes.write_file("s-1", "/work/answer.txt", b"42\n")
    calls = []

    async def flaky(runtime, state):
        calls.append(runtime.box_id)
        boxes.broken.add(runtime.box_id)          # the box vanished under us: a retryable failure of ours
        boxes.fail_remove = True                  # and it cannot be removed either
        await runtime.run(["check"], {})

    grader = Grader(SimpleNamespace(episode_grader_output_bytes=1024, episode_grading_attempts=2), boxes)
    grader.pool = GradingPool(Reservation(1_000, 4 * GIB))
    task = dataclasses.replace(sandbox_task("train", 0), grade=flaky)
    outcome = grader.grade("s-1", "s-1", task, spec, env="fake-env", owner="a")
    assert outcome.status == "aborted" and "could not be removed" in outcome.reason
    assert calls == [Grader.pristine_box_id("s-1", 0)]                     # no second box beside it
    assert outcome.cleanup == (Grader.pristine_box_id("s-1", 0),)
    assert grader.pool.snapshot()["used_cpu_millis"] == 500                # the box still holds it
    grader.release_cleanup(outcome.cleanup[0])
    grader.release_cleanup(outcome.cleanup[0])                             # duplicate cleanup is harmless
    assert grader.pool.snapshot()["used_cpu_millis"] == 0
