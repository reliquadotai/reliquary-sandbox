"""CPU and memory reservations of episodes: each episode reserves its
agent box (the task's CPUs and the token's memory); the grading box draws from the grading pool.
An open that does not fit is refused before its token is spent; a task asking for more CPUs per box than
the machine gives is refused, never clamped, when the machine accounts resources."""

import pytest
from episode_fixtures import budgets
from episode_manager_fixtures import build, restarted
from fastapi import HTTPException

from reliquary_sandbox_service.episodes.capacity import capacity_report
from reliquary_sandbox_service.episodes.resources import (
    BOXES_PER_EPISODE,
    Reservation,
    in_use,
    resources_report,
)
from reliquary_sandbox_service.episodes.store import CLOSED, OPENING

GIB = 1024**3
LIMITED = {"fake-env": "limited_episode_env:sandbox_task"}


def harness(tmp_path, **overrides):
    values = dict(episode_envs=LIMITED, episode_nano_cpus=2_000_000_000, episode_capacity=8,
                  episode_cpu_millis_total=1_000, episode_memory_bytes_total=16 * GIB,
                  episode_grading_cpu_millis_total=2_000, episode_grading_memory_bytes_total=8 * GIB,
                  episode_present_window_s=120)
    values.update(overrides)
    return build(tmp_path, **values)


def open_(h, session_id, index=7, memory=3 * GIB):
    return h.open(session_id, index=index, budgets=budgets(memory_bytes=memory))


def refused(h, session_id, **kwargs):
    with pytest.raises(HTTPException) as error:
        open_(h, session_id, **kwargs)
    row = h.manager.store.get(session_id)       # a signed refusal leaves a tombstone row
    assert (row is None or row.get("final_status") == "refused") and session_id not in h.boxes.boxes
    return error.value


def test_two_half_cpu_tasks_fill_one_cpu_and_the_third_waits(tmp_path):
    h = harness(tmp_path)
    open_(h, "s-1")
    open_(h, "s-2")                     # 2 x (1 box x 500 millis) = 1000
    error = refused(h, "s-3")
    assert error.status_code == 503 and error.detail == "no free resources"
    assert error.headers["Retry-After"] == "5"


def test_the_overcommit_multiplies_the_reservable_cpu(tmp_path):
    h = harness(tmp_path, episode_cpu_overcommit=2.0)
    for n in range(4):                  # 4 x 500 = 2000 = 1000 x 2
        open_(h, f"s-{n}")
    assert refused(h, "s-4").status_code == 503


def test_memory_bounds_the_machine_too(tmp_path):
    h = harness(tmp_path, episode_cpu_millis_total=64_000, episode_memory_bytes_total=5 * GIB)
    open_(h, "s-1")                     # 1 box x 3 GiB
    assert refused(h, "s-2").status_code == 503


def test_a_task_asking_more_cpus_than_a_box_gives_is_refused_not_clamped(tmp_path):
    h = harness(tmp_path, episode_cpu_millis_total=64_000)
    error = refused(h, "s-1", index=8)  # 8 CPUs asked, boxes of 2 CPUs
    assert error.status_code == 422 and "more CPUs per box" in error.detail
    open_(h, "s-2")                     # the machine is still free


def test_without_accounting_the_box_size_is_still_clamped(tmp_path):
    h = build(tmp_path, episode_envs=LIMITED, episode_nano_cpus=2_000_000_000, episode_capacity=2)
    open_(h, "s-1", index=8)            # legacy machines keep today's clamp


def test_a_task_that_can_never_fit_is_refused_without_spending_the_token(tmp_path):
    h = harness(tmp_path, episode_memory_bytes_total=2 * GIB)
    error = refused(h, "s-1")           # 3 GiB > 2 GiB
    assert error.status_code == 422 and "more CPU or memory" in error.detail


def test_a_finished_episode_frees_its_reservation(tmp_path):
    h = harness(tmp_path)
    key = open_(h, "s-1")["episode_key"]
    open_(h, "s-2")
    h.manager.finish("s-1", key)
    open_(h, "s-3")


def _abort(h, key):                 # the box vanished under us: the call aborts the episode
    h.boxes.broken.add("s-1")
    with pytest.raises(HTTPException) as error:
        h.manager.call("s-1", key, 0, 0, "bash", {"command": "echo x"})
    assert error.value.detail["final"]["body"]["status"] == "aborted"


def _box_failure(h, key):           # the agent killed its own box
    h.manager.call("s-1", key, 0, 0, "bash", {"command": "echo ok"})
    h.boxes.dead.add("s-1")
    with pytest.raises(HTTPException) as error:
        h.manager.call("s-1", key, 1, 0, "bash", {"command": "echo x"})
    assert error.value.detail["final"]["body"]["status"] == "box_failed"


def _expiry(h, key):                # its wall budget ran out; the reaper closes it
    h.clock.now += 61
    h.manager.reap()
    assert h.manager.store.get("s-1")["final_status"] == "expired"


def _open_left_opening(h, key):     # an open nobody finished: the reaper fails it
    episode = h.manager.store.get("s-1")
    episode.update(status=OPENING, open_by=h.clock.now - 1)
    h.manager.store.commit(episode)
    h.manager.reap()
    assert h.manager.store.get("s-1")["final_status"] == "open_failed"


@pytest.mark.parametrize("end", [_abort, _box_failure, _expiry, _open_left_opening],
                         ids=["abort", "box_failed", "expiry", "open_failed"])
def test_every_way_an_episode_ends_frees_its_reservation(tmp_path, end):
    """Not only `finish` frees the agent box's reservation."""
    h = harness(tmp_path)
    key = h.open("s-1", budgets=budgets(memory_bytes=3 * GIB, wall_s=60))["episode_key"]
    open_(h, "s-2")                     # the machine is full (2 x 500 millis)
    assert refused(h, "s-3").status_code == 503
    end(h, key)
    assert h.manager.store.get("s-1")["status"] == CLOSED
    open_(h, "s-5")                     # s-1's share is free again (s-3's token was spent by its refusal)
    assert refused(h, "s-4").status_code == 503     # and s-2's still held


def test_a_failed_open_holds_no_reservation(tmp_path):
    h = harness(tmp_path)
    open_(h, "s-0")
    h.boxes.broken.add("s-1")           # the box starts, then `prepare` fails in it
    with pytest.raises(HTTPException) as error:
        open_(h, "s-1")
    assert error.value.status_code == 503
    assert h.manager.store.get("s-1")["final_status"] == "open_failed"
    open_(h, "s-2")                     # s-1 left nothing behind
    assert refused(h, "s-3").status_code == 503


def test_a_restart_counts_the_live_rows(tmp_path):
    h = harness(tmp_path)
    open_(h, "s-1")
    open_(h, "s-2")
    again = restarted(h)
    assert refused(again, "s-3").status_code == 503


def test_the_report_carries_totals_use_free_and_the_box_size(tmp_path):
    h = harness(tmp_path, episode_cpu_overcommit=1.5)
    open_(h, "s-1")
    h.manager.host_facts = h.boxes.preflight()
    document = capacity_report(h.manager)["document"]
    assert document["resources"] == {
        "cpu_millis": 1_500, "memory_bytes": 16 * GIB, "used_cpu_millis": 500,
        "used_memory_bytes": 3 * GIB, "free_cpu_millis": 1_000, "free_memory_bytes": 13 * GIB,
        "max_box_cpu_millis": 2_000, "boxes_per_episode": BOXES_PER_EPISODE,
        "cpu_overcommit_permille": 1_500, "present_window_s": 120, "sessions": ["s-1"]}
    assert h.manager.store.get("s-1")["reserve"] == {"cpu_millis": 500, "memory_bytes": 3 * GIB}


def test_without_totals_the_report_and_the_opens_are_unchanged(tmp_path):
    h = build(tmp_path, episode_envs=LIMITED, episode_capacity=2)
    open_(h, "s-1")
    open_(h, "s-2", index=8)
    h.manager.host_facts = h.boxes.preflight()
    assert "resources" not in capacity_report(h.manager)["document"]
    assert resources_report(h.settings, h.manager.store.active()) is None


def test_a_row_without_a_stored_reservation_is_charged_the_largest_box(tmp_path):
    h = harness(tmp_path)
    rows = [{"claims": {"budgets": {"memory_bytes": GIB}}}]
    assert in_use(rows, h.settings) == Reservation(2_000, GIB)


@pytest.mark.parametrize("values", [{"episode_cpu_millis_total": 2_000},
                                    {"episode_cpu_millis_total": 2_000, "episode_memory_bytes_total": GIB}])
def test_the_totals_go_together_and_need_a_grading_pool(tmp_path, values):
    with pytest.raises(ValueError, match="go together|grading pool"):
        build(tmp_path, **values)


GRADING = {"episode_grading_cpu_millis_total": 2_000, "episode_grading_memory_bytes_total": GIB}


@pytest.mark.parametrize("window", [None, 121, 600])
def test_totals_need_a_present_window_of_at_most_120_s(tmp_path, window):
    with pytest.raises(ValueError, match="episode_present_window_s"):
        build(tmp_path, episode_cpu_millis_total=2_000, episode_memory_bytes_total=GIB,
              episode_present_window_s=window, **GRADING)


def test_the_window_is_free_without_totals_and_published_with_them(tmp_path):
    build(tmp_path, episode_envs=LIMITED, episode_capacity=2)
    build(tmp_path, episode_envs=LIMITED, episode_capacity=2, episode_present_window_s=600)
    h = harness(tmp_path, episode_present_window_s=45)
    h.manager.host_facts = h.boxes.preflight()
    assert capacity_report(h.manager)["document"]["resources"]["present_window_s"] == 45
