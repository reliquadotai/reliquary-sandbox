import threading
import time

import pytest

from reliquary_sandbox_service.episodes.store import CLOSED, RUNNING, EpisodeStore
from reliquary_sandbox_service.store import Store


def episode(episode_id="s-1", token="t1", status=RUNNING, expires_at=100.0, **extra):
    return {"id": episode_id, "token_sha256": token, "status": status,
            "claims": {"expires_at": expires_at}, **extra}


def record(i, turn=None, k=None):
    return {"body": {"i": i, "turn": turn, "k": k}, "signature": "x"}


def test_tokens_and_sessions_are_single_use(tmp_path):
    store = EpisodeStore(Store(tmp_path))
    assert store.insert(episode())
    assert not store.insert(episode(token="t2"))
    assert not store.insert(episode("s-2", token="t1"))


def test_records_commit_with_the_episode_and_keep_order(tmp_path):
    store = EpisodeStore(Store(tmp_path))
    store.insert(episode())
    state = episode(next_i=1)
    store.commit(state, record(0))
    store.commit(state, record(1, 0, 0))
    store.commit(state, record(2, 0, 1))
    assert [r["body"]["i"] for r in store.records("s-1")] == [0, 1, 2]
    assert store.call_record("s-1", 0, 1)["body"]["i"] == 2
    assert store.call_record("s-1", 1, 0) is None


def test_state_survives_reopening_the_database(tmp_path):
    store = Store(tmp_path)
    EpisodeStore(store).insert(episode())
    store.close()
    reopened = EpisodeStore(Store(tmp_path))
    assert reopened.get("s-1")["status"] == RUNNING
    assert [e["id"] for e in reopened.active()] == ["s-1"]


def test_prune_uses_closed_at_plus_retention_and_keeps_pending_cleanup(tmp_path):
    store = EpisodeStore(Store(tmp_path))
    store.insert(episode("old", "t1", CLOSED, closed_at=100.0))
    store.insert(episode("fresh", "t2", CLOSED, closed_at=990.0, expires_at=10))
    store.insert(episode("dirty", "t3", CLOSED, closed_at=100.0, cleanup=["dirty"]))
    store.commit(episode("old", "t1", CLOSED, closed_at=100.0), record(0))
    store.commit(episode("fresh", "t2", CLOSED, closed_at=990.0, expires_at=10), record(0))
    assert store.prune(now=1000.0, retention_s=500.0) == 1
    assert store.get("old") is None and store.records("old") == []
    assert store.get("fresh") and store.get("dirty")
    assert [e["id"] for e in store.with_cleanup()] == ["dirty"]


def test_a_closed_episode_is_kept_while_its_token_is_valid(tmp_path):
    """Pruning a row whose token could still be presented would make it reusable."""
    store = EpisodeStore(Store(tmp_path))
    store.insert(episode("a", "t1", CLOSED, closed_at=100.0, expires_at=5000.0))
    store.commit(episode("a", "t1", CLOSED, closed_at=100.0, expires_at=5000.0), record(0))
    assert store.prune(now=1000.0, retention_s=500.0) == 0
    assert store.prune(now=5001.0, retention_s=500.0) == 1


def test_just_closed_episode_survives_even_with_an_expired_token(tmp_path):
    store = EpisodeStore(Store(tmp_path))
    store.insert(episode("a", "t1", RUNNING, expires_at=10))
    store.commit(episode("a", "t1", CLOSED, expires_at=10), record(0))
    assert store.prune(now=time.time() + 1, retention_s=3600) == 0


def test_closed_episode_that_never_recorded_goes_once_its_token_expires(tmp_path):
    store = EpisodeStore(Store(tmp_path))
    store.insert(episode("a", "t1", CLOSED, closed_at=999.0, expires_at=10))
    assert store.prune(now=1000.0, retention_s=500.0) == 1


def test_commit_of_an_unknown_episode_fails_and_writes_nothing(tmp_path):
    store = EpisodeStore(Store(tmp_path))
    with pytest.raises(KeyError):
        store.commit(episode("ghost"), record(0))
    assert store.records("ghost") == []


def test_concurrent_double_insert_of_a_token_has_one_winner(tmp_path):
    store = EpisodeStore(Store(tmp_path))
    results = []
    barrier = threading.Barrier(2)

    def attempt(episode_id):
        barrier.wait()
        results.append(store.insert(episode(episode_id, token="same")))

    threads = [threading.Thread(target=attempt, args=(f"s-{n}",)) for n in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(results) == [False, True]


def test_the_published_state_commits_with_the_final_record_and_is_pruned_with_it(tmp_path):
    store = EpisodeStore(Store(tmp_path))
    store.insert(episode("a", "t1", RUNNING))
    store.commit(episode("a", "t1", CLOSED, closed_at=100.0), record(0), b"\x00state")
    assert store.state("a") == b"\x00state" and store.state("b") is None
    assert "state" not in store.get("a")
    assert store.prune(now=1000.0, retention_s=500.0) == 1
    assert store.state("a") is None
