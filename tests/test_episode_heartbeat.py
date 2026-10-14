"""Signed capacity reports (served and pushed as heartbeats) and their pure check."""

import asyncio
import logging

import httpx
import pytest
from episode_fixtures import IMAGE, MACHINE, directory_for
from episode_manager_fixtures import build

from reliquary_sandbox.attest import (
    HEARTBEAT_DOMAIN,
    RECORD_DOMAIN,
    TOKEN_DOMAIN,
    HeartbeatCheck,
    HeartbeatReason,
    heartbeat_valid,
)
from reliquary_sandbox.observation import MAX_ARGUMENT_BYTES
from reliquary_sandbox_service.episodes.capacity import (
    CapacityCache,
    capacity_report,
    push_heartbeats,
)
from reliquary_sandbox_service.episodes.tools import TOOLS_VERSION

MAX_AGE = 60


def fresh(h):
    h.manager.host_facts = h.boxes.preflight()
    return capacity_report(h.manager)


def check(report, h, now=None, max_age_s=MAX_AGE):
    now = int(h.clock.now) if now is None else now
    return heartbeat_valid(report, directory_for(h.machine), now=now, max_age_s=max_age_s)


def test_capacity_report_is_signed_and_counts_slots(tmp_path):
    h = build(tmp_path)
    h.open()
    report = fresh(h)
    document = report["document"]
    assert document["machine_id"] == MACHINE and document["key_id"] == "m-1"
    assert document["at"] == int(h.clock.now)
    assert (document["capacity"], document["active"], document["free"]) == (4, 1, 3)
    assert document["images"] == [IMAGE] and document["tools_version"] == TOOLS_VERSION
    assert document["env_packages"] == {"fake-env": "fake_episode_env==unpackaged"}
    assert document["public_base_url"] == h.settings.public_base_url
    assert len(document["load"]) == 3 and document["cpu_count"] >= 1
    result = check(report, h)
    assert isinstance(result, HeartbeatCheck) and result.ok and result
    document["images"] = []  # any change after signing
    assert check(report, h).reason == "bad_signature"


def test_capacity_report_carries_the_host_versions_and_machine_caps(tmp_path):
    h = build(tmp_path)
    document = fresh(h)["document"]
    settings = h.settings
    assert document["runsc_version"] == "release-20260928.0"
    assert document["busybox_version"] == "1.37.0"
    assert document["helper_busybox_sha256"] == settings.episode_helper_busybox_sha256
    assert document["caps"] == {
        "max_calls": settings.episode_max_calls,
        "max_cpu_s": settings.episode_max_cpu_s,
        "max_call_timeout_s": settings.episode_max_call_timeout_s,
        "max_wall_s": settings.episode_max_wall_s,
        "max_memory_bytes": settings.episode_max_memory_bytes,
        "max_pids": settings.episode_max_pids,
        "max_disk_bytes": settings.episode_max_disk_bytes,
        "max_argument_bytes": MAX_ARGUMENT_BYTES,
        "output_cap_bytes": settings.episode_output_cap_bytes,
        "max_token_validity_s": settings.episode_retention_days * 86400,
        "max_transcript_bytes": settings.episode_max_transcript_bytes,
    }


def test_capacity_report_without_host_facts_says_unknown(tmp_path):
    h = build(tmp_path)
    document = capacity_report(h.manager)["document"]
    assert document["runsc_version"] is None and document["busybox_version"] is None
    assert check(capacity_report(h.manager), h)


def test_a_full_machine_reports_no_free_slot(tmp_path):
    h = build(tmp_path, episode_capacity=1)
    h.open()
    document = fresh(h)["document"]
    assert (document["capacity"], document["active"], document["free"]) == (1, 1, 0)


def test_the_capacity_report_is_signed_under_the_heartbeat_domain_only(tmp_path):
    h = build(tmp_path)
    report = fresh(h)
    for domain in (RECORD_DOMAIN, TOKEN_DOMAIN):
        forged = {"document": report["document"],
                  "signature": h.machine.sign(domain, report["document"])}
        assert check(forged, h).reason == "bad_signature"
    assert h.machine.sign(HEARTBEAT_DOMAIN, report["document"]) == report["signature"]


def test_stale_and_future_reports_are_refused(tmp_path):
    h = build(tmp_path)
    report = fresh(h)
    at = report["document"]["at"]
    assert check(report, h, now=at + MAX_AGE)
    assert check(report, h, now=at + MAX_AGE + 1).reason == "stale"
    assert check(report, h, now=at - 30)  # validator clock behind ours, within the skew
    assert check(report, h, now=at - 31).reason == "from_the_future"


def test_a_report_signed_with_an_unlisted_or_retired_key_is_refused(tmp_path):
    h = build(tmp_path)
    report = fresh(h)
    at = report["document"]["at"]
    retired = directory_for(h.machine, valid_until=at)
    assert heartbeat_valid(report, retired, now=at, max_age_s=MAX_AGE).reason == "unknown_key"
    (tmp_path / "other").mkdir()
    other = build(tmp_path / "other")
    assert check(report, other).reason == "bad_signature"


@pytest.mark.parametrize("mutate", [
    lambda r: "not a report",
    lambda r: {**r, "extra": 1},
    lambda r: {"document": r["document"]},
    lambda r: {**r, "signature": 7},
    lambda r: {**r, "document": [1, 2]},
    lambda r: {**r, "document": {**r["document"], "at": True}},
    lambda r: {**r, "document": {**r["document"], "at": "1"}},
    lambda r: {**r, "document": {**r["document"], "machine_id": None}},
    lambda r: {**r, "document": {**r["document"], "free": -1}},
    lambda r: {**r, "document": {**r["document"], "free": 5, "capacity": 4}},

    lambda r: {**r, "document": {**r["document"], "capacity": 1.5}},
    lambda r: {**r, "document": {**r["document"], "load": float("nan")}},
    lambda r: {**r, "signature": "not base64!"},
])
def test_malformed_reports_are_refused_without_raising(tmp_path, mutate):
    h = build(tmp_path)
    result = check(mutate(fresh(h)), h)
    assert not result
    assert result.reason in {HeartbeatReason.MALFORMED, HeartbeatReason.BAD_SIGNATURE}


@pytest.mark.parametrize("counts", [
    {"active": 5, "capacity": 4, "free": 1},
    {"active": 0, "capacity": 4, "free": 1},  # free != capacity - active
    {"active": 1, "capacity": 4, "free": 4},
])
def test_inconsistent_slot_counts_are_malformed_even_when_signed(tmp_path, counts):
    h = build(tmp_path)
    document = {**fresh(h)["document"], **counts}
    report = {"document": document, "signature": h.machine.sign(HEARTBEAT_DOMAIN, document)}
    assert check(report, h).reason is HeartbeatReason.MALFORMED


def test_more_active_than_capacity_after_lowering_it_is_a_valid_report(tmp_path):
    # An operator lowered the capacity and restarted: surviving episodes keep
    # running, so active exceeds capacity and nothing is free.
    h = build(tmp_path)
    document = {**fresh(h)["document"], "active": 3, "capacity": 1, "free": 0}
    report = {"document": document, "signature": h.machine.sign(HEARTBEAT_DOMAIN, document)}
    assert check(report, h)


def test_a_directory_that_raises_is_a_refusal_not_a_crash(tmp_path):
    h = build(tmp_path)

    class Broken:
        def public_key(self, machine_id, key_id, at):
            raise RuntimeError("directory down")

    result = heartbeat_valid(fresh(h), Broken(), now=int(h.clock.now), max_age_s=MAX_AGE)
    assert result.reason is HeartbeatReason.UNKNOWN_KEY


def test_heartbeat_reasons_are_a_closed_exported_set():
    assert {reason.value for reason in HeartbeatReason} == {
        "malformed", "stale", "from_the_future", "unknown_key", "bad_signature"}


def test_heartbeat_check_requires_the_callers_clock(tmp_path):
    h = build(tmp_path)
    with pytest.raises(TypeError):
        heartbeat_valid(fresh(h), directory_for(h.machine))


def run_pushes(handler, interval=0.5, until=2, report=None, limit=100):
    seen = []

    async def default_report():
        return {"document": {"n": len(seen)}, "signature": "s"}

    def record(request):
        seen.append(request)
        return handler(request, len(seen))

    async def scenario():
        stop = asyncio.Event()
        state = {}
        task = asyncio.create_task(push_heartbeats(
            "https://validator.example/heartbeat", interval,
            report or default_report,
            stop, state, transport=httpx.MockTransport(record)))
        for _ in range(limit):
            if len(seen) >= until:
                break
            await asyncio.sleep(0.05)
        stop.set()
        await asyncio.wait_for(task, 5)
        return state

    return seen, asyncio.run(scenario())


def test_heartbeats_are_pushed_until_stopped_and_failures_recorded():
    seen, state = run_pushes(lambda request, n: httpx.Response(200 if n == 1 else 503))
    assert len(seen) >= 2
    assert seen[0].url == "https://validator.example/heartbeat"
    assert state.get("heartbeat_error") is True


def test_a_successful_push_clears_the_error():
    seen, state = run_pushes(lambda request, n: httpx.Response(503 if n == 1 else 200),
                             interval=0.1)
    assert len(seen) >= 2 and "heartbeat_error" not in state


def test_a_report_that_cannot_be_built_is_a_failed_push_not_a_dead_loop(caplog):
    calls = []

    async def report():
        calls.append(1)
        raise RuntimeError("docker down")

    caplog.set_level(logging.WARNING)
    seen, state = run_pushes(lambda request, n: httpx.Response(200), interval=0.1,
                             report=report, until=1, limit=10)
    assert seen == [] and calls and state.get("heartbeat_error") is True
    assert "heartbeat" in caplog.text and "RuntimeError" in caplog.text


def test_failed_pushes_back_off():
    from reliquary_sandbox_service.episodes.capacity import backoff_s

    assert backoff_s(5, 0) == 5
    assert backoff_s(5, 1) == 10 and backoff_s(5, 2) == 20
    assert backoff_s(5, 50) == 60  # capped, so the validator hears again soon
    assert backoff_s(300, 3) == 300  # never shorter than the interval


class Ticks:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_the_capacity_cache_builds_off_the_event_loop_and_serves_within_its_ttl(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    h = build(tmp_path)
    builds = []
    original = h.boxes.images_present

    def counted():
        import threading

        builds.append(threading.current_thread().name)
        return original()

    h.boxes.images_present = counted
    ticks = Ticks()
    pool = ThreadPoolExecutor(1, thread_name_prefix="maintenance")

    async def scenario():
        cache = CapacityCache(h.manager, pool, ttl_s=1.0, clock=ticks)
        first, second = await asyncio.gather(cache.get(), cache.get())  # one build, shared
        third = await cache.get()
        ticks.now += 1.5
        fourth = await cache.get()
        return first, second, third, fourth

    try:
        first, second, third, fourth = asyncio.run(scenario())
    finally:
        pool.shutdown()
    assert first is second is third and fourth is not first
    assert len(builds) == 2 and all(name.startswith("maintenance") for name in builds)
    assert check(fourth, h)


@pytest.mark.parametrize("mutate", [
    lambda d: {k: v for k, v in d.items() if k != "caps"},
    lambda d: {**d, "caps": [1]},
    lambda d: {**d, "caps": {k: v for k, v in d["caps"].items() if k != "max_transcript_bytes"}},
    lambda d: {**d, "caps": {**d["caps"], "max_transcript_bytes": True}},
    lambda d: {**d, "caps": {**d["caps"], "max_transcript_bytes": -1}},
    lambda d: {**d, "caps": {**d["caps"], "max_calls": "512"}},
])
def test_caps_must_be_complete_non_negative_integers_even_when_signed(tmp_path, mutate):
    h = build(tmp_path)
    document = mutate(fresh(h)["document"])
    report = {"document": document, "signature": h.machine.sign(HEARTBEAT_DOMAIN, document)}
    assert check(report, h).reason is HeartbeatReason.MALFORMED


def test_the_transcript_cap_is_reported_for_validators():
    from reliquary_sandbox.attest.verify import HEARTBEAT_CAPS

    assert "max_transcript_bytes" in HEARTBEAT_CAPS
