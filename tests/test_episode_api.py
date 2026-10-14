"""The episode gateway over HTTP, with fake tenant and task runtimes."""

import base64
import threading
import time

import pytest
from episode_fixtures import IMAGE, MACHINE, budgets, claims, directory_for, episode_settings
from fake_boxes import FakeBoxes
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_service import FakeRuntime

from reliquary_sandbox.attest import (
    Ed25519TokenVerifier,
    Expected,
    Signer,
    heartbeat_valid,
    issue_session_token,
    load_private_key,
    verify_transcript,
)
from reliquary_sandbox.observation import MAX_ARGUMENT_BYTES
from reliquary_sandbox_service.app import create_app
from reliquary_sandbox_service.episodes.manager import OPEN_FAILED, EpisodeManager
from reliquary_sandbox_service.episodes.registry import TaskRegistry
from reliquary_sandbox_service.episodes.routes import (
    CALL_OVERHEAD_BYTES,
    TOKEN_BODY_BYTES,
    register_episode_routes,
)
from reliquary_sandbox_service.store import Store

EXPECTED = dict(hotkey="5Hotkey", engagement="corpus:job-1:7", env="fake-env", split="train",
                index=7, checkpoint="ck-1")
ANSWER = {"turn": 0, "k": 0, "tool": "bash", "arguments": {"command": "write /work/answer.txt 42"}}


@pytest.fixture
def api(tmp_path):
    settings, validator = episode_settings(tmp_path)
    boxes = FakeBoxes([IMAGE])
    with TestClient(create_app(settings, FakeRuntime(), task_runtime=boxes)) as client:
        yield client, settings, validator, boxes


def bearer(key):
    return {"Authorization": f"Bearer {key}"}


def opened(client, validator, session_id="s-1", **overrides):
    response = client.post("/episodes",
                           json={"token": issue_session_token(claims(session_id, **overrides), validator)})
    assert response.status_code == 200, response.text
    return response.json()["episode_key"]


def machine_of(settings):
    return Signer(settings.machine_key_id, load_private_key(settings.machine_key_file))


def verify(transcript, settings, validator):
    return verify_transcript(transcript, directory_for(machine_of(settings)),
                             Ed25519TokenVerifier({"v-1": validator.public_key_b64}),
                             Expected(**EXPECTED))


def test_full_episode_over_http_produces_a_verifiable_transcript(api):
    client, settings, validator, boxes = api
    key = opened(client, validator)
    call = client.post("/episodes/s-1/calls", headers=bearer(key), json=ANSWER)
    assert call.status_code == 200 and call.json()["final"] is None
    assert call.json()["record"]["body"]["kind"] == "call"
    finished = client.post("/episodes/s-1/finish", headers=bearer(key))
    assert finished.status_code == 200
    assert finished.json()["record"]["body"]["reward"] == 1.0
    assert base64.b64decode(finished.json()["state"]) == b"42\n"
    response = client.get("/episodes/s-1", headers=bearer(key)).json()
    assert response["status"] == "CLOSED" and response["final_status"] == "graded"
    result = verify(response["transcript"], settings, validator)
    assert result.ok, result.reasons


def test_a_resent_call_returns_the_stored_record(api):
    client, _, validator, _ = api
    key = opened(client, validator)
    first = client.post("/episodes/s-1/calls", headers=bearer(key), json=ANSWER).json()
    again = client.post("/episodes/s-1/calls", headers=bearer(key), json=ANSWER).json()
    assert again["record"] == first["record"]


def test_closed_episode_errors_carry_the_final_record(api):
    client, _, validator, _ = api
    key = opened(client, validator)
    client.post("/episodes/s-1/finish", headers=bearer(key))
    late = client.post("/episodes/s-1/calls", headers=bearer(key),
                       json={"turn": 3, "k": 0, "tool": "bash", "arguments": {"command": "echo"}})
    assert late.status_code == 409
    assert late.json()["detail"]["final"]["body"]["kind"] == "final"


def test_finish_after_the_deadline_is_409_with_the_final_then_200(api):
    client, settings, validator, _ = api
    key = opened(client, validator, budgets=budgets(wall_s=1))
    time.sleep(1.2)
    first = client.post("/episodes/s-1/finish", headers=bearer(key))
    assert first.status_code == 409
    final = first.json()["detail"]["final"]
    assert final["body"]["status"] == "expired"
    second = client.post("/episodes/s-1/finish", headers=bearer(key))
    assert second.status_code == 200 and second.json()["record"] == final


def test_a_concurrent_call_on_the_same_episode_is_409_busy(api):
    client, _, validator, boxes = api
    key = opened(client, validator)
    running = threading.Event()
    bash = boxes.bash

    def observed(box_id, call_id, command, **kwargs):
        running.set()
        return bash(box_id, call_id, command, **kwargs)

    boxes.bash = observed
    slow = {"turn": 0, "k": 0, "tool": "bash", "arguments": {"command": "hang"}}
    results = []
    worker = threading.Thread(target=lambda: results.append(
        client.post("/episodes/s-1/calls", headers=bearer(key), json=slow)))
    worker.start()
    try:
        assert running.wait(5)
        for request in (lambda: client.post("/episodes/s-1/calls", headers=bearer(key),
                                            json={**slow, "k": 1}),
                        lambda: client.post("/episodes/s-1/finish", headers=bearer(key))):
            busy = request()
            assert busy.status_code == 409 and busy.json()["detail"] == "episode is busy"
    finally:
        boxes.block.set()
        worker.join(10)
    assert results[0].status_code == 200
    assert client.post("/episodes/s-1/finish", headers=bearer(key)).status_code == 200


def test_a_full_machine_answers_503_with_retry_after(tmp_path):
    settings, validator = episode_settings(tmp_path, episode_capacity=1)
    with TestClient(create_app(settings, FakeRuntime(), task_runtime=FakeBoxes([IMAGE]))) as client:
        opened(client, validator)
        refused = client.post("/episodes", json={"token": issue_session_token(claims("s-2"), validator)})
    assert refused.status_code == 503 and refused.headers["Retry-After"]


def test_a_lost_open_response_is_recovered_by_sending_the_token_again(tmp_path):
    """Even on a full machine: the episode already holds its slot."""
    settings, validator = episode_settings(tmp_path, episode_capacity=1)
    token = issue_session_token(claims(), validator)
    with TestClient(create_app(settings, FakeRuntime(), task_runtime=FakeBoxes([IMAGE]))) as client:
        first = client.post("/episodes", json={"token": token})
        again = client.post("/episodes", json={"token": token})
        assert first.status_code == again.status_code == 200
        assert again.json() == first.json()
        key = again.json()["episode_key"]
        assert client.post("/episodes/s-1/calls", headers=bearer(key), json=ANSWER).status_code == 200


def test_a_failed_open_spends_the_token_without_retry_after(api):
    client, _, validator, boxes = api
    boxes.fail_create = True
    token = issue_session_token(claims(), validator)
    failed = client.post("/episodes", json={"token": token})
    assert failed.status_code == 503 and failed.json()["detail"] == OPEN_FAILED
    assert "token spent; request a new token" in OPEN_FAILED
    assert "retry-after" not in failed.headers
    boxes.fail_create = False
    assert client.post("/episodes", json={"token": token}).status_code == 409


@pytest.mark.parametrize("body", [
    {"turn": -1, "k": 0, "tool": "bash", "arguments": {}},
    {"turn": 0, "k": 0, "tool": "", "arguments": {}},
    {"turn": 0, "k": 0, "tool": "bash", "arguments": []},
    {"turn": 0, "k": 0, "tool": "bash", "arguments": {}, "extra": 1},
    {"turn": "0", "k": 0, "tool": "bash", "arguments": {}},
    {"turn": True, "k": 0, "tool": "bash", "arguments": {}},
    {"turn": 0.0, "k": 0, "tool": "bash", "arguments": {}},
    {"turn": 0, "k": 2**63, "tool": "bash", "arguments": {}},
    {"turn": 0, "k": 0, "arguments": {}},
])
def test_malformed_call_requests_are_refused(api, body):
    client, _, validator, _ = api
    key = opened(client, validator)
    assert client.post("/episodes/s-1/calls", headers=bearer(key), json=body).status_code == 422


@pytest.mark.parametrize("body", [{}, {"token": "x"}, {"token": {}, "extra": 1}])
def test_malformed_open_requests_are_refused(api, body):
    client, *_ = api
    assert client.post("/episodes", json=body).status_code in (401, 422)


def test_episode_endpoints_require_the_episode_key(api):
    client, _, validator, _ = api
    key = opened(client, validator)
    missing = client.get("/episodes/s-1")
    wrong = client.get("/episodes/s-1", headers=bearer("wrong"))
    unknown = client.get("/episodes/unknown", headers=bearer("wrong"))
    assert missing.status_code == wrong.status_code == unknown.status_code == 401
    # An unknown episode and a wrong key are indistinguishable.
    assert wrong.json() == unknown.json()
    assert client.post("/episodes/s-1/finish", headers=bearer("wrong")).status_code == 401
    assert client.post("/episodes/s-1/calls", headers=bearer("wrong"), json=ANSWER).status_code == 401
    # The key is accepted only from the header, never from the query string.
    assert client.get("/episodes/s-1", params={"key": key}).status_code == 401
    assert client.get("/episodes/s-1", params={"authorization": f"Bearer {key}"}).status_code == 401
    assert client.get("/episodes/s-1", headers={"Authorization": key}).status_code == 401


def test_oversized_bodies_are_refused_before_parsing(api):
    client, settings, validator, _ = api
    key = opened(client, validator, session_id="s-1")
    big_token = {"token": {"payload": "x" * TOKEN_BODY_BYTES, "signature": "s", "key_id": "v-1"}}
    assert client.post("/episodes", json=big_token).status_code == 413
    limit = 6 * MAX_ARGUMENT_BYTES + CALL_OVERHEAD_BYTES
    big_call = {**ANSWER, "arguments": {"command": "echo " + "x" * limit}}
    assert client.post("/episodes/s-1/calls", headers=bearer(key), json=big_call).status_code == 413
    chunked = client.post("/episodes/s-1/calls", headers={**bearer(key), "Content-Type": "application/json"},
                          content=iter([b"{" + b" " * limit, b"}"]))
    assert chunked.status_code == 413
    assert client.post("/episodes/s-1/finish", headers=bearer(key), content=b"x" * 4096).status_code == 413
    records = client.get("/episodes/s-1", headers=bearer(key)).json()["transcript"]["records"]
    assert len(records) == 1  # nothing ran


def test_a_call_with_arguments_at_the_limit_passes_the_http_layer(api):
    client, settings, validator, _ = api
    key = opened(client, validator)
    command = "echo " + "x" * (MAX_ARGUMENT_BYTES - 64)
    call = client.post("/episodes/s-1/calls", headers=bearer(key),
                       json={**ANSWER, "arguments": {"command": command}})
    assert call.status_code == 200, call.text


def test_clients_that_escape_non_ascii_text_pass_the_http_layer(api):
    """`json.dumps` (and `requests`) escape non-ASCII as \\uXXXX: up to six body
    bytes per canonical byte. Arguments are still bounded canonically after parsing."""
    import json

    client, settings, validator, _ = api
    key = opened(client, validator)
    command = "echo " + "\u00e9" * ((MAX_ARGUMENT_BYTES - 64) // 2)
    body = json.dumps({**ANSWER, "arguments": {"command": command}})  # ensure_ascii=True
    assert len(body) > 2 * MAX_ARGUMENT_BYTES
    call = client.post("/episodes/s-1/calls", content=body,
                       headers={**bearer(key), "Content-Type": "application/json"})
    assert call.status_code == 200, call.text
    over = json.dumps({**ANSWER, "turn": 1,
                       "arguments": {"command": "x" * (MAX_ARGUMENT_BYTES + 1)}})
    refused = client.post("/episodes/s-1/calls", content=over,
                          headers={**bearer(key), "Content-Type": "application/json"})
    assert refused.status_code == 413  # by the manager's canonical bound


def test_transcripts_are_read_on_their_own_pool_as_prebuilt_json(api, monkeypatch):
    client, settings, validator, boxes = api
    key = opened(client, validator)
    threads = []
    original = EpisodeManager.transcript_json

    def recorded(self, *args):
        threads.append(threading.current_thread().name)
        return original(self, *args)

    monkeypatch.setattr(EpisodeManager, "transcript_json", recorded)
    response = client.get("/episodes/s-1", headers=bearer(key))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["transcript"]["records"][0]["body"]["kind"] == "open"
    assert threads and threads[0].startswith("transcripts")


def test_hanging_calls_cannot_starve_capacity_or_the_reaper(tmp_path, monkeypatch):
    """Calls run on the episode pool; the reaper and the capacity report on their own
    small pool, so a saturated episode pool (and the default executor) cannot stall them."""
    release = threading.Event()
    hanging = []
    reaps = []

    def hang(self, *args):
        hanging.append(1)
        release.wait(20)
        raise RuntimeError("released")

    original_reap = EpisodeManager.reap

    def counted_reap(self):
        reaps.append(time.monotonic())
        return original_reap(self)

    monkeypatch.setattr(EpisodeManager, "call", hang)
    monkeypatch.setattr(EpisodeManager, "reap", counted_reap)
    settings, _ = episode_settings(tmp_path, reaper_interval_seconds=0.1)
    pool_size = settings.episode_capacity * 2 + 4
    with TestClient(create_app(settings, FakeRuntime(), task_runtime=FakeBoxes([IMAGE])),
                    raise_server_exceptions=False) as client:
        workers = [threading.Thread(target=lambda: client.post(
            "/episodes/s-1/calls", headers=bearer("k"), json=ANSWER)) for _ in range(40)]
        for worker in workers:
            worker.start()
        try:
            assert wait_for(lambda: len(hanging) >= pool_size)
            time.sleep(0.2)
            assert len(hanging) == pool_size  # the episode pool is saturated, the rest queue
            started = time.monotonic()
            report = client.get("/capacity")
            assert report.status_code == 200 and time.monotonic() - started < 1.0
            seen = len(reaps)
            assert wait_for(lambda: len(reaps) >= seen + 2, timeout=2.0)
            assert client.get("/readyz").status_code == 200
        finally:
            release.set()
            for worker in workers:
                worker.join(30)


def test_the_capacity_route_serves_a_cached_report(api):
    client, _, _, boxes = api
    builds = []
    original = boxes.images_present

    def counted():
        builds.append(1)
        return original()

    boxes.images_present = counted
    first = client.get("/capacity").json()
    for _ in range(5):
        assert client.get("/capacity").json() == first
    assert len(builds) <= 1


def test_tenant_requests_keep_their_own_body_limit(api):
    client, *_ = api
    variables = {name: "x" * 8000 for name in "ABC"}  # larger than any episode body limit
    created = client.post("/api/v1/sandbox", headers={"Authorization": "Bearer " + "e" * 32},
                          json={"name": "x", "docker_image": "reliquary-sandbox-guest:0.1.0",
                                "environment_vars": variables})
    assert created.status_code == 200, created.text


def test_capacity_endpoint_serves_the_signed_report(api):
    client, settings, validator, _ = api
    opened(client, validator)
    report = client.get("/capacity").json()
    document = report["document"]
    assert document["machine_id"] == MACHINE and document["free"] == 3
    assert document["runsc_version"] == "release-20260928.0"
    assert document["public_base_url"] == settings.public_base_url
    assert heartbeat_valid(report, directory_for(machine_of(settings)),
                           now=int(time.time()), max_age_s=60)


def test_readiness_reads_the_cached_host_verdict_without_running_checks(api):
    client, _, _, boxes = api
    before = boxes.preflight_calls
    for _ in range(5):
        assert client.get("/readyz").status_code == 200
    assert boxes.preflight_calls == before


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_periodic_host_checks_gate_readiness_and_log_only_changes(tmp_path, caplog):
    import logging

    caplog.set_level(logging.DEBUG, logger="reliquary_sandbox_service")
    settings, _ = episode_settings(tmp_path, episode_host_check_seconds=0.05)
    boxes = FakeBoxes([IMAGE])
    healthy = boxes.preflight

    def broken(*, probe=True):
        assert probe is False  # periodic checks never run the quota probe
        boxes.preflight_calls += 1
        raise RuntimeError("runsc gone")

    with TestClient(create_app(settings, FakeRuntime(), task_runtime=boxes)) as client:
        start = boxes.preflight_calls
        assert wait_for(lambda: boxes.preflight_calls >= start + 3)  # periodic, verdict unchanged
        assert client.get("/readyz").status_code == 200
        boxes.preflight = broken
        assert wait_for(lambda: client.get("/readyz").status_code == 503)
        failing = boxes.preflight_calls
        assert wait_for(lambda: boxes.preflight_calls >= failing + 3)
        boxes.preflight = healthy
        assert wait_for(lambda: client.get("/readyz").status_code == 200)
    changes = [r for r in caplog.records
               if r.levelno == logging.INFO and "episode host checks" in r.getMessage()]
    assert len(changes) == 2  # failing, then passing again; repeats are DEBUG
    assert any(r.levelno == logging.DEBUG and "episode host checks" in r.getMessage()
               for r in caplog.records)


def test_startup_preflights_the_host_with_the_quota_probe(api):
    _, _, _, boxes = api
    assert boxes.preflight_calls >= 1


def test_a_host_that_fails_preflight_does_not_start(tmp_path):
    settings, _ = episode_settings(tmp_path)
    boxes = FakeBoxes([IMAGE])

    def broken(*, probe=True):
        raise RuntimeError("no gVisor")

    boxes.preflight = broken
    with pytest.raises(RuntimeError, match="no gVisor"):
        with TestClient(create_app(settings, FakeRuntime(), task_runtime=boxes)):
            pass


def test_a_legacy_only_config_starts_without_a_scoring_forkserver(tmp_path, monkeypatch):
    """Only verifiers entries are scored in a forkserver: a config of `module:callable`
    envs never starts one, and the scoring path never imports their modules (an env that
    reads a credential at import is the service's business, as before the bridge)."""
    from reliquary_sandbox_service import scoring_server

    calls = []
    monkeypatch.setattr(scoring_server, "start", lambda preload=(): calls.append(("start", preload)))
    monkeypatch.setattr(scoring_server, "check_preload",
                        lambda modules: calls.append(("check", modules)))
    monkeypatch.setenv("RELIQUARY_TEST_IMPORT_CREDENTIAL", "token")
    settings, _ = episode_settings(tmp_path, episode_envs={
        "fake-env": "fake_episode_env:sandbox_task",
        "credential-env": "needs_a_credential_env:sandbox_task"})
    with TestClient(create_app(settings, FakeRuntime(), task_runtime=FakeBoxes([IMAGE]))) as client:
        assert client.get("/readyz").status_code == 200
    assert calls == []


def test_only_verifiers_entries_are_preloaded_in_the_scoring_forkserver(tmp_path, monkeypatch):
    from reliquary_sandbox_service import scoring_server

    class Started(Exception):
        pass

    preloaded = []

    def start(preload=()):
        preloaded.append(list(preload))
        raise Started

    monkeypatch.setattr(scoring_server, "start", start)
    settings, _ = episode_settings(tmp_path, episode_allow_unpackaged_verifiers=True,
                                   episode_envs={
                                       "fake-env": "fake_episode_env:sandbox_task",
                                       "toy": "verifiers:toy-bridge-env==unpackaged"})
    with pytest.raises(Started):
        with TestClient(create_app(settings, FakeRuntime(), task_runtime=FakeBoxes([IMAGE]))):
            pass
    assert preloaded == [["toy_bridge_env"]]


def test_restart_recovery_runs_before_the_routes_serve(tmp_path):
    """Episodes left by a previous controller are settled at startup: one
    interrupted mid-call is aborted, a healthy one has its box user restored and
    goes on over HTTP."""
    settings, validator = episode_settings(tmp_path)
    settings.state_dir.mkdir(parents=True)
    boxes = FakeBoxes([IMAGE])
    previous = EpisodeManager(settings, boxes, Store(settings.state_dir), machine_of(settings),
                              TaskRegistry(settings.episode_envs))
    interrupted = previous.open(issue_session_token(claims("s-1"), validator))
    healthy = previous.open(issue_session_token(claims("s-2"), validator))
    episode = previous.store.get("s-1")
    episode["pending"] = {"turn": 0, "k": 0}
    previous.store.commit(episode)
    previous.store.db.close()
    boxes.users.clear()  # the old process's memory is gone

    with TestClient(create_app(settings, FakeRuntime(), task_runtime=boxes)) as client:
        first = client.get("/episodes/s-1", headers=bearer(interrupted["episode_key"])).json()
        assert first["final_status"] == "aborted"
        assert [box for box, *_ in boxes.restored] == ["s-2"]
        key = healthy["episode_key"]
        assert client.post("/episodes/s-2/calls", headers=bearer(key), json=ANSWER).status_code == 200
        assert client.post("/episodes/s-2/finish", headers=bearer(key)).json()["record"]["body"]["reward"] == 1.0
        transcript = client.get("/episodes/s-2", headers=bearer(key)).json()["transcript"]
    assert verify(transcript, settings, validator).ok


def test_episode_routes_answer_503_recovering_until_the_gateway_is_ready():
    app = FastAPI()
    state = {}

    register_episode_routes(app, state, object())
    with TestClient(app) as client:
        for response in (client.get("/capacity"),
                         client.post("/episodes", json={"token": {}}),
                         client.get("/episodes/s-1", headers=bearer("k"))):
            assert response.status_code == 503 and "recovering" in response.text
            assert response.headers["Retry-After"]


def test_tenant_api_is_unchanged_on_episode_hosts(api):
    client, *_ = api
    created = client.post("/api/v1/sandbox", headers={"Authorization": "Bearer " + "e" * 32},
                          json={"name": "x", "docker_image": "reliquary-sandbox-guest:0.1.0"})
    assert created.status_code == 200


def test_heartbeats_are_pushed_when_configured(tmp_path, monkeypatch):
    import reliquary_sandbox_service.episodes.capacity as capacity

    pushed = []

    async def fake_push(url, interval_s, report, stop, state, transport=None):
        pushed.append((url, interval_s, await report()))
        await stop.wait()

    monkeypatch.setattr(capacity, "push_heartbeats", fake_push)
    settings, _ = episode_settings(tmp_path, heartbeat_url="https://validator.example/hb",
                                   heartbeat_interval_seconds=7)
    with TestClient(create_app(settings, FakeRuntime(), task_runtime=FakeBoxes([IMAGE]))) as client:
        for _ in range(100):
            if pushed:
                break
            time.sleep(0.02)
        assert client.get("/health").status_code == 200
    url, interval, report = pushed[0]
    assert (url, interval) == ("https://validator.example/hb", 7)
    assert report["document"]["machine_id"] == MACHINE


def test_episode_routes_are_absent_when_disabled(tmp_path):
    from reliquary_sandbox_service.settings import Settings

    settings = Settings(tokens={"evals": "e" * 32}, state_dir=tmp_path, reaper_interval_seconds=3600)
    with TestClient(create_app(settings, FakeRuntime())) as client:
        assert client.get("/capacity").status_code == 404
        assert client.post("/episodes", json={"token": {}}).status_code == 404
        assert client.get("/readyz").status_code == 200
