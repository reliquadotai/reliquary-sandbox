import asyncio
import dataclasses
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import fake_episode_env
import pytest
from episode_fixtures import IMAGE, budgets, claims, make_signer
from episode_manager_fixtures import build, restarted
from fake_boxes import FakeBoxes
from fastapi import HTTPException

from reliquary_sandbox.attest import Reason, issue_session_token, sha256_hex
from reliquary_sandbox_service.episodes.manager import OPEN_REFUSAL_HEADER
from reliquary_sandbox_service.episodes.store import CLOSED
from reliquary_sandbox_service.episodes.tools import TOOLS_VERSION

OTHER_IMAGE = "registry.example/other@sha256:" + "c" * 64


def test_open_starts_a_prepared_box_and_signs_record_zero(tmp_path):
    h = build(tmp_path)
    opened = h.open()
    record = opened["record"]
    body = record["body"]
    assert body["kind"] == "open" and body["i"] == 0 and body["prev"] is None
    assert body["tools_version"] == TOOLS_VERSION
    assert body["tools"] == ["bash", "edit"]  # the task's tools, sorted
    assert body["env_package"] == "fake_episode_env==unpackaged"
    assert body["setup_output_sha256"] == sha256_hex(b"prepared\n")
    assert record["key_id"] == h.machine.key_id and record["machine_id"] == "machine-1"
    assert h.boxes.boxes["s-1"]["files"]["/work/prepared"] == b"1"
    spec = h.boxes.boxes["s-1"]["spec"]
    assert spec.workdir == "/work" and spec.disk_bytes == budgets().disk_bytes
    assert len(opened["episode_key"]) >= 32


def test_the_box_user_and_its_name_are_persisted_at_open(tmp_path):
    h = build(tmp_path)
    h.open()
    episode = h.manager.store.get("s-1")
    assert episode["box_user"] == h.boxes.box_user("s-1") == "0:0"
    assert "box_user_name" in episode and episode["box_user_name"] is None  # task.user


def test_the_episode_key_is_not_stored_in_clear(tmp_path):
    h = build(tmp_path)
    key = h.open()["episode_key"]
    stored = b"".join(path.read_bytes() for path in (tmp_path / "state").glob("state.sqlite3*"))
    assert stored and key.encode() not in stored


def test_token_is_single_use(tmp_path):
    """A token opens one episode: presenting it again never opens another box."""
    h = build(tmp_path)
    token = h.token()
    h.manager.open(token)
    h.manager.open(token)
    assert list(h.boxes.specs) == ["s-1"]


# -- idempotent open: a lost response does not burn the token ------------------


def test_a_resent_token_of_a_running_episode_returns_record_zero_and_the_same_key(tmp_path):
    h = build(tmp_path)
    token = h.token()
    first = h.manager.open(token)
    again = h.manager.open(dict(token))
    assert again == first
    h.manager.call("s-1", again["episode_key"], 0, 0, "bash", {"command": "echo hi"})
    assert h.manager.open(token)["record"] == first["record"]  # still record 0


def test_the_key_is_derived_from_a_persisted_machine_secret(tmp_path):
    h = build(tmp_path)
    token = h.token()
    key = h.manager.open(token)["episode_key"]
    secret = tmp_path / "state" / "episode-secret"
    assert secret.stat().st_mode & 0o777 == 0o600 and len(secret.read_bytes()) == 32
    created = secret.read_bytes()
    h2 = restarted(h)
    h2.manager.reconcile()
    assert h2.manager.open(token)["episode_key"] == key  # same key after a restart
    assert secret.read_bytes() == created  # created once, never rewritten


def test_a_resent_token_while_opening_is_busy(tmp_path):
    h = build(tmp_path)
    token = h.token()
    h.manager.open(token)
    episode = h.manager.store.get("s-1")
    episode["status"] = "OPENING"
    h.manager.store.commit(episode)
    with pytest.raises(HTTPException) as error:
        h.manager.open(token)
    assert error.value.status_code == 409 and "busy" in str(error.value.detail)


def test_a_resent_token_of_a_closed_episode_gets_the_final(tmp_path):
    h = build(tmp_path)
    token = h.token()
    key = h.manager.open(token)["episode_key"]
    final = h.manager.finish("s-1", key)["record"]
    with pytest.raises(HTTPException) as error:
        h.manager.open(token)
    assert error.value.status_code == 409
    assert error.value.detail["final"] == final and "episode_key" not in str(error.value.detail)


def test_a_resent_token_of_a_failed_open_stays_spent(tmp_path):
    h = build(tmp_path)
    token = h.token()
    h.boxes.fail_create = True
    with pytest.raises(HTTPException):
        h.manager.open(token)
    h.boxes.fail_create = False
    with pytest.raises(HTTPException) as error:
        h.manager.open(token)
    assert error.value.status_code == 409 and error.value.detail["final"] is None


def test_another_token_for_the_same_session_never_gets_the_key(tmp_path):
    h = build(tmp_path)
    h.manager.open(h.token())
    other = h.token(expires_at=int(h.clock.now) + 3000)  # same session id, other token
    with pytest.raises(HTTPException) as error:
        h.manager.open(other)
    assert error.value.status_code == 409 and "final" not in str(error.value.detail)


def test_a_second_token_for_a_used_session_id_is_refused(tmp_path):
    h = build(tmp_path)
    h.open("s-1")
    h.clock.now += 1  # another token (different issued_at), same session id
    with pytest.raises(HTTPException) as error:
        h.open("s-1")
    assert error.value.status_code == 409


@pytest.mark.parametrize("overrides,status", [
    ({"machine_id": "machine-2"}, 403),
    ({"env": "other-env"}, 422),
    ({"image": OTHER_IMAGE}, 422),
    ({"budgets": budgets(disk_bytes=11 * 1024**3)}, 422),
    ({"budgets": budgets(wall_s=5 * 3600)}, 422),
    ({"budgets": budgets(max_calls=513)}, 422),
    ({"budgets": budgets(cpu_s=3601)}, 422),
    ({"budgets": budgets(per_call_timeout_s=601)}, 422),
    ({"session_id": "bad id!"}, 422),
])
def test_tokens_this_machine_cannot_serve_are_refused(tmp_path, overrides, status):
    h = build(tmp_path)
    session = overrides.pop("session_id", "s-1")
    with pytest.raises(HTTPException) as error:
        h.open(session, **overrides)
    assert error.value.status_code == status
    assert h.boxes.boxes == {}


def test_the_image_must_be_approved_on_this_machine(tmp_path):
    # The task's image is the token's, but the operator did not list it.
    h = build(tmp_path, episode_images=[OTHER_IMAGE])
    with pytest.raises(HTTPException) as error:
        h.open()
    assert error.value.status_code == 422 and "approved" in error.value.detail
    assert h.boxes.boxes == {}


def test_an_image_that_is_not_present_is_refused(tmp_path):
    h = build(tmp_path, boxes=FakeBoxes([]))
    with pytest.raises(HTTPException) as error:
        h.open()
    assert error.value.status_code == 409
    assert h.boxes.boxes == {}


def test_expired_and_forged_tokens_are_refused(tmp_path):
    h = build(tmp_path)
    token = h.token()
    h.clock.now += 7200
    with pytest.raises(HTTPException) as error:
        h.manager.open(token)
    assert error.value.status_code == 401
    forged = h.token("s-2")
    forged["claims"]["hotkey"] = "5Thief"
    with pytest.raises(HTTPException) as error:
        h.manager.open(forged)
    assert error.value.status_code == 401
    assert h.boxes.boxes == {}


def test_a_token_is_refused_at_its_expiry_second(tmp_path):
    h = build(tmp_path)
    token = h.token()
    h.clock.now = token["claims"]["expires_at"]
    with pytest.raises(HTTPException) as error:
        h.manager.open(token)
    assert error.value.status_code == 401


def test_a_token_signed_by_an_unknown_validator_key_is_refused(tmp_path):
    h = build(tmp_path)
    stranger, _ = make_signer(tmp_path, "stranger", "v-2")
    with pytest.raises(HTTPException) as error:
        h.manager.open(issue_session_token(claims("s-1", now=int(h.clock.now)), stranger))
    assert error.value.status_code == 401
    # Same key id as the real validator, different key: the signature fails.
    impostor, _ = make_signer(tmp_path, "impostor", "v-1")
    with pytest.raises(HTTPException) as error:
        h.manager.open(issue_session_token(claims("s-1", now=int(h.clock.now)), impostor))
    assert error.value.status_code == 401
    assert h.boxes.boxes == {} and h.manager.store.get("s-1") is None


def test_capacity_is_enforced_with_retry_after(tmp_path):
    h = build(tmp_path, episode_capacity=1)
    h.open("s-1")
    with pytest.raises(HTTPException) as error:
        h.open("s-2")
    assert error.value.status_code == 503 and error.value.headers["Retry-After"] == "5"
    # A full machine signs its refusal and spends the token (a tombstone row, no box).
    assert "s-2" not in h.boxes.boxes and h.manager.store.get("s-2")["final_status"] == "refused"


def test_failed_creation_consumes_the_token_and_frees_the_slot(tmp_path):
    h = build(tmp_path, episode_capacity=1)
    h.boxes.fail_create = True
    token = h.token("s-1")
    with pytest.raises(HTTPException) as error:
        h.manager.open(token)
    assert error.value.status_code == 503
    h.boxes.fail_create = False
    with pytest.raises(HTTPException) as error:
        h.manager.open(token)
    assert error.value.status_code == 409
    h.open("s-2")


@pytest.mark.parametrize("overrides", [
    {"budgets": budgets(max_calls=513)}, {"budgets": budgets(cpu_s=3601)},
    {"budgets": budgets(per_call_timeout_s=601)},
    {"expires_at": 1_800_000_000 + 7 * 86400 + 1},  # outlives the single-use memory
])
def test_tokens_beyond_this_machines_limits_are_refused_signed_and_spent(tmp_path, overrides):
    h = build(tmp_path)
    with pytest.raises(HTTPException) as error:
        h.open("s-1", **overrides)
    assert error.value.status_code == 422
    # The validator's claims, not the miner's doing: signed and spent.
    assert OPEN_REFUSAL_HEADER in error.value.headers
    assert h.manager.store.get("s-1")["final_status"] == "refused" and "s-1" not in h.boxes.boxes
    h.open("s-2")  # the validator issues a serviceable token for a new session


def test_a_token_valid_as_long_as_the_retention_opens(tmp_path):
    h = build(tmp_path, episode_retention_days=7)
    h.open("s-1", expires_at=1_800_000_000 + 7 * 86400 - 5)


def slow_prepare(h, seconds, prepare_timeout_s=3600):
    """`prepare` takes `seconds` of the host clock."""
    real_resolve, real_exec = h.manager.registry.resolve, h.boxes.exec

    def resolve(*args):
        resolved = real_resolve(*args)
        task = dataclasses.replace(resolved.task, prepare_timeout_s=prepare_timeout_s)
        return dataclasses.replace(resolved, task=task)

    def exec_(box_id, argv, **kwargs):
        if argv == ["prepare"]:
            h.clock.now += seconds
        return real_exec(box_id, argv, **kwargs)

    h.manager.registry.resolve = resolve
    h.boxes.exec = exec_


def test_preparation_past_the_token_expiry_fails_the_open(tmp_path):
    h = build(tmp_path)
    slow_prepare(h, 3700, prepare_timeout_s=600)  # the token expires 3600 s after it was issued
    token = h.token()
    with pytest.raises(HTTPException) as error:
        h.manager.open(token)
    assert error.value.status_code == 503 and "request a new token" in error.value.detail
    assert OPEN_REFUSAL_HEADER not in (error.value.headers or {})   # a late token: never signed
    assert h.manager.store.records("s-1") == []
    assert h.manager.store.get("s-1")["final_status"] == "open_failed"
    assert "s-1" not in h.boxes.boxes


def test_preparation_does_not_consume_the_agents_wall_time(tmp_path):
    h = build(tmp_path)
    slow_prepare(h, 3000, prepare_timeout_s=3000)  # a 3600 s token leaves time to prepare
    start = int(h.clock.now)
    h.open("s-1", expires_at=start + 10_000)
    assert h.manager.store.get("s-1")["deadline"] == start + 3000 + 3600
    assert h.boxes.boxes["s-1"]["spec"].lifetime_s >= 3000 + 3600 + 600
    h.clock.now = start
    h.open("s-2")  # the token expires at start + 3600: the expiry caps the deadline
    episode = h.manager.store.get("s-2")
    assert episode["deadline"] == episode["claims"]["expires_at"] == start + 3600


def test_a_failed_open_tells_the_miner_the_token_is_spent(tmp_path):
    h = build(tmp_path)
    h.boxes.fail_create = True
    with pytest.raises(HTTPException) as error:
        h.open()
    assert error.value.status_code == 503
    assert "token spent; request a new token" in error.value.detail
    assert not (error.value.headers or {}).get("Retry-After")


def test_a_failure_after_preparation_marks_the_open_failed_and_frees_the_slot(tmp_path, monkeypatch):
    h = build(tmp_path, episode_capacity=1)

    def broken_sign(body):
        raise ValueError("signer failed")

    real_sign = h.manager._sign
    monkeypatch.setattr(h.manager, "_sign", broken_sign)
    with pytest.raises(HTTPException) as error:
        h.open("s-1")
    assert error.value.status_code == 503
    episode = h.manager.store.get("s-1")
    assert episode["status"] == CLOSED and episode["final_status"] == "open_failed"
    assert h.manager.store.records("s-1") == [] and "s-1" not in h.boxes.boxes
    monkeypatch.setattr(h.manager, "_sign", real_sign)
    h.open("s-2")


def test_failed_preparation_burns_the_token_and_writes_no_record_zero(tmp_path):
    h = build(tmp_path, episode_capacity=1)
    h.boxes.broken.add("s-1")  # the box starts, then `prepare` fails in it
    token = h.token("s-1")
    with pytest.raises(HTTPException) as error:
        h.manager.open(token)
    assert error.value.status_code == 503
    assert h.manager.store.records("s-1") == []
    episode = h.manager.store.get("s-1")
    assert episode["status"] == CLOSED and episode["final_status"] == "open_failed"
    assert episode["closed_at"] == h.clock.now
    assert "s-1" in h.boxes.removed and "s-1" not in h.boxes.boxes
    h.boxes.broken.clear()
    with pytest.raises(HTTPException) as error:
        h.manager.open(token)
    assert error.value.status_code == 409
    h.open("s-2")  # the slot was freed


def test_a_box_that_cannot_be_removed_after_a_failed_open_is_left_for_the_reaper(tmp_path):
    h = build(tmp_path)
    h.boxes.broken.add("s-1")
    h.boxes.fail_remove = True
    with pytest.raises(HTTPException):
        h.open()
    assert h.manager.store.get("s-1")["cleanup"] == ["s-1"]
    assert [e["id"] for e in h.manager.store.with_cleanup()] == ["s-1"]


def test_opened_transcript_has_record_zero_only(tmp_path):
    h = build(tmp_path)
    h.open()
    result = h.verify("s-1", require_graded=False)
    assert result.reasons == (Reason.NO_FINAL,)


def test_opens_do_not_serialize_on_a_global_lock(tmp_path):
    h = build(tmp_path, episode_capacity=8)
    h.boxes.create_delay = 0.4
    tokens = [h.token(f"s-{n}") for n in range(4)]
    started = time.monotonic()
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(h.manager.open, tokens))
    assert len({r["episode_id"] for r in results}) == 4
    assert time.monotonic() - started < 1.2


def test_small_validator_clock_skew_is_tolerated(tmp_path):
    h = build(tmp_path)
    ahead = claims("s-1", now=int(h.clock.now) + 25)
    h.manager.open(issue_session_token(ahead, h.validator))
    too_far = claims("s-2", now=int(h.clock.now) + 60)
    with pytest.raises(HTTPException) as error:
        h.manager.open(issue_session_token(too_far, h.validator))
    assert error.value.status_code == 401


def test_the_deadline_is_the_earlier_of_wall_budget_and_token_expiry(tmp_path):
    h = build(tmp_path)
    h.open("s-1", budgets=budgets(wall_s=60))
    assert h.manager.store.get("s-1")["deadline"] == int(h.clock.now) + 60
    h.open("s-2")  # wall 3600 s, token expires in 3600 s
    token_expiry = h.manager.store.get("s-2")["claims"]["expires_at"]
    assert h.manager.store.get("s-2")["deadline"] == token_expiry
    # prepare timeout (600) + slack (60) + wall (60) + grace: it outlives any deadline,
    # and an extract started just before it (grading timeout 600 + slack 60)
    assert h.boxes.boxes["s-1"]["spec"].lifetime_s == 600 + 60 + 60 + 600 + 60


def test_the_env_factory_runs_off_the_event_loop(tmp_path):
    h = build(tmp_path)
    fake_episode_env.FACTORY_CALLS.clear()

    async def serve(token):
        return await asyncio.to_thread(h.manager.open, token)

    asyncio.run(serve(h.token("s-1")))
    assert fake_episode_env.FACTORY_CALLS
    assert all(not loop and name != threading.main_thread().name
               for name, loop in fake_episode_env.FACTORY_CALLS)

    # Called on the event loop itself (a wiring bug), open refuses before the
    # factory or Docker runs: it would block every other request.
    fake_episode_env.FACTORY_CALLS.clear()

    async def on_the_loop(token):
        return h.manager.open(token)

    with pytest.raises(RuntimeError, match="event loop"):
        asyncio.run(on_the_loop(h.token("s-2")))
    assert fake_episode_env.FACTORY_CALLS == [] and "s-2" not in h.boxes.boxes


def test_record_zero_is_signed_by_the_machine_key_with_the_open_time(tmp_path):
    h = build(tmp_path)
    record = h.open()["record"]
    assert record["body"]["at"] == int(h.clock.now)
    assert record["body"]["image"] == IMAGE
    assert h.verify("s-1", require_graded=False).open.at == int(h.clock.now)


def test_a_symlinked_machine_secret_is_refused(tmp_path):
    from reliquary_sandbox_service.episodes.manager import SECRET_FILE, load_machine_secret

    target = tmp_path / "elsewhere"
    target.write_bytes(b"x" * 32)
    target.chmod(0o600)
    (tmp_path / SECRET_FILE).symlink_to(target)
    with pytest.raises(RuntimeError, match="symlink|regular"):
        load_machine_secret(tmp_path)


def test_a_machine_secret_owned_by_someone_else_is_refused(tmp_path, monkeypatch):
    import os

    from reliquary_sandbox_service.episodes.manager import load_machine_secret

    load_machine_secret(tmp_path)
    monkeypatch.setattr(os, "geteuid", lambda: os.stat(tmp_path).st_uid + 1)
    with pytest.raises(RuntimeError, match="owner"):
        load_machine_secret(tmp_path)
