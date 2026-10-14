"""Signed open refusals: the gateway signs every refusal of a fresh token that is
not the miner's doing, bound to the token, and spends the token; a validator verifies it with the machine's
directory key."""

import asyncio
import base64
import json

import httpx
import pytest
from episode_fixtures import budgets
from episode_manager_fixtures import build
from fastapi import HTTPException

from reliquary_sandbox import episode_client
from reliquary_sandbox.attest import OPEN_REFUSAL_DOMAIN, open_refusal_valid, token_sha256
from reliquary_sandbox.episode_client import EpisodeClient, EpisodeError, decode_open_refusal
from reliquary_sandbox_service.episodes import manager as manager_module
from reliquary_sandbox_service.episodes.manager import OPEN_REFUSAL_HEADER

GIB = 1024**3
LIMITED = {"fake-env": "limited_episode_env:sandbox_task"}
FITS = budgets(memory_bytes=3 * GIB)          # index 7 needs 3 GiB
PREPARE_S = 600                               # SandboxTask.prepare_timeout_s, the limited task's default


class Directory:
    def __init__(self, signer, machine_id):
        self.signer, self.machine_id = signer, machine_id

    def public_key(self, machine_id, key_id, at):
        if machine_id != self.machine_id or key_id != self.signer.key_id:
            return None
        return self.signer.private_key.public_key()


def refusal_of(error: HTTPException) -> dict:
    receipt = decode_open_refusal(error.headers[OPEN_REFUSAL_HEADER])
    assert receipt is not None
    return receipt


def refused(h, token) -> HTTPException:
    with pytest.raises(HTTPException) as error:
        h.manager.open(token)
    return error.value


def resources(tmp_path, **overrides):
    """The harness: reservations on, 1 CPU and 16 GiB for agent boxes."""
    values = dict(episode_envs=LIMITED, episode_nano_cpus=2_000_000_000, episode_capacity=8,
                  episode_cpu_millis_total=1_000, episode_memory_bytes_total=16 * GIB,
                  episode_grading_cpu_millis_total=2_000, episode_grading_memory_bytes_total=8 * GIB,
                  episode_present_window_s=120)
    values.update(overrides)
    return build(tmp_path, **values)


def test_a_full_machine_signs_its_refusal(tmp_path):
    h = build(tmp_path, episode_envs=LIMITED, episode_capacity=1)
    h.open("s-1", index=7, budgets=FITS)
    token = h.token("s-2", index=7, budgets=FITS)
    error = refused(h, token)
    assert error.status_code == 503 and error.detail == "no free episode slot"
    assert error.headers["Retry-After"] == "5"
    receipt = refusal_of(error)
    document = receipt["document"]
    assert (document["kind"], document["reason"], document["session_id"]) == ("open_refusal", "no_free_slot", "s-2")
    token_sha = document["token_sha256"]
    assert token_sha == token_sha256(token)
    directory = Directory(h.manager.signer, h.settings.machine_id)
    assert open_refusal_valid(receipt, directory, machine_id=h.settings.machine_id, session_id="s-2",
                              token_sha256=token_sha, now=document["at"])
    # bound to its session, its token and its machine
    assert not open_refusal_valid(receipt, directory, machine_id=h.settings.machine_id, session_id="s-3",
                                  token_sha256=token_sha, now=document["at"])
    assert not open_refusal_valid(receipt, directory, machine_id=h.settings.machine_id, session_id="s-2",
                                  token_sha256="0" * 64, now=document["at"])
    assert not open_refusal_valid(receipt, directory, machine_id="other", session_id="s-2",
                                  token_sha256=token_sha, now=document["at"])


def test_a_machine_without_free_resources_signs_its_refusal(tmp_path):
    h = resources(tmp_path)
    h.open("s-1", index=7, budgets=FITS)
    h.open("s-2", index=7, budgets=FITS)                                    # 2 x 500 millis = 1000
    error = refused(h, h.token("s-3", index=7, budgets=FITS))
    assert error.status_code == 503 and error.detail == "no free resources"
    assert error.headers["Retry-After"] == "5"
    assert refusal_of(error)["document"]["reason"] == "no_free_resources"
    assert h.manager.store.get("s-3")["final_status"] == "refused"


def test_a_signed_refusal_spends_the_token(tmp_path):
    h = build(tmp_path, episode_envs=LIMITED, episode_capacity=1)
    first = h.open("s-1", index=7, budgets=FITS)
    token = h.token("s-2", index=7, budgets=FITS)
    signed = refused(h, token)
    assert OPEN_REFUSAL_HEADER in signed.headers
    h.manager.finish("s-1", first["episode_key"])                           # the slot is free again
    again = refused(h, token)                                               # ... but this token never opens
    assert again.status_code == 409 and again.detail["final"] is None
    # A resend of the very token (its first answer may have been lost) gets the same receipt, nothing new.
    assert again.headers[OPEN_REFUSAL_HEADER] == signed.headers[OPEN_REFUSAL_HEADER]
    assert "s-2" not in h.boxes.boxes and h.manager.active_count() == 0
    other = refused(h, h.token("s-2", index=7, budgets=budgets(memory_bytes=4 * GIB)))   # same id, other token
    assert other.status_code == 409 and OPEN_REFUSAL_HEADER not in (other.headers or {})


def test_a_refusal_of_the_validators_claims_is_signed_and_spends_the_token(tmp_path):
    h = build(tmp_path, episode_envs=LIMITED, episode_capacity=2)
    error = refused(h, h.token("s-1", index=7, budgets=budgets(memory_bytes=GIB)))      # the order's budget is short
    assert error.status_code == 422 and "below the task's needs" in str(error.detail)
    assert refusal_of(error)["document"]["reason"] == "task_refused"
    assert h.manager.store.get("s-1")["final_status"] == "refused"


@pytest.mark.parametrize("overrides, detail", [
    (dict(index=99), None),                                                  # unresolvable task
    (dict(image="registry.example/other@sha256:" + "b" * 64), "the token's image is not this task's image"),
    (dict(budgets=budgets(memory_bytes=3 * GIB, pids=10**6)), "session budgets exceed this machine's limits"),
    (dict(expires_at=10**10), "session token validity exceeds this machine's retention"),
])
def test_every_claim_this_machine_cannot_serve_is_signed(tmp_path, overrides, detail):
    h = build(tmp_path, episode_envs=LIMITED, episode_capacity=2)
    error = refused(h, h.token("s-1", **{"index": 7, "budgets": FITS, **overrides}))
    assert error.status_code == 422
    if detail is not None:
        assert error.detail == detail
    assert refusal_of(error)["document"]["reason"] == "task_refused"
    assert h.manager.store.get("s-1")["final_status"] == "refused" and "s-1" not in h.boxes.boxes


def test_an_absent_image_is_signed_and_a_later_pull_does_not_reopen_the_token(tmp_path):
    h = build(tmp_path, episode_envs=LIMITED, episode_capacity=2)
    h.boxes.image_present = lambda image: False
    token = h.token("s-1", index=7, budgets=FITS)
    error = refused(h, token)
    assert error.status_code == 409 and error.detail == "image not present on this machine"
    assert refusal_of(error)["document"]["reason"] == "task_refused"
    del h.boxes.image_present                                                # pulled
    assert refused(h, token).status_code == 409 and "s-1" not in h.boxes.boxes


def test_a_box_smaller_than_the_task_declares_is_signed(tmp_path):
    h = resources(tmp_path, episode_nano_cpus=1_000_000_000)
    error = refused(h, h.token("s-1", index=8, budgets=FITS))               # index 8 wants 8 CPUs
    assert error.detail == "the task needs more CPUs per box than this machine gives"
    assert refusal_of(error)["document"]["reason"] == "task_refused"


def test_a_failed_open_signs_its_failure(tmp_path, monkeypatch):
    h = build(tmp_path, episode_envs=LIMITED, episode_capacity=2)

    def broken(*args, **kwargs):
        raise RuntimeError("prepare failed")

    monkeypatch.setattr(h.manager, "_start", broken)
    token = h.token("s-1", index=7, budgets=FITS)
    error = refused(h, token)
    assert error.status_code == 503 and error.detail == manager_module.OPEN_FAILED
    assert "Retry-After" not in error.headers
    assert refusal_of(error)["document"]["reason"] == "open_failed"
    assert h.manager.store.get("s-1")["final_status"] == "open_failed"
    again = refused(h, token)                                               # a resend: same receipt
    assert again.status_code == 409 and again.headers[OPEN_REFUSAL_HEADER] == error.headers[OPEN_REFUSAL_HEADER]


def test_a_preparation_that_outlasts_the_token_is_not_signed(tmp_path, monkeypatch):
    h = build(tmp_path, episode_envs=LIMITED, episode_capacity=2)

    def late(*args, **kwargs):
        raise manager_module._Expired("preparation outlasted the session token")

    monkeypatch.setattr(h.manager, "_start", late)
    token = h.token("s-1", index=7, budgets=FITS)
    error = refused(h, token)
    assert error.status_code == 503 and OPEN_REFUSAL_HEADER not in (error.headers or {})
    assert OPEN_REFUSAL_HEADER not in (refused(h, token).headers or {})     # nor on a resend


def test_a_token_presented_too_close_to_its_expiry_buys_nothing(tmp_path):
    h = build(tmp_path, episode_envs=LIMITED, episode_capacity=2)
    token = h.token("s-1", index=7, budgets=FITS, expires_at=int(h.clock.now) + 30)   # no time to prepare a box
    error = refused(h, token)
    assert error.status_code == 401 and OPEN_REFUSAL_HEADER not in (error.headers or {})
    assert h.manager.store.get("s-1") is None and "s-1" not in h.boxes.boxes          # neither signed nor spent


def test_a_late_token_buys_nothing_even_when_this_machine_would_refuse_it(tmp_path):
    """The late check comes before every signed refusal. A token held until one second
    before its expiry on a full machine, or with claims this machine cannot serve, earns no receipt."""
    h = build(tmp_path, episode_envs=LIMITED, episode_capacity=1)
    h.open("s-1", index=7, budgets=FITS)
    now = int(h.clock.now)
    for session_id, overrides in [("s-2", dict(budgets=FITS)),                         # full machine
                                  ("s-3", dict(budgets=budgets(memory_bytes=GIB))),     # short budget
                                  ("s-4", dict(index=99, budgets=FITS))]:               # unresolvable
        error = refused(h, h.token(session_id, **{"index": 7, **overrides}, expires_at=now + 1))
        assert error.status_code == 401 and OPEN_REFUSAL_HEADER not in (error.headers or {})
        assert h.manager.store.get(session_id) is None


def test_the_late_boundary_is_the_preparation_time(tmp_path):
    h = build(tmp_path, episode_envs=LIMITED, episode_capacity=1)
    h.open("s-1", index=7, budgets=FITS)
    now = int(h.clock.now)
    boundary = now + PREPARE_S + manager_module.PREPARE_SLACK_S
    assert refused(h, h.token("s-2", index=7, budgets=FITS, expires_at=boundary)).status_code == 401
    error = refused(h, h.token("s-3", index=7, budgets=FITS, expires_at=boundary + 1))
    assert error.status_code == 503 and refusal_of(error)["document"]["reason"] == "no_free_slot"


def test_a_token_presented_after_its_present_window_buys_nothing(tmp_path):
    h = build(tmp_path, episode_envs=LIMITED, episode_capacity=2, episode_present_window_s=120)
    token = h.token("s-1", index=7, budgets=FITS, issued_at=int(h.clock.now) - 121)
    error = refused(h, token)
    assert error.status_code == 401 and OPEN_REFUSAL_HEADER not in (error.headers or {})
    assert h.manager.store.get("s-1") is None
    h.open("s-2", index=7, budgets=FITS, issued_at=int(h.clock.now) - 120)     # at the window: opens


def test_a_refusal_for_the_miners_own_reasons_carries_no_receipt(tmp_path):
    h = build(tmp_path, episode_envs=LIMITED, episode_capacity=2)
    h.open("s-1", index=7, budgets=FITS)
    error = refused(h, h.token("s-1", index=7, budgets=budgets(memory_bytes=4 * GIB)))   # another token, same id
    assert error.status_code == 409 and OPEN_REFUSAL_HEADER not in (error.headers or {})
    now = int(h.clock.now)
    for token, status in [(h.token("s-5", expires_at=now - 1, issued_at=now - 100), 401),   # expired
                          (h.token("s-6", machine_id="other"), 403),                         # foreign
                          (h.token("bad id!"), 422)]:                                        # invalid id
        error = refused(h, token)
        assert error.status_code == status and OPEN_REFUSAL_HEADER not in (error.headers or {})
    assert all(h.manager.store.get(i) is None for i in ("s-5", "s-6", "bad id!"))


def test_a_token_that_cannot_be_spent_is_never_signed(tmp_path):
    """A concurrent open of the same token got its row first: its box may be running, so nothing is
    signed (the receipt would let the miner keep the box and be cleared)."""
    h = build(tmp_path, episode_envs=LIMITED, episode_capacity=2)
    token = h.token("s-1", index=7, budgets=FITS)
    h.manager.open(token)
    claims = h.manager._admissible(token)
    error = h.manager._refuse_signed(claims, token_sha256(token), token, 503, "no free episode slot",
                                     "no_free_slot")
    assert error.status_code == 409 and OPEN_REFUSAL_HEADER not in (error.headers or {})
    assert h.manager.store.get("s-1")["status"] == "RUNNING"


def test_a_tombstone_answers_like_an_unknown_episode_and_is_pruned_with_its_token(tmp_path):
    h = build(tmp_path, episode_envs=LIMITED, episode_capacity=2)
    refused(h, h.token("s-1", index=7, budgets=budgets(memory_bytes=GIB)))
    with pytest.raises(HTTPException) as error:
        h.manager.finish("s-1", "0" * 64)
    assert error.value.status_code == 401
    assert h.manager.active_count() == 0
    h.clock.now += 3600 + 1                                                  # its token expired
    h.manager.reap()
    assert h.manager.store.get("s-1") is None


def test_a_forged_or_malformed_receipt_is_refused(tmp_path):
    h = build(tmp_path, episode_envs=LIMITED, episode_capacity=1)
    h.open("s-1", index=7, budgets=FITS)
    receipt = refusal_of(refused(h, h.token("s-2", index=7, budgets=FITS)))
    directory = Directory(h.manager.signer, h.settings.machine_id)
    kwargs = dict(machine_id=h.settings.machine_id, session_id="s-2",
                  token_sha256=receipt["document"]["token_sha256"], now=receipt["document"]["at"])
    assert open_refusal_valid(receipt, directory, **kwargs)
    forged = {"document": {**receipt["document"], "reason": "open_failed"}, "signature": receipt["signature"]}
    assert not open_refusal_valid(forged, directory, **kwargs)
    unknown = {"document": {**receipt["document"], "reason": "miner_quit"}, "signature": receipt["signature"]}
    assert not open_refusal_valid(unknown, directory, **kwargs)
    extra = {"document": {**receipt["document"], "x": 1}, "signature": receipt["signature"]}
    assert not open_refusal_valid(extra, directory, **kwargs)
    assert not open_refusal_valid({"document": receipt["document"]}, directory, **kwargs)
    assert not open_refusal_valid("x", directory, **kwargs)
    assert not open_refusal_valid(receipt, directory, **{**kwargs, "now": receipt["document"]["at"] + 8 * 86_400})
    assert not open_refusal_valid(receipt, directory, **{**kwargs, "now": receipt["document"]["at"] - 31})

    class Broken:
        def public_key(self, *args):
            raise RuntimeError("directory down")

    assert not open_refusal_valid(receipt, Broken(), **kwargs)
    assert decode_open_refusal("not base64 json") is None
    assert decode_open_refusal(base64.b64encode(b"[1]").decode()) is None
    assert decode_open_refusal(None) is None
    assert OPEN_REFUSAL_DOMAIN.endswith(b"\x00")


def test_the_header_is_base64_of_canonical_json_and_the_client_reads_the_same_name(tmp_path):
    h = build(tmp_path, episode_envs=LIMITED, episode_capacity=1)
    h.open("s-1", index=7, budgets=FITS)
    error = refused(h, h.token("s-2", index=7, budgets=FITS))
    raw = base64.b64decode(error.headers[OPEN_REFUSAL_HEADER])
    assert set(json.loads(raw)) == {"document", "signature"}
    assert manager_module.OPEN_REFUSAL_HEADER == episode_client.OPEN_REFUSAL_HEADER


def test_the_client_exposes_the_receipt_on_its_error(tmp_path):
    h = build(tmp_path, episode_envs=LIMITED, episode_capacity=1)
    h.open("s-1", index=7, budgets=FITS)
    error = refused(h, h.token("s-2", index=7, budgets=FITS))

    def full(request):
        return httpx.Response(error.status_code, json={"detail": error.detail}, headers=error.headers)

    def plain(request):
        return httpx.Response(403, json={"detail": "session token is for another machine"})

    async def open_through(handler):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            with pytest.raises(EpisodeError) as raised:
                await EpisodeClient("http://gateway", http=http).open({"any": "token"})
        return raised.value

    signed = asyncio.run(open_through(full))
    assert signed.status == 503 and signed.open_refusal == refusal_of(error)
    assert asyncio.run(open_through(plain)).open_refusal is None
