"""CPU and memory reservations on a real gVisor host. Opt-in, like test_episode_live.py:
RUN_EPISODE_LIVE=1 on the episode test host, against the isolated daemon of scripts/test-xfs-docker.sh."""

import asyncio
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from episode_fixtures import budgets, claims, directory_for
from fastapi import HTTPException
from fastapi.testclient import TestClient
from test_episode_live import Gateway, client, image, live_settings, manager_for, report  # noqa: F401
from test_service import FakeRuntime

from reliquary_sandbox.attest import (
    Signer,
    issue_session_token,
    load_private_key,
    open_refusal_valid,
    token_sha256,
)
from reliquary_sandbox.episode_client import EpisodeClient, EpisodeError, decode_open_refusal
from reliquary_sandbox_service.app import create_app
from reliquary_sandbox_service.episodes.capacity import capacity_report
from reliquary_sandbox_service.episodes.manager import OPEN_REFUSAL_HEADER
from reliquary_sandbox_service.task_runtime import TaskBoxRuntime

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("RUN_EPISODE_LIVE") != "1",
                       reason="set RUN_EPISODE_LIVE=1 on the episode test host"),
]
GIB = 1024**3
HOST_MILLIS = (os.cpu_count() or 1) * 1000
UNAPPROVED = "ubuntu@sha256:" + "0" * 64


def receipt_valid(receipt, machine, session_id, token):
    return open_refusal_valid(receipt, directory_for(machine), machine_id=receipt["document"]["machine_id"],
                              session_id=session_id, token_sha256=token_sha256(token))


def token_for(validator, image, session_id, memory_bytes=GIB):  # noqa: F811
    return issue_session_token(claims(session_id, env="live-env", image=image,
                                      budgets=budgets(per_call_timeout_s=10, memory_bytes=memory_bytes)),
                               validator)


@pytest.fixture
def reserved(tmp_path, client, image):  # noqa: F811
    host_millis = (os.cpu_count() or 1) * 1000
    settings, validator = live_settings(tmp_path, image, episode_capacity=4, episode_nano_cpus=1_000_000_000,
                                        episode_cpu_millis_total=min(host_millis, 1_000),
                                        episode_memory_bytes_total=3 * GIB,
                                        episode_grading_cpu_millis_total=1_000,
                                        episode_grading_memory_bytes_total=2 * GIB,
                                        episode_present_window_s=120)
    manager, boxes, machine = manager_for(settings, client, uuid.uuid4().hex)
    yield Gateway(manager, boxes, validator, machine, image)
    for box_id in boxes.managed():
        boxes.remove(box_id)


def test_a_reserved_episode_holds_the_host_until_it_ends(reserved):
    g = reserved
    g.manager.host_facts = g.boxes.preflight()
    key = g.open("s-1", memory_bytes=GIB)                           # its agent box: 1 CPU, 1 GiB
    host_config = g.boxes.client.containers.get(g.boxes._name("s-1")).attrs["HostConfig"]
    report("agent_box", runtime=host_config["Runtime"], nano_cpus=host_config["NanoCpus"],
           memory=host_config["Memory"])
    assert host_config["Runtime"] == "runsc" and host_config["NanoCpus"] == 1_000_000_000
    held = capacity_report(g.manager)["document"]["resources"]
    report("held", **held)
    assert held["free_cpu_millis"] == held["cpu_millis"] - 1_000
    with pytest.raises(HTTPException) as error:
        g.open("s-2", memory_bytes=GIB)
    receipt = decode_open_refusal((error.value.headers or {}).get(OPEN_REFUSAL_HEADER))
    report("second_open", status=error.value.status_code, detail=error.value.detail,
           receipt=OPEN_REFUSAL_HEADER in (error.value.headers or {}),
           reason=(receipt or {}).get("document", {}).get("reason"))
    assert error.value.status_code == 503 and error.value.detail == "no free resources"
    assert OPEN_REFUSAL_HEADER in error.value.headers                # signed: the miner is not charged
    assert receipt["document"]["reason"] == "no_free_resources" and receipt["document"]["session_id"] == "s-2"
    assert held["sessions"] == ["s-1"]                              # the report lists what it counts
    assert g.call("s-1", key, 0, "bash", {"command": "nproc"})["exit_code"] == 0
    still = capacity_report(g.manager)["document"]["resources"]
    assert still["free_cpu_millis"] == held["free_cpu_millis"]       # held while the episode runs
    final = g.manager.finish("s-1", key)["record"]["body"]
    pool = capacity_report(g.manager)["document"]["grading_pool"]
    freed = capacity_report(g.manager)["document"]["resources"]
    report("first_final", status=final["status"], reward=final.get("reward"), pool=pool,
           free_cpu_millis=freed["free_cpu_millis"], sessions=freed["sessions"])
    assert final["status"] == "graded" and pool["timeouts"] == 0 and pool["waiting"] == 0
    assert pool["used_cpu_millis"] == 0 and freed["free_cpu_millis"] == freed["cpu_millis"]
    with pytest.raises(HTTPException) as spent:
        g.open("s-2", memory_bytes=GIB)                             # the refused session id stays spent
    assert spent.value.status_code == 409
    g.open("s-3", memory_bytes=GIB)
    report("after", **capacity_report(g.manager)["document"]["resources"])


def test_the_signed_refusal_reaches_the_client_through_the_route(tmp_path, client, image):  # noqa: F811
    """The real app (lifespan, recovery, routes, middleware) on the real gVisor runtime: a full machine's
    refusal carries the receipt in the HTTP response, and the episode client exposes it."""
    settings, validator = live_settings(tmp_path, image, episode_capacity=4, episode_nano_cpus=1_000_000_000,
                                        episode_cpu_millis_total=min(HOST_MILLIS, 1_000),
                                        episode_memory_bytes_total=3 * GIB,
                                        episode_grading_cpu_millis_total=1_000,
                                        episode_grading_memory_bytes_total=2 * GIB,
                                        episode_present_window_s=120)
    machine = Signer(settings.machine_key_id, load_private_key(settings.machine_key_file))
    boxes = TaskBoxRuntime(settings, client=client, namespace=uuid.uuid4().hex)
    app = create_app(settings, FakeRuntime(), task_runtime=boxes)
    try:
        with TestClient(app) as http:
            opened = http.post("/episodes", json={"token": token_for(validator, image, "r-1")})
            assert opened.status_code == 200, opened.text
            refused_token = token_for(validator, image, "r-2")
            refused = http.post("/episodes", json={"token": refused_token})
            receipt = decode_open_refusal(refused.headers.get(OPEN_REFUSAL_HEADER))
            report("route_refusal", status=refused.status_code, detail=refused.json().get("detail"),
                   retry_after=refused.headers.get("Retry-After"), header=receipt is not None,
                   reason=(receipt or {}).get("document", {}).get("reason"),
                   valid=receipt is not None and receipt_valid(receipt, machine, "r-2", refused_token))
            assert refused.status_code == 503 and receipt is not None
            assert receipt_valid(receipt, machine, "r-2", refused_token)
            again = http.post("/episodes", json={"token": refused_token})   # a resend: the same receipt
            assert again.status_code == 409 and again.headers[OPEN_REFUSAL_HEADER] == refused.headers[
                OPEN_REFUSAL_HEADER]

            client_token = token_for(validator, image, "r-3")

            async def through_the_client():
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport) as session:
                    with pytest.raises(EpisodeError) as error:
                        await EpisodeClient("http://gateway", http=session).open(client_token)
                return error.value

            error = asyncio.run(through_the_client())
            report("client_refusal", status=error.status, detail=error.detail,
                   receipt=error.open_refusal is not None,
                   valid=error.open_refusal is not None
                   and receipt_valid(error.open_refusal, machine, "r-3", client_token))
            assert error.status == 503 and receipt_valid(error.open_refusal, machine, "r-3", client_token)

            key = opened.json()["episode_key"]
            finished = http.post("/episodes/r-1/finish", headers={"Authorization": f"Bearer {key}"})
            assert finished.status_code == 200 and finished.json()["record"]["body"]["status"] == "graded"
    finally:
        for box_id in boxes.managed():
            boxes.remove(box_id)


def test_an_unapproved_image_open_is_signed(tmp_path, client, image):  # noqa: F811
    settings, validator = live_settings(tmp_path, image, episode_images=[UNAPPROVED],
                                        episode_cpu_millis_total=min(HOST_MILLIS, 1_000),
                                        episode_memory_bytes_total=3 * GIB,
                                        episode_grading_cpu_millis_total=1_000,
                                        episode_grading_memory_bytes_total=2 * GIB,
                                        episode_present_window_s=120)
    manager, boxes, machine = manager_for(settings, client, uuid.uuid4().hex)
    token = token_for(validator, image, "u-1")
    with pytest.raises(HTTPException) as error:
        manager.open(token)
    receipt = decode_open_refusal((error.value.headers or {}).get(OPEN_REFUSAL_HEADER))
    report("unapproved_image", status=error.value.status_code, detail=error.value.detail,
           reason=(receipt or {}).get("document", {}).get("reason"),
           valid=receipt is not None and receipt_valid(receipt, machine, "u-1", token))
    assert error.value.status_code == 422 and error.value.detail == "image not approved on this machine"
    assert receipt_valid(receipt, machine, "u-1", token) and receipt["document"]["reason"] == "task_refused"
    with pytest.raises(HTTPException) as again:
        manager.open(token)                                          # spent; the same receipt again
    assert again.value.status_code == 409
    assert again.value.headers[OPEN_REFUSAL_HEADER] == error.value.headers[OPEN_REFUSAL_HEADER]
    assert boxes.managed() == {}


def test_two_gradings_share_the_grading_pool_while_both_agents_hold_the_host(tmp_path, client, image):  # noqa: F811
    """Two 1-CPU episodes fill a 2-CPU host; a third is refused. Their gradings run in a 1-CPU pool, so the
    second waits for the first (waits >= 1, no timeout), and the agent reservations hold until each final."""
    if HOST_MILLIS < 2_000:
        pytest.skip("needs two CPUs")
    settings, validator = live_settings(tmp_path, image, episode_capacity=4, episode_nano_cpus=1_000_000_000,
                                        episode_cpu_millis_total=2_000, episode_memory_bytes_total=3 * GIB,
                                        episode_grading_cpu_millis_total=1_000,
                                        episode_grading_memory_bytes_total=2 * GIB,
                                        episode_present_window_s=120)
    manager, boxes, machine = manager_for(settings, client, uuid.uuid4().hex)
    g = Gateway(manager, boxes, validator, machine, image)
    try:
        keys = {sid: g.open(sid, memory_bytes=GIB) for sid in ("w-1", "w-2")}
        with pytest.raises(HTTPException) as full:
            g.open("w-3", memory_bytes=GIB)
        assert full.value.status_code == 503 and OPEN_REFUSAL_HEADER in full.value.headers
        for sid, key in keys.items():
            assert g.call(sid, key, 0, "bash", {"command": "echo 42 > answer.txt"})["exit_code"] == 0
        samples, done = [], threading.Event()

        def watch():
            while not done.is_set():
                document = capacity_report(manager)["document"]
                samples.append((time.monotonic(), document["grading_pool"], document["resources"]))
                time.sleep(0.1)

        watcher = threading.Thread(target=watch)
        watcher.start()
        began = time.monotonic()
        try:
            with ThreadPoolExecutor(2) as pool:
                futures = {sid: pool.submit(manager.finish, sid, key) for sid, key in keys.items()}
                finals = {sid: f.result()["record"]["body"] for sid, f in futures.items()}
        finally:
            done.set()
            watcher.join()
        finish_s = time.monotonic() - began
        end = capacity_report(manager)["document"]
        report("pool_two_gradings", finish_s=round(finish_s, 2), samples=len(samples),
               max_waiting=max(s[1]["waiting"] for s in samples),
               max_used_grading_cpu=max(s[1]["used_cpu_millis"] for s in samples),
               min_agent_free_cpu=min(s[2]["free_cpu_millis"] for s in samples),
               finals={sid: [f["status"], f.get("reward")] for sid, f in finals.items()},
               pool=end["grading_pool"], resources_free_cpu=end["resources"]["free_cpu_millis"])
        assert all(f["status"] == "graded" and f["reward"] == 1.0 for f in finals.values()), finals
        assert max(s[1]["used_cpu_millis"] for s in samples) <= 1_000      # never two grading boxes at once
        assert end["grading_pool"]["waits"] >= 1 and end["grading_pool"]["timeouts"] == 0
        assert end["grading_pool"]["waiting"] == 0 and end["grading_pool"]["used_cpu_millis"] == 0
        assert end["resources"]["free_cpu_millis"] == 2_000
        g.open("w-4", memory_bytes=GIB)                                   # the host is free again
    finally:
        for box_id in boxes.managed():
            boxes.remove(box_id)
