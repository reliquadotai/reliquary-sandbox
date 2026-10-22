"""Episode composition contracts; no CLI, Docker or model execution."""

import asyncio
import contextlib
import json
import signal
import time
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("verifiers.v1")

from reliquary_sandbox_verifiers import cli
from reliquary_sandbox_verifiers import cli_runner as runner

pytest_plugins = ("test_cli_bridge",)


@pytest.fixture
def composition(monkeypatch, capability):
    capability = replace(capability, date_recipe=cli.CLI_DATE_RECIPE)
    events = []
    episode = SimpleNamespace(id="signed-fixture", calls=0, finishes=0, transcripts=0,
        record0={"body": {"tools_version": capability.tools_version, "tools": ["bash", "edit"],
                           "budgets": {"wall_s": 600}}})

    async def finish(**kwargs):
        episode.finishes += 1
        events.append("finish")
        return SimpleNamespace(record={"body": {"status": "graded"}}, state=b"signed-state")

    async def transcript(**kwargs):
        episode.transcripts += 1
        return {"transcript": {"signed": True}}

    episode.finish, episode.transcript = finish, transcript

    class Gateway:
        def __init__(self, url):
            assert url == "https://signed-gateway.invalid"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            events.append("gateway-closed")

        async def open(self, token, *, deadline):
            events.append("open")
            assert token == {"signed": "token"}
            return episode

    @contextlib.asynccontextmanager
    async def mcp(app, deadline):
        events.append("mcp-start")
        try:
            yield "http://127.0.0.1:2"
        finally:
            events.append("mcp-closed")

    @contextlib.asynccontextmanager
    async def model(client, identity, budget):
        assert identity == "fixture/model@" + "a" * 40 and budget == 65536
        events.append("proxy-start")
        try:
            yield "http://127.0.0.1:1/v1", "a" * 22, "trace"
        finally:
            events.append("proxy-closed")

    class Transport:
        def __init__(self, protocol):
            assert protocol == capability.protocol_id

        async def run(self, **kwargs):
            events.append("transport")
            state.transport_args = kwargs

    def make_client(bridge, deadline):
        events.append("client")
        state.bridge, state.effective_deadline = bridge, deadline
        return SimpleNamespace()

    monkeypatch.setattr(cli, "_QUALIFIED", {capability.protocol_id: capability})
    monkeypatch.setattr(runner, "EpisodeClient", Gateway)
    monkeypatch.setattr(runner, "SignedCLITransport", Transport)
    monkeypatch.setattr(runner, "_mcp_endpoint", mcp)
    monkeypatch.setattr(runner, "_model_endpoint", model)
    monkeypatch.setattr(runner, "LEASE_CHECK_SECONDS", 0.001)
    state = SimpleNamespace(events=events, episode=episode, gateway_type=Gateway,
                            transport_type=Transport)
    state.args = dict(protocol_id=capability.protocol_id,
        expected_capability_sha256=runner.capability_sha256(capability),
        context_date=cli.utc_context_date(), gateway_url="https://signed-gateway.invalid",
        token={"signed": "token"}, model="fixture/model@" + "a" * 40, prompt="the trusted task",
        make_model_client=make_client, max_turns=48, max_episode_tokens=65536,
        deadline=time.monotonic() + 180)
    return state


def test_capability_identity_binds_complete_artifacts_prompt_and_runtime(capability):
    identity = runner.capability_identity(capability)
    assert identity["model_tools"] == capability.model_tools
    assert "model_tools_json" not in identity
    assert identity["versions"] == list(capability.versions)
    assert identity["max_turns"] == 48
    assert identity["runtime_image"] == capability.runtime_image
    assert identity["system_prompt_sha256"] == capability.system_prompt_sha256
    assert json.loads(json.dumps(identity)) == identity
    for changed in (replace(capability, qualification_sha256="f" * 64),
                    replace(capability, runtime_image="other@sha256:" + "d" * 64),
                    replace(capability, initial_template_sha256="e" * 64)):
        assert runner.capability_sha256(changed) != runner.capability_sha256(capability)


async def test_unapproved_or_changed_capability_never_opens_or_starts(composition, monkeypatch):
    monkeypatch.delitem(cli._QUALIFIED, composition.args["protocol_id"])
    with pytest.raises(cli.CLIUnavailable):
        await runner.run_cli_episode(**composition.args)
    assert composition.events == []


@pytest.mark.parametrize("change", [
    {"expected_capability_sha256": "0" * 64}, {"context_date": "2000-01-01"},
    {"max_turns": 49}, {"max_turns": True}, {"max_episode_tokens": 65537},
    {"cancelled": lambda: True},
])
async def test_frozen_contract_and_lease_refuse_before_open(composition, change):
    with pytest.raises((cli.CLIUnavailable, ValueError, runner.CLILeaseLost)):
        await runner.run_cli_episode(**{**composition.args, **change})
    assert composition.events == []


async def test_one_episode_reuses_bridge_and_reserves_signed_wall_for_finish(composition):
    loop = asyncio.get_running_loop()
    composition.episode.record0["body"]["budgets"]["wall_s"] = 100
    started = loop.time()
    result = await runner.run_cli_episode(**composition.args)
    assert result.final == {"body": {"status": "graded"}}
    assert result.state == b"signed-state" and result.transcript == {"signed": True}
    assert result.trace == "trace" and result.error is None
    assert composition.events.count("open") == composition.events.count("transport") == 1
    assert composition.episode.finishes == composition.episode.transcripts == 1
    assert composition.bridge.max_turns == 48
    assert composition.bridge.context_date == composition.args["context_date"]
    assert started + 39 <= composition.effective_deadline <= loop.time() + 40
    assert composition.transport_args["bridge"] is composition.bridge
    assert composition.events[-4:] == ["proxy-closed", "mcp-closed", "finish", "gateway-closed"]


@pytest.mark.parametrize("phase", ["open", "transport", "finish"])
async def test_lost_lease_cancels_whole_episode_and_awaits_cleanup(composition, monkeypatch, phase):
    began, released = asyncio.Event(), asyncio.Event()
    lost = False

    async def blocked(*args, **kwargs):
        began.set()
        try:
            await asyncio.Future()
        finally:
            released.set()

    if phase == "open":
        monkeypatch.setattr(composition.gateway_type, "open", blocked)
    elif phase == "transport":
        monkeypatch.setattr(composition.transport_type, "run", blocked)
    else:
        normal_finish = composition.episode.finish

        async def first_finish(**kwargs):
            if not began.is_set():
                await blocked()
            return await normal_finish(**kwargs)

        composition.episode.finish = first_finish
    task = asyncio.create_task(runner.run_cli_episode(**composition.args, cancelled=lambda: lost))
    await asyncio.wait_for(began.wait(), 1)
    lost = True
    with pytest.raises(runner.CLILeaseLost):
        await asyncio.wait_for(task, 1)
    assert released.is_set() and composition.events[-1] == "gateway-closed"
    assert composition.events.count("open") <= 1
    assert composition.events.count("transport") <= 1
    assert composition.episode.finishes == (0 if phase == "open" else 1)
    assert composition.episode.transcripts == 0
    if phase != "open":
        assert composition.events.index("proxy-closed") < composition.events.index("finish")


async def test_execution_failure_finishes_once_and_never_relaunches(composition, monkeypatch):
    async def failure(self, **kwargs):
        composition.events.append("failed-transport")
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(composition.transport_type, "run", failure)
    with pytest.raises(RuntimeError, match="fixture failure"):
        await runner.run_cli_episode(**composition.args)
    assert composition.events.count("open") == composition.events.count("failed-transport") == 1
    assert composition.episode.finishes == 1 and composition.episode.transcripts == 0
    assert composition.events[-1] == "gateway-closed"


@pytest.mark.parametrize("helper", ["_mcp_endpoint", "_model_endpoint"])
async def test_listener_or_proxy_cleanup_failure_cannot_return_success(composition, monkeypatch, helper):
    normal = getattr(runner, helper)

    @contextlib.asynccontextmanager
    async def failed_cleanup(*args):
        async with normal(*args) as value:
            yield value
        raise RuntimeError("cleanup could not be verified")

    monkeypatch.setattr(runner, helper, failed_cleanup)
    with pytest.raises(RuntimeError, match="cleanup could not be verified"):
        await runner.run_cli_episode(**composition.args)
    assert composition.events.count("transport") == composition.episode.finishes == 1
    assert composition.episode.transcripts == 0
    assert composition.events[-1] == "gateway-closed"


async def test_hard_wall_stops_execution_before_finish_reserve(composition, monkeypatch):
    stopped = asyncio.Event()

    async def hang(self, **kwargs):
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    monkeypatch.setattr(runner, "FINISH_MARGIN_S", 0.02)
    monkeypatch.setattr(composition.transport_type, "run", hang)
    with pytest.raises(TimeoutError):
        await runner.run_cli_episode(**{**composition.args,
            "deadline": asyncio.get_running_loop().time() + 0.05})
    assert stopped.is_set() and composition.episode.finishes == 1
    assert composition.events[-1] == "gateway-closed"


async def test_cancellation_finish_is_bounded_and_preserves_failure(composition, monkeypatch):
    stopped = asyncio.Event()

    async def bad_transport(self, **kwargs):
        raise RuntimeError("original failure")

    async def hanging_finish(**kwargs):
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    monkeypatch.setattr(composition.transport_type, "run", bad_transport)
    monkeypatch.setattr(runner, "CANCELLED_FINISH_TIMEOUT_S", 0.01)
    composition.episode.finish = hanging_finish
    with pytest.raises(RuntimeError, match="original failure"):
        await asyncio.wait_for(runner.run_cli_episode(**composition.args), 1)
    assert stopped.is_set() and composition.events[-1] == "gateway-closed"


async def test_caller_cancellation_waits_for_owned_cleanup_without_another_generation(composition, monkeypatch):
    started, finishing, release_finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    normal_finish = composition.episode.finish
    before = asyncio.all_tasks()

    async def generating(self, **kwargs):
        composition.events.append("generation")
        started.set()
        try:
            await asyncio.Future()
        finally:
            composition.events.append("runtime-closed")

    async def finishing_episode(**kwargs):
        finishing.set()
        await release_finish.wait()
        return await normal_finish(**kwargs)

    monkeypatch.setattr(composition.transport_type, "run", generating)
    composition.episode.finish = finishing_episode
    task = asyncio.create_task(runner.run_cli_episode(**composition.args))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    await asyncio.wait_for(finishing.wait(), 1)
    assert not task.done()
    assert composition.events[-2:] == ["proxy-closed", "mcp-closed"]
    release_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    await asyncio.sleep(0)
    assert composition.events.count("generation") == composition.episode.finishes == 1
    assert composition.episode.transcripts == 0
    assert asyncio.all_tasks() <= before


async def test_mcp_listener_is_loopback_and_stops_without_owning_signals():
    pytest.importorskip("uvicorn")
    before = signal.getsignal(signal.SIGTERM)
    path = "a" * 48
    app = cli.SignedMCPApp(SimpleNamespace(), path)
    async with runner._mcp_endpoint(app, asyncio.get_running_loop().time() + 5) as url:
        assert url.startswith("http://127.0.0.1:")
        async with httpx.AsyncClient(trust_env=False, timeout=1) as http:
            response = await http.post(url + "/" + path,
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
            assert response.status_code == 200
        assert signal.getsignal(signal.SIGTERM) == before
    async with httpx.AsyncClient(trust_env=False, timeout=1) as http:
        with pytest.raises(httpx.ConnectError):
            await http.post(url + "/" + path)
    assert signal.getsignal(signal.SIGTERM) == before
