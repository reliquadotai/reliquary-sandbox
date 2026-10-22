"""Signed CLI mediation contracts; these fixtures do not qualify a real CLI."""

import asyncio
import copy
import json
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("verifiers.v1")

from reliquary_sandbox.attest.canonical import canonical_json
from reliquary_sandbox.tool_routing import RoutingError, ToolRouter
from reliquary_sandbox_verifiers.cli import (
    MCP_TOOLS,
    CLIUnavailable,
    SignedCLICapability,
    SignedCLIToolBridge,
    SignedMCPApp,
    initial_template_sha256,
    qualified_capability,
)

PREFIX = [{"role": "system", "content": "qualified static prefix"},
          {"role": "user", "content": "the trusted task"}]


@pytest.fixture
def capability():
    names = (("mcp__signed_episode__bash", "bash"), ("mcp__signed_episode__edit", "edit"))
    tools = [{"type": "function", "function": {"name": name,
              "parameters": tool["inputSchema"], "description": tool["description"]}}
             for (name, _), tool in zip(names, MCP_TOOLS, strict=True)]
    return SignedCLICapability(
        protocol_id="signed-codex-cli-v1", runtime_image="test@sha256:" + "a" * 64,
        model_tools_json=canonical_json(tools), model_tool_names=names,
        artifacts=(("@openai/codex", "0.147.0", "sha512-unit-fixture"),
                   ("@agentclientprotocol/codex-acp", "1.2.0", "sha512-unit-fixture")),
        qualification_sha256="b" * 64,
        runner_argv=("/opt/signed-cli/runtime/bin/python", "/opt/signed-cli/acp-runner.py"),
        runner_sha256="c" * 64,
        initial_template_sha256=initial_template_sha256(PREFIX, "the trusted task"))


class Episode:
    def __init__(self):
        self.calls = []

    async def call(self, turn, k, tool, arguments):
        self.calls.append((turn, k, tool, arguments))
        return SimpleNamespace(record={"body": {
            "turn": turn, "k": k, "tool": tool, "arguments": arguments,
            "output": "signed output", "truncated": False, "timed_out": False}}, final=None)


def make_bridge(capability):
    episode = Episode()
    bridge = SignedCLIToolBridge(ToolRouter(episode, offered=("bash", "edit")), capability,
                                 prompt="the trusted task")
    bridge.validate_messages(PREFIX)
    return bridge, episode


def message(call_id="call0", command="pwd"):
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": call_id, "type": "function", "function": {
            "name": "mcp__signed_episode__bash", "arguments": json.dumps({"command": command})}}]}


def register(bridge, turn=0, assistant=None):
    assistant = assistant or message()
    calls = [(call["id"], call["function"]["name"], call["function"]["arguments"])
             for call in assistant["tool_calls"]]
    bridge.register_turn(turn, calls, assistant_message=assistant)
    return assistant


def test_no_actual_cli_is_enabled_without_a_reviewed_probe():
    for protocol in ("signed-codex-cli-v1", "signed-claude-code-cli-v1", "bash"):
        with pytest.raises(CLIUnavailable, match="no approved"):
            qualified_capability(protocol)


def test_exact_model_schema_refuses_a_builtin_or_modified_mcp_tool(capability):
    capability.require_model_tools(capability.model_tools)
    changed = capability.model_tools
    changed[0]["function"]["description"] += "changed"
    with pytest.raises(RoutingError):
        capability.require_model_tools(changed)
    with pytest.raises(RoutingError):
        capability.require_model_tools(capability.model_tools + [{"type": "custom"}])


async def test_only_the_sampled_call_gets_one_signed_observation(capability):
    bridge, episode = make_bridge(capability)
    bridge.validate_messages(PREFIX)
    assistant = register(bridge)
    with pytest.raises(RoutingError):
        await bridge.call("bash", {"command": "touch /host-marker"})
    assert episode.calls == []
    result = await bridge.call("bash", {"command": "pwd"})
    assert result == bridge.expected_result(0, "call0")
    assert episode.calls == [(0, 0, "bash", {"command": "pwd"})]
    with pytest.raises(RoutingError):
        await bridge.call("bash", {"command": "pwd"})
    history = [*PREFIX, assistant,
               {"role": "tool", "tool_call_id": "call0", "content": "signed output"}]
    bridge.validate_messages(history)
    folded = copy.deepcopy(history)
    folded[-2]["content"] = None
    bridge.validate_messages(folded)
    for alter in (lambda h: h[-1].update(content="unsigned"),
                  lambda h: h[-2].update(content="changed assistant"),
                  lambda h: h.append({"role": "user", "content": "local tool output"}),
                  lambda h: h.pop(), lambda h: h.pop(0)):
        changed = copy.deepcopy(history)
        alter(changed)
        with pytest.raises(RoutingError):
            bridge.validate_messages(changed)


def test_unsigned_startup_context_is_rejected(capability):
    bridge, _ = make_bridge(capability)
    with pytest.raises(RoutingError):
        bridge.validate_messages([*PREFIX, {"role": "user", "content": "local file"}])


def test_bad_calls_and_more_than_48_turns_fail_closed(capability):
    bridge, _ = make_bridge(capability)
    assistant = message()
    with pytest.raises(RoutingError):
        bridge.register_turn(0, [("a", "Bash", '{}')], assistant_message=assistant)
    with pytest.raises(RoutingError):
        bridge.register_turn(0, [("a", "mcp__signed_episode__bash", '{}')],
                             assistant_message=assistant)
    with pytest.raises(RoutingError):
        bridge.register_turn(0, [("a", "mcp__signed_episode__bash", '{"command":"a"}'),
                                 ("b", "mcp__signed_episode__bash", '{"command":"a"}')],
                             assistant_message=assistant)
    assert not bridge.has_pending_call and bridge.model_turns == 0
    for turn in range(48):
        bridge.register_turn(turn, [], assistant_message={"role": "assistant", "content": "ok"})
    with pytest.raises(RoutingError):
        bridge.register_turn(48, [], assistant_message={"role": "assistant", "content": "ok"})


async def test_mcp_endpoint_requires_secret_path_and_exact_registered_call(capability):
    bridge, episode = make_bridge(capability)
    app = SignedMCPApp(bridge, "a" * 32)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                base_url="http://test") as client:
        request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        assert (await client.post("/wrong", json=request)).status_code == 404
        assert (await client.get("/" + "a" * 32)).status_code == 405
        response = await client.post("/" + "a" * 32, json=request)
        assert response.json()["result"]["tools"] == list(MCP_TOOLS)
        request.update(method="tools/call", params={"name": "bash", "arguments": {"command": "pwd"}})
        refused = await client.post("/" + "a" * 32, json=request)
        assert "error" in refused.json() and not episode.calls
        register(bridge)
        accepted = await client.post("/" + "a" * 32, json=request)
        assert accepted.json()["result"] == bridge.expected_result(0, "call0")
        assert len(episode.calls) == 1
        oversized = await client.post("/" + "a" * 32, content=b"x" * (128 * 1024 + 1))
        assert oversized.status_code == 413


async def test_acp_final_reply_must_be_the_final_sampled_non_tool_turn(capability):
    bridge, _ = make_bridge(capability)
    register(bridge)
    await bridge.call("bash", {"command": "pwd"})
    with pytest.raises(RoutingError):
        bridge.require_final_reply("invented answer after a genuine tool")
    bridge.register_turn(1, [], assistant_message={"role": "assistant", "content": "done"})
    bridge.require_final_reply("done")
    with pytest.raises(RoutingError):
        bridge.require_final_reply("done but rewritten")


async def test_an_unsigned_router_refusal_never_becomes_an_mcp_result(capability):
    episode = Episode()
    bridge = SignedCLIToolBridge(ToolRouter(episode, offered=("bash",)), capability,
                                prompt="the trusted task")
    bridge.validate_messages(PREFIX)
    assistant = {"role": "assistant", "content": "", "tool_calls": [{
        "id": "e", "type": "function", "function": {"name": "mcp__signed_episode__edit",
        "arguments": '{"path":"x","old_str":"a","new_str":"b"}'}}]}
    register(bridge, assistant=assistant)
    with pytest.raises(RoutingError, match="signed episode record"):
        await bridge.call("edit", {"path": "x", "old_str": "a", "new_str": "b"})
    assert episode.calls == []


async def test_threaded_mcp_listener_executes_gateway_calls_on_the_owner_loop(capability):
    owner = asyncio.get_running_loop()

    class OwnedEpisode(Episode):
        async def call(self, *args):
            assert asyncio.get_running_loop() is owner
            return await super().call(*args)

    episode = OwnedEpisode()
    bridge = SignedCLIToolBridge(ToolRouter(episode), capability, prompt="the trusted task")
    bridge.validate_messages(PREFIX)
    register(bridge)
    app = SignedMCPApp(bridge, "a" * 32)

    async def request_from_listener_loop():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://test") as client:
            response = await client.post("/" + "a" * 32, json={
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "bash", "arguments": {"command": "pwd"}}})
            return response.json()

    result = await asyncio.to_thread(lambda: asyncio.run(request_from_listener_loop()))
    assert result["result"] == bridge.expected_result(0, "call0")
    assert len(episode.calls) == 1


async def test_mcp_app_accepts_the_existing_servers_required_lifespan(capability):
    bridge, _ = make_bridge(capability)
    app = SignedMCPApp(bridge, "a" * 32)
    events = iter(({"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}))
    sent = []

    async def receive():
        return next(events)

    async def send(value):
        sent.append(value)

    await app({"type": "lifespan"}, receive, send)
    assert sent == [{"type": "lifespan.startup.complete"},
                    {"type": "lifespan.shutdown.complete"}]
