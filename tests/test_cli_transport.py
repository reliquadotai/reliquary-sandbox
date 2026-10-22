"""Transport safety checks without launching a CLI, container or server."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("verifiers.v1")

from reliquary_sandbox_verifiers.cli import CLIUnavailable
from reliquary_sandbox_verifiers.cli_transport import (
    MAX_PACKET_BYTES,
    SignedCLITransport,
    _cleanup,
    _packet,
    _PacketReader,
    docker_runtime,
    require_isolated_runtime,
    require_pinned_cli_sources,
)

pytest_plugins = ("test_cli_bridge",)


def test_pinned_actual_cli_sources_match():
    require_pinned_cli_sources()


def test_runtime_gate_has_no_local_or_unrestricted_fallback(capability):
    runtime = docker_runtime(capability)
    require_isolated_runtime(runtime, capability)
    with pytest.raises(CLIUnavailable):
        require_isolated_runtime(SimpleNamespace(config=runtime.config), capability)
    runtime.env["HOST_AUTH"] = "must not cross"
    with pytest.raises(CLIUnavailable):
        require_isolated_runtime(runtime, capability)
    runtime.env.clear()
    runtime.config.allow = ["*"]
    with pytest.raises(CLIUnavailable):
        require_isolated_runtime(runtime, capability)


async def chunks(*items):
    for item in items:
        yield item


async def test_acp_framing_is_finite_and_rejects_truncation_and_oversize():
    packet = _packet({"ok": True, "reply": "done"})
    assert await _PacketReader(chunks(packet[:3], packet[3:])).read() == {"ok": True, "reply": "done"}
    with pytest.raises(EOFError):
        await _PacketReader(chunks(packet[:-1])).read()
    with pytest.raises(ValueError, match="size bound"):
        await _PacketReader(chunks((MAX_PACKET_BYTES + 1).to_bytes(8, "big"))).read()
    with pytest.raises(ValueError, match="bound"):
        _packet({"reply": "x" * MAX_PACKET_BYTES})


async def test_cleanup_removes_the_runtime_even_when_process_stop_fails():
    calls = []

    async def terminate():
        calls.append("terminate")
        raise RuntimeError("lost session")

    async def kill():
        calls.append("kill")

    async def wait():
        calls.append("wait")
        if calls.count("wait") == 1:
            raise RuntimeError("still alive")
        return 0

    async def stop():
        calls.append("runtime.stop")

    stderr = asyncio.create_task(asyncio.sleep(60))
    await _cleanup(SimpleNamespace(stop=stop),
                   SimpleNamespace(terminate=terminate, kill=kill, wait=wait), stderr)
    assert calls == ["terminate", "wait", "kill", "wait", "runtime.stop"]
    assert stderr.cancelled()


async def test_exact_upstream_proxy_token_is_accepted_without_opening_a_runtime(capability):
    class ReachedFactory(Exception):
        pass

    class Probe(SignedCLITransport):
        def __init__(self):
            self.capability = capability
            self.runtime_factory = self.factory

        def factory(self, cap):
            raise ReachedFactory

        def _require_approved(self, bridge):
            assert bridge.capability == self.capability

    args = dict(model="fixture", prompt="task", bridge=SimpleNamespace(capability=capability),
                endpoint="http://127.0.0.1:1/v1", mcp_url="http://127.0.0.1:2/mcp",
                deadline=asyncio.get_running_loop().time() + 30)
    with pytest.raises(ReachedFactory):
        await Probe().run(secret="a" * 22, **args)
    for invalid in ("a" * 21, "a" * 21 + "!"):
        with pytest.raises(ValueError, match="fresh per-run proxy secret"):
            await Probe().run(secret=invalid, **args)
