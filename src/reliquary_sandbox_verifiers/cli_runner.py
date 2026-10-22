"""Compose one approved CLI with signed tools and an injected checkpoint client.

This does not qualify a CLI, a model endpoint or an operator placement. The caller
binds the capability and context date in its signed grant and owns model budgets.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import secrets
import socket
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.interception import InterceptionServer
from verifiers.v1.session import RolloutSession
from verifiers.v1.trace import AgentInfo, Trace, TraceTask
from verifiers.v1.types import Sampling

from reliquary_sandbox.attest.canonical import canonical_json
from reliquary_sandbox.episode_client import EpisodeClient
from reliquary_sandbox.observation import require_tools_version
from reliquary_sandbox.tool_routing import ToolRouter

from .cli import (
    CLI_DATE_RECIPE,
    MAX_CLI_TURNS,
    CLIUnavailable,
    SignedCLICapability,
    SignedCLIToolBridge,
    SignedMCPApp,
    qualified_capability,
    require_context_date,
    utc_context_date,
)
from .cli_transport import SignedCLITransport
from .runner import CANCELLED_FINISH_TIMEOUT_S, FINISH_MARGIN_S, SandboxEpisodeResult

SERVER_TIMEOUT_S = 5.0
LEASE_CHECK_SECONDS = 0.1


class CLILeaseLost(RuntimeError):
    """The caller's lease ended; no result from this episode may be submitted."""


def capability_identity(capability: SignedCLICapability) -> dict[str, Any]:
    identity = asdict(capability)
    del identity["model_tools_json"]
    identity = {"schema": "reliquary/signed-cli-capability/v1", **identity,
            "model_tools": capability.model_tools, "versions": list(capability.versions),
            "date_template_sha256": capability.date_template_sha256, "max_turns": MAX_CLI_TURNS}
    return json.loads(canonical_json(identity))


def capability_sha256(capability: SignedCLICapability) -> str:
    return hashlib.sha256(canonical_json(capability_identity(capability))).hexdigest()


@contextlib.asynccontextmanager
async def _mcp_endpoint(app: SignedMCPApp, deadline: float):
    import uvicorn

    class LocalServer(uvicorn.Server):
        def capture_signals(self):
            # The operator owns process signals, including lease shutdown.
            return contextlib.nullcontext()

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.setblocking(False)
        port = listener.getsockname()[1]
        server = LocalServer(uvicorn.Config(app, lifespan="on", ws="none", access_log=False,
            log_level="critical", timeout_keep_alive=1, timeout_graceful_shutdown=1))
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout_at(min(deadline, asyncio.get_running_loop().time() + SERVER_TIMEOUT_S)):
                while not server.started:
                    if serving.done():
                        await serving
                        raise CLIUnavailable("the signed MCP listener did not start")
                    await asyncio.sleep(0.01)
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            try:
                await asyncio.wait_for(asyncio.shield(serving), SERVER_TIMEOUT_S)
            except TimeoutError:
                serving.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await serving
                raise CLIUnavailable("the signed MCP listener did not stop") from None


@contextlib.asynccontextmanager
async def _model_endpoint(client, model: str, max_episode_tokens: int):
    config = EvalClientConfig(base_url="http://127.0.0.1:1", api_key_var="UNUSED_SIGNED_CLI_KEY")
    context = ModelContext(model=model, client=config, sampling=Sampling(max_tokens=max_episode_tokens))
    trace = Trace(task=TraceTask(type="signed-cli", data={}), agent=AgentInfo(config=AgentConfig()))
    async with InterceptionServer() as server:
        server.clients[config.model_dump_json()] = client
        async with server.acquire(RolloutSession(ctx=context, trace=trace)) as (url, secret, _):
            yield url + "/v1", secret, trace


async def _finish_cancelled(episode) -> None:
    async def finish():
        async with asyncio.timeout(CANCELLED_FINISH_TIMEOUT_S):
            await episode.finish()

    cleanup = asyncio.create_task(finish())
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        await asyncio.shield(cleanup)


async def run_cli_episode(*, protocol_id: str, expected_capability_sha256: str,
                          context_date: str, gateway_url: str, token: dict[str, Any],
                          model: str, prompt: str, make_model_client: Callable,
                          max_turns: int, max_episode_tokens: int, deadline: float,
                          cancelled: Callable[[], bool] = lambda: False) -> SandboxEpisodeResult:
    """Run once, or raise after bounded cleanup; ambiguous failures are never relaunched.

    ``make_model_client(bridge, effective_deadline)`` must return the checkpoint-only
    client enforcing exact model/tools/history, rendered token and generation limits.
    Neither upstream API credentials nor a model endpoint enter the CLI process.
    An open without a reply has no episode key; its signed gateway expiry owns cleanup.
    """
    capability = qualified_capability(protocol_id)
    if (capability.date_recipe != CLI_DATE_RECIPE
            or capability_sha256(capability) != expected_capability_sha256):
        raise CLIUnavailable("the signed CLI capability differs from the frozen grant")
    if (type(max_turns) is not int or not 1 <= max_turns <= MAX_CLI_TURNS
            or type(max_episode_tokens) is not int
            or not 1 <= max_episode_tokens <= capability.max_episode_tokens):
        raise ValueError("the signed CLI exceeds its turn or context limit")
    if require_context_date(context_date) != utc_context_date():
        raise CLIUnavailable("the signed CLI context date is no longer current")
    loop = asyncio.get_running_loop()
    if not 0 < deadline - loop.time() <= 3600:
        raise ValueError("the signed CLI deadline must be within one hour")
    if cancelled():
        raise CLILeaseLost("the signed CLI lease was lost before execution")
    transport = SignedCLITransport(protocol_id)

    async def execute():
        started = loop.time()
        async with EpisodeClient(gateway_url) as gateway:
            episode = None
            try:
                async with asyncio.timeout_at(deadline):
                    episode = await gateway.open(token, deadline=deadline)
                    body = episode.record0["body"]
                    require_tools_version(body["tools_version"])
                    if (set(body["tools"]) != {"bash", "edit"} or len(body["tools"]) != 2
                            or type(body["budgets"]["wall_s"]) is not int
                            or body["budgets"]["wall_s"] <= FINISH_MARGIN_S):
                        raise CLIUnavailable("the signed episode tools or wall budget are unsupported")
                    effective_deadline = min(deadline - FINISH_MARGIN_S,
                        started + body["budgets"]["wall_s"] - FINISH_MARGIN_S)
                    if loop.time() >= effective_deadline:
                        raise TimeoutError("the signed episode has no remaining CLI budget")
                    router = ToolRouter(episode, offered=tuple(body["tools"]))
                    bridge = SignedCLIToolBridge(router, capability, prompt=prompt,
                        max_turns=max_turns, context_date=context_date)
                    client = make_model_client(bridge, effective_deadline)
                    path = secrets.token_hex(24)
                    async with asyncio.timeout_at(effective_deadline):
                        async with _mcp_endpoint(SignedMCPApp(bridge, path), effective_deadline) as mcp:
                            async with _model_endpoint(client, model, max_episode_tokens) as (endpoint, secret, trace):
                                await transport.run(model=model, endpoint=endpoint, secret=secret,
                                    mcp_url=mcp + "/" + path, prompt=prompt, bridge=bridge,
                                    deadline=effective_deadline)
                    finished = await episode.finish(deadline=deadline)
                    transcript = (await episode.transcript(deadline=deadline))["transcript"]
                return SandboxEpisodeResult(episode.id, trace, finished.record, finished.state,
                                            transcript, tuple(router.refusals), None)
            except BaseException:
                if episode is not None:
                    with contextlib.suppress(Exception):
                        await _finish_cancelled(episode)
                raise

    async def watch_lease():
        while not cancelled():
            await asyncio.sleep(LEASE_CHECK_SECONDS)
        raise CLILeaseLost("the signed CLI lease was lost during execution")

    running = asyncio.create_task(execute())
    watcher = asyncio.create_task(watch_lease())
    try:
        done, _ = await asyncio.wait((running, watcher), return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            watcher.result()
        result = await running
        if cancelled():
            raise CLILeaseLost("the signed CLI lease was lost before completion")
        return result
    finally:
        watcher.cancel()
        if not running.done():
            running.cancel()
        settled = asyncio.gather(running, watcher, return_exceptions=True)
        try:
            await asyncio.shield(settled)
        except asyncio.CancelledError:
            await asyncio.shield(settled)
            raise
