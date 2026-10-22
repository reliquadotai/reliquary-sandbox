"""Bounded ACP transport for an actual CLI in a disposable, prepared Docker box.

No CLI, uv or npm installation happens while an episode is running. The approved
image already contains the exact CLI artifacts and upstream ACP runner.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

from verifiers.v1.acp import ACP_SOURCE
from verifiers.v1.harnesses.claude_code.harness import (
    ClaudeCodeHarness,
    ClaudeCodeHarnessConfig,
)
from verifiers.v1.harnesses.codex.harness import CodexHarness, CodexHarnessConfig
from verifiers.v1.runtimes import Runtime, RuntimeProcess
from verifiers.v1.runtimes.docker import DockerConfig, DockerRuntime, docker
from verifiers.v1.task import TaskData

from .cli import CLIUnavailable, SignedCLICapability, SignedCLIToolBridge, qualified_capability

MAX_PACKET_BYTES = 1024 * 1024
CLEANUP_TIMEOUT_S = 40.0
ACP_RUNNER_SHA256 = "68fef2c97303a759e453e2f6d014aa730a91a3d60484d9e933cbbf7567c5a42c"
UPSTREAM_MODULES = {
    "verifiers.v1.acp": "f90421e594f4a3d8fdf02cea96db2179c7263375eaab29debbebcdd54d0d277f",
    "verifiers.v1.harnesses.codex.harness": "6fe094337f10a56eb41691a49e7230b8ad6ecd3aab3c0c2323e261f0652cf049",
    "verifiers.v1.harnesses.claude_code.harness": "6e461b41ed2d8a1f6d2064d4a7e91e4a3525c780c51976235fb2e2b1ddb5d9da",
    "verifiers.v1.runtimes.docker": "2bac5b2404f5177f8fb603f91ae9b6509290f117b5633f7404ff1473b36979ce",
}


def require_pinned_cli_sources() -> None:
    import importlib

    for name, expected in UPSTREAM_MODULES.items():
        module = importlib.import_module(name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise CLIUnavailable("the installed upstream CLI transport source changed")
    if hashlib.sha256(ACP_SOURCE.encode()).hexdigest() != ACP_RUNNER_SHA256:
        raise CLIUnavailable("the installed upstream ACP runner changed")


def docker_runtime(capability: SignedCLICapability) -> DockerRuntime:
    """One fresh no-GPU/no-bind runtime; the upstream Docker type has no host mounts."""
    return DockerRuntime(DockerConfig(image=capability.runtime_image, workdir="/workspace",
                                     cpu=2, memory=4, allow=[], block=["*"]))


def require_isolated_runtime(runtime: Runtime, capability: SignedCLICapability) -> None:
    # Type identity prevents a subclass from silently providing local processes or mounts.
    if type(runtime) is not DockerRuntime:
        raise CLIUnavailable("actual signed CLIs require the dedicated Docker runtime")
    config = runtime.config
    if (config.image != capability.runtime_image or config.workdir != "/workspace"
            or config.gpu is not None or config.allow != [] or config.block != ["*"]
            or config.cpu is None or not 0 < config.cpu <= 2
            or config.memory is None or not 0 < config.memory <= 4 or runtime.env):
        raise CLIUnavailable("the CLI runtime differs from its isolated capability")


class _PacketReader:
    def __init__(self, source) -> None:
        self.source = source.__aiter__()
        self.buffer = bytearray()

    async def _exact(self, count: int) -> bytes:
        while len(self.buffer) < count:
            try:
                chunk = await anext(self.source)
            except StopAsyncIteration:
                raise EOFError("the CLI ACP process closed its output") from None
            self.buffer.extend(chunk)
            if len(self.buffer) > MAX_PACKET_BYTES + 8:
                raise ValueError("the CLI ACP output exceeded its packet bound")
        value = bytes(self.buffer[:count])
        del self.buffer[:count]
        return value

    async def read(self) -> dict[str, Any]:
        size = int.from_bytes(await self._exact(8), "big")
        if size > MAX_PACKET_BYTES:
            raise ValueError("the CLI ACP packet exceeded its size bound")
        result = json.loads(await self._exact(size))
        if not isinstance(result, dict):
            raise ValueError("the CLI ACP response must be an object")
        return result


def _packet(value: dict[str, Any]) -> bytes:
    data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError("the CLI ACP input exceeded its packet bound")
    return len(data).to_bytes(8, "big") + data


async def _stop_process(process: RuntimeProcess) -> None:
    for timeout, operation in ((2.0, process.terminate), (5.0, process.kill)):
        with contextlib.suppress(Exception):
            await asyncio.wait_for(operation(), timeout=timeout)
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
            return
        except (Exception, asyncio.CancelledError):
            continue


async def _cleanup(runtime: Runtime, process: RuntimeProcess | None,
                   stderr_task: asyncio.Task | None) -> None:
    try:
        if process is not None:
            await _stop_process(process)
    finally:
        if stderr_task is not None:
            stderr_task.cancel()
            with contextlib.suppress(BaseException):
                await stderr_task
        await runtime.stop()
        listing = await asyncio.wait_for(docker(
            "ps", "-a", "--filter", "name=^/" + runtime.name + "$",
            "--format", "{{.Names}}"), timeout=10)
        if listing.exit_code or listing.stdout.strip():
            raise CLIUnavailable("the CLI container removal could not be verified")


@dataclass(frozen=True)
class CLITransportResult:
    reply: str
    model_turns: int
    qualification_sha256: str


class SignedCLITransport:
    def __init__(self, protocol_id: str, *,
                 runtime_factory: Callable[[SignedCLICapability], Runtime] = docker_runtime):
        self.capability = qualified_capability(protocol_id)
        self.runtime_factory = runtime_factory

    def _require_approved(self, bridge: SignedCLIToolBridge) -> None:
        if (qualified_capability(self.capability.protocol_id) != self.capability
                or bridge.capability != self.capability):
            raise CLIUnavailable("the CLI capability is no longer approved")

    async def run(self, *, model: str, endpoint: str, secret: str, mcp_url: str,
                  prompt: str, bridge: SignedCLIToolBridge, deadline: float) -> CLITransportResult:
        """Run once; never relaunch after an ambiguous ACP/transport failure.

        The endpoint is the trusted checkpoint-only proxy, not an upstream provider.
        Its caller owns schema/history/token-budget validation through ``bridge``.
        """
        require_pinned_cli_sources()
        cap = self.capability
        self._require_approved(bridge)
        remaining = deadline - asyncio.get_running_loop().time()
        if not 0 < remaining <= 3600:
            raise ValueError("the CLI execution deadline must be within one hour")
        for url in (endpoint, mcp_url):
            parsed = urlsplit(url)
            if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}
                    or parsed.username or parsed.password or parsed.query or parsed.fragment):
                raise ValueError("CLI model and MCP endpoints must be local HTTP services")
        # The pinned interception server mints token_urlsafe(16): 128 bits, 22 chars.
        if (not isinstance(secret, str) or not re.fullmatch(r"[A-Za-z0-9_-]{22,}", secret)
                or not model or not prompt):
            raise ValueError("the CLI needs a model, prompt and fresh per-run proxy secret")
        runtime = self.runtime_factory(cap)
        require_isolated_runtime(runtime, cap)
        process, stderr_task = None, None
        try:
            async with asyncio.timeout_at(deadline):
                await runtime.start()
                limit = await docker("update", "--pids-limit", "128", runtime.name)
                if limit.exit_code:
                    raise CLIUnavailable("the CLI runtime PID bound could not be installed")
                endpoint, mcp_url = runtime.host_url(endpoint), runtime.host_url(mcp_url)
                await runtime.prepare_execution([endpoint, mcp_url])
                # The trusted image contains this exact upstream runner; no uv/npm setup.
                check = await runtime.run(["sha256sum", cap.runner_argv[-1]], {})
                if (check.exit_code or check.stdout.split()[:1] != [cap.runner_sha256]
                        or cap.runner_sha256 != ACP_RUNNER_SHA256):
                    raise CLIUnavailable("the prepared image has a different ACP runner")
                user = await runtime.run(["id", "-u"], {})
                if user.exit_code or user.stdout.strip() != "10001":
                    raise CLIUnavailable("the actual CLI runtime must run as its dedicated nonroot user")
                cli = cap.protocol_id == "signed-codex-cli-v1"
                binary = ("/opt/signed-cli/node/node_modules/.bin/codex" if cli
                          else "/opt/signed-cli/node/node_modules/"
                               "@anthropic-ai/claude-code-linux-x64/claude")
                version = await runtime.run([binary, "--version"], {})
                if version.exit_code or not re.search(
                        r"(?:^|\s)" + re.escape(cap.versions[1]) + r"(?:\s|$)", version.stdout):
                    raise CLIUnavailable("the prepared image contains another CLI binary version")
                harness = (CodexHarness(CodexHarnessConfig(version=cap.versions[1], multi_agent=False))
                           if cli else ClaudeCodeHarness(ClaudeCodeHarnessConfig(
                               version=cap.versions[1], disabled_tools=[
                                   "Bash", "Read", "Write", "Edit", "MultiEdit", "Glob", "Grep",
                                   "WebFetch", "WebSearch", "Task", "Agent", "NotebookEdit"])))
                trace = SimpleNamespace(id=uuid.uuid4().hex)
                config = await harness.prepare_acp(
                    SimpleNamespace(model=model), trace, runtime, endpoint, secret,
                    {"signed_episode": mcp_url}, TaskData(prompt=prompt))
                # The exact effective config is part of the real CLI probe. The model
                # schema gate, not these switches alone, enforces absence of builtins.
                if cli:
                    settings = json.loads(config.env["CODEX_CONFIG"])
                    settings["features"].update({"shell_tool": False, "apply_patch_freeform": False,
                                                "view_image": False, "web_search": False,
                                                "code_mode": False, "tool_search": False,
                                                "js_repl": False, "browser_use": False})
                    settings["tools"] = {"update_plan": {"enabled": False},
                                         "experimental_request_user_input": {"enabled": False}}
                    settings["web_search"] = "disabled"
                    config.env["CODEX_CONFIG"] = json.dumps(settings)
                else:
                    # ACP otherwise chooses its SDK's bundled native executable.
                    config.env["CLAUDE_CODE_EXECUTABLE"] = binary
                    config.session_meta["claudeCode"]["options"]["tools"] = []
                    config.session_meta["claudeCode"]["options"]["settingSources"] = []
                process = await runtime.open_process(list(cap.runner_argv), config.env)

                async def drain_stderr() -> None:
                    # Stderr can contain prompts, paths or transient URLs; never expose it.
                    total = 0
                    async for chunk in process.stderr:
                        total += len(chunk)
                        if total > MAX_PACKET_BYTES:
                            raise ValueError("the CLI stderr exceeded its byte bound")

                stderr_task = asyncio.create_task(drain_stderr())
                reader = _PacketReader(process.stdout)
                await process.write(_packet({"operation": "prompt", "config": {
                    "command": config.command, "user_contents": [config.prompt],
                    "mcp_urls": ({"signed_episode": mcp_url} if config.mcp_urls is None
                                 else config.mcp_urls),
                    "system_prompt": config.system_prompt or "",
                    "session_meta": config.session_meta or {},
                }}))
                reading = asyncio.create_task(reader.read())
                try:
                    done, _ = await asyncio.wait((reading, stderr_task),
                                                 return_when=asyncio.FIRST_COMPLETED)
                    if stderr_task in done:
                        stderr_task.result()
                    response = await reading
                finally:
                    if not reading.done():
                        reading.cancel()
                        with contextlib.suppress(BaseException):
                            await reading
                if not response.get("ok") or not isinstance(response.get("reply"), str):
                    raise RuntimeError("the actual CLI ACP execution failed")
                if stderr_task.done():
                    stderr_task.result()
                bridge.require_final_reply(response["reply"])
                return CLITransportResult(response["reply"], bridge.model_turns,
                                          cap.qualification_sha256)
        finally:
            cleanup = asyncio.create_task(asyncio.wait_for(
                _cleanup(runtime, process, stderr_task), timeout=CLEANUP_TIMEOUT_S))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await asyncio.shield(cleanup)
                raise
