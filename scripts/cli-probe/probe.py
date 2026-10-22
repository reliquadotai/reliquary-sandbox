"""Opt-in real-CLI CPU fixture. No model weights, provider credentials or paid resources.

Raw synthetic requests stay in the caller's private output directory. The summary
contains only public package/tool identities, hashes and acceptance counts.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import secrets
import time
from pathlib import Path

from reliquary.eval.protocol import digest
from reliquary.eval.signed_agentic import model_client
from verifiers.v1.clients import Client, ModelContext
from verifiers.v1.clients.train import message_to_wire, tool_to_wire
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.dialects import AnthropicDialect, ResponsesDialect
from verifiers.v1.interception import InterceptionServer
from verifiers.v1.runtimes.docker import docker
from verifiers.v1.session import RolloutSession
from verifiers.v1.trace import AgentInfo, Trace, TraceTask
from verifiers.v1.types import Sampling

from reliquary_sandbox.attest.canonical import canonical_json
from reliquary_sandbox.episode_client import EpisodeClient
from reliquary_sandbox.tool_routing import ToolRouter
from reliquary_sandbox_service.episodes.local_gateway import (
    default_budgets,
    local_gateway,
    serve_app,
)
from reliquary_sandbox_verifiers.cli import (
    CLI_VERSIONS,
    MCP_TOOLS,
    SignedCLICapability,
    SignedCLIToolBridge,
    SignedMCPApp,
    initial_template_sha256,
)
from reliquary_sandbox_verifiers.cli_transport import (
    ACP_RUNNER_SHA256,
    SignedCLITransport,
    docker_runtime,
)

PROMPT = "Use the signed workspace tools to write 41, edit it to 42, read it, then reply done."
MODEL = "fixture/signed-cli@" + "a" * 40


class CleanupFailure(Exception):
    pass


class ProbeTransport(SignedCLITransport):
    """Explicit unapproved CPU probe; never used by the normal protocol lookup."""

    def __init__(self, capability):
        self.capability = capability
        self.owned = None

        def factory(cap):
            self.owned = docker_runtime(cap)
            return self.owned

        self.runtime_factory = factory

    def _require_approved(self, bridge):
        if bridge.capability != self.capability:
            raise ValueError("probe capability changed")

    async def run(self, **kwargs):
        try:
            return await super().run(**kwargs)
        finally:
            if self.owned is not None:
                listing = await asyncio.wait_for(docker(
                    "ps", "-a", "--filter", "name=^/" + self.owned.name + "$",
                    "--format", "{{.Names}}"), timeout=10)
                if listing.exit_code or listing.stdout.strip():
                    raise CleanupFailure("the CPU probe could not verify its CLI container removal")


class NoCalls:
    async def call(self, *args, **kwargs):
        raise AssertionError("discovery must never execute a tool")


class Discovery(Client):
    def __init__(self):
        self.found = asyncio.Event()
        self.messages, self.tools = None, None
        self.raw_tools_count = 0

    async def get_response(self, dialect, body, sampling, **kwargs):
        request = dialect.parse_request(body)
        self.messages = [message_to_wire(message) for message in request.messages]
        self.tools = [tool_to_wire(tool) for tool in request.tools or []]
        self.raw_tools_count = len(body.get("tools") or [])
        self.found.set()
        raise ValueError("CPU schema discovery stops before model generation")

    async def relay(self, dialect, body, **kwargs):
        return await self.get_response(dialect, body, None)

    async def relay_aux(self, dialect, route, body, **kwargs):
        if route != "/v1/messages/count_tokens":
            raise ValueError("unsupported discovery auxiliary route")
        return {"input_tokens": 1}


class ScriptedGenerator:
    """Fixed CPU test messages; token integers are fixtures, not tokenizer measurements."""

    def __init__(self, capability, total_turns=4):
        self.names = {tool: name for name, tool in capability.model_tool_names}
        self.calls = 0
        self.total_turns = total_turns

    def render_messages(self, messages, **kwargs):
        return [1, 2]

    def generate_turn(self, ids, **kwargs):
        steps = [("bash", {"command": "printf 41 > /root/answer.txt; cat /root/answer.txt"}),
                 ("edit", {"path": "/root/answer.txt", "old_str": "41", "new_str": "42"}),
                 ("bash", {"command": "cat /root/answer.txt"})]
        if self.total_turns == 48:
            steps = [("bash", {"command": "printf 42 > /root/answer.txt; printf ok"}),
                     *[("bash", {"command": "printf ok"}) for _ in range(46)]]
        index = self.calls
        self.calls += 1
        message = {"content": "done", "reasoning_content": None, "tool_calls": None}
        if index < len(steps):
            tool, arguments = steps[index]
            message.update(content="", tool_calls=[{
                "id": f"signed-call-{index}", "name": self.names[tool],
                "arguments": canonical_json(arguments).decode()}])
        return {"token_ids": [10 + index], "message": message,
                "finish_reason": "tool_calls" if index < len(steps) else "stop"}

    def close(self):
        pass


def capability(args):
    lock = json.loads((Path(__file__).parent / "package-lock.json").read_text())
    artifacts = tuple((path.removeprefix("node_modules/"), item["version"], item["integrity"])
                      for path, item in lock["packages"].items() if path and "integrity" in item)
    names = (("mcp__signed_episode__bash", "bash"), ("mcp__signed_episode__edit", "edit"))
    tools = [{"type": "function", "function": {"name": name,
              "description": tool["description"], "parameters": tool["inputSchema"]}}
             for (name, _), tool in zip(names, MCP_TOOLS, strict=True)]
    return SignedCLICapability(args.protocol, args.image, canonical_json(tools), names,
                              artifacts, "0" * 64,
                              ("/opt/signed-cli/runtime/bin/python", "/opt/signed-cli/acp-runner.py"),
                              ACP_RUNNER_SHA256, "0" * 64)


@contextlib.asynccontextmanager
async def model_endpoint(client):
    config = EvalClientConfig(base_url="http://127.0.0.1:1", api_key_var="UNUSED_CLI_PROBE_KEY")
    context = ModelContext(model=MODEL, client=config, sampling=Sampling(max_tokens=512))
    trace = Trace(task=TraceTask(type="synthetic-cpu-cli", data={}),
                  agent=AgentInfo(config=AgentConfig()))
    async with InterceptionServer() as server:
        server.clients[config.model_dump_json()] = client
        async with server.acquire(RolloutSession(ctx=context, trace=trace)) as (url, token, _):
            yield url + "/v1", token


async def signed_acceptance(args, cap, gateway, turns):
    token = gateway.issue(env="live-env", split="train", index=0, image=args.task_image,
                          budgets=default_budgets(wall_s=600, cpu_s=120, max_calls=64))
    async with EpisodeClient(gateway.url) as client:
        episode = await client.open(token)
        try:
            bridge = SignedCLIToolBridge(ToolRouter(episode, offered=("bash", "edit")),
                                         cap, prompt=PROMPT)
            generator = ScriptedGenerator(cap, turns)
            native = model_client(generator, task={
                "model": {"repo": "fixture/signed-cli", "revision": "a" * 40},
                "protocol": {"max_turns": 48, "max_episode_tokens": 65536},
                "thinking": False, "max_new_tokens": 512, "sampling": {}},
                source={"tools_sha256": digest(cap.model_tools),
                        "renderer_id": "renderers:qwen38@0.1.11"},
                deadline=time.monotonic() + 240, seed=1, cli_bridge=bridge)
            with serve_app(SignedMCPApp(bridge, path := secrets.token_hex(24))) as mcp:
                async with model_endpoint(native) as (endpoint, secret):
                    result = await ProbeTransport(cap).run(
                        model=MODEL, endpoint=endpoint, secret=secret,
                        mcp_url=mcp + "/" + path, prompt=PROMPT, bridge=bridge,
                        deadline=asyncio.get_running_loop().time() + 240)
            if turns == 48:
                dialect = (ResponsesDialect() if args.protocol == "signed-codex-cli-v1"
                           else AnthropicDialect())
                try:
                    await native.get_response(dialect, {}, None)
                except TimeoutError:
                    pass
                else:
                    raise AssertionError("a 49th generation was not refused")
                if generator.calls != 48:
                    raise AssertionError("a 49th call reached the scripted generator")
        finally:
            final = await episode.finish()
            transcript = (await episode.transcript())["transcript"]
    verified = gateway.verify(transcript)
    if (not verified.ok or final.record["body"].get("reward") != 1.0
            or result.reply != "done" or native.turns != turns or native.tool_calls != turns - 1):
        raise ValueError("the actual CLI did not pass signed CPU acceptance")
    (args.out / f"synthetic-signed-{turns}-turns.private.json").write_bytes(canonical_json(transcript))
    return {"signed_calls": turns - 1, "model_turns": turns, "verified_transcript": True,
            "graded_reward": 1.0, "attempt_49_refused_before_generator": turns == 48}


async def execute(args):
    args.stage = "schema-discovery"
    cap = capability(args)
    discovery = Discovery()
    dummy = SignedCLIToolBridge(ToolRouter(NoCalls()), cap, prompt=PROMPT)
    with serve_app(SignedMCPApp(dummy, path := secrets.token_hex(24))) as mcp:
        async with model_endpoint(discovery) as (endpoint, secret):
            task = asyncio.create_task(ProbeTransport(cap).run(
                model=MODEL, endpoint=endpoint, secret=secret, mcp_url=mcp + "/" + path,
                prompt=PROMPT, bridge=dummy, deadline=asyncio.get_running_loop().time() + 120))
            found = asyncio.create_task(discovery.found.wait())
            try:
                done, _ = await asyncio.wait((task, found), return_when=asyncio.FIRST_COMPLETED)
                if found not in done:
                    task.result()
                    raise RuntimeError("the actual CLI made no model request")
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                    await task
                found.cancel()
    (args.out / "synthetic-initial-request.private.json").write_bytes(canonical_json({
        "messages": discovery.messages, "tools": discovery.tools}))
    names = {tool["function"]["name"] for tool in discovery.tools}
    args.stage = "exact-mcp-schema-check"
    if (names != {name for name, _ in cap.model_tool_names} or len(discovery.tools) != 2
            or discovery.raw_tools_count != 2):
        raise ValueError("the actual CLI did not expose exactly the two signed MCP tools")
    cap = dataclasses.replace(cap, model_tools_json=canonical_json(discovery.tools),
        initial_template_sha256=initial_template_sha256(discovery.messages, PROMPT))
    args.stage = "signed-real-workspace-acceptance"
    busybox = Path(args.busybox)
    with local_gateway(args.out / "gateway", envs={"live-env": "live_episode_env:sandbox_task"},
                       images=[args.task_image], episode_allow_unquoted_disk=False,
                       episode_cpu_source="cgroupfs", episode_docker_pidfile=Path(args.pidfile),
                       episode_helper_busybox=busybox,
                       episode_helper_busybox_sha256=hashlib.sha256(busybox.read_bytes()).hexdigest()
                       ) as gateway:
        smoke = await signed_acceptance(args, cap, gateway, 4)
        args.stage = "signed-48-turn-acceptance"
        full_budget = await signed_acceptance(args, cap, gateway, 48)
    return {"status": "passed", "protocol_id": args.protocol, "approved": False,
            "model_execution": "scripted CPU fixture; no model weights or inference",
            "cli_and_acp": CLI_VERSIONS[args.protocol], "runtime_image": args.image,
            "artifact_lock_sha256": hashlib.sha256(
                (Path(__file__).parent / "package-lock.json").read_bytes()).hexdigest(),
            "model_tools": cap.model_tools, "model_tools_sha256": cap.model_tools_sha256,
            "initial_template_sha256": cap.initial_template_sha256,
            "smoke": smoke, "full_budget": full_budget,
            "process_cleanup": "owned CLI containers verified absent"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=CLI_VERSIONS, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--task-image", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--busybox", default="/usr/bin/busybox")
    parser.add_argument("--pidfile", default="/run/reliquary-xfs-docker.pid")
    args = parser.parse_args()
    os.umask(0o077)
    if Path(__file__).resolve().parents[2] in args.out.resolve().parents:
        raise ValueError("probe evidence must be outside the repository")
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.environ["EPISODE_LIVE_IMAGE"] = args.task_image
    logging.disable(logging.CRITICAL)
    try:
        summary = asyncio.run(asyncio.wait_for(execute(args), timeout=600))
    except Exception as error:
        summary = {"status": "failed", "protocol_id": args.protocol, "approved": False,
                   "exception_class": type(error).__name__,
                   "stage": getattr(args, "stage", "preflight"),
                   "detail": "actual CLI CPU probe failed; raw synthetic evidence retained privately"}
    (args.out / "summary.json").write_bytes(canonical_json(summary) + b"\n")
    print(json.dumps({key: summary[key] for key in ("status", "protocol_id", "approved")}))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
