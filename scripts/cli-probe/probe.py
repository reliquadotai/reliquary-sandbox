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
import inspect
import json
import logging
import os
import platform
import re
import secrets
import traceback
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from reliquary.eval import model_wire
from reliquary.eval.protocol import digest
from reliquary.eval.signed_agentic import model_client
from verifiers.v1.clients import Client, ModelContext
from verifiers.v1.clients.train import message_to_wire, tool_to_wire
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.dialects import AnthropicDialect, ResponsesDialect
from verifiers.v1.interception import InterceptionServer
from verifiers.v1.session import RolloutSession
from verifiers.v1.trace import AgentInfo, Trace, TraceTask
from verifiers.v1.types import Sampling

from reliquary_sandbox.attest.canonical import canonical_json
from reliquary_sandbox.tool_routing import ToolRouter
from reliquary_sandbox_service.episodes.local_gateway import (
    default_budgets,
    local_gateway,
    serve_app,
)
from reliquary_sandbox_verifiers.cli import (
    CLI_DATE_RECIPE,
    CLI_VERSIONS,
    MCP_TOOLS,
    CLIUnavailable,
    SignedCLICapability,
    SignedCLIToolBridge,
    SignedMCPApp,
    initial_template_sha256,
    utc_context_date,
)
from reliquary_sandbox_verifiers.cli_runner import capability_sha256, run_cli_episode
from reliquary_sandbox_verifiers.cli_transport import (
    ACP_RUNNER_SHA256,
    SignedCLITransport,
    docker_runtime,
)

PROMPT = "Use the signed workspace tools to write 41, edit it to 42, read it, then reply done."
MODEL = "fixture/signed-cli@" + "a" * 40
ALTERNATE_PROMPT = "With the signed workspace tools, write 41, replace it with 42, read it, and finish with done."
ALTERNATE_MODEL = "fixture/another-checkpoint@" + "b" * 40


class ProbeTransport(SignedCLITransport):
    """Explicit unapproved CPU probe; never used by the normal protocol lookup."""

    def __init__(self, capability):
        self.capability = capability
        self.runtime_factory = docker_runtime

    def _require_approved(self, bridge):
        if bridge.capability != self.capability:
            raise ValueError("probe capability changed")


class NoCalls:
    async def call(self, *args, **kwargs):
        raise AssertionError("discovery must never execute a tool")


class Discovery(Client):
    def __init__(self, model):
        self.model = model
        self.found = asyncio.Event()
        self.messages, self.tools = None, None
        self.raw_tools_count = 0
        self.raw_tools = []

    async def get_response(self, dialect, body, sampling, **kwargs):
        if body.get("model") != self.model:
            raise ValueError("the actual CLI requested another model identity")
        self.raw_tools = body.get("tools") or []
        self.raw_tools_count = len(self.raw_tools)
        request = dialect.parse_request(body)
        self.messages = [message_to_wire(message) for message in request.messages]
        self.tools = [tool_to_wire(tool) for tool in request.tools or []]
        self.found.set()
        raise ValueError("CPU schema discovery stops before model generation")

    async def relay(self, dialect, body, **kwargs):
        return await self.get_response(dialect, body, None)

    async def relay_aux(self, dialect, route, body, **kwargs):
        if route != "/v1/messages/count_tokens":
            raise ValueError("unsupported discovery auxiliary route")
        return {"input_tokens": 1}


class CaptureFailure(Client):
    """Retain a failed synthetic request privately without changing native behavior."""

    def __init__(self, client, path):
        self.client, self.path = client, path

    async def _capture(self, body, operation):
        try:
            return await operation
        except Exception:
            if not self.path.exists():
                data = canonical_json({"body": body, "traceback": traceback.format_exc()})
                if len(data) <= 2 * 1024 * 1024:
                    self.path.write_bytes(data)
            raise

    async def get_response(self, dialect, body, sampling, **kwargs):
        return await self._capture(body, self.client.get_response(dialect, body, sampling, **kwargs))

    async def relay(self, dialect, body, **kwargs):
        return await self._capture(body, self.client.relay(dialect, body, **kwargs))

    async def relay_aux(self, dialect, route, body, **kwargs):
        return await self._capture(body, self.client.relay_aux(dialect, route, body, **kwargs))


class ScriptedGenerator:
    """Fixed CPU test messages; token integers are fixtures, not tokenizer measurements."""

    def __init__(self, capability, total_turns=4):
        self.names = {tool: name for name, tool in capability.model_tool_names}
        self.calls = 0
        self.total_turns = total_turns

    def render_messages(self, messages, **kwargs):
        if kwargs.get("thinking") is not True:
            raise AssertionError("the CPU fixture must exercise frozen thinking=true")
        if self.calls and not any(message.get("reasoning_content") == "Synthetic CPU reasoning fixture."
                                  for message in messages if message.get("role") == "assistant"):
            raise AssertionError("the actual CLI did not preserve complete sampled reasoning")
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
            if index == 0:
                message["reasoning_content"] = "Synthetic CPU reasoning fixture."
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
                              ACP_RUNNER_SHA256, "0" * 64, CLI_DATE_RECIPE)


@contextlib.asynccontextmanager
async def model_endpoint(client, model=MODEL):
    config = EvalClientConfig(base_url="http://127.0.0.1:1", api_key_var="UNUSED_CLI_PROBE_KEY")
    context = ModelContext(model=model, client=config, sampling=Sampling(max_tokens=512))
    trace = Trace(task=TraceTask(type="synthetic-cpu-cli", data={}),
                  agent=AgentInfo(config=AgentConfig()))
    async with InterceptionServer() as server:
        server.clients[config.model_dump_json()] = client
        async with server.acquire(RolloutSession(ctx=context, trace=trace)) as (url, token, _):
            yield url + "/v1", token


async def signed_acceptance(args, cap, gateway, turns, *, model=MODEL, prompt=PROMPT):
    token = gateway.issue(env="live-env", split="train", index=0, image=args.task_image,
                          budgets=default_budgets(wall_s=600, cpu_s=120, max_calls=64))
    generator = ScriptedGenerator(cap, turns)
    args.native, args.generator = None, generator
    repo, revision = model.rsplit("@", 1)

    def make_model_client(bridge, effective_deadline):
        args.native = model_client(generator, task={
            "model": {"repo": repo, "revision": revision},
            "protocol": {"max_turns": 48, "max_episode_tokens": 65536},
            "thinking": True, "max_new_tokens": 512, "sampling": {}},
            source={"tools_sha256": digest(cap.model_tools),
                    "renderer_id": "renderers:qwen38@0.1.11"},
            deadline=effective_deadline, seed=1, cli_bridge=bridge)
        return CaptureFailure(args.native, args.out / "native-failure.private.json")

    # Qualification-only scoped lookups: the real composer/transport run unchanged,
    # and no candidate enters the persistent production capability registry.
    with patch("reliquary_sandbox_verifiers.cli_runner.qualified_capability", return_value=cap), \
         patch("reliquary_sandbox_verifiers.cli_transport.qualified_capability", return_value=cap):
        result = await run_cli_episode(protocol_id=cap.protocol_id,
            expected_capability_sha256=capability_sha256(cap), context_date=args.context_date,
            gateway_url=gateway.url, token=token, model=model, prompt=prompt,
            make_model_client=make_model_client, max_turns=48, max_episode_tokens=65536,
            deadline=asyncio.get_running_loop().time() + 240)
    native = args.native
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
    verified = gateway.verify(result.transcript)
    if (not verified.ok or result.final["body"].get("reward") != 1.0 or result.error is not None
            or result.refusals or native.finish_reason != "stop"
            or native.turns != turns or native.tool_calls != turns - 1):
        raise ValueError("the actual CLI did not pass signed CPU acceptance")
    (args.out / f"synthetic-signed-{turns}-turns.private.json").write_bytes(canonical_json(result.transcript))
    return {"signed_calls": turns - 1, "model_turns": turns, "verified_transcript": True,
            "graded_reward": 1.0, "attempt_49_refused_before_generator": turns == 48,
            "frozen_thinking": True, "nonempty_reasoning_history_roundtrip": True}


async def discover(args, cap, *, model, prompt, filename):
    discovery = Discovery(model)
    args.discovery = discovery
    dummy = SignedCLIToolBridge(ToolRouter(NoCalls()), cap, prompt=prompt,
                                context_date=args.context_date)
    with serve_app(SignedMCPApp(dummy, path := secrets.token_hex(24))) as mcp:
        async with model_endpoint(discovery, model) as (endpoint, secret):
            task = asyncio.create_task(ProbeTransport(cap).run(
                model=model, endpoint=endpoint, secret=secret, mcp_url=mcp + "/" + path,
                prompt=prompt, bridge=dummy, deadline=asyncio.get_running_loop().time() + 120))
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
    (args.out / filename).write_bytes(canonical_json({
        "messages": discovery.messages, "tools": discovery.tools, "raw_tools": discovery.raw_tools}))
    return discovery


async def execute(args):
    args.stage = "schema-discovery"
    cap = capability(args)
    discovery = await discover(args, cap, model=MODEL, prompt=PROMPT,
                               filename="synthetic-initial-request.private.json")
    names = {tool["function"]["name"] for tool in discovery.tools}
    args.stage = "exact-mcp-schema-check"
    if (names != {name for name, _ in cap.model_tool_names} or len(discovery.tools) != 2
            or discovery.raw_tools_count != 2):
        raise ValueError("the actual CLI did not expose exactly the two signed MCP tools")
    cap = dataclasses.replace(cap, model_tools_json=canonical_json(discovery.tools),
        initial_template_sha256=initial_template_sha256(discovery.messages, PROMPT,
            context_date=args.context_date, date_recipe=cap.date_recipe))
    args.stage = "fresh-model-task-prefix-repeatability"
    alternate = await discover(args, cap, model=ALTERNATE_MODEL, prompt=ALTERNATE_PROMPT,
                               filename="synthetic-alternate-initial-request.private.json")
    cap.require_model_tools(alternate.tools)
    if (alternate.raw_tools_count != 2 or initial_template_sha256(
            alternate.messages, ALTERNATE_PROMPT, context_date=args.context_date,
            date_recipe=cap.date_recipe) != cap.initial_template_sha256):
        raise ValueError("the actual CLI prefix or tools changed across model/task identities")
    tomorrow = (date.fromisoformat(args.context_date) + timedelta(days=1)).isoformat()
    controlled = json.loads(json.dumps(discovery.messages))
    user = next(message for message in controlled if message["role"] == "user")
    user["content"][0]["text"] = user["content"][0]["text"].replace(args.context_date, tomorrow)
    if initial_template_sha256(controlled, PROMPT, context_date=tomorrow,
                              date_recipe=cap.date_recipe) != cap.initial_template_sha256:
        raise ValueError("the pinned native-date template is not stable across controlled dates")
    prefix = canonical_json(discovery.messages).decode().replace(PROMPT, "<signed-task-prompt>")
    portable = {"no_model_identity": MODEL not in prefix and ALTERNATE_MODEL not in prefix,
                "no_kernel": platform.release() not in prefix,
                "no_workspace_path": "/workspace" not in prefix,
                "one_signed_context_date": prefix.count(args.context_date) == 1,
                "one_iso_date": len(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", prefix)) == 1}
    if not all(portable.values()):
        raise ValueError("the actual CLI retained default environment/model/date context")
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
        full_budget = await signed_acceptance(args, cap, gateway, 48,
                                               model=ALTERNATE_MODEL, prompt=ALTERNATE_PROMPT)
    return {"status": "passed", "protocol_id": args.protocol, "approved": False,
            "model_execution": "scripted CPU fixture; no model weights or inference",
            "cli_and_acp": CLI_VERSIONS[args.protocol], "runtime_image": args.image,
            "artifact_lock_sha256": hashlib.sha256(
                (Path(__file__).parent / "package-lock.json").read_bytes()).hexdigest(),
            "artifact_lock_scope": "all optional platforms; image uses Linux x64 glibc",
            "excluded_unused_optionals": ["@anthropic-ai/claude-code-linux-x64-musl",
                "@anthropic-ai/claude-agent-sdk-linux-x64-musl",
                "@anthropic-ai/claude-agent-sdk-linux-x64"],
            "model_tools_sha256": cap.model_tools_sha256,
            "model_tools_count": len(cap.model_tools),
            "prompt_recipe": cap.prompt_recipe, "system_prompt_sha256": cap.system_prompt_sha256,
            "date_recipe": cap.date_recipe, "date_template_sha256": cap.date_template_sha256,
            "context_date": args.context_date, "controlled_date_templates_equal": True,
            "capability_sha256": capability_sha256(cap), "production_composer": "run_cli_episode",
            "native_model_client_source_sha256": hashlib.sha256(inspect.getsource(model_client).encode()).hexdigest(),
            "native_model_wire_source_sha256": hashlib.sha256(Path(inspect.getfile(model_wire)).read_bytes()).hexdigest(),
            "fresh_model_and_task_prefix_equal": True, "default_context_absence": portable,
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
    args.context_date = utc_context_date()
    os.umask(0o077)
    if Path(__file__).resolve().parents[2] in args.out.resolve().parents:
        raise ValueError("probe evidence must be outside the repository")
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.environ["EPISODE_LIVE_IMAGE"] = args.task_image
    logging.disable(logging.CRITICAL)
    try:
        summary = asyncio.run(asyncio.wait_for(execute(args), timeout=600))
    except Exception as error:
        (args.out / "failure.private.json").write_bytes(canonical_json({
            "traceback": traceback.format_exc(),
            "messages": getattr(getattr(args, "discovery", None), "messages", None),
            "tools": getattr(getattr(args, "discovery", None), "tools", None),
            "raw_tools": getattr(getattr(args, "discovery", None), "raw_tools", None)}))
        site = traceback.extract_tb(error.__traceback__)[-1]
        summary = {"status": "failed", "protocol_id": args.protocol, "approved": False,
                   "exception_class": type(error).__name__,
                   "failure_site": {"module": Path(site.filename).name,
                                    "line": site.lineno, "function": site.name},
                   "stage": getattr(args, "stage", "preflight"),
                   "detail": "actual CLI CPU probe failed; raw synthetic evidence retained privately"}
        if isinstance(error, CLIUnavailable):
            summary["preflight_reason"] = {
                "the CLI runtime PID bound could not be installed": "pid_bound",
                "the prepared image has a different ACP runner": "runner_hash",
                "the actual CLI runtime must run as its dedicated nonroot user": "nonroot_uid",
                "the prepared image contains another CLI binary version": "binary_version",
            }.get(str(error), "other_capability_gate")
        if getattr(args, "discovery", None) is not None:
            discovery = args.discovery
            expected = {"mcp__signed_episode__bash", "mcp__signed_episode__edit"}
            names = {tool.get("function", {}).get("name") for tool in discovery.tools or []}
            summary["offered_tools"] = {
                "raw_count": discovery.raw_tools_count,
                "parsed_count": len(discovery.tools or []),
                "expected_mcp_names_match": names == expected,
                "expected_mcp_names_present": len(names & expected),
                "raw_types": sorted({tool.get("type") if tool.get("type") in {
                    "function", "custom", "namespace"} else "other"
                    for tool in discovery.raw_tools}),
                "sha256": hashlib.sha256(canonical_json(discovery.raw_tools)).hexdigest()}
        if getattr(args, "native", None) is not None:
            summary["scripted_generator_calls"] = args.generator.calls
            summary["native_model_turns"] = args.native.turns
    (args.out / "summary.json").write_bytes(canonical_json(summary) + b"\n")
    print(json.dumps({key: summary[key] for key in ("status", "protocol_id", "approved")}))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
