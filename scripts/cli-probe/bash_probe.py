"""Synthetic CPU acceptance of the real native signed Bash harness, not a benchmark."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
from unittest.mock import patch

PROMPT = "Use the signed workspace to write 42 to /root/answer.txt, then finish with done."
MODEL = {"repo": "fixture/signed-bash", "revision": "a" * 40}
TURNS = 48


class ScriptedBashGenerator:
    """Fixture token IDs exercise transport/counting, not tokenizer or model accuracy."""

    def __init__(self, tools):
        self.tools = tools
        self.calls = 0
        self.renders = 0

    def render_messages(self, messages, *, tools, thinking, renderer_id):
        if (tools != self.tools or thinking is not True
                or renderer_id != "renderers:qwen38@0.1.11"
                or sum(m["role"] == "assistant" for m in messages) != self.calls
                or sum(m["role"] == "tool" for m in messages) != self.calls):
            raise ValueError("the signed Bash fixture history or rendering policy changed")
        self.renders += 1
        return [1, 2]

    def generate_turn(self, ids, **kwargs):
        if ids != [1, 2] or self.calls >= TURNS:
            raise AssertionError("an extra generation reached the scripted Bash model")
        index = self.calls
        self.calls += 1
        message = {"content": "done"}
        if index < TURNS - 1:
            command = "printf 42 > /root/answer.txt; printf ok" if index == 0 else "printf ok"
            message = {"content": None, "tool_calls": [{"id": f"bash-fixture-{index}",
                "name": "bash", "arguments": json.dumps({"command": command})}]}
        return {"token_ids": [10 + index], "message": message,
                "finish_reason": "tool_calls" if index < TURNS - 1 else "stop"}

    def close(self):
        pass


@contextlib.contextmanager
def _observe_bash_processes(runtime_type):
    """Observe unchanged upstream calls only within this sequential CPU fixture."""
    workdirs, processes = set(), []
    start, spawn = runtime_type.start, asyncio.create_subprocess_exec

    async def observed_start(runtime):
        await start(runtime)
        workdirs.add(Path(runtime.workdir).resolve())

    async def observed_spawn(*argv, **kwargs):
        process = await spawn(*argv, **kwargs)
        if kwargs.get("cwd") is not None and Path(kwargs["cwd"]).resolve() in workdirs:
            processes.append((process, kwargs.get("start_new_session") is True))
        return process

    with patch.object(runtime_type, "start", observed_start), \
            patch("asyncio.create_subprocess_exec", observed_spawn):
        yield workdirs, processes


def _assert_owned_cleanup(workdirs, processes):
    if not workdirs or not processes or any(path.exists() for path in workdirs):
        raise RuntimeError("the signed Bash runtime cleanup could not be verified")
    for process, owns_group in processes:
        if process.returncode is None or not owns_group:
            raise RuntimeError("the signed Bash process ownership or exit could not be verified")
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            continue
        raise RuntimeError("an owned signed Bash process group still exists")


async def signed_bash_acceptance(args, gateway):
    """Use the existing isolated CI gateway; never provision or start a daemon here.

    The local grant and task are synthetic. This proves native Bash 48-turn signed
    execution/cleanup, not platform grant authority, Terminal data or model quality.
    """
    from reliquary.environment.agentic_swe import harness_refusal, harness_tools
    from reliquary.eval import signed_agentic
    from reliquary.eval.protocol import digest
    from reliquary.eval.sets import prompt_sha256
    from reliquary.sandbox.evaluation_engagements import _engagement
    from verifiers.v1.dialects import ChatDialect
    from verifiers.v1.runtimes.subprocess import SubprocessRuntime

    from reliquary_sandbox.attest.canonical import canonical_json
    from reliquary_sandbox_service.episodes.local_gateway import default_budgets

    out = Path(args.out).resolve()
    repository = Path(__file__).resolve().parents[2]
    if out == repository or repository in out.parents:
        raise ValueError("synthetic Bash evidence must remain outside the repository")
    if harness_refusal() is not None:
        raise ValueError("the native Bash harness spec differs from its installed pin")
    tools = list(harness_tools(("bash", "edit")))
    source = {"renderer_id": "renderers:qwen38@0.1.11", "tools_sha256": digest(tools)}
    protocol = {"harness": "signed-bash-v2", "interaction": "signed_episode",
        "max_turns": TURNS, "max_episode_tokens": 65536, "episode_timeout_seconds": 300}
    task = {"eval_id": "synthetic-bash48", "task_id": "synthetic-bash48-task", "model": MODEL,
        "protocol": protocol, "protocol_sha256": digest(protocol),
        "set_protocols": {"synthetic-bash48": {"protocol": {"source": {"episode": source}}}},
        "thinking": True, "max_new_tokens": 512, "sampling": {}}
    row = {"set_id": "synthetic-bash48", "problem_id": "synthetic-bash48-0",
           "source_index": 0, "messages": [{"role": "user", "content": PROMPT}]}
    binding = {"eval_id": task["eval_id"], "task_id": task["task_id"], "set_id": row["set_id"],
        "problem_id": row["problem_id"], "sample_index": 0, "model": MODEL,
        "protocol_sha256": task["protocol_sha256"], "identity_kind": "hf-snapshot/v1",
        "checkpoint": signed_agentic.checkpoint_identity(MODEL), "prompt_sha256": prompt_sha256(PROMPT),
        "reference_sha256": digest({"prompt": PROMPT, "task": {"fixture": "signed-cpu-workspace/v1",
            "image": args.task_image, "expected_state_sha256": hashlib.sha256(b"42").hexdigest()}})}
    token = gateway.issue(env="live-env", split="train", index=0, image=args.task_image,
        engagement=_engagement(binding), checkpoint=binding["checkpoint"],
        budgets=default_budgets(wall_s=300, per_call_timeout_s=30, cpu_s=120, max_calls=64))
    grant = {"binding": binding, "grant": {"gateway_url": gateway.url, "token": token,
        "session_id": token["claims"]["session_id"], "expires_at": token["claims"]["expires_at"]}}
    generator, clients = ScriptedBashGenerator(tools), []
    args.bash_generator, args.bash_native = generator, None
    native_model_client = signed_agentic.model_client

    def capture_client(*positional, **kwargs):
        if clients:
            raise ValueError("the Bash fixture created another model client")
        client = native_model_client(*positional, **kwargs)
        clients.append(client)
        args.bash_native = client
        return client

    with _observe_bash_processes(SubprocessRuntime) as (workdirs, processes), \
            patch.object(signed_agentic, "model_client", capture_client):
        result = await signed_agentic.run_signed_episode(generator=generator, task=task, row=row,
            sample_index=0, grant=grant, seed=7, deadline=asyncio.get_running_loop().time() + 240)
    _assert_owned_cleanup(workdirs, processes)
    client = clients[0]
    try:
        await client.get_response(ChatDialect(), {}, None)
    except TimeoutError:
        pass
    else:
        raise AssertionError("the native Bash boundary accepted a 49th generation")
    evidence = result["episode"]
    verified = gateway.verify(evidence["transcript"])
    if (not verified.ok or verified.final.reward != 1.0 or len(verified.calls) != TURNS - 1
            or any(call.tool != "bash" for call in verified.calls)
            or generator.calls != TURNS or client.turns != TURNS or result["turns"] != TURNS
            or result["tool_calls"] != TURNS - 1 or result["finish_reason"] != "stop"
            or base64.b64decode(evidence["state_base64"], validate=True) != b"42"
            or len(evidence["generated_spans"]) != TURNS or len(evidence["model_token_ids"]) != TURNS):
        raise ValueError("the native signed Bash fixture did not pass CPU acceptance")
    manager = gateway.app.state.service["episodes"]
    saved = await asyncio.to_thread(manager.store.get, grant["grant"]["session_id"])
    managed = await asyncio.to_thread(manager.boxes.managed)
    if saved.get("cleanup") or grant["grant"]["session_id"] in managed.values():
        raise RuntimeError("the signed Bash fixture boxes were not removed")
    descriptor = os.open(out / "synthetic-bash-48.private.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical_json(evidence))
    return {"status": "passed", "harness": "signed-bash-v2", "actual_driver": "native signed Bash",
        "model_execution": "scripted CPU fixture; no model weights or inference",
        "grant_authority": "synthetic local gateway; not platform-issued",
        "source_qualification": "synthetic workspace; not a Terminal benchmark result",
        "model_turns": TURNS, "signed_calls": len(verified.calls), "verified_transcript": True,
        "graded_reward": verified.final.reward, "attempt_49_refused_before_generator": True,
        "owned_process_groups_absent": len(processes), "owned_runtime_directories_removed": len(workdirs),
        "owned_task_and_grading_boxes_absent": True,
        "native_runner_source_sha256": hashlib.sha256(inspect.getsource(signed_agentic.run_signed_episode).encode()).hexdigest(),
        "native_model_client_source_sha256": hashlib.sha256(inspect.getsource(native_model_client).encode()).hexdigest()}
