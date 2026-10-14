"""verifiers' real bash program, a scripted fake model and a local gateway over FakeBoxes.
The text the model sees for each call must be the pinned rendering of the signed record
(the miner/validator parity check). Opt-in: RUN_VERIFIERS_BRIDGE=1. It prepares the
program with uv (network), so it runs on a dev host but not in CI."""

import os

import pytest

pytest.importorskip("verifiers.v1")
pytestmark = pytest.mark.skipif(os.getenv("RUN_VERIFIERS_BRIDGE") != "1",
                                reason="set RUN_VERIFIERS_BRIDGE=1 (needs uv and network)")

from episode_fixtures import IMAGE, budgets  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from gateway_fixtures import fake_gateway  # noqa: E402
from verifiers.v1.clients import ModelContext  # noqa: E402
from verifiers.v1.configs.client import EvalClientConfig  # noqa: E402

from reliquary_sandbox.observation import INVALID_JSON, NO_OUTPUT, render_observation  # noqa: E402
from reliquary_sandbox_service.episodes.local_gateway import serve_app  # noqa: E402
from reliquary_sandbox_verifiers import SandboxEpisodeRunner  # noqa: E402

SCRIPT = [
    {"tool_calls": [("bash", '{"command": "write /work/answer.txt 42"}'),
                    ("bash", '{"command": "echo hi"}'),
                    ("bash", '{"command": ')]},
    {"tool_calls": [("bash", '{"command": "cat /work/answer.txt"}'),
                    ("bash", '{"command": "echo bye"}')]},
    {"content": "done"},
]
BURN = [
    {"tool_calls": [("bash", '{"command": "echo hi"}'), ("bash", '{"command": "burn 1500"}'),
                    ("bash", '{"command": "echo never"}')]},
    {"content": "a turn that must never be asked for"},
]


class FakeModel:
    """Tool call ids restart at `call_0` every turn, as many servers number them."""

    def __init__(self, script=SCRIPT):
        self.requests = []
        self.app = FastAPI()

        @self.app.post("/v1/chat/completions")
        async def complete(body: dict):
            self.requests.append(body)
            step = script[min(len(self.requests) - 1, len(script) - 1)]
            message = {"role": "assistant", "content": step.get("content")}
            if "tool_calls" in step:
                message["tool_calls"] = [
                    {"id": f"call_{i}", "type": "function",
                     "function": {"name": name, "arguments": arguments}}
                    for i, (name, arguments) in enumerate(step["tool_calls"])]
            return {"id": "fake", "object": "chat.completion", "created": 0,
                    "model": body.get("model", "fake"),
                    "choices": [{"index": 0, "message": message,
                                 "finish_reason": "tool_calls" if "tool_calls" in step else "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


async def test_the_model_sees_exactly_the_rendering_of_the_signed_records(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODEL_KEY", "fake")
    model = FakeModel()
    with fake_gateway(tmp_path) as (gateway, _), serve_app(model.app) as model_url:
        ctx = ModelContext(model="fake", client=EvalClientConfig(base_url=f"{model_url}/v1",
                                                                 api_key_var="FAKE_MODEL_KEY"))
        token = gateway.issue(env="fake-env", split="train", index=7, image=IMAGE,
                              budgets=budgets())
        async with SandboxEpisodeRunner(ctx=ctx, gateway_url=gateway.url, max_turns=4) as runner:
            result = await runner.run(token=token, prompt="Write 42 to /work/answer.txt.")
        verified = gateway.verify(result.transcript)
    assert result.final["body"]["status"] == "graded", result
    assert result.final["body"]["reward"] == 1.0
    assert verified.ok, verified.reasons
    calls = [r["body"] for r in result.transcript["records"] if r["body"]["kind"] == "call"]
    seen = [m["content"] for m in model.requests[1]["messages"] if m["role"] == "tool"]
    assert seen == [render_observation(calls[0]), render_observation(calls[1]), INVALID_JSON]
    # The second turn reuses the ids call_0 and call_1: each is answered afresh.
    assert [(c["turn"], c["k"]) for c in calls] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    later = [m["content"] for m in model.requests[2]["messages"] if m["role"] == "tool"]
    assert later == [*seen, render_observation(calls[2]), render_observation(calls[3])]
    assert later[3:] == ["42\n", "bye\n"]
    assert len(model.requests) == 3 and result.error is None
    # `write` prints nothing: the real program got "(no output)" from the hook, never ""
    # (an "allow"), so its local run_bash (which exits) never ran.
    assert seen[0] == NO_OUTPUT
    assert result.refusals == ((0, 2, "invalid_json"),)
    first_user = next(m for m in model.requests[0]["messages"] if m["role"] == "user")
    assert first_user["content"] == "Write 42 to /work/answer.txt."


async def test_a_final_mid_turn_stops_the_agent_before_another_model_call(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODEL_KEY", "fake")
    model = FakeModel(BURN)
    with fake_gateway(tmp_path) as (gateway, _), serve_app(model.app) as model_url:
        ctx = ModelContext(model="fake", client=EvalClientConfig(base_url=f"{model_url}/v1",
                                                                 api_key_var="FAKE_MODEL_KEY"))
        # FakeBoxes: `burn N` uses N ms of CPU, past the 1 s budget.
        token = gateway.issue(env="fake-env", split="train", index=7, image=IMAGE,
                              budgets=budgets(cpu_s=1))
        async with SandboxEpisodeRunner(ctx=ctx, gateway_url=gateway.url, max_turns=4) as runner:
            result = await runner.run(token=token, prompt="p")
        verified = gateway.verify(result.transcript, require_graded=False)
    assert verified.ok, verified.reasons
    assert result.final["body"]["status"] == "budget_exhausted", result
    calls = [r["body"]["arguments"]["command"] for r in result.transcript["records"]
             if r["body"]["kind"] == "call"]
    assert calls == ["echo hi", "burn 1500"]
    assert len(model.requests) == 1, "no model turn may run after the final"
    # A clean stop by the task's hook, not a routing failure.
    assert result.error is None and result.trace.stop_condition == "episode_closed", result
