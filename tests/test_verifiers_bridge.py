"""The verifiers bridge, without running a model."""

import pytest

vf = pytest.importorskip("verifiers.v1")

from verifiers.v1.types import (  # noqa: E402
    AssistantMessage,
    Request,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from reliquary_sandbox_verifiers import (  # noqa: E402
    HARNESS_ID,
    SandboxBashHarness,
    SandboxEpisodeTaskset,
    env_config,
    program_source,
    route_request,
)


class FakeRouter:
    def __init__(self):
        self.asked = []

    async def answer(self, turn, calls, call_id):
        self.asked.append((turn, [c[0] for c in calls], call_id))
        return f"answer {call_id}"


def conversation(*tail):
    assistant = AssistantMessage(content="", tool_calls=[
        ToolCall(id="a", name="bash", arguments='{"command": "ls"}'),
        ToolCall(id="b", name="bash", arguments='{"command": "pwd"}')])
    return Request(messages=[SystemMessage(content="s"), UserMessage(content="u"), assistant, *tail])


def test_the_program_never_runs_a_tool_locally():
    source = program_source()
    compile(source, "program.py", "exec")
    assert source.endswith('if __name__ == "__main__":\n    asyncio.run(main())\n')
    assert source.index("reached local execution") > source.index("def run_bash(command: str)")


@pytest.mark.parametrize("tool, arguments", [
    ("run_bash", ("touch {marker}",)),
    ("run_edit", ("{marker}", "a", "b")),
])
def test_the_patched_local_tools_exit_without_running(tmp_path, tool, arguments):
    marker = tmp_path / "ran"
    namespace = {"__name__": "reliquary_program_under_test"}
    exec(compile(program_source(), "program.py", "exec"), namespace)
    with pytest.raises(SystemExit, match="reached local execution"):
        namespace[tool](*(a.format(marker=marker) for a in arguments))
    assert not marker.exists()


async def test_an_empty_or_blank_output_is_never_an_empty_tool_message():
    from types import SimpleNamespace

    from reliquary_sandbox.observation import NO_OUTPUT
    from reliquary_sandbox.tool_routing import ToolRouter

    class SilentEpisode:
        async def call(self, turn, k, tool, arguments):
            output = {"ls": "", "pwd": " \n\t"}[arguments["command"]]
            body = {"turn": turn, "k": k, "tool": tool, "arguments": arguments,
                    "output": output, "truncated": False, "timed_out": False, "exit_code": 0}
            return SimpleNamespace(record={"body": body}, final=None)

    router = ToolRouter(SilentEpisode())
    request = conversation(ToolMessage(tool_call_id="a", content=""),
                           ToolMessage(tool_call_id="b", content=""))
    rewritten = await route_request(router, request)
    # verifiers answers "allow" (run locally) when the rewritten message equals the incoming
    # one, whose content is "": a rewrite must always change it.
    assert rewritten is not None
    assert [m.content for m in rewritten.messages[-2:]] == [NO_OUTPUT, NO_OUTPUT]


async def test_route_request_answers_each_new_tool_message_once():
    router = FakeRouter()
    first = await route_request(router, conversation(ToolMessage(tool_call_id="a", content="")))
    assert first.messages[-1].content == "answer a"
    assert router.asked == [(0, ["a", "b"], "a")]
    second = await route_request(router, conversation(
        ToolMessage(tool_call_id="a", content="answer a"),
        ToolMessage(tool_call_id="b", content="")))
    assert [m.content for m in second.messages[-2:]] == ["answer a", "answer b"]


async def test_route_request_leaves_a_request_without_tool_messages_alone():
    assert await route_request(FakeRouter(), conversation()) is None
    plain = Request(messages=[UserMessage(content="u")])
    assert await route_request(FakeRouter(), plain) is None


def test_the_turn_counts_earlier_assistant_messages():
    import asyncio

    router = FakeRouter()
    earlier = AssistantMessage(content="", tool_calls=[ToolCall(id="z", name="bash", arguments="{}")])
    request = Request(messages=[UserMessage(content="u"), earlier,
                                ToolMessage(tool_call_id="z", content="answer z"),
                                *conversation(ToolMessage(tool_call_id="a", content="")).messages[2:]])
    asyncio.run(route_request(router, request))
    assert router.asked[-1][0] == 1


def test_the_bridge_loads_as_a_verifiers_environment():
    from verifiers.v1.utils.loaders import load_environment, resolve_env_config

    env = load_environment(resolve_env_config(env_config(tools=("bash",), max_turns=3)))
    assert env.config.agent.harness.id == HARNESS_ID
    assert env.config.agent.harness.edit is False
    assert SandboxBashHarness.__mro__[1].__name__ == "BashHarness"
    assert [type(h) for h in env._harnesses.values()] == [SandboxBashHarness]
    assert list(SandboxEpisodeTaskset(env.config.taskset).load()) == []


def test_the_task_stops_the_agent_once_the_episode_has_a_final():
    from reliquary_sandbox.tool_routing import ToolRouter
    from reliquary_sandbox_verifiers import SandboxEpisodeData, SandboxEpisodeTask

    router = ToolRouter(object())
    task = SandboxEpisodeTask(SandboxEpisodeData(prompt="p", env="e", split="s", index=0), router)
    [stop] = task.hooks("stop")
    request = conversation()
    assert stop(request) is False
    router.final = {"body": {"status": "budget_exhausted"}}
    assert stop(request) is True


PINNED = ('{"url":"https://github.com/PrimeIntellect-ai/verifiers.git","vcs_info":{"vcs":"git",'
          '"commit_id":"b2e4e8157783b2c0dffc7821044c87f29f1c3ccf"}}')


def test_the_installed_verifiers_is_the_pinned_commit():
    from reliquary_sandbox_verifiers import harness

    harness.require_pinned_verifiers(harness.installed_direct_url(), harness.installed_harness())
    harness.require_pinned_verifiers(PINNED, harness.installed_harness())


@pytest.mark.parametrize("direct_url", [
    None, "not json", '{"url": "file:///src/verifiers"}',
    PINNED.replace("b2e4e8157783", "000000000000"),
])
def test_another_verifiers_install_is_refused(direct_url):
    from reliquary_sandbox_verifiers import harness

    with pytest.raises(RuntimeError, match="verifiers"):
        harness.require_pinned_verifiers(direct_url, harness.installed_harness())


def test_a_changed_bash_harness_is_refused():
    from reliquary_sandbox_verifiers import harness

    with pytest.raises(RuntimeError, match="verifiers"):
        harness.require_pinned_verifiers(PINNED, harness.installed_harness() + b"\n# changed\n")


async def test_the_episode_is_finished_even_when_the_run_cannot_start(tmp_path, monkeypatch):
    from episode_fixtures import IMAGE, budgets
    from gateway_fixtures import fake_gateway
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.configs.client import EvalClientConfig

    from reliquary_sandbox_verifiers import SandboxEpisodeRunner, runner

    def broken(*args, **kwargs):
        raise ValueError("malformed claims")

    monkeypatch.setattr(runner, "SandboxEpisodeTask", broken)
    ctx = ModelContext(model="none", client=EvalClientConfig(base_url="http://127.0.0.1:9/v1"))
    with fake_gateway(tmp_path) as (gateway, _):
        token = gateway.issue(env="fake-env", split="train", index=7, image=IMAGE,
                              budgets=budgets())
        async with SandboxEpisodeRunner(ctx=ctx, gateway_url=gateway.url, max_turns=2) as run:
            result = await run.run(token=token, prompt="p")
    assert "malformed claims" in result.error
    assert result.final["body"]["kind"] == "final"
    assert result.transcript["records"][-1] == result.final


async def test_the_offered_tools_are_record_zeros(tmp_path, monkeypatch):
    from episode_fixtures import IMAGE, budgets
    from gateway_fixtures import fake_gateway
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.configs.client import EvalClientConfig

    from reliquary_sandbox_verifiers import SandboxEpisodeRunner, runner

    seen = {}

    def router(episode, offered):
        seen["offered"] = offered
        seen["record0"] = episode.record0["body"]["tools"]
        raise ValueError("stop here")

    monkeypatch.setattr(runner, "ToolRouter", router)
    ctx = ModelContext(model="none", client=EvalClientConfig(base_url="http://127.0.0.1:9/v1"))
    with fake_gateway(tmp_path) as (gateway, _):
        token = gateway.issue(env="fake-env", split="train", index=7, image=IMAGE,
                              budgets=budgets())
        async with SandboxEpisodeRunner(ctx=ctx, gateway_url=gateway.url, max_turns=2) as run:
            result = await run.run(token=token, prompt="p")
    assert seen == {"offered": ("bash", "edit"), "record0": ["bash", "edit"]}
    assert "stop here" in result.error and result.final["body"]["kind"] == "final"


def test_the_env_config_offers_exactly_the_given_tools():
    assert env_config(tools=("bash",), max_turns=1)["agent"]["harness"]["edit"] is False
    assert env_config(tools=("bash", "edit"), max_turns=1)["agent"]["harness"]["edit"] is True
    with pytest.raises(ValueError):
        env_config(tools=("edit",), max_turns=1)  # the bash program always offers bash
