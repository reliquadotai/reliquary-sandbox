"""ToolRouter: each tool call answered once, in order, through the episode."""

import asyncio
from types import SimpleNamespace

import pytest

from reliquary_sandbox import tool_routing
from reliquary_sandbox.episode_client import EpisodeClosed
from reliquary_sandbox.observation import INVALID_JSON, NO_OUTPUT, NOT_OFFERED
from reliquary_sandbox.tool_routing import RoutingError, ToolRouter


class FakeEpisode:
    def __init__(self, outputs=None, final_after=None):
        self.calls = []
        self.outputs = outputs or {}
        self.final_after = final_after

    async def call(self, turn, k, tool, arguments):
        self.calls.append((turn, k, tool, arguments))
        body = {"turn": turn, "k": k, "tool": tool, "arguments": arguments,
                "output": self.outputs.get(arguments.get("command"), ""), "truncated": False,
                "timed_out": False, "exit_code": 0}
        final = {"body": {"status": "budget_exhausted"}} if len(self.calls) == self.final_after else None
        return SimpleNamespace(record={"body": body}, final=final)


CALLS = [("a", "bash", '{"command": "ls"}'), ("b", "bash", "{bad"),
         ("c", "edit", '{"path": "x", "old_str": "1", "new_str": "2"}')]


async def test_sent_calls_take_consecutive_k_and_refusals_none():
    episode = FakeEpisode({"ls": "x\n"})
    router = ToolRouter(episode)
    assert await router.answer(3, CALLS, "a") == "x\n"
    assert await router.answer(3, CALLS, "b") == INVALID_JSON
    assert await router.answer(3, CALLS, "c") == NO_OUTPUT
    assert [(t, k, tool) for t, k, tool, _ in episode.calls] == [(3, 0, "bash"), (3, 1, "edit")]
    assert router.refusals == [(3, 1, "invalid_json")]


async def test_an_answer_is_given_once():
    episode = FakeEpisode()
    router = ToolRouter(episode)
    first = await router.answer(0, CALLS[:1], "a")
    assert await router.answer(0, CALLS[:1], "a") == first
    assert len(episode.calls) == 1


async def test_calls_must_be_answered_in_order():
    with pytest.raises(RoutingError):
        await ToolRouter(FakeEpisode()).answer(0, CALLS, "c")


async def test_a_turn_seen_with_other_calls_is_refused():
    router = ToolRouter(FakeEpisode())
    await router.answer(0, CALLS[:1], "a")
    with pytest.raises(RoutingError):
        await router.answer(0, CALLS[1:], "b")


async def test_a_gateway_record_for_another_call_is_refused():
    class Wrong(FakeEpisode):
        async def call(self, turn, k, tool, arguments):
            result = await super().call(turn, k, tool, arguments)
            result.record["body"]["arguments"] = {"command": "rm -rf /"}
            return result

    with pytest.raises(RoutingError):
        await ToolRouter(Wrong()).answer(0, CALLS[:1], "a")


async def test_the_final_is_kept_and_no_call_runs_after_it():
    router = ToolRouter(FakeEpisode(final_after=1))
    await router.answer(0, CALLS[:1], "a")
    assert router.final["body"]["status"] == "budget_exhausted"
    with pytest.raises(RoutingError):
        await router.answer(1, [("d", "bash", '{"command": "ls"}')], "d")


async def test_call_ids_repeated_across_turns_are_answered_per_turn():
    episode = FakeEpisode({"ls": "x\n", "pwd": "/w\n"})
    router = ToolRouter(episode)
    assert await router.answer(0, [("call_0", "bash", '{"command": "ls"}')], "call_0") == "x\n"
    assert await router.answer(1, [("call_0", "bash", '{"command": "pwd"}')], "call_0") == "/w\n"
    assert [(t, k, a["command"]) for t, k, _, a in episode.calls] == [(0, 0, "ls"), (1, 0, "pwd")]
    # The order check is per turn too: turn 2's call_1 waits for turn 2's call_0.
    calls = [("call_0", "bash", '{"command": "ls"}'), ("call_1", "bash", '{"command": "ls"}')]
    with pytest.raises(RoutingError):
        await router.answer(2, calls, "call_1")


async def test_a_closed_episode_sets_the_final_and_ends_the_routing():
    final = {"body": {"status": "expired"}}

    class Closed(FakeEpisode):
        async def call(self, turn, k, tool, arguments):
            raise EpisodeClosed(409, {"final": final})

    router = ToolRouter(Closed())
    with pytest.raises(EpisodeClosed):
        await router.answer(0, CALLS[:1], "a")
    assert router.final == final
    with pytest.raises(RoutingError):
        await router.answer(1, [("d", "bash", '{"command": "ls"}')], "d")


async def test_tools_outside_the_offered_set_are_refused_and_take_no_position():
    episode = FakeEpisode({"ls": "x\n"})
    router = ToolRouter(episode, offered=("bash",))
    calls = [("c", "edit", '{"path": "x", "old_str": "1", "new_str": "2"}'),
             ("a", "bash", '{"command": "ls"}')]
    assert await router.answer(0, calls, "c") == NOT_OFFERED
    assert await router.answer(0, calls, "a") == "x\n"
    assert [(t, k, tool) for t, k, tool, _ in episode.calls] == [(0, 0, "bash")]
    assert router.refusals == [(0, 0, "not_offered")]


async def test_an_empty_rendering_fails_closed(monkeypatch):
    monkeypatch.setattr(tool_routing, "render_observation", lambda body: "")
    with pytest.raises(RoutingError):
        await ToolRouter(FakeEpisode()).answer(0, CALLS[:1], "a")


async def test_concurrent_answers_are_served_one_at_a_time():
    class Slow(FakeEpisode):
        async def call(self, turn, k, tool, arguments):
            await asyncio.sleep(0.05)
            return await super().call(turn, k, tool, arguments)

    router = ToolRouter(Slow({"ls": "x\n"}))
    calls = [("a", "bash", '{"command": "ls"}'), ("b", "bash", '{"command": "ls"}')]
    first = asyncio.create_task(router.answer(0, calls, "a"))
    await asyncio.sleep(0.01)
    assert await router.answer(0, calls, "b") == "x\n"
    assert await first == "x\n"
