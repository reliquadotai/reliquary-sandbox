"""The loop's stops and call order, over a fake renderer, engine and gateway."""

from __future__ import annotations

import pytest
from reliquary_harness.engine import OverlongPrompt, Turn
from reliquary_harness.loop import BridgeFailed, play


class Renderer:
    def __init__(self, bridge=True):
        self.bridges = []
        self._bridge = bridge

    def initial_ids(self, prompt):
        return [1, 2]

    def next_prompt(self, prompt_ids, completion_ids, observations):
        self.bridges.append((list(prompt_ids), list(completion_ids), list(observations)))
        if not self._bridge:
            return None
        return [*prompt_ids, *completion_ids, 9 * len(observations)]


def calls(*names):
    return tuple((f"call_{i}", name, "{}") for i, name in enumerate(names))


class World:
    """Turn n of the script; `final_on` closes the episode on that (turn, k)."""

    def __init__(self, script, final_on=None, overlong_at=None):
        self.script, self.final_on, self.overlong_at = script, final_on, overlong_at
        self.prompts, self.bridged, self.answered = [], [], []
        self.final = None

    async def generate(self, prompt_ids, bridged_from):
        n = len(self.prompts)
        self.prompts.append(list(prompt_ids))
        self.bridged.append(bridged_from)
        if n == self.overlong_at:
            raise OverlongPrompt("too long")
        return Turn(tuple(prompt_ids), (100 + n,), self.script[n], "stop")

    async def answer(self, turn, turn_calls, call_id):
        k = [c[0] for c in turn_calls].index(call_id)
        self.answered.append((turn, k))
        if (turn, k) == self.final_on:
            self.final = {"status": "budget_exhausted"}
        return f"obs {turn}.{k}"

    def closed(self):
        return self.final is not None


async def run(world, renderer=None, max_turns=6):
    return await play(renderer=renderer or Renderer(), prompt="p", max_turns=max_turns,
                      generate=world.generate, answer=world.answer, closed=world.closed)


async def test_a_turn_without_a_call_completes_the_agent():
    world = World([calls("bash", "edit"), calls("bash"), ()])
    renderer = Renderer()
    outcome = await run(world, renderer)
    assert outcome.stop == "agent_completed" and len(outcome.turns) == 3
    assert world.answered == [(0, 0), (0, 1), (1, 0)]
    assert world.bridged == [None, 3, 5]
    assert renderer.bridges[0] == ([1, 2], [100], ["obs 0.0", "obs 0.1"])
    assert world.prompts[1] == [1, 2, 100, 18]


async def test_the_turn_reaching_max_turns_has_its_calls_answered_then_stops():
    world = World([calls("bash"), calls("bash", "bash"), calls("bash")])
    outcome = await run(world, max_turns=2)
    assert outcome.stop == "max_turns" and len(world.prompts) == 2
    assert world.answered == [(0, 0), (1, 0), (1, 1)]


async def test_a_final_on_a_call_stops_before_the_next_call():
    world = World([calls("bash", "bash", "bash"), ()], final_on=(0, 1))
    outcome = await run(world)
    assert outcome.stop == "episode_closed" and len(world.prompts) == 1
    assert world.answered == [(0, 0), (0, 1)]


async def test_a_final_on_the_last_turns_call_closes_it_even_at_max_turns():
    # verifiers' task stop runs right after the call that brought the final.
    world = World([calls("bash")], final_on=(0, 0))
    assert (await run(world, max_turns=1)).stop == "episode_closed"


async def test_an_overlong_prompt_is_a_context_length_stop():
    world = World([calls("bash"), calls("bash")], overlong_at=1)
    outcome = await run(world)
    assert outcome.stop == "context_length" and len(outcome.turns) == 1
    assert world.answered == [(0, 0)]


async def test_a_failed_bridge_is_an_error_not_a_stop():
    with pytest.raises(BridgeFailed):
        await run(World([calls("bash"), ()]), Renderer(bridge=False))


async def test_what_answer_raises_propagates():
    class Closed(Exception):
        pass

    world = World([calls("bash")])

    async def answer(*_):
        raise Closed

    with pytest.raises(Closed):
        await play(renderer=Renderer(), prompt="p", max_turns=3, generate=world.generate, answer=answer,
                   closed=world.closed)
