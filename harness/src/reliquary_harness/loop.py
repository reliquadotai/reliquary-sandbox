"""The episode loop of a harness spec: model turn, tool calls, observations, next turn.

It reproduces verifiers b2e4e81's bash harness as its interception layer drove it
(`harnesses/bash/program.py`'s chat loop, `session.py`'s limits and the task's stop):

* before each model call: `max_turns` once `max_turns` turns were served (so the turn
  that reaches it has its calls answered, and nothing renders their observations), then
  `episode_closed` once the episode is closed;
* a prompt longer than the engine's context: `context_length` (that turn never runs);
* a turn with no call (after the spec's filter): `agent_completed`;
* each call in order is answered by `answer(turn, calls, call_id)` (the gateway's
  rendering of its signed record, or the refusal text of a call never sent); once an
  answer closed the episode (a final on that call), the loop stops at once
  (`episode_closed`): the turn's later calls are never sent and no observation is
  rendered;
* otherwise the next prompt is the renderer's bridge from the turn's own prompt and
  completion ids with the observations as tool messages.

Anything `generate` or `answer` raises (an engine error, a routing failure, a gateway
close without a final on a call) propagates: the episode did not complete.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from .engine import OverlongPrompt, Turn
from .qwen import QwenTurnRenderer

AGENT_COMPLETED, MAX_TURNS, CONTEXT_LENGTH, EPISODE_CLOSED = (
    "agent_completed", "max_turns", "context_length", "episode_closed")


class BridgeFailed(RuntimeError):
    """The renderer could not bridge to the next turn: no prompt the validator renders."""


@dataclass(frozen=True)
class Outcome:
    stop: str
    turns: tuple[Turn, ...]


Generate = Callable[[Sequence[int], "int | None"], Awaitable[Turn]]
Answer = Callable[[int, Sequence[tuple[str, str, str]], str], Awaitable[str]]


async def play(*, renderer: QwenTurnRenderer, prompt: str, max_turns: int, generate: Generate,
               answer: Answer, closed: Callable[[], bool]) -> Outcome:
    """`generate(prompt_ids, bridged_from)` runs one turn; `closed()` says whether the
    episode has ended (a final was received)."""
    prompt_ids = await asyncio.to_thread(renderer.initial_ids, prompt)
    bridged_from: int | None = None
    turns: list[Turn] = []
    while True:
        if len(turns) >= max_turns:
            return Outcome(MAX_TURNS, tuple(turns))
        if closed():
            return Outcome(EPISODE_CLOSED, tuple(turns))
        try:
            turn = await generate(prompt_ids, bridged_from)
        except OverlongPrompt:
            return Outcome(CONTEXT_LENGTH, tuple(turns))
        turns.append(turn)
        if not turn.calls:
            return Outcome(AGENT_COMPLETED, tuple(turns))
        index = len(turns) - 1
        observations = []
        for call_id, _, _ in turn.calls:
            observations.append(await answer(index, turn.calls, call_id))
            if closed():
                return Outcome(EPISODE_CLOSED, tuple(turns))
        following = await asyncio.to_thread(renderer.next_prompt, turn.prompt_ids, turn.completion_ids,
                                            observations)
        if following is None:
            raise BridgeFailed(f"turn {index}: the renderer cannot bridge to the next turn")
        bridged_from = len(turn.prompt_ids) + len(turn.completion_ids)
        prompt_ids = following


__all__ = ["AGENT_COMPLETED", "BridgeFailed", "CONTEXT_LENGTH", "EPISODE_CLOSED", "MAX_TURNS", "Outcome", "play"]
