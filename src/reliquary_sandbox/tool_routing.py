"""Answering a model's tool calls through a signed episode: once each, in order.

A harness hands `answer` the assistant message's calls as `(id, name, raw arguments)`
and the id it needs answered. Calls are planned with `observation.plan_call` (over the
episode's offered tools, when given); refused ones are answered with their pinned text,
sent ones run as `(turn, k)` with `k` from `observation.sent_positions`, and the answer
is `observation.render_observation` of the signed record (a refusal consumes
no record). Answers are cached by `(turn, call id)` (models reuse ids such as `call_0`
across turns), so a harness that asks twice (before and after a tool runs) gets the
same text and the gateway sees one call. Answers are served one at a time.

An answer is never empty: verifiers' tool hook would read an empty rewrite as "allow"
and run the call on the miner's host. A closed episode (a final on a call, or
`EpisodeClosed`) ends the routing: `final` holds the signed final when there is one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection, Sequence
from typing import Any, Protocol

from .attest.canonical import canonical_json
from .episode_client import EpisodeClosed
from .observation import Refuse, plan_call, render_observation, sent_positions


class RoutingError(RuntimeError):
    """The harness asked for something the episode cannot answer consistently."""


class CallingEpisode(Protocol):
    async def call(self, turn: int, k: int, tool: str, arguments: dict[str, Any]) -> Any: ...


class ToolRouter:
    def __init__(self, episode: CallingEpisode, offered: Collection[str] | None = None):
        self._episode = episode
        self._offered = None if offered is None else frozenset(offered)
        self._answers: dict[tuple[int, str], str] = {}
        self._turns: dict[int, tuple[str, ...]] = {}
        self._lock = asyncio.Lock()
        self._closed = False
        self.records: list[dict[str, Any]] = []
        self.refusals: list[tuple[int, int, str]] = []
        """`(turn, j, reason)`: `j` is the call's index in its assistant message."""
        self.final: dict[str, Any] | None = None

    async def answer(self, turn: int, calls: Sequence[tuple[str, str, str]], call_id: str) -> str:
        async with self._lock:
            return await self._answer(turn, calls, call_id)

    async def _answer(self, turn: int, calls: Sequence[tuple[str, str, str]],
                      call_id: str) -> str:
        ids = tuple(call[0] for call in calls)
        if len(set(ids)) != len(ids):
            raise RoutingError("an assistant message repeats a tool call id")
        if self._turns.setdefault(turn, ids) != ids:
            raise RoutingError(f"turn {turn} was seen with other tool calls")
        if (turn, call_id) in self._answers:
            return self._answers[(turn, call_id)]
        if call_id not in ids:
            raise RoutingError(f"unknown tool call id {call_id!r}")
        index = ids.index(call_id)
        if any((turn, earlier) not in self._answers for earlier in ids[:index]):
            raise RoutingError("tool calls must be answered in order")
        if self._closed or self.final is not None:
            raise RoutingError("the episode already ended")
        plans = [plan_call(name, arguments, self._offered) for _, name, arguments in calls]
        plan = plans[index]
        if isinstance(plan, Refuse):
            self.refusals.append((turn, index, plan.reason))
            text = plan.observation
        else:
            k = sent_positions(plans)[index]
            try:
                result = await self._episode.call(turn, k, plan.tool, plan.arguments)
            except EpisodeClosed as closed:
                self._closed = True
                self.final = closed.final
                raise
            body = result.record["body"]
            if ((body["turn"], body["k"], body["tool"]) != (turn, k, plan.tool)
                    or canonical_json(body["arguments"]) != canonical_json(plan.arguments)):
                raise RoutingError("the gateway recorded a different call")
            self.records.append(result.record)
            if result.final is not None:
                self.final = result.final
            text = render_observation(body)
        if not text.strip():
            raise RoutingError("an observation rendered empty; refusing to answer")
        self._answers[(turn, call_id)] = text
        return text
