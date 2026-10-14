"""Run one signed episode with verifiers' bash harness: the miner does inference only."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass
from typing import Any

from verifiers.v1.utils.loaders import load_environment, resolve_env_config

from reliquary_sandbox.episode_client import EpisodeClient
from reliquary_sandbox.observation import require_tools_version
from reliquary_sandbox.tool_routing import ToolRouter

from .task import SandboxEpisodeData, SandboxEpisodeTask

TASKSET_ID = "reliquary-sandbox-verifiers"
HARNESS_ID = "reliquary-sandbox-verifiers"
FINISH_MARGIN_S = 60.0
"""The agent stops this long before the episode's wall deadline, so `finish` is in time."""
CANCELLED_FINISH_TIMEOUT_S = 120.0
"""When the run is cancelled, `finish` (which frees the box) gets at most this long
before the cancellation propagates; the gateway's own deadline closes the episode
otherwise."""


@dataclass(frozen=True)
class SandboxEpisodeResult:
    """One episode's outcome. `final`, `state` and `transcript` are the gateway's.
    `error` set means the agent run did not complete cleanly (a framework, harness or
    routing failure, or the wall budget): do not submit the episode."""

    episode_id: str
    trace: Any
    final: dict[str, Any]
    state: bytes | None
    transcript: dict[str, Any]
    refusals: tuple[tuple[int, int, str], ...]
    error: str | None


def env_config(*, tools: tuple[str, ...], max_turns: int,
               harness_env: dict[str, str] | None = None) -> dict[str, Any]:
    """The verifiers env config offering exactly `tools`: `bash`, or `bash` and `edit`
    (the bash program always lists `bash`, so an edit-only set cannot be offered)."""
    if set(tools) not in ({"bash"}, {"bash", "edit"}):
        raise ValueError(f"the bash harness offers bash or bash and edit, not {sorted(tools)}")
    return {
        "taskset": {"id": TASKSET_ID},
        "agent": {"harness": {"id": HARNESS_ID, "edit": "edit" in tools, "search": False,
                              "env": dict(harness_env or {})},
                  "runtime": {"type": "subprocess"},
                  "max_turns": max_turns},
    }


def _errors(errors) -> str:
    return "; ".join(f"{type(e).__name__}: {getattr(e, 'message', e)}" for e in errors)


class SandboxEpisodeRunner:
    def __init__(self, *, ctx: Any, gateway_url: str, max_turns: int,
                 harness_env: dict[str, str] | None = None) -> None:
        self._ctx = ctx
        self._url = gateway_url
        self._max_turns = max_turns
        self._harness_env = dict(harness_env or {})
        self._envs: dict[tuple[str, ...], Any] = {}
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> SandboxEpisodeRunner:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._stack.aclose()

    async def _env(self, tools: tuple[str, ...]) -> Any:
        key = tuple(sorted(tools))
        if key not in self._envs:
            env = load_environment(resolve_env_config(
                env_config(tools=key, max_turns=self._max_turns, harness_env=self._harness_env)))
            await self._stack.enter_async_context(env.serving())
            self._envs[key] = env
        return self._envs[key]

    async def run(self, *, token: dict[str, Any], prompt: str,
                  on_trace=None) -> SandboxEpisodeResult:
        """Open the token's episode and run the agent on it. The offered tools are
        record 0's (signed), for both the harness's tool list and the router's refusals."""
        loop = asyncio.get_running_loop()
        # Taken before the open: the gateway's wall time starts no earlier (when the
        # box is prepared), so this deadline is never later than the gateway's.
        started = loop.time()
        async with EpisodeClient(self._url) as client:
            episode = await client.open(token)
            router = None
            trace, error = None, None
            try:
                body = episode.record0["body"]
                deadline = started + max(1.0, body["budgets"]["wall_s"] - FINISH_MARGIN_S)
                require_tools_version(body["tools_version"])
                tools = tuple(body["tools"])
                router = ToolRouter(episode, offered=tools)
                env = await self._env(tools)
                claims = token["claims"]
                task = SandboxEpisodeTask(
                    SandboxEpisodeData(prompt=prompt, env=claims["env"], split=claims["split"],
                                       index=claims["index"]),
                    router)
                async with asyncio.timeout_at(deadline):
                    result = await env.run_episode(task, self._ctx, on_trace=on_trace)
                trace = result.traces[0] if result.traces else None
                if not (result.ok and trace is not None and trace.ok):
                    errors = [*(trace.errors if trace is not None else []), *result.errors]
                    error = _errors(errors) or "episode not ok"
            except TimeoutError:
                error = "stopped before the episode's wall budget ran out"
            except Exception as failure:  # the episode is still finished below
                error = f"{type(failure).__name__}: {failure}"
            except BaseException:
                # Cancelled or interrupted: try to free the box, briefly, then propagate.
                with suppress(Exception):
                    async with asyncio.timeout(CANCELLED_FINISH_TIMEOUT_S):
                        await episode.finish()
                raise
            finished = await episode.finish()
            transcript = (await episode.transcript())["transcript"]
        refusals = tuple(router.refusals) if router is not None else ()
        return SandboxEpisodeResult(episode.id, trace, finished.record, finished.state,
                                    transcript, refusals, error)
