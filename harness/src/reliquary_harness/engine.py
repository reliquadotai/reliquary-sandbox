"""One model turn on a vLLM `/inference/v1/generate` endpoint, exactly as verifiers b2e4e81's
train client sends it (`clients/train.py:TrainClient.get_response`).

The prompt ids come from the episode loop (the spec's initial render, then each bridge);
`renderers.client.generate` posts them with the client's sampling (`temperature`,
`top_p`, `max_tokens`), the renderer's stop ids, `logprobs=1`,
`skip_special_tokens=False`, and, on a bridged turn, `routed_experts_prompt_start`; every
turn names the episode in `X-Session-ID`. Before posting, it refuses a prompt longer than
the engine's `max_model_len` (`GET /v1/models`) with `OverlongPromptError`: the episode's
`context_length` stop. The calls are the spec's filter of the parse `generate` returns.

`renderers` and `openai` (a dependency of `renderers`) are imported inside functions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .qwen import QwenTurnRenderer

SESSION_ID_HEADER = "X-Session-ID"
CONNECT_TIMEOUT_S = 5.0
"""No read timeout (agentic turns are slow; the episode's wall budget is the backstop), a
connect bound so an unreachable endpoint fails fast; no client-side retries."""


class OverlongPrompt(Exception):
    """The next prompt does not fit the engine's context: the episode stops (context_length)."""


@dataclass(frozen=True)
class Turn:
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    calls: tuple[tuple[str, str, str], ...]
    """`(id, name, arguments)` the spec's filter keeps, in order."""
    finish_reason: str | None


def openai_client(base_url: str, api_key: str):
    """The engine connection verifiers' train client opens (`build_async_openai`)."""
    import httpx
    from openai import AsyncOpenAI

    timeout = httpx.Timeout(connect=CONNECT_TIMEOUT_S, read=None, write=None, pool=None)
    return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0,
                       http_client=httpx.AsyncClient(
                           timeout=timeout,
                           limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)))


class GenerateEngine:
    """Turns of one model on one endpoint. `sampling` is the client's: `temperature`,
    `top_p`, `max_tokens` (None values are not sent, as `Sampling.wire_args`)."""

    def __init__(self, renderer: QwenTurnRenderer, *, client, model: str,
                 sampling: Mapping[str, Any]) -> None:
        self._renderer = renderer
        self._client = client
        self.model = model
        self._sampling = {key: value for key, value in dict(sampling).items() if value is not None}

    async def turn(self, prompt_ids, *, session_id: str, bridged_from: int | None = None) -> Turn:
        """One turn. `bridged_from` is the length of the previous turn's prompt and
        completion when `prompt_ids` bridges from it (None on the first turn)."""
        from renderers import OverlongPromptError
        from renderers.client import generate

        sampling = dict(self._sampling)
        if bridged_from is not None:
            sampling["routed_experts_prompt_start"] = max(int(bridged_from) - 1, 0)
        try:
            result = await generate(client=self._client, renderer=self._renderer.renderer, messages=[],
                                    model=self.model, prompt_ids=list(prompt_ids),
                                    tools=self._renderer.tools, sampling_params=sampling,
                                    extra_headers={SESSION_ID_HEADER: session_id})
        except OverlongPromptError as exc:
            raise OverlongPrompt(str(exc)) from exc
        return Turn(tuple(int(t) for t in result["prompt_ids"]),
                    tuple(int(t) for t in result["completion_ids"]),
                    tuple(self._renderer.calls_of(result.get("tool_calls"))),
                    result.get("finish_reason"))

    async def aclose(self) -> None:
        await self._client.close()


__all__ = ["GenerateEngine", "OverlongPrompt", "SESSION_ID_HEADER", "Turn", "openai_client"]
