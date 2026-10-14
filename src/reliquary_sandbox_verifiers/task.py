"""The verifiers task of one opened episode: every tool message is the gateway's answer."""

from __future__ import annotations

import verifiers.v1 as vf
from verifiers.v1.types import AssistantMessage, Request, ToolMessage

from reliquary_sandbox.tool_routing import ToolRouter


class SandboxEpisodeData(vf.TaskData):
    env: str
    split: str
    index: int


async def route_request(router: ToolRouter, request: Request) -> Request | None:
    """Answer the tool messages after the last assistant message through `router`.
    Unchanged messages (already answered) stay as they are; None when nothing changes."""
    messages = list(request.messages)
    last = max((i for i, m in enumerate(messages) if isinstance(m, AssistantMessage)),
               default=None)
    if last is None or not messages[last].tool_calls:
        return None
    calls = [(call.id, call.name, call.arguments) for call in messages[last].tool_calls]
    turn = sum(1 for m in messages[:last] if isinstance(m, AssistantMessage))
    changed = False
    for position in range(last + 1, len(messages)):
        message = messages[position]
        if not isinstance(message, ToolMessage):
            continue
        text = await router.answer(turn, calls, message.tool_call_id)
        if message.content != text:
            messages[position] = message.model_copy(update={"content": text})
            changed = True
    return request.model_copy(update={"messages": messages}) if changed else None


class SandboxEpisodeTask(vf.Task[SandboxEpisodeData]):
    NEEDS_CONTAINER = False

    def __init__(self, data: SandboxEpisodeData, router: ToolRouter, config=None) -> None:
        super().__init__(data, config)
        self.router = router

    @property
    def key(self) -> str:
        return f"{self.data.env}/{self.data.split}/{self.data.index}"

    @vf.intercept
    async def answer_through_the_gateway(self, request: Request) -> Request | None:
        return await route_request(self.router, request)

    @vf.stop
    def episode_closed(self, request: Request) -> bool:
        """Stop the agent once the gateway closed the episode (budget, deadline, cap):
        no further model turn runs after the final."""
        return self.router.final is not None


class SandboxEpisodeTasksetConfig(vf.TasksetConfig):
    pass


class SandboxEpisodeTaskset(vf.Taskset[SandboxEpisodeTask, SandboxEpisodeTasksetConfig]):
    """Tasks exist only per opened episode (SandboxEpisodeRunner builds them)."""

    def load(self):
        return iter(())
