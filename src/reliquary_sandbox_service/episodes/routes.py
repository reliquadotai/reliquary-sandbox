"""HTTP surface of the episode gateway.

No `from __future__ import annotations` here: FastAPI must resolve the
route-local `Key` dependency annotation at registration time (a string
annotation silently turns the episode key into a required query parameter).

Every manager call runs on the episode pool (`state["episode_pool"]`, sized
from `episode_capacity`): the manager blocks on Docker and env hooks and
refuses to run on the event loop. The capacity report is served from a cache
built on the separate maintenance pool, so hanging calls cannot stall it.
It lives at `GET /capacity`, outside the episode id namespace. Transcripts
are read on their own two-thread pool (`state["transcript_pool"]`).

The episode key travels only in the `Authorization: Bearer` header, never in
the URL, so access logs (which record the path and query) never contain it.

Errors come from the manager unchanged: 401 (one answer for an unknown
episode and a wrong key), 409 busy, 409 with the signed final for a closed
episode or a finish after the deadline (later finishes: 200), 413, 422,
503 + Retry-After for a full machine, 503 without Retry-After for a failed
open (the token is spent; the validator must issue a new one). Every
refusal of a fresh token that is not the miner's doing (failed open, full
machine, claims the machine cannot serve) spends the token and carries the
signed receipt in the `X-Reliquary-Open-Refusal` header (default handler: the
exception's headers pass through).
"""

import asyncio
import functools
import logging
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from reliquary_sandbox.observation import MAX_ARGUMENT_BYTES

_log = logging.getLogger(__name__)

_INT_MAX = 2**63 - 1
TOKEN_BODY_BYTES = 16 * 1024
"""A session token (claims, budgets, signature) is well under 2 KiB."""
CALL_BODY_FACTOR = 6
CALL_OVERHEAD_BYTES = 64 * 1024
"""A call body may be `CALL_BODY_FACTOR * MAX_ARGUMENT_BYTES +
CALL_OVERHEAD_BYTES`: a client that escapes every character as `\\uXXXX`
(`json.dumps` defaults) sends up to six bytes per canonical byte. The
arguments are still bounded canonically by the manager after parsing."""
EMPTY_BODY_BYTES = 1024
"""Every other episode route takes no body."""
_KEY_HEADER_BYTES = 512
RECOVERING = "episodes are recovering"
RETRY_AFTER_S = "5"


class OpenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    token: dict[str, Any]


class CallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    turn: int = Field(ge=0, le=_INT_MAX)
    k: int = Field(ge=0, le=_INT_MAX)
    tool: str = Field(min_length=1, max_length=64)
    arguments: dict[str, Any]


class EpisodeBodyLimit:
    """Per-route body limits for the episode routes, enforced before parsing
    (Content-Length, then the bytes actually received). Other paths pass
    through to the controller's global limit."""

    def __init__(self, app, call_limit: int):
        self.app = app
        self.call_limit = call_limit

    def limit(self, method: str, path: str) -> int | None:
        if path == "/capacity":
            return EMPTY_BODY_BYTES
        if path != "/episodes" and not path.startswith("/episodes/"):
            return None
        if method == "POST" and path == "/episodes":
            return TOKEN_BODY_BYTES
        if method == "POST" and path.endswith("/calls") and path.count("/") == 3:
            return self.call_limit
        return EMPTY_BODY_BYTES

    async def __call__(self, scope, receive, send):
        limit = self.limit(scope.get("method", ""), scope.get("path", "")) \
            if scope["type"] == "http" else None
        if limit is None:
            return await self.app(scope, receive, send)
        detail = "request body exceeds the episode limit"
        too_large = JSONResponse({"detail": detail}, 413)
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    if int(value) > limit:
                        return await too_large(scope, receive, send)
                except ValueError:
                    return await JSONResponse({"detail": "invalid content length"}, 400)(
                        scope, receive, send)
        total = 0
        started = False

        async def bounded_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > limit:  # FastAPI re-raises an HTTPException from the body read
                raise HTTPException(413, detail)
            return message

        async def observed_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, bounded_receive, observed_send)
        except HTTPException as exc:
            if started or exc.status_code != 413:
                raise
            await too_large(scope, receive, send)


def register_episode_routes(app: FastAPI, state: dict[str, Any], settings: Any) -> None:
    app.add_middleware(
        EpisodeBodyLimit,
        call_limit=CALL_BODY_FACTOR * MAX_ARGUMENT_BYTES + CALL_OVERHEAD_BYTES)

    def recovering() -> HTTPException:
        return HTTPException(503, RECOVERING, headers={"Retry-After": RETRY_AFTER_S})

    async def run(method: str, *args, pool_name: str = "episode_pool"):
        """Run a manager method on the episode pool (or the named one)."""
        manager, pool = state.get("episodes"), state.get(pool_name)
        if manager is None or pool is None:  # starting (recovery not finished) or stopping
            raise recovering()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(pool, functools.partial(getattr(manager, method), *args))

    def episode_key(authorization: Annotated[str | None, Header()] = None) -> str:
        if (not authorization or not authorization.startswith("Bearer ")
                or len(authorization) > _KEY_HEADER_BYTES):
            raise HTTPException(401, "valid episode key required",
                                headers={"WWW-Authenticate": "Bearer"})
        return authorization[7:]

    Key = Annotated[str, Depends(episode_key)]

    @app.get("/capacity")
    async def capacity():
        cache = state.get("capacity")
        if cache is None:
            raise recovering()
        try:
            return await cache.get()
        except Exception:
            _log.exception("the capacity report could not be built")
            raise HTTPException(503, "capacity unavailable",
                                headers={"Retry-After": RETRY_AFTER_S}) from None

    @app.post("/episodes")
    async def open_episode(request: OpenRequest):
        return await run("open", request.token)

    @app.post("/episodes/{episode_id}/calls")
    async def call(episode_id: str, request: CallRequest, key: Key):
        return await run("call", episode_id, key, request.turn, request.k, request.tool,
                         request.arguments)

    @app.post("/episodes/{episode_id}/finish")
    async def finish(episode_id: str, key: Key):
        return await run("finish", episode_id, key)

    @app.get("/episodes/{episode_id}")
    async def transcript(episode_id: str, key: Key):
        # Built as JSON bytes on the two-thread transcript pool: a large
        # transcript never occupies an episode worker or the event loop.
        body = await run("transcript_json", episode_id, key, pool_name="transcript_pool")
        return Response(content=body, media_type="application/json")
