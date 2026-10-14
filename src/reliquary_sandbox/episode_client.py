"""The miner's client for a signed-episode gateway (reliquary-sandbox's /episodes routes).

Wire rules the gateway relies on:
* bodies are compact UTF-8 JSON (`ensure_ascii=False`). A `\\uXXXX`-escaped body is up
  to six times larger and can hit the call body limit while the canonical arguments fit;
* a call is identified by `(turn, k)`. Resending one is always safe: the gateway returns
  the stored record without running it again. Network errors, timeouts, `503 episodes
  are recovering` and `409 episode is busy` (this call or another still runs; the gateway
  answers busy instead of queueing) are retried with the same `(turn, k)`: at least
  MIN_RESENDS times, then within `retry_s` (`busy_wait_s` for busy) of the first
  failure, never past the caller's `deadline`;
* an open is idempotent per token: resending the same token returns the same record 0
  and key while the episode runs, `409 episode is busy...` while it still opens (waited
  out), and `409` with the signed final once it is closed (`EpisodeClosed`). A lost open
  response is therefore resent like a call. A 503 without Retry-After means the open
  failed and the token is spent (a resend then answers `409` with no final): ask the
  validator for a new one. A full machine (`503` with Retry-After) is not retried: the
  token is spent too. Every refusal of a fresh token that is not the miner's doing (a
  failed preparation, a full machine, claims the machine cannot serve) carries the
  machine's signed receipt (`EpisodeError.open_refusal`, from the `X-Reliquary-Open-Refusal`
  header; a resend of the same token answers the same receipt): forward it when reporting
  the failed open;
* send only what `reliquary_sandbox.observation.plan_call` says to send: arguments with
  no canonical form are refused with nothing recorded;
* `finish` is idempotent. A finish after the deadline answers 409 with the signed
  `expired` final, which `finish()` returns like any other final;
* a call may return a final (`CallResult.final`): the CPU budget ran out, the deadline
  passed, or the stored transcript passed the machine's cap (`budget_exhausted`, reason
  `transcript_bytes`; the call itself is recorded). Later calls raise `EpisodeClosed`;
* an episode whose machine key rotated is void and unsigned (`aborted_unsigned`): calls
  and `finish()` raise `EpisodeClosed` with `final` None.
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

BUSY = "episode is busy"
RECOVERING = "episodes are recovering"
READING = "the transcript is already being read"
MIN_RESENDS = 3
"""Resends always allowed after a failure, however long the failed attempts took (a hung
connection fails only at its read timeout, which may exceed `retry_s`), unless the
caller's deadline has passed."""
RETRYABLE = (httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError)
"""Transport failures worth a resend. Other transport errors (unsupported protocol,
proxy or local protocol errors) are configuration faults and raise at once."""
OPEN_TIMEOUT_S = 900.0
"""Box creation plus the env's `prepare` (up to its `prepare_timeout_s`)."""
FINISH_TIMEOUT_S = 3900.0
"""Extract plus grading, which may run up to GRADING_GRACE_S past the token's expiry."""
CALL_MARGIN_S = 60.0
TRANSCRIPT_TIMEOUT_S = 60.0


def encode_body(body: Any) -> bytes:
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


class EpisodeError(Exception):
    """The gateway refused a request. `open_refusal` is the gateway's signed receipt when it refused
    (or failed) a fresh token for a reason that is not the miner's (open only; None otherwise)."""

    def __init__(self, status: int, detail: Any, open_refusal: dict[str, Any] | None = None):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail
        self.open_refusal = open_refusal


class EpisodeClosed(EpisodeError):
    """The episode is over; `final` is its signed final record (None if it never opened)."""

    @property
    def final(self) -> dict[str, Any] | None:
        return self.detail.get("final") if isinstance(self.detail, dict) else None


OPEN_REFUSAL_HEADER = "X-Reliquary-Open-Refusal"
"""Equal to the gateway's `episodes.manager.OPEN_REFUSAL_HEADER` (the client cannot import the service)."""


def decode_open_refusal(value: Any) -> dict[str, Any] | None:
    """The gateway's signed open refusal from its response header, or None (absent or malformed). Not
    verified here: the validator verifies it (`reliquary_sandbox.attest.open_refusal_valid`)."""
    if not isinstance(value, str) or not value:
        return None
    try:
        receipt = json.loads(base64.b64decode(value, validate=True))
    except (ValueError, TypeError):
        return None
    return receipt if isinstance(receipt, dict) and set(receipt) == {"document", "signature"} else None


@dataclass(frozen=True)
class CallResult:
    record: dict[str, Any]
    final: dict[str, Any] | None


@dataclass(frozen=True)
class Finished:
    record: dict[str, Any]
    state: bytes | None


def _detail(response: httpx.Response) -> Any:
    try:
        return response.json().get("detail")
    except (ValueError, AttributeError):
        return response.text


def _closed_or_error(response: httpx.Response) -> EpisodeError:
    detail = _detail(response)
    if isinstance(detail, dict) and "final" in detail:
        return EpisodeClosed(response.status_code, detail)
    return EpisodeError(response.status_code, detail)


class EpisodeClient:
    def __init__(self, base_url: str, *, http: httpx.AsyncClient | None = None,
                 retry_s: float = 300.0, busy_wait_s: float = 900.0,
                 sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
                 clock: Callable[[], float] = time.monotonic):
        """`retry_s` bounds resends after a transport error or `recovering`, and
        `busy_wait_s` after `busy`, both counted from the end of the first failed attempt
        (see MIN_RESENDS). Deadlines passed to the methods are on `clock`."""
        self.base_url = base_url.rstrip("/")
        self._http = http or httpx.AsyncClient()
        self._owned = http is None
        self.retry_s = retry_s
        self.busy_wait_s = busy_wait_s
        self._sleep = sleep
        self._clock = clock

    async def __aenter__(self) -> EpisodeClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owned:
            await self._http.aclose()

    async def _request(self, method: str, path: str, *, key: str | None = None,
                       body: Any = None, timeout: float) -> httpx.Response:
        headers: dict[str, str] = {}
        if key is not None:
            headers["Authorization"] = f"Bearer {key}"
        content = None
        if body is not None:
            content = encode_body(body)
            headers["Content-Type"] = "application/json"
        return await self._http.request(method, self.base_url + path, content=content,
                                        headers=headers, timeout=timeout)

    async def _backoff(self, attempt: int) -> None:
        await self._sleep(min(5.0, 0.25 * 2 ** min(attempt, 5)) * (0.5 + random.random() / 2))

    async def _retrying(self, send: Callable[[], Awaitable[httpx.Response]], *,
                        waits: tuple[str, ...], deadline: float | None) -> httpx.Response:
        """Resend until an answer that is not a retryable transport error, `recovering`
        or a 409 whose detail starts with one of `waits`. The budget (`retry_s`; for a
        409, `busy_wait_s`) starts at the end of the first failed attempt, and at least
        MIN_RESENDS resends are made whatever it says; nothing is resent once
        `deadline` (on the client's clock) has passed. The last failure is then raised
        (transport) or returned (answer)."""
        first_failure: float | None = None
        resends = 0
        while True:
            error: Exception | None = None
            budget = self.retry_s
            try:
                response = await send()
            except RETRYABLE as exc:
                error = exc
            else:
                detail = _detail(response) if response.status_code != 200 else None
                waiting = (response.status_code == 409 and isinstance(detail, str)
                           and detail.startswith(waits))
                recovering = response.status_code == 503 and detail == RECOVERING
                if not (waiting or recovering):
                    return response
                if waiting:
                    budget = self.busy_wait_s
            now = self._clock()
            if first_failure is None:
                first_failure = now
            out_of_time = deadline is not None and now >= deadline
            if out_of_time or (resends >= MIN_RESENDS and now - first_failure > budget):
                if error is not None:
                    raise error
                return response
            resends += 1
            await self._backoff(resends)

    async def open(self, token: dict[str, Any], *, deadline: float | None = None) -> Episode:
        """Open the token's episode. Resending the token is safe (the gateway answers
        the same episode while it runs), so lost responses and opens still in progress
        are retried."""
        response = await self._retrying(
            lambda: self._request("POST", "/episodes", body={"token": token},
                                  timeout=OPEN_TIMEOUT_S),
            waits=(BUSY,), deadline=deadline)
        if response.status_code != 200:
            error = _closed_or_error(response)
            error.open_refusal = decode_open_refusal(response.headers.get(OPEN_REFUSAL_HEADER))
            raise error
        data = response.json()
        return Episode(self, data["episode_id"], data["episode_key"], data["record"])


class Episode:
    def __init__(self, client: EpisodeClient, episode_id: str, key: str,
                 record0: dict[str, Any]):
        self.client = client
        self.id = episode_id
        self.key = key
        self.record0 = record0

    @property
    def budgets(self) -> dict[str, int]:
        return self.record0["body"]["budgets"]

    async def call(self, turn: int, k: int, tool: str, arguments: dict[str, Any], *,
                   deadline: float | None = None) -> CallResult:
        body = {"turn": turn, "k": k, "tool": tool, "arguments": arguments}
        timeout = self.budgets["per_call_timeout_s"] + CALL_MARGIN_S
        response = await self.client._retrying(
            lambda: self.client._request("POST", f"/episodes/{self.id}/calls", key=self.key,
                                         body=body, timeout=timeout),
            waits=(BUSY,), deadline=deadline)
        if response.status_code == 200:
            data = response.json()
            return CallResult(data["record"], data.get("final"))
        raise _closed_or_error(response)

    async def finish(self, *, deadline: float | None = None) -> Finished:
        response = await self.client._retrying(
            lambda: self.client._request("POST", f"/episodes/{self.id}/finish", key=self.key,
                                         timeout=FINISH_TIMEOUT_S),
            waits=(BUSY,), deadline=deadline)
        if response.status_code == 200:
            data = response.json()
            state = data.get("state")
            return Finished(data["record"], base64.b64decode(state) if state is not None else None)
        error = _closed_or_error(response)
        if isinstance(error, EpisodeClosed) and error.final is not None:
            return Finished(error.final, None)
        raise error

    async def transcript(self, *, deadline: float | None = None) -> dict[str, Any]:
        """The transcript. A concurrent read of the same episode (409) is waited out."""
        response = await self.client._retrying(
            lambda: self.client._request("GET", f"/episodes/{self.id}", key=self.key,
                                         timeout=TRANSCRIPT_TIMEOUT_S),
            waits=(READING,), deadline=deadline)
        if response.status_code != 200:
            raise EpisodeError(response.status_code, _detail(response))
        return response.json()
