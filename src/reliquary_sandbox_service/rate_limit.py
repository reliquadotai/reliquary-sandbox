"""Per-credential and per-address request rate limits for the HTTP API (ASGI middleware).

A token bucket per credential (the `Authorization` header: an episode key or a tenant token)
and per client address. Behind Caddy on loopback the address is the last `X-Forwarded-For`
entry (Caddy's, the client's own entries are not trusted); otherwise the peer. Over a bucket:
429 with Retry-After, before the app (and its thread pools) sees the request. Bucket tables
are bounded (least recently used dropped first).
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict

from starlette.responses import JSONResponse

MAX_KEYS = 50_000
LOOPBACK = {"127.0.0.1", "::1"}


class _Buckets:
    def __init__(self, rate: float, burst: float, clock):
        self.rate, self.burst, self.clock = rate, burst, clock
        self._table: OrderedDict[str, tuple[float, float]] = OrderedDict()

    def take(self, key: str) -> float:
        """0 when a request may pass, else the seconds until it could."""
        now = self.clock()
        tokens, at = self._table.pop(key, (self.burst, now))
        tokens = min(self.burst, tokens + (now - at) * self.rate)
        wait = 0.0
        if tokens >= 1:
            tokens -= 1
        else:
            wait = (1 - tokens) / self.rate
        self._table[key] = (tokens, now)
        while len(self._table) > MAX_KEYS:
            self._table.popitem(last=False)
        return wait


class RateLimit:
    def __init__(self, app, *, rate_per_key: float = 0, burst_per_key: float = 40,
                 rate_per_ip: float = 0, burst_per_ip: float = 200, clock=time.monotonic):
        self.app = app
        self.keys = _Buckets(rate_per_key, burst_per_key, clock) if rate_per_key > 0 else None
        self.ips = _Buckets(rate_per_ip, burst_per_ip, clock) if rate_per_ip > 0 else None

    @staticmethod
    def _address(scope) -> str:
        peer = (scope.get("client") or ("", 0))[0]
        if peer in LOOPBACK:
            for name, value in scope.get("headers", []):
                if name == b"x-forwarded-for":
                    return value.decode("latin-1").split(",")[-1].strip() or peer
        return peer

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or (self.keys is None and self.ips is None):
            return await self.app(scope, receive, send)
        wait = 0.0
        if self.ips is not None:
            wait = self.ips.take(self._address(scope))
        if not wait and self.keys is not None:
            for name, value in scope.get("headers", []):
                if name == b"authorization":
                    wait = self.keys.take(hashlib.sha256(value).hexdigest())
                    break
        if wait:
            response = JSONResponse({"detail": "too many requests"}, 429,
                                    headers={"Retry-After": str(max(1, math.ceil(wait)))})
            return await response(scope, receive, send)
        return await self.app(scope, receive, send)
