"""Signed capacity reports: served at GET /capacity and pushed as heartbeats.

The report says what this machine can serve now (free slots, images, env
packages and the digest of each bridged env's options, tool and isolation versions) and the caps it enforces, so a validator
issues only tokens this machine will admit. It is signed under the heartbeat
domain (never the record or token domain) and carries no secret.
`reliquary_sandbox.attest.heartbeat_valid` checks it on the validator side.

The report is cached for `CAPACITY_TTL_S` and built on the maintenance pool,
never the episode pool: the unauthenticated route costs no Docker work per
request, and hanging episode calls cannot stall it. A build reads the image
inventory of `TaskBoxRuntime` (one `images.list()` call per
`IMAGE_INVENTORY_TTL_S`, shared with episode opens), never one Docker lookup
per approved image.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import Executor
from typing import Any

import httpx

from reliquary_sandbox.attest import HEARTBEAT_DOMAIN
from reliquary_sandbox.observation import MAX_ARGUMENT_BYTES

from .resources import resources_report
from .tools import TOOLS_VERSION

logger = logging.getLogger(__name__)

CAPACITY_TTL_S = 1.0
"""How long a built report is served before the next request rebuilds it."""
BACKOFF_CAP_S = 60
"""Failed pushes back off up to this long (or the interval, if longer), so a
validator that comes back hears from the machine again soon."""


def capacity_report(manager: Any) -> dict[str, Any]:
    """The signed report. Blocks (SQLite, a Docker image listing when the inventory is
    stale): call it from a worker thread."""
    settings = manager.settings
    host = getattr(manager, "host_facts", None) or {}
    capacity = settings.episode_capacity
    active = manager.active_count()
    document = {
        "machine_id": settings.machine_id,
        # Signed for the record; a validator takes addresses from its directory.
        "public_base_url": settings.public_base_url,
        "key_id": manager.signer.key_id,
        "at": int(manager.clock()),
        "capacity": capacity,
        "active": active,
        "free": max(0, capacity - active),
        "load": [round(value, 2) for value in os.getloadavg()],
        "cpu_count": os.cpu_count() or 0,
        "images": sorted(manager.boxes.images_present()),
        "env_packages": manager.registry.versions(),
        # Per bridged env: sha256 of its effective `episode_env_options`, so a validator
        # that resolves tasks with its own copy of the options can refuse a mismatch.
        "env_options_sha256": manager.registry.options_digests(),
        "tools_version": TOOLS_VERSION,
        "runsc_version": host.get("runsc_version"),
        "busybox_version": host.get("busybox_version"),
        "helper_busybox_sha256": settings.episode_helper_busybox_sha256,
        "caps": {
            "max_calls": settings.episode_max_calls,
            "max_cpu_s": settings.episode_max_cpu_s,
            "max_call_timeout_s": settings.episode_max_call_timeout_s,
            "max_wall_s": settings.episode_max_wall_s,
            "max_memory_bytes": settings.episode_max_memory_bytes,
            "max_pids": settings.episode_max_pids,
            "max_disk_bytes": settings.episode_max_disk_bytes,
            "max_argument_bytes": MAX_ARGUMENT_BYTES,
            "output_cap_bytes": settings.episode_output_cap_bytes,
            "max_token_validity_s": settings.episode_retention_days * 86400,
            "max_transcript_bytes": settings.episode_max_transcript_bytes,
        },
    }
    resources = resources_report(settings, manager.store.active())
    if resources is not None:   # absent when nothing is accounted: older reports keep their bytes
        document["resources"] = resources
    pool = getattr(manager, "grading_pool", None)
    if pool is not None:        # absent without a pool: older reports keep their bytes
        document["grading_pool"] = pool.snapshot()
    image_locks = getattr(manager.registry, "image_locks", None)
    locks = image_locks() if callable(image_locks) else {}
    if locks:   # absent on a machine serving no bridged env: older reports keep their bytes
        present = set(document["images"])
        document["env_image_locks"] = {
            env: {"sha256": lock.sha256(), "images": len(set(lock.images.values())),
                  "present": len(set(lock.images.values()) & present)}
            for env, lock in sorted(locks.items())}
    return {"document": document, "signature": manager.signer.sign(HEARTBEAT_DOMAIN, document)}


class CapacityCache:
    """The signed report, rebuilt at most once per `ttl_s` on `executor`.
    Concurrent requests for a stale report share one build."""

    def __init__(self, manager: Any, executor: Executor, ttl_s: float = CAPACITY_TTL_S,
                 clock: Callable[[], float] = time.monotonic):
        self.manager = manager
        self.executor = executor
        self.ttl_s = ttl_s
        self.clock = clock
        self._report: dict[str, Any] | None = None
        self._built_at = 0.0
        self._lock = asyncio.Lock()

    def _fresh(self) -> bool:
        return self._report is not None and self.clock() - self._built_at < self.ttl_s

    async def get(self) -> dict[str, Any]:
        if self._fresh():
            return self._report
        async with self._lock:
            if not self._fresh():
                loop = asyncio.get_running_loop()
                report = await loop.run_in_executor(self.executor, capacity_report, self.manager)
                self._report, self._built_at = report, self.clock()
            return self._report


def backoff_s(interval_s: float, failures: int) -> float:
    """Wait before the next push after `failures` consecutive failures."""
    if failures <= 0:
        return interval_s
    return max(interval_s, min(interval_s * 2 ** min(failures, 16), BACKOFF_CAP_S))


async def push_heartbeats(url: str, interval_s: float,
                          report: Callable[[], Awaitable[dict[str, Any]]],
                          stop: asyncio.Event, state: dict[str, Any],
                          transport: httpx.AsyncBaseTransport | None = None) -> None:
    """POST the signed report every `interval_s` until `stop`. `report` is a
    coroutine function (`CapacityCache.get`: built off the event loop, on the
    maintenance pool). A failure sets `state["heartbeat_error"]`, is logged, and the
    next push backs off; a success clears it."""
    failures = 0
    async with httpx.AsyncClient(timeout=10, transport=transport) as client:
        while not stop.is_set():
            try:
                payload = await report()
                response = await client.post(url, json=payload)
                response.raise_for_status()
            except Exception as exc:  # the validator being down must not stop the machine
                failures += 1
                state["heartbeat_error"] = True
                logger.warning("heartbeat to %s failed (%d in a row): %s", url, failures,
                               type(exc).__name__)
            else:
                failures = 0
                state.pop("heartbeat_error", None)
            try:
                await asyncio.wait_for(stop.wait(), timeout=backoff_s(interval_s, failures))
            except TimeoutError:
                pass
