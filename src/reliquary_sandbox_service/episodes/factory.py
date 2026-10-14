"""Assemble the episode gateway from settings at controller start."""

from __future__ import annotations

import logging
from typing import Any

from reliquary_sandbox.attest import Signer, load_private_key

from .manager import EpisodeManager
from .registry import TaskRegistry

_log = logging.getLogger(__name__)


def build_episode_manager(settings: Any, backend: Any, store: Any, boxes: Any = None) -> EpisodeManager:
    """Blocks (Docker, subprocesses, env imports): call it from a worker thread.
    Refuses a host that fails the preflight (with the quota probe)."""
    if boxes is None:
        from ..task_runtime import TaskBoxRuntime

        boxes = TaskBoxRuntime(settings, client=backend.client, namespace=backend.namespace)
    host_facts = boxes.preflight()
    signer = Signer(settings.machine_key_id, load_private_key(settings.machine_key_file))
    registry = TaskRegistry(settings.episode_envs, settings.episode_env_options,
                            allow_unpackaged=settings.episode_allow_unpackaged_verifiers)
    manager = EpisodeManager(settings, boxes, store, signer, registry)
    manager.host_facts = host_facts
    return manager


class HostChecks:
    """The periodic episode host checks (preflight without the quota probe).
    Blocking: run `check` on the maintenance pool. The verdict gates the whole
    node's readiness; it is logged at INFO only when it changes."""

    def __init__(self, manager: EpisodeManager, ok: bool = True):
        self.manager = manager
        self.ok = ok

    def check(self) -> bool:
        try:
            facts = self.manager.boxes.preflight(probe=False)
        except Exception as exc:
            ok, detail = False, f"fail: {type(exc).__name__}: {exc}"
        else:
            ok, detail = True, "pass"
            self.manager.host_facts = facts
        level = logging.INFO if ok != self.ok else logging.DEBUG
        _log.log(level, "episode host checks %s", detail)
        self.ok = ok
        return ok
