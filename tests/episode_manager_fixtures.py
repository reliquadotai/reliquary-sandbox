"""An EpisodeManager over FakeBoxes with a controllable clock."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from episode_fixtures import IMAGE, claims, directory_for, episode_settings
from fake_boxes import FakeBoxes

from reliquary_sandbox.attest import (
    Ed25519TokenVerifier,
    Expected,
    Signer,
    issue_session_token,
    load_private_key,
    verify_transcript,
)
from reliquary_sandbox_service.episodes.manager import EpisodeManager
from reliquary_sandbox_service.episodes.registry import TaskRegistry
from reliquary_sandbox_service.store import Store

# What `episode_fixtures.claims` binds by default: what an honest submission claims.
EXPECTED = dict(hotkey="5Hotkey", engagement="corpus:job-1:7", env="fake-env", split="train",
                index=7, checkpoint="ck-1")


class Clock:
    def __init__(self, now=1_800_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


@dataclass
class Harness:
    manager: EpisodeManager
    boxes: FakeBoxes
    clock: Clock
    validator: Signer
    machine: Signer
    settings: object

    def token(self, session_id="s-1", **overrides):
        return issue_session_token(claims(session_id, now=int(self.clock.now), **overrides),
                                   self.validator)

    def open(self, session_id="s-1", **overrides):
        return self.manager.open(self.token(session_id, **overrides))

    def transcript(self, session_id):
        episode = self.manager.store.get(session_id)
        return {"token": episode["token"], "records": self.manager.store.records(session_id)}

    def verify(self, session_id, **expected):
        """Verify what the machine stored, exactly as a validator would."""
        return verify_transcript(
            self.transcript(session_id), directory_for(self.machine),
            Ed25519TokenVerifier({self.validator.key_id: self.validator.public_key_b64}),
            Expected(**{**EXPECTED, **expected}))


def build(tmp_path, store=None, boxes=None, clock=None, **overrides) -> Harness:
    settings, validator = episode_settings(tmp_path, **overrides)
    machine = Signer(settings.machine_key_id, load_private_key(settings.machine_key_file))
    boxes = boxes or FakeBoxes([IMAGE])
    clock = clock or Clock()
    manager = EpisodeManager(settings, boxes, store or Store(settings.state_dir), machine,
                             TaskRegistry(settings.episode_envs), clock=clock)
    return Harness(manager, boxes, clock, validator, machine, settings)


def restarted(h: Harness) -> Harness:
    """The same machine after a controller restart: same keys, state dir and boxes;
    the runtime's in-memory state (resolved box users) is gone with the process."""
    h.manager.store.db.close()
    h.boxes.users.clear()
    manager = EpisodeManager(h.settings, h.boxes, Store(h.settings.state_dir), h.machine,
                             TaskRegistry(h.settings.episode_envs), clock=h.clock)
    return dataclasses.replace(h, manager=manager)
