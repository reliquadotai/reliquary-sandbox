"""A local gateway over FakeBoxes, for client and bridge tests (no Docker)."""

import hashlib
from contextlib import contextmanager

from episode_fixtures import IMAGE
from fake_boxes import FakeBoxes

from reliquary_sandbox_service.episodes.local_gateway import local_gateway


@contextmanager
def fake_gateway(tmp_path, **overrides):
    helper = tmp_path / "busybox"
    helper.write_bytes(b"not really busybox")
    boxes = FakeBoxes([IMAGE])
    settings = dict(episode_allow_unquoted_disk=True, episode_helper_busybox=helper,
                    episode_helper_busybox_sha256=hashlib.sha256(helper.read_bytes()).hexdigest())
    settings.update(overrides)
    with local_gateway(tmp_path / "gateway", envs={"fake-env": "fake_episode_env:sandbox_task"},
                       images=[IMAGE], task_runtime=boxes, **settings) as gateway:
        yield gateway, boxes
