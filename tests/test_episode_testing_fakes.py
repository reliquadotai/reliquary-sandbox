"""The fakes downstream repositories use to run a real gateway without Docker."""

import asyncio

from reliquary_sandbox.attest import FINAL_STATUSES, STATUS_BOX_FAILED
from reliquary_sandbox_service.episodes import testing
from reliquary_sandbox_service.episodes.local_gateway import run_scripted_episode


def test_box_failed_is_exported_with_the_other_statuses():
    assert STATUS_BOX_FAILED == "box_failed" and STATUS_BOX_FAILED in FINAL_STATUSES


def test_the_shipped_fakes_serve_a_verifiable_graded_episode(tmp_path):
    with testing.fake_gateway(tmp_path, envs={"fake-env": testing.FAKE_ENV_ENTRY}) as (gateway, boxes):
        episode = asyncio.run(run_scripted_episode(
            gateway, env="fake-env", split="train", index=7,
            calls=[("bash", {"command": "write /work/answer.txt 42"})]))
    assert episode.final["status"] == "graded" and episode.final["reward"] == 1.0
    assert episode.verification.ok and episode.state == b"42\n"   # FakeBoxes writes a line
    assert isinstance(boxes, testing.FakeBoxes)


def test_an_env_name_can_be_mapped_to_the_fake_task(tmp_path):
    with testing.fake_gateway(tmp_path, envs={"reliquary-swe": testing.FAKE_ENV_ENTRY}) as (gateway, _):
        episode = asyncio.run(run_scripted_episode(
            gateway, env="reliquary-swe", split="train:20", index=0,
            calls=[("bash", {"command": "write /work/answer.txt 41"})]))
    assert episode.final["status"] == "graded" and episode.final["reward"] == 0.0
