"""Each bridged env's image lock, as the signed capacity report states it."""

import json
from types import SimpleNamespace

import pytest
from episode_fixtures import IMAGE
from episode_manager_fixtures import build

from reliquary_sandbox_service.episodes.capacity import capacity_report
from reliquary_sandbox_service.episodes.registry import TaskRegistry

pytest.importorskip("reliquary_sandbox_verifiers")
from reliquary_sandbox_verifiers.image_lock import ImageLock  # noqa: E402

OTHER = "registry.example/other@sha256:" + "b" * 64


def lock(images):
    return ImageLock.parse(json.dumps({"version": 1, "images": images}).encode())


def test_the_lock_digest_ignores_key_order_and_follows_the_content():
    a = lock({"base": IMAGE, "other": OTHER})
    b = ImageLock.parse(b'{"images": {"other": "' + OTHER.encode() + b'", "base": "'
                        + IMAGE.encode() + b'"}, "version": 1}')
    assert a.sha256() == b.sha256() and len(a.sha256()) == 64
    assert lock({"base": IMAGE}).sha256() != a.sha256()


# Golden vector: Catalyst recomputes this digest from its own install and must get the same hex.
GOLDEN_LOCK = (b'{"version": 1, "images": {"other": "registry.example/other@sha256:' + b"b" * 64
               + b'", "base": "registry.example/task@sha256:' + b"a" * 64 + b'"}}')
GOLDEN_SHA256 = "e5459c4580f905f85e61aa8114aec37194919bbf2484a2443357edb429330731"


def test_the_lock_digest_of_a_fixed_lock_is_a_fixed_hex():
    assert ImageLock.parse(GOLDEN_LOCK).sha256() == GOLDEN_SHA256


def test_the_registry_lists_the_locks_of_its_bridged_envs_only():
    registry = TaskRegistry({})
    registry._bridged["tmax"] = SimpleNamespace(lock=lock({"base": IMAGE}))
    assert set(registry.image_locks()) == {"tmax"}


def test_the_report_says_which_lock_and_how_many_images_are_present(tmp_path):
    h = build(tmp_path)
    # The report also states the options digest of each bridged env (`options_digests`).
    options = SimpleNamespace(sha256=lambda: "0" * 64)
    h.manager.registry._bridged["tmax"] = SimpleNamespace(
        lock=lock({"base": IMAGE, "other": OTHER}), options=options)
    h.manager.host_facts = h.boxes.preflight()
    document = capacity_report(h.manager)["document"]
    assert document["env_image_locks"] == {
        "tmax": {"sha256": lock({"base": IMAGE, "other": OTHER}).sha256(), "images": 2, "present": 1}}


def test_a_machine_with_no_bridged_env_reports_no_lock(tmp_path):
    h = build(tmp_path)
    h.manager.host_facts = h.boxes.preflight()
    assert "env_image_locks" not in capacity_report(h.manager)["document"]


def test_two_tags_on_one_digest_count_once(tmp_path):
    h = build(tmp_path)
    options = SimpleNamespace(sha256=lambda: "0" * 64)
    h.manager.registry._bridged["tmax"] = SimpleNamespace(
        lock=lock({"base": IMAGE, "alias": IMAGE, "other": OTHER}), options=options)
    h.manager.host_facts = h.boxes.preflight()
    entry = capacity_report(h.manager)["document"]["env_image_locks"]["tmax"]
    assert (entry["images"], entry["present"]) == (2, 1)
