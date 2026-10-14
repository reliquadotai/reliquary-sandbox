"""The image lock of a verifiers env package: tag -> pinned digest, or refused."""

import json
import types

import pytest

from reliquary_sandbox_verifiers.image_lock import LOCK_FILE, ImageLock, ImageLockError

DIGEST = "registry.example/task@sha256:" + "a" * 64
OTHER = "registry.example/other@sha256:" + "b" * 64


def lock_bytes(images, version=1):
    return json.dumps({"version": version, "images": images}).encode()


def test_a_locked_tag_resolves_to_its_digest_and_a_locked_digest_to_itself():
    lock = ImageLock.parse(lock_bytes({"toy/task:1": DIGEST, "toy/other:2": OTHER}))
    assert lock.digest("toy/task:1") == DIGEST
    assert lock.digest(OTHER) == OTHER
    assert lock.digests() == sorted([DIGEST, OTHER])


@pytest.mark.parametrize("image", ["toy/task:2", "toy/task", "registry.example/task@sha256:" + "c" * 64])
def test_an_image_the_lock_does_not_pin_is_refused(image):
    lock = ImageLock.parse(lock_bytes({"toy/task:1": DIGEST}))
    with pytest.raises(ImageLockError, match="not pinned"):
        lock.digest(image)


@pytest.mark.parametrize("data", [
    b"not json",
    b"[]",
    json.dumps({"images": {"a:1": DIGEST}}).encode(),
    lock_bytes({"a:1": DIGEST}, version=2),
    lock_bytes({}),
    lock_bytes({"a:1": "registry.example/task:latest"}),
    lock_bytes({"a:1": "registry.example/task@sha256:" + "A" * 64}),
    lock_bytes({"a 1": DIGEST}),
    json.dumps({"version": 1, "images": {"a:1": DIGEST}, "extra": 1}).encode(),
])
def test_a_malformed_lock_is_refused(data):
    with pytest.raises(ImageLockError):
        ImageLock.parse(data)


def test_a_duplicate_key_is_refused_at_every_level():
    tag = ('{"version": 1, "images": {"a:1": "%s", "a:1": "%s"}}' % (DIGEST, OTHER)).encode()
    top = ('{"version": 1, "version": 1, "images": {"a:1": "%s"}}' % DIGEST).encode()
    for data in (tag, top):
        with pytest.raises(ImageLockError, match="duplicate"):
            ImageLock.parse(data)


def test_a_digest_key_must_map_to_itself():
    a = "r@sha256:" + "a" * 64
    b = "r@sha256:" + "b" * 64
    with pytest.raises(ImageLockError, match="itself"):
        ImageLock.parse(lock_bytes({a: b}))
    assert ImageLock.parse(lock_bytes({a: a})).digest(a) == a


def test_the_lock_ships_in_the_tasksets_top_level_package(tmp_path, monkeypatch):
    package = tmp_path / "locked_env"
    (package / "tasks").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "tasks" / "__init__.py").write_text("")
    (package / LOCK_FILE).write_bytes(lock_bytes({"toy/task:1": DIGEST}))
    monkeypatch.syspath_prepend(str(tmp_path))
    import locked_env.tasks

    assert ImageLock.of_module(locked_env.tasks).digest("toy/task:1") == DIGEST


def test_a_package_without_a_lock_is_refused():
    with pytest.raises(ImageLockError, match=LOCK_FILE):
        ImageLock.of_module(json)
    with pytest.raises(ImageLockError, match=LOCK_FILE):
        ImageLock.of_module(types.ModuleType("no_such_package_here"))
