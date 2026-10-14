"""The local image inventory of TaskBoxRuntime: one `images.list()` per refresh, read by
the capacity report and by every episode open as a set lookup."""

import threading
from types import SimpleNamespace

import pytest
from episode_fixtures import IMAGE
from episode_manager_fixtures import build

from reliquary_sandbox_service.task_runtime import (
    IMAGE_INVENTORY_RETRY_S,
    IMAGE_INVENTORY_TTL_S,
    TaskBoxRuntime,
)

OTHER = "registry.example/other@sha256:" + "b" * 64
UNAPPROVED = "registry.example/extra@sha256:" + "c" * 64


class Docker:
    """Counts list and per-image calls; lists what `local` holds."""

    def __init__(self, local):
        self.local = list(local)
        self.lists = 0
        self.gets = 0
        self.fail = False
        self.images = SimpleNamespace(list=self.list, get=self.get)

    def list(self):
        self.lists += 1
        if self.fail:
            raise ConnectionError("docker down")
        return [SimpleNamespace(id="sha256:" + str(i) * 64, attrs={"RepoDigests": digests})
                for i, digests in enumerate(self.local)]

    def get(self, name):
        self.gets += 1
        raise AssertionError("no per-image lookup")


class Ticks:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def runtime(local, approved=(IMAGE, OTHER)):
    client = Docker(local)
    boxes = TaskBoxRuntime(SimpleNamespace(episode_images=list(approved)), client=client,
                           namespace="n" * 32, exec_transport=object())
    boxes._clock = Ticks()
    return boxes, client


def test_one_listing_serves_every_lookup_within_the_window():
    boxes, client = runtime([[IMAGE], [UNAPPROVED]])
    assert boxes.images_present() == [IMAGE]          # approved AND listed, unapproved invisible
    for _ in range(50):
        assert boxes.image_present(IMAGE) and not boxes.image_present(OTHER)
    assert (client.lists, client.gets) == (1, 0)
    boxes._clock.now += IMAGE_INVENTORY_TTL_S
    client.local.append([OTHER])
    assert boxes.images_present() == [IMAGE, OTHER]   # refreshed once the window is over
    assert client.lists == 2


def test_two_tags_or_default_registry_forms_match_the_listed_digest():
    hub = "docker.io/library/ubuntu@sha256:" + "d" * 64
    tagged = "registry.example/tagged:v1@sha256:" + "e" * 64
    boxes, _ = runtime([["ubuntu@sha256:" + "d" * 64],
                        ["registry.example/tagged@sha256:" + "e" * 64]], approved=(hub, tagged))
    assert boxes.images_present() == [hub, tagged]


def test_invalidation_lists_again_on_the_next_lookup():
    boxes, client = runtime([[IMAGE]])
    assert boxes.image_present(IMAGE)
    client.local = []
    assert boxes.image_present(IMAGE)        # still the cached inventory
    boxes.invalidate_images()
    assert not boxes.image_present(IMAGE)
    assert client.lists == 2


def test_a_failed_listing_keeps_the_last_inventory_and_says_its_age(caplog):
    boxes, client = runtime([[IMAGE]])
    assert boxes.image_present(IMAGE)
    boxes._clock.now += IMAGE_INVENTORY_TTL_S + 20
    client.fail = True
    assert boxes.image_present(IMAGE)        # last inventory kept
    assert "65 s old" in caplog.text
    assert boxes.image_inventory_age_s() == IMAGE_INVENTORY_TTL_S + 20
    assert boxes.image_present(IMAGE) and client.lists == 2   # no retry before the retry delay
    boxes._clock.now += IMAGE_INVENTORY_RETRY_S
    client.fail = False
    client.local = []
    assert not boxes.image_present(IMAGE) and client.lists == 3


def test_a_first_listing_that_fails_means_no_image_until_it_works():
    boxes, client = runtime([[IMAGE]])
    client.fail = True
    assert boxes.images_present() == [] and boxes.image_inventory_age_s() is None
    boxes._clock.now += IMAGE_INVENTORY_RETRY_S
    client.fail = False
    assert boxes.images_present() == [IMAGE]


def test_concurrent_lookups_share_one_listing():
    boxes, client = runtime([[IMAGE]])
    started, release = threading.Event(), threading.Event()
    original = client.list

    def slow():
        started.set()
        release.wait(5)
        return original()

    client.images.list = slow
    first = threading.Thread(target=boxes.image_present, args=(IMAGE,))
    first.start()
    assert started.wait(5)
    others = [threading.Thread(target=boxes.image_present, args=(IMAGE,)) for _ in range(8)]
    for thread in others:   # no inventory yet: they wait for the one listing
        thread.start()
    release.set()
    for thread in [first, *others]:
        thread.join(5)
    assert client.lists == 1 and boxes.image_present(IMAGE)


def test_episode_opens_read_the_inventory_without_listing_again(tmp_path):
    h = build(tmp_path, episode_capacity=5)
    boxes, client = runtime([[IMAGE]], approved=(IMAGE,))
    h.boxes.images_present = lambda: pytest.fail("open must not scan the approved images")
    h.boxes.image_present = boxes.image_present
    assert boxes.images_present() == [IMAGE] and client.lists == 1   # the heartbeat's listing
    for number in range(5):
        h.open(f"s-{number}")
    assert (client.lists, client.gets) == (1, 0)


def test_an_invalidation_during_a_listing_is_not_lost():
    """Review of 3b: a listing that started before a pull or removal must not hold the inventory for a
    whole TTL (generation counter)."""
    boxes, client = runtime([[IMAGE]])
    started, release = threading.Event(), threading.Event()
    original = client.list

    def slow():
        started.set()
        release.wait(5)
        return original()

    client.images.list = slow
    lister = threading.Thread(target=boxes.image_present, args=(IMAGE,))
    lister.start()
    assert started.wait(5)
    boxes.invalidate_images()            # an image is removed while the listing runs
    release.set()                        # the listing, from before the removal (IMAGE present), lands
    lister.join(5)
    client.images.list = original
    client.local = []
    assert not boxes.image_present(IMAGE) and client.lists == 2   # listed again at once


def test_a_create_that_finds_its_image_gone_invalidates_the_inventory(tmp_path):
    from docker.errors import ImageNotFound

    from reliquary_sandbox_service.task_runtime import InfraUnavailable

    boxes, client = runtime([[IMAGE]])
    boxes.settings.docker_runtime, boxes.settings.episode_allow_unquoted_disk = "runsc", False
    boxes._check_helper = lambda: None

    def gone(name):
        client.gets += 1
        raise ImageNotFound("gone")

    client.images.get = gone
    assert boxes.image_present(IMAGE) and client.lists == 1
    client.local = []
    with pytest.raises(InfraUnavailable):
        boxes.create("box-1", "s-1", SimpleNamespace(image=IMAGE))
    assert client.gets == 1
    assert not boxes.image_present(IMAGE) and client.lists == 2   # the next lookup lists again
