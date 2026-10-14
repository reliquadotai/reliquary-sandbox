"""`env_package_of`: the public identity function validators use to allow-list env packages
(the value record 0 and the capacity report carry)."""

import importlib

import fake_episode_env
import pytest

from reliquary_sandbox_service.episodes import env_package_of as reexported
from reliquary_sandbox_service.episodes.registry import TaskRegistry, env_package_of


def test_it_is_the_registrys_identity():
    registry = TaskRegistry({"fake-env": "fake_episode_env:sandbox_task"})
    expected = registry.versions()["fake-env"]
    assert env_package_of("fake_episode_env:sandbox_task") == expected
    assert env_package_of("fake_episode_env") == expected
    assert env_package_of(fake_episode_env) == expected
    assert reexported is env_package_of


def test_an_installed_distribution_is_named_versioned_and_digested():
    identity = env_package_of(importlib.import_module("httpx"))
    name, _, rest = identity.partition("==")
    version, _, digest = rest.partition("+g")
    assert name == "httpx" and version and len(digest) == 16
    int(digest, 16)


@pytest.mark.parametrize("bad", [None, 3, "", ":attr"])
def test_a_value_that_names_no_module_is_refused(bad):
    with pytest.raises((TypeError, ValueError)):
        env_package_of(bad)
