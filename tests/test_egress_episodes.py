"""Tasks with controlled egress: refused where the profile is not offered, online boxes
join only the egress network with the profile's proxy env, offline phases stay offline."""

import pytest
from episode_manager_fixtures import build
from fastapi import HTTPException

from reliquary_sandbox.episode_task import TaskNetwork
from reliquary_sandbox_service.episodes.registry import task_fingerprint
from reliquary_sandbox_service.task_runtime import BoxSpec, InfraUnavailable, TaskBoxRuntime

ONLINE = {"fake-env": "online_episode_env:sandbox_task"}
PROFILES = {"packages": {"proxy": "http://172.31.250.1:3128",
                         "env": {"UV_EXCLUDE_NEWER": "2026-11-01T00:00:00Z"}}}


def egress(tmp_path, **overrides):
    values = dict(episode_envs=ONLINE, episode_egress_network="rq-egress",
                  episode_egress_profiles=PROFILES, episode_egress_envs={"fake-env": ["packages"]})
    values.update(overrides)
    return build(tmp_path, **values)


def test_task_network_validation():
    assert TaskNetwork("packages").online("grading") and not TaskNetwork("packages").online("agent")
    with pytest.raises(ValueError):
        TaskNetwork("Bad Name")
    with pytest.raises(ValueError):
        TaskNetwork("packages", agent=False, grading=False)


def test_refused_where_the_profile_is_not_offered(tmp_path):
    h = build(tmp_path, episode_envs=ONLINE)
    with pytest.raises(HTTPException) as error:
        h.open(index=1)
    assert error.value.status_code == 422 and "egress profile" in str(error.value.detail)


def test_agent_box_offline_unless_the_task_says_agent(tmp_path):
    h = egress(tmp_path)
    h.open("s-1", index=1)
    h.open("s-2", index=2)
    offline, online = h.boxes.specs.values()
    assert offline.network is None and "HTTPS_PROXY" not in offline.env
    assert online.network == "rq-egress"
    assert online.env["HTTPS_PROXY"] == "http://172.31.250.1:3128"
    assert online.env["UV_EXCLUDE_NEWER"] == "2026-11-01T00:00:00Z" and online.env["TASK"] == "1"


def test_grading_box_online(tmp_path):
    h = egress(tmp_path)
    task = h.manager.registry.resolve("fake-env", "train", 1).task
    from episode_fixtures import budgets
    spec = h.manager._spec(task, budgets(), 60, env="fake-env", phase="grading")
    assert spec.network == "rq-egress" and spec.env["http_proxy"] == "http://172.31.250.1:3128"


def test_network_enters_the_fingerprint_only_when_set(tmp_path):
    h = egress(tmp_path)
    resolved = h.manager.registry.resolve("fake-env", "train", 1)
    from dataclasses import replace
    offline = replace(resolved, task=replace(resolved.task, network=None))
    assert task_fingerprint(resolved) != task_fingerprint(offline)


@pytest.mark.parametrize("bad", [
    {"episode_egress_profiles": {"packages": {"proxy": "https://x:1"}}},
    {"episode_egress_profiles": {"packages": {"proxy": "http://x"}}},
    {"episode_egress_envs": {"fake-env": ["other"]}},
    {"episode_egress_network": None},
])
def test_settings_refuse_bad_egress(tmp_path, bad):
    with pytest.raises(Exception):
        egress(tmp_path, **bad)


class _Net:
    def __init__(self, attrs):
        self.attrs = attrs


class _Networks:
    def __init__(self, attrs):
        self._attrs = attrs

    def get(self, name):
        if self._attrs is None:
            raise KeyError(name)
        return _Net(self._attrs)


class _Client:
    def __init__(self, attrs):
        self.networks = _Networks(attrs)


GOOD = {"Driver": "bridge", "Internal": True, "EnableIPv6": False,
        "Options": {"com.docker.network.bridge.enable_icc": "false"}}


@pytest.mark.parametrize("attrs,ok", [
    (GOOD, True), (None, False), ({**GOOD, "Internal": False}, False),
    ({**GOOD, "Options": {}}, False), ({**GOOD, "EnableIPv6": True}, False),
])
def test_egress_network_host_check(attrs, ok):
    class S:
        episode_egress_network = "rq-egress"
    runtime = TaskBoxRuntime.__new__(TaskBoxRuntime)
    runtime.settings, runtime.client = S(), _Client(attrs)
    if ok:
        runtime.check_egress_network()
        spec = BoxSpec("i@sha256:" + "0" * 64, "/", {}, None, 1, 1, 1, 1, 1, network="rq-egress")
        assert runtime._network_mode(spec) == "rq-egress"
        with pytest.raises(ValueError):
            runtime._network_mode(BoxSpec("i", "/", {}, None, 1, 1, 1, 1, 1, network="bridge"))
    else:
        with pytest.raises(InfraUnavailable):
            runtime.check_egress_network()


def test_verifiers_options_network():
    from reliquary_sandbox_verifiers.loading import VerifiersEnvOptions, options_sha256
    plain = options_sha256({"tools": ["bash"]})
    assert options_sha256({"tools": ["bash"], "network": None}) == plain
    online = VerifiersEnvOptions.from_mapping({"tools": ["bash"], "network": {"profile": "packages"}})
    assert online.network == TaskNetwork("packages") and online.sha256() != plain
    assert online.effective()["network"] == {"profile": "packages", "agent": False, "grading": True}
    for bad in ({"profile": "packages", "x": 1}, {"profile": "UP"}, "packages"):
        with pytest.raises(ValueError):
            VerifiersEnvOptions.from_mapping({"network": bad})
