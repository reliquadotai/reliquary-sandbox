"""A verifiers env declared in `episode_envs` runs through the bridge on the real gateway
(routes, manager, grader, signing) over FakeBoxes."""

import hashlib

import pytest

pytest.importorskip("verifiers.v1")

from episode_fixtures import IMAGE  # noqa: E402

from reliquary_sandbox_service.episodes import bridged_env_package_of  # noqa: E402
from reliquary_sandbox_service.episodes.local_gateway import (  # noqa: E402
    resolve_task,
    run_scripted_episode,
)
from reliquary_sandbox_service.episodes.registry import (  # noqa: E402
    TaskRegistry,
    TaskResolutionError,
    env_package_of,
    task_fingerprint,
)
from reliquary_sandbox_service.episodes.testing import fake_gateway  # noqa: E402
from reliquary_sandbox_verifiers.declaration import setup_output  # noqa: E402
from reliquary_sandbox_verifiers.loading import VerifiersEnv, options_sha256  # noqa: E402

TOY = "verifiers:toy-bridge-env==unpackaged"
UNP = {"allow_unpackaged": True}
GW = {"episode_allow_unpackaged_verifiers": True}
ENVS = {"toy": TOY, "fake-env": "fake_episode_env:sandbox_task"}


def test_a_bridged_identity_names_the_package_and_the_bridge_version():
    assert bridged_env_package_of("toy_bridge_env") == "toy_bridge_env==unpackaged.vfb2"
    assert bridged_env_package_of("toy_bridge_env").startswith(env_package_of("toy_bridge_env"))


def test_the_registry_serves_verifiers_entries_next_to_entry_points():
    registry = TaskRegistry(ENVS, {"toy": {"tools": ["bash"]}}, **UNP)
    assert registry.versions()["toy"] == bridged_env_package_of("toy_bridge_env")
    assert registry.versions()["fake-env"] == env_package_of("fake_episode_env")
    assert isinstance(registry.bridged("toy"), VerifiersEnv)
    assert registry.bridged("fake-env") is None
    resolved = registry.resolve("toy", "rl", 0)
    assert resolved.task.image == IMAGE and resolved.task.tools == ("bash",)
    assert resolved.env_package == registry.versions()["toy"]


def test_a_restarted_registry_resolves_the_same_task():
    first = TaskRegistry(ENVS, **UNP).resolve("toy", "rl", 0)
    again = TaskRegistry(ENVS, **UNP).resolve("toy", "rl", 0)
    assert task_fingerprint(first) == task_fingerprint(again)
    assert task_fingerprint(first) != task_fingerprint(TaskRegistry(ENVS, **UNP).resolve("toy", "rl", 1))


@pytest.mark.parametrize("split, index", [("eval", 0), ("rl", 99), ("train", 0)])
def test_unserved_tasks_are_resolution_errors(split, index):
    with pytest.raises(TaskResolutionError):
        TaskRegistry(ENVS, **UNP).resolve("toy", split, index)


@pytest.mark.parametrize("entries, options", [
    ({"toy": "verifiers:toy-bridge-env==9.9.9"}, None),
    ({"toy": "verifiers:no-such-env==1"}, None),
    ({"toy": TOY}, {"toy": {"tools": ["python"]}}),
    (ENVS, {"fake-env": {"tools": ["bash"]}}),
])
def test_an_env_that_cannot_be_served_fails_at_start(entries, options):
    with pytest.raises(TaskResolutionError):
        TaskRegistry(entries, options, **UNP)


async def test_a_reference_episode_is_graded_one_and_signed(tmp_path):
    with fake_gateway(tmp_path, envs=ENVS, **GW) as (gateway, _boxes):
        episode = await run_scripted_episode(
            gateway, env="toy", split="rl", index=0,
            calls=[("bash", {"command": "write /work/answer.txt 42"})])
        task = VerifiersEnv(TOY, allow_unpackaged=True).task("rl", 0)
    assert episode.verification.ok, episode.verification.reasons
    assert (episode.final["status"], episode.final["reward"]) == ("graded", 1.0)
    assert episode.final["grading"] == {"rewards": {"solved": 1.0}, "metrics": {}}
    record0 = episode.transcript["records"][0]["body"]
    assert record0["env_package"] == bridged_env_package_of("toy_bridge_env")
    assert record0["setup_output_sha256"] == hashlib.sha256(
        setup_output(task).encode()).hexdigest()
    assert record0["tools"] == ["bash", "edit"]


async def test_a_wrong_answer_and_an_untouched_box_grade_zero(tmp_path):
    with fake_gateway(tmp_path, envs=ENVS, **GW) as (gateway, _boxes):
        wrong = await run_scripted_episode(
            gateway, env="toy", split="rl", index=0,
            calls=[("bash", {"command": "write /work/answer.txt 41"})])
        untouched = await run_scripted_episode(gateway, env="toy", split="rl", index=0, calls=[])
    for episode in (wrong, untouched):
        assert episode.verification.ok
        assert (episode.final["status"], episode.final["reward"]) == ("graded", 0.0)


async def test_local_gateway_tasks_resolve_through_the_registry_with_options(tmp_path):
    with fake_gateway(tmp_path, envs=ENVS, **GW,
                      episode_env_options={"toy": {"tools": ["bash"]}}) as (gateway, _boxes):
        assert resolve_task(gateway, "toy", "rl", 0).tools == ("bash",)
        assert resolve_task(gateway, "fake-env", "train", 0).image == IMAGE


def test_the_capacity_report_lists_the_bridged_identity(tmp_path):
    with fake_gateway(tmp_path, envs=ENVS, **GW) as (gateway, _boxes):
        import httpx

        document = httpx.get(gateway.url + "/capacity").json()["document"]
    assert document["env_packages"]["toy"] == bridged_env_package_of("toy_bridge_env")
    # Only bridged envs have options; the digest is the one a client computes.
    assert document["env_options_sha256"] == {"toy": options_sha256(None)}


def test_the_capacity_report_publishes_the_options_each_env_is_served_with(tmp_path):
    options = {"tools": ["bash"], "defaults": {"per_call_timeout_s": 600}}
    with fake_gateway(tmp_path, envs=ENVS, **GW,
                      episode_env_options={"toy": options}) as (gateway, _boxes):
        import httpx

        document = httpx.get(gateway.url + "/capacity").json()["document"]
    assert document["env_options_sha256"]["toy"] == options_sha256(options)
    assert document["env_options_sha256"]["toy"] != options_sha256(None)


def test_the_registry_digests_each_bridged_envs_options():
    registry = TaskRegistry({"toy": TOY, "toy-bash": TOY, "fake-env": ENVS["fake-env"]},
                            {"toy-bash": {"tools": ["bash"]}}, **UNP)
    assert registry.options_digests() == {"toy": options_sha256(None),
                                          "toy-bash": options_sha256({"tools": ["bash"]})}


def test_one_taskset_may_be_served_twice_with_different_options():
    registry = TaskRegistry({"toy": TOY, "toy-bash": TOY}, {"toy-bash": {"tools": ["bash"]}}, **UNP)
    assert registry.resolve("toy", "rl", 0).task.tools == ("bash", "edit")
    assert registry.resolve("toy-bash", "rl", 0).task.tools == ("bash",)
    assert registry.bridged("toy") is not registry.bridged("toy-bash")
    assert registry.versions()["toy"] == registry.versions()["toy-bash"]


def test_unpackaged_verifiers_envs_are_refused_unless_opted_in():
    with pytest.raises(TaskResolutionError):
        TaskRegistry(ENVS)


def test_the_default_settings_refuse_an_unpackaged_env_when_building_the_manager(tmp_path):
    from episode_fixtures import episode_settings

    from reliquary_sandbox_service.episodes.factory import build_episode_manager

    class Boxes:
        def preflight(self, **_):
            return {}

    settings, _validator = episode_settings(tmp_path, episode_envs=ENVS)
    assert settings.episode_allow_unpackaged_verifiers is False
    with pytest.raises(TaskResolutionError, match="unpackaged"):
        build_episode_manager(settings, backend=None, store=None, boxes=Boxes())


def test_local_gateway_builds_one_registry_per_env(tmp_path):
    with fake_gateway(tmp_path, envs=ENVS, **GW) as (gateway, _boxes):
        resolve_task(gateway, "toy", "rl", 0)
        first = gateway.registries["toy"]
        resolve_task(gateway, "toy", "rl", 0)
        assert gateway.registries["toy"] is first
        resolve_task(gateway, "fake-env", "train", 0)
        assert set(gateway.registries) == {"toy", "fake-env"}
