"""What the bridge derives from a verifiers Task's declarative data."""

import math

import pytest

pytest.importorskip("verifiers.v1")

import verifiers.v1 as vf  # noqa: E402
from toy_bridge_env import ToyData, ToyTask  # noqa: E402
from verifiers.v1.task import TaskResources, TaskTimeout  # noqa: E402
from verifiers.v1.tasksets.harbor import HarborData, HarborTask  # noqa: E402

from reliquary_sandbox_service.episodes.manager import grading_fits_the_window  # noqa: E402
from reliquary_sandbox_verifiers.declaration import (  # noqa: E402
    MAX_GRADING_TIMEOUT_S,
    MAX_PREPARE_TIMEOUT_S,
    MIN_GRADING_TIMEOUT_S,
    ArtifactRoot,
    BridgeDefaults,
    TaskDeclarationError,
    artifact_roots,
    grading_timeout_s,
    literal_runtime_env,
    prepare_timeout_s,
    setup_output,
    task_limits,
)

DEFAULTS = BridgeDefaults()


def toy(**fields):
    values = dict(idx=0, prompt="p", image="toy/task:1", workdir="/work")
    values.update(fields)
    return ToyTask(ToyData(**values))


def test_declared_artifacts_resolve_against_the_workdir_and_the_convention_dir_is_added():
    task = toy(artifacts=[vf.Artifact(source="out"), vf.Artifact(source="/etc/app.conf",
                                                                 required=False)])
    assert artifact_roots(task, "/work") == (
        ArtifactRoot("/work/out", True), ArtifactRoot("/etc/app.conf", False),
        ArtifactRoot("/logs/artifacts", False))


@pytest.mark.parametrize("source", ["/logs", "/logs/artifacts", "/logs/artifacts/patch.diff"])
def test_the_convention_dir_is_left_out_when_a_declared_root_overlaps_it(source):
    roots = artifact_roots(toy(artifacts=[vf.Artifact(source=source)]), "/work")
    assert [root.path for root in roots] == [source]


@pytest.mark.parametrize("artifacts, message", [
    ([vf.Artifact(source="out", exclude=["*.pyc"])], "exclude"),
    ([vf.Artifact(source="..")], "resolves to '/'"),
    ([vf.Artifact(source="out"), vf.Artifact(source="out/sub")], "overlap"),
    ([vf.Artifact(source="out"), vf.Artifact(source="/work/out/")], "overlap"),
])
def test_artifacts_the_sandbox_cannot_archive_are_refused(artifacts, message):
    with pytest.raises(TaskDeclarationError, match=message):
        artifact_roots(toy(artifacts=artifacts), "/work")


class EnvTask(ToyTask):
    def runtime_env(self):
        return dict(self.data.env_values)


class EnvData(ToyData):
    env_values: dict[str, str] = {}
    declared: str = "literal"


def test_runtime_env_values_must_be_literals_of_the_task():
    task = EnvTask(EnvData(idx=0, image="toy/task:1", env_values={"MODE": "literal", "EMPTY": ""}))
    assert literal_runtime_env(task) == {"MODE": "literal", "EMPTY": ""}


def test_a_runtime_env_value_resolved_on_the_gateway_is_refused():
    class Resolving(EnvTask):
        def runtime_env(self):
            return {"TOKEN": "a-secret-of-the-gateway"}

    with pytest.raises(TaskDeclarationError, match="TOKEN"):
        literal_runtime_env(Resolving(EnvData(idx=0, image="toy/task:1")))


def test_limits_come_from_resources_and_timeouts_and_the_rest_from_defaults():
    data = ToyData(idx=0, image="toy/task:1",
                   resources=TaskResources(cpu=2, memory=1.5, disk=10),
                   timeout=TaskTimeout(agent=900.5))
    limits = task_limits(data, BridgeDefaults(pids=512, per_call_timeout_s=60, max_calls=99))
    assert (limits.cpus, limits.memory_bytes, limits.disk_bytes) == (2, 3 * 1024**3 // 2,
                                                                     10 * 1024**3)
    assert (limits.pids, limits.wall_s, limits.per_call_timeout_s, limits.max_calls) == (
        512, 901, 60, 99)
    bare = task_limits(ToyData(idx=0, image="toy/task:1"), DEFAULTS)
    assert (bare.cpus, bare.memory_bytes, bare.disk_bytes, bare.wall_s) == (None,) * 4
    assert (bare.pids, bare.per_call_timeout_s) == (1024, 180)


def test_gpu_tasks_and_impossible_resources_are_refused():
    with pytest.raises(TaskDeclarationError, match="GPU"):
        task_limits(ToyData(idx=0, resources=TaskResources(gpu="A100")), DEFAULTS)
    with pytest.raises(TaskDeclarationError, match="resources"):
        task_limits(ToyData(idx=0, resources=TaskResources(cpu=-1)), DEFAULTS)


def test_grading_gets_the_larger_of_finalize_and_scoring_up_to_the_window():
    assert grading_timeout_s(ToyData(idx=0), DEFAULTS) == 600.0
    assert grading_timeout_s(ToyData(idx=0, timeout=TaskTimeout(finalize=30, scoring=700)),
                             DEFAULTS) == 700.0
    with pytest.raises(TaskDeclarationError, match="810"):
        grading_timeout_s(ToyData(idx=0, timeout=TaskTimeout(scoring=1800)), DEFAULTS)
    assert grading_fits_the_window(MAX_GRADING_TIMEOUT_S)
    assert not grading_fits_the_window(MAX_GRADING_TIMEOUT_S + 1)


def test_prepare_gets_the_setup_timeout_or_the_default():
    assert prepare_timeout_s(ToyData(idx=0), DEFAULTS) == 600.0
    assert prepare_timeout_s(ToyData(idx=0, timeout=TaskTimeout(setup=40)), DEFAULTS) == 40.0


def test_a_setup_timeout_too_short_for_a_scoring_process_is_refused():
    with pytest.raises(TaskDeclarationError, match="at least 35"):
        prepare_timeout_s(ToyData(idx=0, timeout=TaskTimeout(setup=20)), DEFAULTS)


def test_setup_output_binds_the_bridge_version_and_the_task_content():
    one, other = toy(), toy(prompt="another prompt")
    assert setup_output(one) == setup_output(toy())
    assert setup_output(one) != setup_output(other)
    assert setup_output(one) == '{"bridge":2,"task_hash":"%s"}' % one.hash


def test_an_unmodified_harbor_task_declares_through_the_same_functions(tmp_path):
    task = HarborTask(HarborData(idx=0, name="hello", prompt="Write hello.", image="toy/task:1",
                                 workdir="/app", task_dir=str(tmp_path),
                                 artifacts=[vf.Artifact(source="/app", required=False)],
                                 resources=TaskResources(cpu=1, memory=2),
                                 timeout=TaskTimeout(scoring=120)))
    assert artifact_roots(task, "/app") == (ArtifactRoot("/app", False),
                                            ArtifactRoot("/logs/artifacts", False))
    assert literal_runtime_env(task) == {}
    assert task_limits(task.data, DEFAULTS).memory_bytes == 2 * 1024**3
    assert grading_timeout_s(task.data, DEFAULTS) == 120.0


def timed(**fields):
    return ToyData.model_construct(idx=0, timeout=TaskTimeout.model_construct(**fields))


@pytest.mark.parametrize("source", ["//", "///", "/work/..//"])
def test_leading_slashes_do_not_widen_a_root_to_the_whole_filesystem(source):
    with pytest.raises(TaskDeclarationError, match="resolves to '/'"):
        artifact_roots(toy(artifacts=[vf.Artifact(source=source)]), "/work")


@pytest.mark.parametrize("first, second", [("//work/out", "out"), ("//logs", "/logs"),
                                           ("out", "//work/out/sub")])
def test_leading_slashes_do_not_escape_the_overlap_check(first, second):
    with pytest.raises(TaskDeclarationError, match="overlap"):
        artifact_roots(toy(artifacts=[vf.Artifact(source=first), vf.Artifact(source=second)]),
                       "/work")


def test_a_double_slash_root_is_normalised():
    roots = artifact_roots(toy(artifacts=[vf.Artifact(source="//work/out")]), "/work")
    assert roots[0].path == "/work/out"


@pytest.mark.parametrize("bad", [math.nan, -5.0, 0.0, math.inf])
def test_a_declared_grading_timeout_must_be_positive_and_finite(bad):
    for field in ("finalize", "scoring"):
        with pytest.raises(TaskDeclarationError, match="timeout"):
            grading_timeout_s(timed(**{field: bad}), DEFAULTS)
    with pytest.raises(TaskDeclarationError, match="timeout"):
        grading_timeout_s(timed(finalize=bad, scoring=100.0), DEFAULTS)


@pytest.mark.parametrize("bad", [math.nan, -3.0, math.inf, 1e9])
def test_a_declared_prepare_timeout_must_be_valid_and_is_capped(bad):
    with pytest.raises(TaskDeclarationError, match="timeout"):
        prepare_timeout_s(timed(setup=bad), DEFAULTS)


def test_prepare_timeout_at_the_cap_is_accepted():
    assert prepare_timeout_s(timed(setup=MAX_PREPARE_TIMEOUT_S), DEFAULTS) == 600.0


@pytest.mark.parametrize("timeout", [dict(agent=math.nan), dict(agent=math.inf),
                                     dict(agent=-1.0)])
def test_an_invalid_agent_timeout_is_a_declaration_error(timeout):
    with pytest.raises(TaskDeclarationError, match="timeout"):
        task_limits(timed(**timeout), DEFAULTS)


@pytest.mark.parametrize("field", ["cpu", "memory", "disk"])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_invalid_resources_are_a_declaration_error(field, bad):
    data = ToyData.model_construct(idx=0, resources=TaskResources.model_construct(**{field: bad}))
    with pytest.raises(TaskDeclarationError, match="resources"):
        task_limits(data, DEFAULTS)


def test_a_grading_step_shorter_than_the_scoring_floor_and_a_margin_is_refused():
    """No scoring process starts with less than SCORING_STARTUP_FLOOR_S left: a grading
    step below the floor plus 5 s would grade every episode 0 `grading_timeout`."""
    from reliquary_sandbox_verifiers.scoring_process import SCORING_STARTUP_FLOOR_S

    assert MIN_GRADING_TIMEOUT_S == SCORING_STARTUP_FLOOR_S + 5 == 35
    for timeout in (TaskTimeout(scoring=20), TaskTimeout(finalize=34.9),
                    TaskTimeout(finalize=10, scoring=34)):
        with pytest.raises(TaskDeclarationError, match="at least 35"):
            grading_timeout_s(ToyData(idx=0, timeout=timeout), DEFAULTS)
    with pytest.raises(TaskDeclarationError, match="at least 35"):
        grading_timeout_s(ToyData(idx=0), BridgeDefaults(grading_timeout_s=30))
    assert grading_timeout_s(ToyData(idx=0, timeout=TaskTimeout(scoring=35)), DEFAULTS) == 35.0
    # each grading step gets the larger of the two: a short finalize alone is fine
    assert grading_timeout_s(ToyData(idx=0, timeout=TaskTimeout(finalize=10, scoring=60)),
                             DEFAULTS) == 60.0
