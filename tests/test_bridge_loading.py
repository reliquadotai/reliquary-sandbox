"""Loading a verifiers env declared as `verifiers:<taskset-id>==<version>`."""

import pytest

pytest.importorskip("verifiers.v1")

from episode_fixtures import IMAGE  # noqa: E402
from toy_bridge_env import ToyTask  # noqa: E402

from reliquary_sandbox.episode_task import SandboxTask  # noqa: E402
from reliquary_sandbox_verifiers.declaration import BridgeDefaults  # noqa: E402
from reliquary_sandbox_verifiers.loading import (  # noqa: E402
    VerifiersEnv,
    VerifiersEnvOptions,
    VerifiersEnvRef,
)

TOY = "verifiers:toy-bridge-env==unpackaged"


def _env(options=None):
    return VerifiersEnv(TOY, options, allow_unpackaged=True)


def test_entries_name_a_taskset_id_and_its_exact_version():
    assert VerifiersEnvRef.parse("verifiers:reliquary-swe==0.3.0") == VerifiersEnvRef(
        "reliquary-swe", "0.3.0")


def test_the_entry_grammar_is_defined_once():
    import reliquary_sandbox.episode_task as episode_task
    import reliquary_sandbox_service.episodes.registry as registry
    import reliquary_sandbox_service.settings as settings
    import reliquary_sandbox_verifiers.loading as loading

    assert settings.VERIFIERS_ENTRY is loading.VERIFIERS_ENTRY is episode_task.VERIFIERS_ENTRY
    assert (settings.VERIFIERS_ENTRY_PREFIX is registry.VERIFIERS_ENTRY_PREFIX
            is loading.VERIFIERS_ENTRY_PREFIX is episode_task.VERIFIERS_ENTRY_PREFIX)
    assert not hasattr(loading, "ENTRY") and not hasattr(loading, "installed_version")


@pytest.mark.parametrize("entry", ["verifiers:reliquary-swe", "verifiers:reliquary-swe>=0.3",
                                   "verifiers:org/env==1", "verifiers:==1", "reliquary-swe==1",
                                   "verifiers:env==1 ; rm -rf /"])
def test_malformed_entries_are_refused(entry):
    with pytest.raises(ValueError):
        VerifiersEnvRef.parse(entry)


def test_default_options_serve_the_rl_split_with_bash_and_edit():
    options = VerifiersEnvOptions.from_mapping(None)
    assert options.tools == ("bash", "edit") and options.publish_state is False
    assert dict(options.splits) == {"rl": {"split": "rl"}}
    assert options.defaults == BridgeDefaults()


def test_options_are_read_from_plain_settings_values():
    options = VerifiersEnvOptions.from_mapping({
        "tools": ["bash"], "splits": {"train": {"split": "rl"}},
        "defaults": {"per_call_timeout_s": 60, "pids": 512}, "publish_state": True})
    assert options.tools == ("bash",) and options.publish_state is True
    assert dict(options.splits["train"]) == {"split": "rl"}
    assert options.defaults == BridgeDefaults(per_call_timeout_s=60, pids=512)


@pytest.mark.parametrize("raw", [
    {"tools": []}, {"tools": ["bash", "bash"]}, {"tools": ["python"]},
    {"splits": {}}, {"splits": {"eval": {}}}, {"splits": {"held": {"split": "eval"}}},
    {"splits": {"rl": {"id": "another-env"}}}, {"splits": {"bad name": {}}},
    {"splits": {"Eval": {"split": "rl"}}}, {"splits": {"EVAL": {"split": "rl"}}},
    {"splits": {"eval ": {"split": "rl"}}}, {"splits": {"rl+eval": {"split": "rl"}}},
    {"splits": {"rl": {"split": "Eval"}}}, {"splits": {"rl": {"split": "EVAL"}}},
    {"splits": {"rl": {"split": "eval "}}}, {"splits": {"rl": {"split": "rl+eval"}}},
    {"splits": {"rl": {"split": "eval[:10]"}}}, {"splits": {"rl": {"split": " rl"}}},
    {"splits": {"rl": {"split": "RL"}}}, {"splits": {"rl": {"split": 3}}},
    {"splits": {"rl": {"split": None}}}, {"splits": {"rl": {"split": ["rl"]}}},
    {"splits": {"rl": {"split": "rl", "shuffle": True}}}, {"splits": {"rl": {"seed": 1}}},
    {"splits": {"rl": {}}},
    {"splits": {"rl": "rl"}}, {"defaults": {"no_such_default": 1}}, {"defaults": 3},
    {"publish_state": "yes"}, {"unknown": 1},
])
def test_invalid_options_are_refused(raw):
    with pytest.raises(ValueError):
        VerifiersEnvOptions.from_mapping(raw)


def test_a_declared_env_serves_its_tasks_by_split_and_index():
    env = _env()
    assert env.module.__name__ == "toy_bridge_env"
    task = env.task("rl", 0)
    assert isinstance(task, ToyTask) and task.data.idx == 0
    assert env.task("rl", 1).data.finalize_copies is False
    assert env.taskset("rl") is env.taskset("rl")
    sandbox = env.sandbox_task("rl", 0)
    assert isinstance(sandbox, SandboxTask) and sandbox.image == IMAGE


@pytest.mark.parametrize("split, index, error", [
    ("rl", 2, IndexError), ("rl", -1, IndexError), ("rl", True, IndexError),
    ("eval", 0, LookupError), ("train", 0, LookupError),
])
def test_tasks_outside_the_served_splits_or_range_are_refused(split, index, error):
    with pytest.raises(error):
        _env().task(split, index)


def test_split_names_map_to_taskset_config_overrides():
    env = _env({"splits": {"train": {"split": "rl"}}})
    assert env.task("train", 0).data.idx == 0
    with pytest.raises(LookupError):
        env.task("rl", 0)


def test_a_version_other_than_the_installed_one_is_refused():
    with pytest.raises(ValueError, match="installed at unpackaged"):
        VerifiersEnv("verifiers:toy-bridge-env==1.0.0", allow_unpackaged=True)


def test_a_taskset_with_task_at_is_indexed_directly():
    env = _env()
    taskset = env.taskset("rl")
    calls = []

    def task_at(index):
        calls.append(index)
        return next(iter(taskset.load()))

    taskset.task_at = task_at
    assert env.task("rl", 5).data.idx == 0 and calls == [5]


def test_concurrent_first_resolves_build_one_taskset():
    import threading

    env = _env()
    seen, barrier = [], threading.Barrier(8)

    def resolve():
        barrier.wait()
        seen.append(env.taskset("rl"))

    threads = [threading.Thread(target=resolve) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(seen) == 8 and all(taskset is seen[0] for taskset in seen)


def test_the_default_refuses_an_unpackaged_module():
    with pytest.raises(ValueError, match="unpackaged"):
        VerifiersEnv(TOY)


@pytest.mark.parametrize("entry", ["verifiers:os.path==1", "verifiers:toy.bridge.env==1",
                                   "verifiers:Toy==1", "verifiers:1toy==1", "verifiers:_toy==1"])
def test_dotted_or_irregular_ids_are_refused_before_any_import(entry, monkeypatch):
    import reliquary_sandbox_verifiers.loading as loading

    def no_import(*args):
        raise AssertionError("imported")

    monkeypatch.setattr(loading, "import_taskset", no_import)
    with pytest.raises(ValueError):
        VerifiersEnv(entry, allow_unpackaged=True)


def test_indexes_are_bounded_by_the_taskset_length_when_it_has_one():
    env = _env()
    taskset = env.taskset("rl")
    taskset.task_at = lambda index: next(iter(taskset.load()))
    type(taskset).__len__ = lambda self: 2
    try:
        assert env.task("rl", 1).data.idx == 0  # task_at is stubbed
        with pytest.raises(IndexError):
            env.task("rl", 2)
    finally:
        del type(taskset).__len__


def test_without_a_length_indexes_above_the_constant_are_refused():
    from reliquary_sandbox_verifiers.loading import MAX_TASK_INDEX

    env = _env()
    taskset = env.taskset("rl")
    taskset.task_at = lambda index: next(iter(taskset.load()))
    assert env.task("rl", MAX_TASK_INDEX).data.idx == 0
    with pytest.raises(IndexError):
        env.task("rl", MAX_TASK_INDEX + 1)


@pytest.mark.parametrize("error", [KeyError, ValueError, StopIteration])
def test_task_at_lookup_errors_become_index_errors(error):
    env = _env()

    def task_at(index):
        raise error(index)

    env.taskset("rl").task_at = task_at
    with pytest.raises(IndexError):
        env.task("rl", 0)


def test_a_taskset_without_a_system_prompt_attribute_is_served():
    from types import SimpleNamespace

    env = _env()
    real = env.taskset("rl")
    env._tasksets["rl"] = SimpleNamespace(task_at=lambda index: next(iter(real.load())))
    assert env.task("rl", 0).data.idx == 0


@pytest.mark.parametrize("split", [["rl"], {"a": 1}, None, 3])
def test_unhashable_or_odd_splits_are_a_lookup_error(split):
    with pytest.raises(LookupError):
        _env().task(split, 0)


def _boom_package(tmp_path, monkeypatch, *, version=None):
    """A taskset module that records and raises when imported; optionally with an installed
    distribution (`boom-dist`) at `version`."""
    (tmp_path / "boom_pkg.py").write_text(
        "import os\nopen(os.environ['BOOM_LOG'], 'a').write('imported')\nraise RuntimeError('imported')\n")
    monkeypatch.setenv("BOOM_LOG", str(tmp_path / "boom.log"))
    if version is not None:
        info = tmp_path / f"boom_dist-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: boom-dist\nVersion: {version}\n")
        (info / "top_level.txt").write_text("boom_pkg\n")
        (info / "RECORD").write_text("boom_pkg.py,,\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path / "boom.log"


@pytest.mark.parametrize("version, entry, allow", [
    (None, "verifiers:boom-pkg==unpackaged", False),
    ("2.0", "verifiers:boom-pkg==1.0", False),
    (None, "verifiers:boom-pkg==1.0", True),
])
def test_a_refused_entry_never_imports_the_module(tmp_path, monkeypatch, version, entry, allow):
    import sys

    log = _boom_package(tmp_path, monkeypatch, version=version)
    assert "boom_pkg" not in sys.modules
    with pytest.raises(ValueError):
        VerifiersEnv(entry, allow_unpackaged=allow)
    assert "boom_pkg" not in sys.modules and not log.exists()


def test_locate_taskset_reads_metadata_without_importing(tmp_path, monkeypatch):
    import sys

    from reliquary_sandbox_verifiers.loading import locate_taskset

    log = _boom_package(tmp_path, monkeypatch, version="2.0")
    assert locate_taskset("boom-pkg") == ("boom_pkg", "2.0")
    assert locate_taskset("toy-bridge-env") == ("toy_bridge_env", "unpackaged")
    assert "boom_pkg" not in sys.modules and not log.exists()
    with pytest.raises(ModuleNotFoundError):
        locate_taskset("no-such-taskset-anywhere")


def test_a_module_shadowing_an_installed_distribution_is_unpackaged(tmp_path, monkeypatch):
    import sys

    from reliquary_sandbox_verifiers.loading import locate_taskset

    real = tmp_path / "site"
    real.mkdir()
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    for directory in (real, shadow):
        (directory / "boom_pkg.py").write_text(
            "import os\nopen(os.environ['BOOM_LOG'], 'a').write('imported')\nraise RuntimeError\n")
    monkeypatch.setenv("BOOM_LOG", str(tmp_path / "boom.log"))
    info = real / "boom_dist-2.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: boom-dist\nVersion: 2.0\n")
    (info / "top_level.txt").write_text("boom_pkg\n")
    (info / "RECORD").write_text("boom_pkg.py,,\n")
    monkeypatch.syspath_prepend(str(real))
    assert locate_taskset("boom-pkg") == ("boom_pkg", "2.0")  # the listed file
    monkeypatch.syspath_prepend(str(shadow))  # now first on the path
    assert locate_taskset("boom-pkg") == ("boom_pkg", "unpackaged")
    with pytest.raises(ValueError):
        VerifiersEnv("verifiers:boom-pkg==2.0")
    assert "boom_pkg" not in sys.modules and not (tmp_path / "boom.log").exists()


def test_the_options_digest_is_over_the_options_in_effect():
    import hashlib

    from reliquary_sandbox.attest import canonical_json
    from reliquary_sandbox_verifiers.loading import options_sha256

    explicit = {"tools": ["edit", "bash"], "splits": {"rl": {"split": "rl"}},
                "defaults": {"prepare_timeout_s": 600, "per_call_timeout_s": 180},
                "publish_state": False}
    assert options_sha256(None) == options_sha256({}) == options_sha256(explicit)
    assert options_sha256(None) == hashlib.sha256(
        canonical_json(VerifiersEnvOptions.from_mapping(None).effective())).hexdigest()
    effective = VerifiersEnvOptions.from_mapping(explicit).effective()
    assert effective["tools"] == ["bash", "edit"]
    assert effective["defaults"]["prepare_timeout_s"] == 600.0
    assert effective["splits"] == {"rl": {"split": "rl"}}
    for changed in ({"tools": ["bash"]}, {"splits": {"train": {"split": "rl"}}},
                    {"defaults": {"per_call_timeout_s": 600}}, {"publish_state": True}):
        assert options_sha256(changed) != options_sha256(None), changed
    with pytest.raises(ValueError):
        options_sha256({"tools": ["rm"]})
