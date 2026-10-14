import pytest
from episode_fixtures import IMAGE

from reliquary_sandbox.episode_task import SandboxTask
from reliquary_sandbox_service.episodes.registry import TaskRegistry, TaskResolutionError


async def noop(*args):
    return ""


@pytest.mark.parametrize("override", [
    {"image": "registry.example/task:latest"},
    {"workdir": "relative"},
    {"tools": ()},
    {"tools": ("bash", "browser")},
    {"max_state_bytes": 0},
])
def test_task_declarations_are_validated(override):
    values = dict(image=IMAGE, workdir="/work", tools=("bash",), prepare=noop, extract=noop, grade=noop)
    values.update(override)
    with pytest.raises(ValueError):
        SandboxTask(**values)


def test_registry_resolves_declared_tasks_and_versions():
    registry = TaskRegistry({"fake-env": "fake_episode_env:sandbox_task"})
    resolved = registry.resolve("fake-env", "train", 3)
    assert resolved.task.workdir == "/work"
    assert resolved.env_package == "fake_episode_env==unpackaged"
    assert registry.versions() == {"fake-env": "fake_episode_env==unpackaged"}


def test_registry_reports_distribution_versions_for_installed_packages():
    registry = TaskRegistry({"pkg": "pydantic:BaseModel"})
    assert registry.versions()["pkg"].startswith("pydantic==2.")


@pytest.mark.parametrize("env,split,index", [("other-env", "train", 1), ("fake-env", "test", 1)])
def test_unknown_env_or_failing_declaration(env, split, index):
    registry = TaskRegistry({"fake-env": "fake_episode_env:sandbox_task"})
    with pytest.raises(TaskResolutionError):
        registry.resolve(env, split, index)


def test_grading_workdir_and_user_default_to_the_agents_and_keep_explicit_values():
    values = dict(image=IMAGE, workdir="/app", tools=("bash",), prepare=noop, extract=noop,
                  grade=noop, user="agent")
    task = SandboxTask(**values)
    assert (task.effective_grading_workdir, task.effective_grading_user) == ("/app", "agent")
    task = SandboxTask(**values, grading_workdir="/", grading_user="root")
    assert (task.effective_grading_workdir, task.effective_grading_user) == ("/", "root")
    with pytest.raises(ValueError):
        SandboxTask(**values, grading_workdir="relative")


def test_an_explicit_empty_grading_user_is_kept_not_replaced():
    task = SandboxTask(image=IMAGE, workdir="/app", tools=("bash",), prepare=noop, extract=noop,
                       grade=noop, user="agent", grading_user="")
    assert task.effective_grading_user == ""


@pytest.mark.parametrize("image", ["a@sha256:zzz", "a@sha256:" + "a" * 63, "a@sha256:" + "A" * 64])
def test_image_digest_must_be_well_formed(image):
    with pytest.raises(ValueError):
        SandboxTask(image=image, workdir="/w", tools=("bash",), prepare=noop, extract=noop, grade=noop)


def test_registry_refuses_unloadable_non_callable_and_wrong_type_entries():
    with pytest.raises(TaskResolutionError):
        TaskRegistry({"a": "no_such_module_xyz:f"})
    with pytest.raises(TaskResolutionError):
        TaskRegistry({"a": "fake_episode_env:no_such_attribute"})
    with pytest.raises(TaskResolutionError):
        TaskRegistry({"a": "fake_episode_env:IMAGE"})
    registry = TaskRegistry({"a": "fake_episode_env:prepare"})
    with pytest.raises(TaskResolutionError):
        registry.resolve("a", "train", 0)


@pytest.mark.asyncio
async def test_agent_attributable_outcome_is_graded_as_zero_with_a_fact():
    import fake_episode_env

    result = await fake_episode_env.grade(None, fake_episode_env.TOO_LARGE)
    assert result.reward == 0.0 and result.facts == {"refused": "state_too_large"}


def _install_fake_env(root, name="rq_digest_env", body="GRADE = 1\n"):
    """A wheel-installed env package: its module and a dist-info whose RECORD lists it."""
    package = root / name
    package.mkdir()
    (package / "__init__.py").write_text(
        "from reliquary_sandbox.episode_task import SandboxTask\n\n"
        "def sandbox_task(split, index):\n    raise LookupError(index)\n")
    (package / "grading.py").write_text(body)
    (package / "data.json").write_text("{}")
    info = root / f"{name}-0.2.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.2.0\n")
    (info / "RECORD").write_text(
        f"{name}/__init__.py,,\n{name}/grading.py,,\n{name}/data.json,,\n"
        f"{name}-0.2.0.dist-info/METADATA,,\n{name}-0.2.0.dist-info/RECORD,,\n")
    (info / "top_level.txt").write_text(name + "\n")
    return package


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    import sys

    monkeypatch.syspath_prepend(str(tmp_path))
    package = _install_fake_env(tmp_path)
    yield package
    sys.modules.pop("rq_digest_env", None)


def test_env_package_carries_a_digest_of_the_installed_python_files(fake_env):
    import re

    registry = TaskRegistry({"e": "rq_digest_env:sandbox_task"})
    package = registry.versions()["e"]
    assert re.fullmatch(r"rq_digest_env==0\.2\.0\+g[0-9a-f]{16}", package)
    assert TaskRegistry({"e": "rq_digest_env:sandbox_task"}).versions()["e"] == package
    (fake_env / "data.json").write_text('{"not": "python"}')
    assert TaskRegistry({"e": "rq_digest_env:sandbox_task"}).versions()["e"] == package


def test_env_package_digest_changes_when_a_grading_file_changes(fake_env):
    before = TaskRegistry({"e": "rq_digest_env:sandbox_task"}).versions()["e"]
    (fake_env / "grading.py").write_text("GRADE = 0\n")
    after = TaskRegistry({"e": "rq_digest_env:sandbox_task"}).versions()["e"]
    assert before.split("+g")[0] == after.split("+g")[0] and before != after


def test_env_package_is_in_the_task_fingerprint():
    import dataclasses

    from reliquary_sandbox_service.episodes.registry import task_fingerprint

    resolved = TaskRegistry({"fake-env": "fake_episode_env:sandbox_task"}).resolve("fake-env",
                                                                                   "train", 3)
    other = dataclasses.replace(resolved, env_package=resolved.env_package + "+g" + "0" * 16)
    assert task_fingerprint(resolved) != task_fingerprint(other)


def test_an_editable_install_is_hashed_from_its_package_directory(tmp_path, monkeypatch):
    """An editable install's RECORD lists a .pth, no module and no top_level.txt: the
    distribution is found by name and its imported package directory is hashed."""
    import re
    import sys

    source = tmp_path / "src"
    source.mkdir()
    package = _install_fake_env(source, name="rq_editable_env")
    site = tmp_path / "site"
    site.mkdir()
    info = source / "rq_editable_env-0.2.0.dist-info"
    (info / "RECORD").write_text("rq_editable_env.pth,,\n")
    (info / "top_level.txt").unlink()
    info.rename(site / info.name)
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.syspath_prepend(str(source))
    try:
        first = TaskRegistry({"e": "rq_editable_env:sandbox_task"}).versions()["e"]
        assert re.fullmatch(r"rq_editable_env==0\.2\.0\+g[0-9a-f]{16}", first)
        (package / "grading.py").write_text("GRADE = 0\n")
        assert TaskRegistry({"e": "rq_editable_env:sandbox_task"}).versions()["e"] != first
    finally:
        sys.modules.pop("rq_editable_env", None)


def test_a_declared_content_hash_is_in_the_task_fingerprint_only_when_set():
    import dataclasses

    from reliquary_sandbox_service.episodes.registry import task_fingerprint

    resolved = TaskRegistry({"fake-env": "fake_episode_env:sandbox_task"}).resolve("fake-env",
                                                                                   "train", 3)
    before = task_fingerprint(resolved)
    assert resolved.task.content_sha256 is None
    with_content = dataclasses.replace(
        resolved, task=dataclasses.replace(resolved.task, content_sha256="1" * 64))
    other_content = dataclasses.replace(
        resolved, task=dataclasses.replace(resolved.task, content_sha256="2" * 64))
    assert task_fingerprint(with_content) not in (before, task_fingerprint(other_content))
    assert task_fingerprint(resolved) == before


@pytest.mark.parametrize("content", ["", "A" * 64, "1" * 63, 7])
def test_a_content_hash_must_be_a_sha256(content):
    with pytest.raises(ValueError):
        SandboxTask(image=IMAGE, workdir="/work", tools=("bash",), prepare=noop, extract=noop,
                    grade=noop, content_sha256=content)


def test_entry_module_resolves_without_importing(tmp_path, monkeypatch):
    import sys

    from reliquary_sandbox_service.episodes.registry import entry_module

    (tmp_path / "boom_pkg.py").write_text(
        "import os\nopen(os.environ['BOOM_LOG'], 'a').write('x')\nraise RuntimeError\n")
    monkeypatch.setenv("BOOM_LOG", str(tmp_path / "boom.log"))
    monkeypatch.syspath_prepend(str(tmp_path))
    assert entry_module("verifiers:boom-pkg==unpackaged", allow_unpackaged=True) == "boom_pkg"
    for refused in ({"entry": "verifiers:boom-pkg==unpackaged"},
                    {"entry": "verifiers:boom-pkg==1", "allow_unpackaged": True}):
        with pytest.raises(ValueError):
            entry_module(refused.pop("entry"), **refused)
    assert entry_module("fake_episode_env:sandbox_task") == "fake_episode_env"
    assert "boom_pkg" not in sys.modules and not (tmp_path / "boom.log").exists()
    with pytest.raises(ModuleNotFoundError):
        entry_module("verifiers:no-such-taskset-anywhere==1")
