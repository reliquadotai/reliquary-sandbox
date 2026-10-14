"""Resolve `(env, split, index)` to a SandboxTask through configured entry points.

Env packages are imported and run in the gateway process, next to the machine signing key, so
only operator-chosen, version-pinned packages may be listed in `episode_envs`. Each env's
identity (`env_package`, in record 0 and the task fingerprint) is computed once, here, from
its installed files: validators allow-list the accepted values.

An entry is either `module:callable` (a `sandbox_task(split, index)` factory, legacy) or `verifiers:<taskset-id>==<version>`: a standard verifiers v1 Taskset served
through the generic bridge (`reliquary_sandbox_verifiers.loading.VerifiersEnv`), with the
optional per-env options of `episode_env_options`. A bridged env's identity is
`bridged_env_package_of(module)`.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from reliquary_sandbox.attest import canonical_json
from reliquary_sandbox.episode_task import VERIFIERS_ENTRY_PREFIX, SandboxTask, SandboxTaskFactory


class TaskResolutionError(ValueError):
    """The env is unknown here or its declaration is invalid."""


@dataclass(frozen=True)
class ResolvedTask:
    task: SandboxTask
    env_package: str
    """`distribution==version+g<digest>` of the package that declared it (record 0): the
    digest is the first 16 hex of `python_files_digest`, so a grading file edited in place
    changes the identity a validator allow-lists."""


def python_files_digest(files: Iterable[tuple[str, bytes]]) -> str:
    """sha256 over `(relative path, bytes)` pairs sorted by path: each pair hashed as
    `path NUL bytes`, the outer hash over those digests (so no file boundary is ambiguous)."""
    outer = hashlib.sha256()
    for path, data in sorted(files):
        outer.update(hashlib.sha256(path.encode() + b"\0" + data).digest())
    return outer.hexdigest()


def _installed_python_files(name: str, top: str) -> list[tuple[str, bytes]]:
    """The distribution's installed `.py` files (its RECORD), by path relative to its
    install root. An editable install's RECORD lists none: then the files of the
    imported top-level package, relative to the directory holding it."""
    listed = [(str(f.as_posix()), f.locate()) for f in importlib.metadata.files(name) or ()
              if f.suffix == ".py"]
    if listed:
        return [(path, Path(located).read_bytes()) for path, located in listed]
    module = importlib.import_module(top)
    roots = [Path(p) for p in getattr(module, "__path__", ())] or [Path(module.__file__)]
    found = []
    for root in roots:
        paths = [root] if root.is_file() else sorted(root.rglob("*.py"))
        found.extend((path.relative_to(root.parent).as_posix(), path.read_bytes())
                     for path in paths if "__pycache__" not in path.parts)
    return found


def _package_of(entry: str) -> str:
    module = entry.split(":", 1)[0]
    top = module.split(".", 1)[0]
    distributions = importlib.metadata.packages_distributions().get(top, [])
    if distributions:
        name = distributions[0]
    else:
        # An editable install maps no module to its distribution: find it by name.
        try:
            name = importlib.metadata.distribution(top).metadata["Name"]
        except importlib.metadata.PackageNotFoundError:
            return f"{top}==unpackaged"
    digest = python_files_digest(_installed_python_files(name, top))[:16]
    return f"{name}=={importlib.metadata.version(name)}+g{digest}"


def env_package_of(module_or_name: ModuleType | str) -> str:
    """The identity record 0, the task fingerprint and the capacity report carry for the
    env package that provides `module_or_name` (a module, a module name, or a
    `module:callable` entry point): `distribution==version+g<digest>`, the digest being
    the first 16 hex of `python_files_digest` over the distribution's installed `.py`
    files (an editable install: its imported package directory), or
    `<module>==unpackaged` outside any distribution. Validators compute the values they
    allow-list with it."""
    if isinstance(module_or_name, ModuleType):
        name = module_or_name.__name__
    elif isinstance(module_or_name, str):
        name = module_or_name.split(":", 1)[0]
    else:
        raise TypeError("env_package_of takes a module, a module name or an entry point")
    if not name:
        raise ValueError("env_package_of needs a module name")
    return _package_of(name)


def bridged_env_package_of(module_or_name: ModuleType | str) -> str:
    """The identity of a verifiers env served through the bridge: `env_package_of` of its
    taskset module, followed by `.vfb<BRIDGE_VERSION>` (a local-version segment), so a
    change to how the bridge grades changes what validators allow-list."""
    from reliquary_sandbox_verifiers.declaration import BRIDGE_VERSION

    return f"{env_package_of(module_or_name)}.vfb{BRIDGE_VERSION}"


def entry_module(entry: str, *, allow_unpackaged: bool = False) -> str:
    """The module an `episode_envs` entry imports (what the scoring forkserver preloads),
    resolved and GATED without importing anything: a verifiers entry that is unpackaged
    (unless `allow_unpackaged`), at another version than it pins, or not found raises
    (ValueError / ModuleNotFoundError)."""
    if not entry.startswith(VERIFIERS_ENTRY_PREFIX):
        return entry.split(":", 1)[0]
    from reliquary_sandbox_verifiers.loading import VerifiersEnvRef, gate_taskset

    ref = VerifiersEnvRef.parse(entry)
    return gate_taskset(ref.taskset_id, ref.version, allow_unpackaged=allow_unpackaged)


class TaskRegistry:
    def __init__(self, entries: Mapping[str, str],
                 options: Mapping[str, Mapping[str, Any]] | None = None,
                 *, allow_unpackaged: bool = False):
        self._entries = dict(entries)
        self._factories: dict[str, SandboxTaskFactory] = {}
        self._packages: dict[str, str] = {}
        self._bridged: dict[str, Any] = {}
        options = dict(options or {})
        for env in options:
            if not self._entries.get(env, "").startswith(VERIFIERS_ENTRY_PREFIX):
                raise TaskResolutionError(f"options are for verifiers envs only, not {env!r}")
        for env, entry in self._entries.items():
            if entry.startswith(VERIFIERS_ENTRY_PREFIX):
                self._load_bridged(env, entry, options.get(env), allow_unpackaged)
                continue
            module, _, attribute = entry.partition(":")
            try:
                factory = getattr(importlib.import_module(module), attribute)
            except (ImportError, AttributeError) as exc:
                raise TaskResolutionError(f"cannot load {entry}") from exc
            if not callable(factory):
                raise TaskResolutionError(f"{entry} is not callable")
            self._factories[env] = factory
            self._packages[env] = _package_of(entry)

    def _load_bridged(self, env: str, entry: str, options: Mapping[str, Any] | None,
                      allow_unpackaged: bool) -> None:
        try:
            from reliquary_sandbox_verifiers.loading import VerifiersEnv

            bridged = VerifiersEnv(entry, options, allow_unpackaged=allow_unpackaged)
        except Exception as exc:
            raise TaskResolutionError(f"cannot load {entry}: {exc}") from exc
        self._bridged[env] = bridged
        self._factories[env] = bridged.sandbox_task
        self._packages[env] = bridged_env_package_of(bridged.module)

    def bridged(self, env: str) -> Any:
        """The `VerifiersEnv` serving `env`, or None for a `module:callable` entry."""
        return self._bridged.get(env)

    def versions(self) -> dict[str, str]:
        return dict(self._packages)

    def options_digests(self) -> dict[str, str]:
        """Per bridged env, the sha256 of the options it is served with in effect
        (`VerifiersEnvOptions.sha256`); `module:callable` entries have none."""
        return {env: bridged.options.sha256() for env, bridged in self._bridged.items()}

    def image_locks(self) -> dict[str, Any]:
        """Per bridged env, the `ImageLock` it is served with; `module:callable` entries have none."""
        return {env: bridged.lock for env, bridged in self._bridged.items()}

    def resolve(self, env: str, split: str, index: int) -> ResolvedTask:
        factory = self._factories.get(env)
        if factory is None:
            raise TaskResolutionError(f"env {env!r} is not served by this machine")
        try:
            task = factory(split, index)
        except Exception as exc:
            raise TaskResolutionError(f"{env} could not declare task {split}/{index}") from exc
        if not isinstance(task, SandboxTask):
            raise TaskResolutionError(f"{env} returned {type(task).__name__}, not a SandboxTask")
        return ResolvedTask(task, self._packages[env])


def _limits(task: SandboxTask) -> dict:
    """The task's limits, `cpus` as a float so 2 and 2.0 fingerprint alike."""
    limits = asdict(task.limits)
    if limits["cpus"] is not None:
        limits["cpus"] = float(limits["cpus"])
    return limits


def task_fingerprint(resolved: ResolvedTask) -> str:
    """sha256 of what an episode's outcome depends on: the declaring package's
    version and code digest (`env_package`), the image, users and workdirs (agent and
    grading), environment, tools, limits and timeouts, and the task's content hash when
    it declares one (`content_sha256`). Persisted at open; a task that resolves
    differently after a restart is not the one opened."""
    task = resolved.task
    declared = {
        "env_package": resolved.env_package, "image": task.image, "workdir": task.workdir,
        "user": task.user, "grading_workdir": task.effective_grading_workdir,
        "grading_user": task.effective_grading_user, "env": dict(task.env),
        "tools": list(task.tools), "max_state_bytes": task.max_state_bytes,
        "prepare_timeout_s": float(task.prepare_timeout_s),
        "grading_timeout_s": float(task.grading_timeout_s), "publish_state": task.publish_state,
        "limits": _limits(task),
    }
    if task.content_sha256 is not None:  # absent otherwise: existing fingerprints unchanged
        declared["content_sha256"] = task.content_sha256
    return hashlib.sha256(canonical_json(declared)).hexdigest()
