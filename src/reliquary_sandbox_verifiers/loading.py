"""Load a verifiers env declared as `verifiers:<taskset-id>==<version>` and serve its tasks.

The gateway's `episode_envs` maps an env name to that entry; `episode_env_options` gives
optional per-env options (`VerifiersEnvOptions`). Miners and validators load the same task
with the same class, so all sides agree on its data (record 0 signs its content hash).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import itertools
import os
import re
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from types import MappingProxyType
from typing import Any

from verifiers.v1.utils.loaders import import_taskset, taskset_class, taskset_config_type

from reliquary_sandbox.attest import canonical_json
from reliquary_sandbox.episode_task import (
    TOOLS,
    VERIFIERS_ENTRY,
    VERIFIERS_ENTRY_PREFIX,
    SandboxTask,
)

from .bridge import task_from_verifiers
from .declaration import DEFAULT_TOOLS, BridgeDefaults
from .image_lock import ImageLock

__all__ = ["EVAL_SPLIT", "VERIFIERS_ENTRY_PREFIX", "VerifiersEnv", "VerifiersEnvOptions",
           "VerifiersEnvRef", "locate_taskset",
           "gate_taskset", "options_sha256"]

EVAL_SPLIT = "eval"
UNPACKAGED = "unpackaged"
MAX_TASK_INDEX = 10_000_000
SPLIT_NAME = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class VerifiersEnvRef:
    taskset_id: str
    version: str

    @classmethod
    def parse(cls, entry: str) -> VerifiersEnvRef:
        match = VERIFIERS_ENTRY.fullmatch(entry)
        if match is None:
            raise ValueError(f"{entry!r} is not 'verifiers:<taskset-id>==<version>'")
        return cls(match["id"], match["version"])


def _split_token(value: Any, what: str) -> str:
    """A split name or value: lower-case `[a-z0-9_]+`, never normalised."""
    if not isinstance(value, str) or SPLIT_NAME.fullmatch(value) is None:
        raise ValueError(f"invalid {what} {value!r}: lower-case [a-z0-9_]+")
    return value


def _says_eval(token: str) -> bool:
    return EVAL_SPLIT in token


def _frozen_splits(raw: Any) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("splits must map at least one split name to taskset config overrides")
    splits = {}
    for name, overrides in raw.items():
        _split_token(name, "split name")
        if not isinstance(overrides, Mapping):
            raise ValueError(f"split {name!r} must map to taskset config overrides")
        if set(overrides) != {"split"}:
            raise ValueError(f"split {name!r}: overrides must be exactly {{'split': ...}}")
        value = _split_token(overrides["split"], "split value")
        if _says_eval(name) != _says_eval(value):
            raise ValueError(f"split {name!r} serves {value!r}: a split serving eval tasks is "
                             "named eval, and only such a split is")
        splits[name] = MappingProxyType({"split": value})
    return MappingProxyType(splits)


@dataclass(frozen=True)
class VerifiersEnvOptions:
    """Per-env options of a bridged env (`episode_env_options[env]`)."""

    tools: tuple[str, ...] = DEFAULT_TOOLS
    splits: Mapping[str, Mapping[str, Any]] = field(
        default_factory=lambda: _frozen_splits({"rl": {"split": "rl"}}))
    """Served split -> taskset config overrides. Default: `rl`, with `split="rl"`."""
    defaults: BridgeDefaults = BridgeDefaults()
    publish_state: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> VerifiersEnvOptions:
        raw = dict(raw or {})
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown options: {sorted(unknown)}")
        values: dict[str, Any] = {}
        if "tools" in raw:
            tools = raw["tools"]
            if (not isinstance(tools, (list, tuple)) or not tools
                    or not set(tools) <= TOOLS or len(set(tools)) != len(tools)):
                raise ValueError(f"tools must be a non-empty subset of {sorted(TOOLS)}")
            values["tools"] = tuple(tools)
        if "splits" in raw:
            values["splits"] = _frozen_splits(raw["splits"])
        if "defaults" in raw:
            if not isinstance(raw["defaults"], Mapping):
                raise ValueError("defaults must be an object")
            try:
                values["defaults"] = BridgeDefaults(**raw["defaults"])
            except TypeError as exc:
                raise ValueError(f"invalid defaults: {exc}") from exc
        if "publish_state" in raw:
            if not isinstance(raw["publish_state"], bool):
                raise ValueError("publish_state must be a boolean")
            values["publish_state"] = raw["publish_state"]
        return cls(**values)

    def effective(self) -> dict[str, Any]:
        """The options in force, defaults filled in, as plain JSON values: tools sorted,
        splits as plain objects, every `BridgeDefaults` field (a float field as a float,
        so `600` and `600.0` read alike)."""
        defaults = asdict(self.defaults)
        for f in fields(BridgeDefaults):
            if isinstance(f.default, float) and isinstance(defaults[f.name], int):
                defaults[f.name] = float(defaults[f.name])
        return {"tools": sorted(self.tools),
                "splits": {name: dict(overrides) for name, overrides in self.splits.items()},
                "defaults": defaults, "publish_state": self.publish_state}

    def sha256(self) -> str:
        """sha256 hex of the canonical JSON of `effective()`: what the capacity report
        publishes per bridged env, so a client building the same env can check it agrees."""
        return hashlib.sha256(canonical_json(self.effective())).hexdigest()


def options_sha256(options: VerifiersEnvOptions | Mapping[str, Any] | None) -> str:
    """`VerifiersEnvOptions.sha256` of `options` (an `episode_env_options[env]` value, or
    None for the defaults): raises ValueError for options the bridge refuses."""
    if not isinstance(options, VerifiersEnvOptions):
        options = VerifiersEnvOptions.from_mapping(options)
    return options.sha256()


def _version_of_spec(spec: Any, top: str) -> str:
    """The version of the distribution providing `top`, but only if that distribution
    LISTS the file `spec` resolved to: a module earlier on sys.path that shadows an
    installed distribution is "unpackaged", whatever the distribution's version."""
    names = importlib.metadata.packages_distributions().get(top, [])
    if not names or not spec.origin:
        return UNPACKAGED
    try:
        dist = importlib.metadata.distribution(names[0])
    except importlib.metadata.PackageNotFoundError:
        return UNPACKAGED
    origin = os.path.realpath(spec.origin)
    listed = {os.path.realpath(dist.locate_file(f)) for f in dist.files or ()}
    return dist.version if origin in listed else UNPACKAGED


def gate_taskset(taskset_id: str, pinned: str, *, allow_unpackaged: bool = False) -> str:
    """The module name of a taskset that passed the checks every entry must pass before
    ANY import (this process, the scoring forkserver, the preload check): found, packaged
    unless `allow_unpackaged`, and installed at exactly `pinned`. Raises ValueError (or
    ModuleNotFoundError) otherwise. Imports nothing."""
    module_name, installed = locate_taskset(taskset_id)
    if installed == UNPACKAGED and not allow_unpackaged:
        raise ValueError(f"{taskset_id} is not an installed distribution "
                         f"({UNPACKAGED}); only a test env may be loaded unpackaged")
    if installed != pinned:
        raise ValueError(f"{taskset_id} is installed at {installed}; the entry pins {pinned}")
    return module_name


def locate_taskset(taskset_id: str) -> tuple[str, str]:
    """`(module name, installed version)` of a taskset, found WITHOUT executing it: the
    module name verifiers' loader would import (`verifiers.v1.tasksets.<mod>` when that
    exists, else `<mod>`, the id with `-` as `_`), and the version of the distribution that
    provides its top-level package ("unpackaged" if none). Raises ModuleNotFoundError when
    nothing provides the module. The controller runs as root: nothing may be imported
    before an entry has passed the checks that use this."""
    name = taskset_id.rsplit("/", 1)[-1].split("@", 1)[0]
    module = name.replace("-", "_").lower()
    for target in (f"verifiers.v1.tasksets.{module}", module):
        try:
            spec = importlib.util.find_spec(target)
        except (ImportError, ValueError):  # a missing parent package; a spec-less module
            spec = None
        if spec is not None:
            return target, _version_of_spec(spec, target.split(".", 1)[0])
    raise ModuleNotFoundError(f"taskset {taskset_id!r} not found (tried {module!r})")


class VerifiersEnv:
    """One verifiers env served through the bridge. Thread-safe."""

    def __init__(self, entry: str, options: VerifiersEnvOptions | Mapping[str, Any] | None = None,
                 *, allow_unpackaged: bool = False):
        self.ref = VerifiersEnvRef.parse(entry)
        self.options = (options if isinstance(options, VerifiersEnvOptions)
                        else VerifiersEnvOptions.from_mapping(options))
        # Gates first, on metadata only: importing runs the module's code in this process.
        module_name = gate_taskset(self.ref.taskset_id, self.ref.version,
                                   allow_unpackaged=allow_unpackaged)
        self.module = import_taskset(self.ref.taskset_id)
        if self.module.__name__ != module_name:
            raise ValueError(f"{self.ref.taskset_id} imported as {self.module.__name__}, "
                             f"not {module_name}")
        self.taskset_class = taskset_class(self.ref.taskset_id)
        self.lock = ImageLock.of_module(self.module)
        self._tasksets: dict[str, Any] = {}
        self._guard = threading.Lock()

    def taskset(self, split: str) -> Any:
        try:
            overrides = self.options.splits.get(split)
        except TypeError:  # unhashable
            overrides = None
        if overrides is None:
            raise LookupError(f"split {split!r} is not served")
        with self._guard:
            if split not in self._tasksets:
                config = taskset_config_type(self.ref.taskset_id).model_validate(
                    {**overrides, "id": self.ref.taskset_id})
                self._tasksets[split] = self.taskset_class(config)
            return self._tasksets[split]

    def task(self, split: str, index: int) -> Any:
        """Task `index` of `split`: the taskset's `task_at(index)` when it defines one
        (with the config's system prompt applied), else the index-th task it yields."""
        if type(index) is not int or index < 0:
            raise IndexError(index)
        taskset = self.taskset(split)
        try:
            size = len(taskset)
        except TypeError:
            size = None
        if index >= (MAX_TASK_INDEX + 1 if size is None else size):
            raise IndexError(index)
        task_at = getattr(taskset, "task_at", None)
        system_prompt = getattr(taskset, "system_prompt", None)
        if callable(task_at):
            try:
                task = task_at(index)
            except (KeyError, ValueError, StopIteration) as exc:
                raise IndexError(index) from exc
            if system_prompt is not None:
                task = task.with_system_prompt(system_prompt)
            return task
        task = next(itertools.islice(iter(taskset), index, None), None)
        if task is None:
            raise IndexError(index)
        return task

    def sandbox_task(self, split: str, index: int) -> SandboxTask:
        return task_from_verifiers(self.task(split, index), lock=self.lock,
                                   tools=self.options.tools, defaults=self.options.defaults,
                                   publish_state=self.options.publish_state)
