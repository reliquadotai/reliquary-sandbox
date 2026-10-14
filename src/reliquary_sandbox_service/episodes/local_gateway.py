"""A complete episode gateway on 127.0.0.1, for env authors' tests and parity goldens.

    with local_gateway(tmp, envs={"my-env": "my_env.sandbox:sandbox_task"},
                       images=["repo@sha256:..."], episode_helper_busybox=Path(...),
                       episode_helper_busybox_sha256="...") as gateway:
        token = gateway.issue(env="my-env", split="train", index=0, image=...,
                              budgets=default_budgets(task.limits))

It runs the real app (routes, manager, grader, signing) under uvicorn in a thread, with
fresh machine and validator keys under `workdir`. Without `task_runtime`, task boxes are
real: Docker at `docker_host` when the settings give one, else from the environment
(`DOCKER_HOST`), runsc, the helper busybox and the quota checks of a production host. On
exit it removes the boxes it created and closes its Docker client, unless the server did
not stop (then both are left and the leak is logged). Tests pass FakeBoxes instead. It serves episodes only:
the tenant API has no runtime behind it. Never run it on a production host.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import threading
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn

from reliquary_sandbox.attest import (
    Budgets,
    Ed25519TokenVerifier,
    Expected,
    MachineKey,
    SessionClaims,
    Signer,
    StaticDirectory,
    VerificationResult,
    generate_private_key_pem,
    issue_session_token,
    load_private_key,
    verify_transcript,
)
from reliquary_sandbox.episode_task import SandboxTask, TaskLimits

from ..app import create_app
from ..settings import Settings
from ..task_runtime import TaskBoxRuntime
from .registry import TaskRegistry

_log = logging.getLogger(__name__)

MACHINE_ID = "local-gateway"
MACHINE_KEY_ID = "local-machine"
VALIDATOR_KEY_ID = "local-validator"
START_TIMEOUT_S = 120.0
DEFAULT_BUDGETS = dict(max_calls=64, per_call_timeout_s=600, cpu_s=3600, wall_s=3600,
                       memory_bytes=2 * 1024**3, pids=1024, disk_bytes=10 * 1024**3)
_FROM_LIMITS = ("memory_bytes", "disk_bytes", "pids", "wall_s", "per_call_timeout_s",
                "max_calls")


def default_budgets(limits: TaskLimits | None = None, **overrides: int) -> Budgets:
    """Budgets for a local episode: DEFAULT_BUDGETS, raised or set to what `limits`
    declares, then `overrides`."""
    values = dict(DEFAULT_BUDGETS)
    if limits is not None:
        for name in _FROM_LIMITS:
            value = getattr(limits, name)
            if value is not None:
                values[name] = value
    values.update(overrides)
    return Budgets(**values)


class _NoTenantRuntime:
    """The tenant API's backend, unused: a local gateway serves episodes only."""

    client = None
    namespace = "local-gateway"

    def readiness(self) -> dict[str, bool]:
        return {"ready": True}

    def managed_sandboxes(self) -> dict[str, str]:
        return {}


def _key_file(directory: Path, name: str) -> Path:
    """A fresh private key, created owner-only; an existing file is never overwritten."""
    path = directory / f"{name}.pem"
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(generate_private_key_pem())
    return path


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


STOP_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class _Served:
    url: str
    thread: threading.Thread


@contextmanager
def serve_app(app: Any, port: int | None = None) -> Iterator[str]:
    """Serve an ASGI app on 127.0.0.1 in a thread (lifespan on); yields its base URL."""
    with _serve(app, port or _free_port()) as served:
        yield served.url


def server_config(app: Any, port: int) -> uvicorn.Config:
    """HTTP only (`ws="none"`): nothing here speaks WebSocket, so no WebSocket stack is
    loaded (the controller image runs uvicorn with `--ws none` too)."""
    return uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                          lifespan="on", ws="none")


@contextmanager
def _serve(app: Any, port: int) -> Iterator[_Served]:
    server = uvicorn.Server(server_config(app, port))
    thread = threading.Thread(target=server.run, name=f"serve-{port}", daemon=True)
    thread.start()
    deadline = time.monotonic() + START_TIMEOUT_S
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("the server failed to start; see its log")
        if time.monotonic() > deadline:
            server.should_exit = True
            raise TimeoutError("the server did not start in time")
        time.sleep(0.05)
    try:
        yield _Served(f"http://127.0.0.1:{port}", thread)
    finally:
        server.should_exit = True
        thread.join(STOP_TIMEOUT_S)
        if thread.is_alive():
            _log.error("the server on port %d is still running after %.0f s", port,
                       STOP_TIMEOUT_S)


def _docker_task_runtime(settings: Settings) -> tuple[Any, Any]:
    """Real task boxes: a Docker client (`settings.docker_host`, else the environment's
    DOCKER_HOST) and the runtime over it."""
    import docker

    if settings.docker_host:
        client = docker.DockerClient(base_url=settings.docker_host, timeout=120)
    else:
        client = docker.from_env(timeout=120)
    runtime = TaskBoxRuntime(settings, client=client,
                             namespace=f"local-{uuid.uuid4().hex[:8]}")
    return runtime, client


def _remove_boxes(runtime: Any) -> None:
    for box_id in list(runtime.managed()):
        try:
            runtime.remove(box_id)
        except Exception:
            _log.exception("could not remove box %s; remove it by hand", box_id)


@dataclass
class LocalGateway:
    url: str
    settings: Settings
    validator: Signer
    machine: Signer
    app: Any = field(default=None, repr=False, compare=False)
    registries: dict[str, TaskRegistry] = field(default_factory=dict, init=False, repr=False,
                                                compare=False)
    """One registry per env, built on first use: a bridged env loads its taskset once."""
    _registry_lock: threading.Lock = field(default_factory=threading.Lock, init=False,
                                           repr=False, compare=False)

    def issue(self, *, env: str, split: str, index: int, image: str, budgets: Budgets,
              hotkey: str = "5LocalHotkey", engagement: str = "local:0",
              checkpoint: str = "local", ttl_s: int = 3600,
              session_id: str | None = None) -> dict[str, Any]:
        now = int(time.time())
        claims = SessionClaims(session_id=session_id or uuid.uuid4().hex, hotkey=hotkey,
                               engagement=engagement, env=env, split=split, index=index,
                               image=image, checkpoint=checkpoint,
                               machine_id=self.settings.machine_id, issued_at=now,
                               expires_at=now + ttl_s, budgets=budgets)
        return issue_session_token(claims, self.validator)

    @contextmanager
    def expect_absent(self, session_id: str, paths: Sequence[str]) -> Iterator[None]:
        """Conformance only: the grader checks, in the pristine grading box once the
        reward ran, that none of `paths` exists, and records it in the grading facts
        (`_grader.canary_checked` / `canary_executed`). The reward is untouched. Nothing
        is registered, so nothing is probed, unless a suite asks."""
        grader = self.app.state.service["episodes"].grader
        grader.canaries[session_id] = tuple(paths)
        try:
            yield
        finally:
            grader.canaries.pop(session_id, None)

    @contextmanager
    def record_retries(self, session_id: str) -> Iterator[dict[str, int]]:
        """Conformance only: counts, for this episode, the grading attempts run again in a
        fresh box (`regraded`) and the read-only helpers run again in place
        (`helper_retries`). Read after the episode; nothing is signed or changed."""
        grader = self.app.state.service["episodes"].grader
        log = grader.retry_log[session_id] = {"regraded": 0, "helper_retries": 0}
        try:
            yield log
        finally:
            grader.retry_log.pop(session_id, None)

    def verify(self, transcript: Any, *, require_graded: bool = True) -> VerificationResult:
        """Signatures, chain and token of a transcript from this gateway (the bindings are
        taken from its own token: a local check, not a validator's)."""
        claims = transcript["token"]["claims"]
        expected = Expected(hotkey=claims["hotkey"], engagement=claims["engagement"],
                            env=claims["env"], split=claims["split"], index=claims["index"],
                            checkpoint=claims["checkpoint"], require_graded=require_graded)
        directory = StaticDirectory([MachineKey(self.settings.machine_id, self.machine.key_id,
                                                self.machine.public_key_b64, 0)])
        tokens = Ed25519TokenVerifier({self.validator.key_id: self.validator.public_key_b64})
        return verify_transcript(transcript, directory, tokens, expected)


@contextmanager
def local_gateway(workdir: Path, *, envs: Mapping[str, str], images: list[str],
                  task_runtime: Any = None, **overrides: Any) -> Iterator[LocalGateway]:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    validator = Signer(VALIDATOR_KEY_ID, load_private_key(_key_file(workdir, "validator")))
    machine_path = _key_file(workdir, "machine")
    machine = Signer(MACHINE_KEY_ID, load_private_key(machine_path))
    port = _free_port()
    values: dict[str, Any] = dict(
        tokens={"local": uuid.uuid4().hex}, state_dir=workdir / "state", episodes_enabled=True,
        machine_id=MACHINE_ID, machine_key_file=machine_path, machine_key_id=MACHINE_KEY_ID,
        validator_public_keys={VALIDATOR_KEY_ID: validator.public_key_b64},
        episode_images=list(images), episode_envs=dict(envs),
        public_base_url=f"http://127.0.0.1:{port}")
    values.update(overrides)
    settings = Settings(**values)
    owned = client = served = None
    if task_runtime is None:
        owned, client = _docker_task_runtime(settings)
        task_runtime = owned
    try:
        app = create_app(settings, _NoTenantRuntime(), task_runtime=task_runtime)
        with _serve(app, port) as served:
            yield LocalGateway(served.url, settings, validator, machine, app)
    finally:
        if served is not None and served.thread.is_alive():
            _log.error("the gateway is still running: its boxes and Docker client are left "
                       "as they are")
        elif owned is not None:
            try:
                _remove_boxes(owned)
            finally:
                client.close()


@dataclass(frozen=True)
class ScriptedEpisode:
    final: dict[str, Any]
    """The final record's body."""
    records: list[dict[str, Any]]
    """The call records' bodies, in order."""
    state: bytes | None
    transcript: dict[str, Any]
    verification: VerificationResult
    retries: dict[str, int] = field(default_factory=dict)
    """`regraded` and `helper_retries` (`LocalGateway.record_retries`)."""


def resolve_task(gateway: LocalGateway, env: str, split: str, index: int) -> SandboxTask:
    """The task the gateway serves for `(env, split, index)`, resolved as its registry does."""
    settings = gateway.settings
    options = settings.episode_env_options
    with gateway._registry_lock:
        registry = gateway.registries.get(env)
        if registry is None:
            registry = gateway.registries[env] = TaskRegistry(
                {env: settings.episode_envs[env]}, {env: options[env]} if env in options else {},
                allow_unpackaged=settings.episode_allow_unpackaged_verifiers)
    return registry.resolve(env, split, index).task


async def run_scripted_episode(gateway: LocalGateway, *, env: str, split: str, index: int,
                               calls: Sequence[tuple[str, dict[str, Any]]],
                               budgets: Budgets | None = None,
                               absent: Sequence[str] = ()) -> ScriptedEpisode:
    """Open, run `calls` (one per turn, k = 0), finish, and read back the transcript:
    an agent without a model, for goldens. With `absent` (conformance), the grading box
    is also checked for those canary paths (`LocalGateway.expect_absent`)."""
    from reliquary_sandbox.episode_client import EpisodeClient

    task = await asyncio.to_thread(resolve_task, gateway, env, split, index)
    session_id = uuid.uuid4().hex
    token = gateway.issue(env=env, split=split, index=index, image=task.image,
                          budgets=budgets or default_budgets(task.limits),
                          session_id=session_id)
    records: list[dict[str, Any]] = []
    with gateway.record_retries(session_id) as retries, \
            gateway.expect_absent(session_id, absent) if absent else nullcontext():
        async with EpisodeClient(gateway.url) as client:
            episode = await client.open(token)
            for turn, (tool, arguments) in enumerate(calls):
                result = await episode.call(turn, 0, tool, arguments)
                records.append(result.record["body"])
                if result.final is not None:
                    break
            finished = await episode.finish()
            transcript = (await episode.transcript())["transcript"]
    return ScriptedEpisode(finished.record["body"], records, finished.state, transcript,
                           gateway.verify(transcript, require_graded=False), dict(retries))
