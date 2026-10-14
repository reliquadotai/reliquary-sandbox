"""A verifiers v1 `Runtime` over one of our signed-episode boxes.

The bridge serves a verifiers Task's hooks on this runtime: `Task.setup` (agent box, at
open), `Task.finalize` (agent box, at finish), the optional `Task.grading_setup` and the
task's `@vf.reward` / `@vf.metric` functions through `Task.score` (pristine grading box).
Every hook runs in a scoring process whose calls the bridge serves here. It delegates every member to the
sandbox's `TaskRuntime` (`LocalRuntime` on a gateway), which bounds each call by the step
deadline and records the faults the grader classifies. Agent tool calls never go through
it: they are the gateway's signed calls.

Faults are recorded by the inner runtime (`inner.faults`), which the bridge's verdict
reads: a decisive fault of the box is classified by the grader whatever the env's code
did with it, while a failure of the env's own scoring code once the agent's state is
restored is graded 0 `grading_failed` (never aborted: the agent's bytes can cause it).

`stop_after_run=True` (grading boxes): after each `run` returns, every process left in
the box is SIGKILLed before the result is handed back, so nothing a command started
(the agent's code under test) can still rewrite what the reward code reads next.

Calls are serialised: verifiers runs a task's rewards concurrently (`asyncio.gather`),
so one lock is held over every delegated call, over a `run` and its stop, and over
`alive` (and its stop). No call starts while another reward's command, or the stop
after it, is in flight: a stop never kills another reward's command, and a read never
races code a command left running.
"""

from __future__ import annotations

import asyncio
from typing import Any

from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.base import BaseRuntimeInfo
from verifiers.v1.runtimes.base import ProgramResult as VerifiersProgramResult

__all__ = ["SandboxRuntime"]


class SandboxRuntime(Runtime):
    is_local = True

    def __init__(self, inner: Any, *, stop_after_run: bool = False) -> None:
        super().__init__(name=getattr(inner, "name", None))
        self.inner = inner
        self.env = inner.env  # the same dict: the task's environment, applied by `inner.run`
        self.stop_after_run = stop_after_run
        self.info = BaseRuntimeInfo(id=self.name)
        self._serial = asyncio.Lock()

    @property
    def config(self) -> Any:
        """`.type` ("reliquary-sandbox") and `.workdir`, as verifiers' helpers read them."""
        return self.inner.config

    @property
    def supports_live_processes(self) -> bool:
        return False

    async def _delegate(self, call: Any) -> Any:
        """Await one inner call, holding the lock (see the module docstring)."""
        async with self._serial:
            return await call

    async def _stop_after(self) -> None:
        """The stop after a command; the caller holds the lock."""
        if self.stop_after_run:
            await self.inner.stop_processes()

    # -- lifecycle: the gateway owns the box -----------------------------------

    async def start(self) -> None:
        """The box is already running (the gateway created it)."""

    def cleanup(self) -> None:
        """The gateway removes the box."""

    # -- the verifiers Runtime surface -----------------------------------------

    async def run(self, argv: list[str], env: dict[str, str]) -> VerifiersProgramResult:
        async with self._serial:  # the command and the stop after it, as one call
            try:
                result = await self.inner.run(list(argv), dict(env))
            except BaseException:
                try:  # best effort; the original exception is the one kept
                    await self._stop_after()
                except Exception:
                    pass
                raise
            await self._stop_after()
        return VerifiersProgramResult(result.exit_code, result.stdout, result.stderr)

    async def run_program(self, argv: list[str], env: dict[str, str]) -> VerifiersProgramResult:
        return await self.run(argv, env)

    async def read(self, path: str, max_bytes: int | None = None) -> bytes:
        return await self._delegate(self.inner.read(path, max_bytes))

    async def _read(self, path: str) -> bytes:
        return await self.read(path)

    async def write(self, path: str, data: bytes) -> None:
        await self._delegate(self.inner.write(path, data))

    async def alive(self) -> bool:
        """The inner probe runs a command: serialised and followed by the stop like `run`."""
        async with self._serial:
            try:
                return bool(await self.inner.alive())
            except Exception:
                return False
            finally:
                try:
                    await self._stop_after()
                except Exception:
                    pass

    async def prepare_setup(self) -> None:
        await self._delegate(self.inner.prepare_setup())

    async def prepare_execution(self, routes: list[str] | None) -> None:
        await self._delegate(self.inner.prepare_execution(routes))

    # Members our boxes do not offer: the inner runtime raises (an integration bug, aborted).

    async def prepare_uv_script(self, script: Any, env: Any = None, *, activate: bool = True):
        return await self._delegate(self.inner.prepare_uv_script(script, env, activate=activate))

    async def run_uv_script(self, script: Any, args: Any = None, env: Any = None):
        return await self._delegate(self.inner.run_uv_script(script, args, env))

    async def open_process(self, argv: list[str], env: dict[str, str]):
        return await self._delegate(self.inner.open_process(argv, env))

    async def run_background(self, argv: list[str], env: dict[str, str], log: str) -> None:
        return await self._delegate(self.inner.run_background(argv, env, log))

    # -- our safe state primitives (see reliquary_sandbox.episode_task) ---------

    async def archive(self, paths: list[str], max_bytes: int) -> bytes:
        return await self._delegate(self.inner.archive(list(paths), max_bytes))

    async def restore_archive(self, data: bytes, roots: list[str], *,
                              contained_links: bool = True) -> None:
        """Always with `contained_links`: the env's own commands (a `test.sh`, `cat`)
        follow links, so no restored link may lead outside the restored roots."""
        await self._delegate(self.inner.restore_archive(data, list(roots), contained_links=True))

    async def exists(self, path: str) -> bool:
        """Whether `path` exists or is a symlink, checked by the sandbox's helper."""
        return bool(await self._delegate(self.inner.exists(path)))

    async def stop_processes(self) -> None:
        await self._delegate(self.inner.stop_processes())
