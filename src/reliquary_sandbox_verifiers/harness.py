"""verifiers' bash harness whose tools can only run through the episode gateway.

The program is the pinned upstream one (verifiers b2e4e81) with `run_bash` and
`run_edit` replaced by functions that exit. Every tool call is answered by the task's
interception hook (`task.route_request`) before the program would run it. If a call
ever reached local execution (an "allow" from the hook), the program stops instead of
running an agent's command on the miner's host. Same patch technique, same pin check,
as reliquary_terminal.harness.

The patch relies on how the pinned harness launches the program, so importing this
module refuses any other verifiers: the installed distribution must come from the
pinned commit (its `direct_url.json`) and the bash harness module must be byte-identical
to the pinned one.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import verifiers.v1.harnesses.bash.harness as _upstream_harness
from verifiers.v1.harnesses.bash import BashHarness
from verifiers.v1.harnesses.bash.harness import PROGRAM_SOURCE as UPSTREAM_PROGRAM

VERIFIERS_COMMIT = "b2e4e8157783b2c0dffc7821044c87f29f1c3ccf"
UPSTREAM_HARNESS_SHA256 = "6a06f21c33a34756e3c00ddbc4f5eaa9c3fd40cf9a2e54fca81a2fd4405fc33d"
UPSTREAM_PROGRAM_SHA256 = "5ddf3f27512ed9519b29c286756bb74fa3dcbd1d4a558fae36b4111e11d94fdb"


def installed_direct_url() -> str | None:
    try:
        return importlib.metadata.distribution("verifiers").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None


def installed_harness() -> bytes:
    return Path(_upstream_harness.__file__).read_bytes()


def require_pinned_verifiers(direct_url: str | None, harness_source: bytes) -> None:
    """Raise unless verifiers was installed from the pinned commit and its bash harness
    module is the pinned one."""
    try:
        commit = json.loads(direct_url or "")["vcs_info"]["commit_id"]
    except (ValueError, KeyError, TypeError):
        commit = None
    if commit != VERIFIERS_COMMIT:
        raise RuntimeError(f"reliquary_sandbox_verifiers needs verifiers at {VERIFIERS_COMMIT} "
                           f"(installed from git); found {commit or 'no pinned commit'}")
    if hashlib.sha256(harness_source).hexdigest() != UPSTREAM_HARNESS_SHA256:
        raise RuntimeError("verifiers' bash harness module changed; re-check "
                           "reliquary_sandbox_verifiers.harness before raising the pin")


require_pinned_verifiers(installed_direct_url(), installed_harness())
_MAIN = '\n\nif __name__ == "__main__":\n    asyncio.run(main())\n'
_NO_LOCAL_TOOLS = '''

def run_bash(command):
    raise SystemExit("reliquary-sandbox: a bash call reached local execution; refused")


def run_edit(path, old_str, new_str):
    raise SystemExit("reliquary-sandbox: an edit call reached local execution; refused")
'''


def program_source() -> str:
    digest = hashlib.sha256(UPSTREAM_PROGRAM.encode()).hexdigest()
    if digest != UPSTREAM_PROGRAM_SHA256 or not UPSTREAM_PROGRAM.endswith(_MAIN):
        raise RuntimeError("verifiers' bash program changed shape; re-check "
                           "reliquary_sandbox_verifiers.harness before raising the pin")
    return UPSTREAM_PROGRAM[: -len(_MAIN)] + _NO_LOCAL_TOOLS + _MAIN


class _ProgramSwap:
    """The runtime, preparing our program where the harness prepares upstream's."""

    def __init__(self, runtime, source: str) -> None:
        self._runtime = runtime
        self._source = source

    async def prepare_uv_script(self, script, env=None, *, activate=True):
        if script != UPSTREAM_PROGRAM:
            raise RuntimeError("the bash harness prepared a script other than the pinned program")
        return await self._runtime.prepare_uv_script(self._source, env, activate=activate)

    def __getattr__(self, name):
        return getattr(self._runtime, name)


class SandboxBashHarness(BashHarness):
    """`bash` (and `edit`) answered by the episode gateway, never run here."""

    async def setup(self, runtime) -> None:
        await runtime.prepare_uv_script(program_source(), self.config.resolved_env)

    async def launch(self, ctx, trace, runtime, *args, **kwargs):
        return await super().launch(ctx, trace, _ProgramSwap(runtime, program_source()),
                                    *args, **kwargs)
