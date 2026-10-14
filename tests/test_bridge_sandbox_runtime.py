"""SandboxRuntime: the verifiers v1 Runtime env hooks see, over a sandbox TaskRuntime."""

import asyncio
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("verifiers.v1")

from episode_fixtures import IMAGE  # noqa: E402
from fake_boxes import FakeBoxes  # noqa: E402
from verifiers.v1.runtimes import Runtime  # noqa: E402
from verifiers.v1.runtimes.base import ProgramResult as VerifiersProgramResult  # noqa: E402

from reliquary_sandbox.episode_task import ProgramResult  # noqa: E402
from reliquary_sandbox_service.episodes.local_runtime import (  # noqa: E402
    FileTooLarge,
    LocalRuntime,
    UnsupportedRuntimeMember,
)
from reliquary_sandbox_service.task_runtime import BoxSpec  # noqa: E402
from reliquary_sandbox_verifiers.sandbox_runtime import SandboxRuntime  # noqa: E402


class Inner:
    """A scripted TaskRuntime: records what it was asked, answers from `files`."""

    def __init__(self):
        self.env = {"TASK": "1"}
        self.config = SimpleNamespace(type="reliquary-sandbox", workdir="/work")
        self.name = "box-1"
        self.calls = []
        self.files = {"/work/a": b"hello"}

    async def run(self, argv, env):
        self.calls.append(("run", argv, env))
        if argv == ["boom"]:
            raise RuntimeError("box unusable")
        return ProgramResult(0, "out", "err")

    async def read(self, path, max_bytes=None):
        self.calls.append(("read", path, max_bytes))
        return self.files[path]

    async def write(self, path, data):
        self.calls.append(("write", path, data))
        self.files[path] = data

    async def alive(self):
        return True

    async def prepare_setup(self):
        self.calls.append(("prepare_setup",))

    async def prepare_execution(self, routes):
        self.calls.append(("prepare_execution", routes))

    async def archive(self, paths, max_bytes):
        self.calls.append(("archive", paths, max_bytes))
        return b"tar"

    async def restore_archive(self, data, roots, *, contained_links=False):
        self.calls.append(("restore_archive", data, roots, contained_links))

    async def stop_processes(self):
        self.calls.append(("stop_processes",))

    async def run_uv_script(self, script, args=None, env=None):
        raise NotImplementedError("no uv here")


async def test_it_is_a_verifiers_runtime_that_delegates_to_the_box():
    inner = Inner()
    box = SandboxRuntime(inner)
    assert isinstance(box, Runtime)
    assert box.type == "reliquary-sandbox" and box.config.workdir == "/work"
    assert box.name == "box-1" and box.info.id == "box-1"
    assert box.env is inner.env and box.supports_live_processes is False
    result = await box.run(["echo", "hi"], {"X": "1"})
    assert isinstance(result, VerifiersProgramResult)
    assert (result.exit_code, result.stdout, result.stderr) == (0, "out", "err")
    assert await box.read("/work/a", max_bytes=10) == b"hello"
    assert await box.read("/work/a") == b"hello"  # verifiers' framework read, unbounded here
    await box.write("/work/b", b"x")
    assert await box.alive() is True
    await box.prepare_setup()
    await box.prepare_execution([])
    assert await box.archive(["/work"], 99) == b"tar"
    await box.restore_archive(b"tar", ["/work"])
    await box.stop_processes()
    assert [c[0] for c in inner.calls] == [
        "run", "read", "read", "write", "prepare_setup", "prepare_execution", "archive",
        "restore_archive", "stop_processes"]
    assert inner.calls[1] == ("read", "/work/a", 10) and inner.calls[2] == ("read", "/work/a", None)
    # the env's commands follow links: the bridge never restores a link leading out
    assert inner.calls[7] == ("restore_archive", b"tar", ["/work"], True)
    await box.start()  # the gateway owns the box: nothing to do
    box.cleanup()


async def test_exceptions_of_the_inner_runtime_reach_the_caller_unchanged():
    box = SandboxRuntime(Inner())
    with pytest.raises(RuntimeError):
        await box.run(["boom"], {})
    with pytest.raises(NotImplementedError):
        await box.run_uv_script("print(1)")
    assert not hasattr(box, "raised") and not hasattr(box, "raised_here")


async def test_a_grading_runtime_stops_every_process_after_each_command():
    inner = Inner()
    box = SandboxRuntime(inner, stop_after_run=True)
    await box.run(["bash", "/tests/test.sh"], {})
    await box.read("/work/a", max_bytes=5)
    assert [c[0] for c in inner.calls] == ["run", "stop_processes", "read"]


async def test_over_a_local_runtime_faults_stay_visible_to_the_grader():
    boxes = FakeBoxes([IMAGE])
    boxes.create("s-1", "s-1", BoxSpec(image=IMAGE, workdir="/work", env={}, user=None,
                                       memory_bytes=1, nano_cpus=1, pids=16, disk_bytes=1,
                                       lifetime_s=60))
    inner = LocalRuntime(boxes, "s-1", workdir="/work", env={}, user="0:0",
                         deadline=time.monotonic() + 10, output_limit=1024)
    box = SandboxRuntime(inner)
    await box.write("/work/big", b"x" * 100)
    with pytest.raises(FileTooLarge) as too_large:
        await box.read("/work/big", max_bytes=10)
    with pytest.raises(UnsupportedRuntimeMember) as unsupported:
        await box.open_process(["sleep", "1"], {})
    assert inner.faults == [too_large.value, unsupported.value]


def test_stop_processes_also_runs_when_the_command_raised():
    import asyncio

    inner = Inner()
    stopped = []

    async def stop_processes():
        stopped.append(True)

    inner.stop_processes = stop_processes
    runtime = SandboxRuntime(inner, stop_after_run=True)
    with pytest.raises(RuntimeError, match="box unusable"):
        asyncio.run(runtime.run(["boom"], {}))
    assert stopped == [True]

    async def failing_stop():
        raise OSError("stop failed too")

    inner.stop_processes = failing_stop
    with pytest.raises(RuntimeError, match="box unusable"):  # the original one is kept
        asyncio.run(runtime.run(["boom"], {}))


class SlowInner(Inner):
    """Every call yields to the loop in the middle, so unserialised calls interleave."""

    async def run(self, argv, env):
        self.calls.append(("run-start", argv[0]))
        await asyncio.sleep(0.01)
        self.calls.append(("run-end", argv[0]))
        return ProgramResult(0, argv[0], "")

    async def read(self, path, max_bytes=None):
        self.calls.append(("read-start", path))
        await asyncio.sleep(0.01)
        self.calls.append(("read-end", path))
        return b"x"

    async def alive(self):
        self.calls.append(("alive",))
        return True

    async def stop_processes(self):
        self.calls.append(("stop_processes",))
        await asyncio.sleep(0.01)


async def test_concurrent_calls_are_serialised_and_a_run_keeps_its_stop():
    inner = SlowInner()
    box = SandboxRuntime(inner, stop_after_run=True)
    results = await asyncio.gather(box.run(["a"], {}), box.read("/r"), box.run(["b"], {}),
                                   box.alive())
    assert [r.stdout for r in (results[0], results[2])] == ["a", "b"]
    assert inner.calls == [
        ("run-start", "a"), ("run-end", "a"), ("stop_processes",),
        ("read-start", "/r"), ("read-end", "/r"),
        ("run-start", "b"), ("run-end", "b"), ("stop_processes",),
        ("alive",), ("stop_processes",)]


async def test_exists_is_delegated_to_the_helper_check():
    boxes = FakeBoxes([IMAGE])
    boxes.create("s-1", "s-1", BoxSpec(image=IMAGE, workdir="/work", env={}, user=None,
                                       memory_bytes=1, nano_cpus=1, pids=16, disk_bytes=1,
                                       lifetime_s=60))
    boxes.boxes["s-1"]["files"]["/work/out/a"] = b"1"
    boxes.boxes["s-1"]["links"] = {"/work/dangling": "/nowhere"}
    inner = LocalRuntime(boxes, "s-1", workdir="/work", env={}, user="0:0",
                         deadline=time.monotonic() + 10, output_limit=1024)
    box = SandboxRuntime(inner)
    assert await box.exists("/work/out") and await box.exists("out/a")
    assert await box.exists("/work/dangling") and not await box.exists("/work/none")
    assert not [argv for _, argv, *_ in boxes.exec_log]  # the image's `test` never runs
