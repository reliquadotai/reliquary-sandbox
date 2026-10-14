"""Every env hook runs in a scoring process, like the rewards: the controller never runs env code on bytes the agent wrote."""

import os
import subprocess
import sys
import time

import pytest

pytest.importorskip("verifiers.v1")

from episode_fixtures import IMAGE  # noqa: E402
from fake_boxes import FakeBoxes  # noqa: E402
from toy_bridge_env import ToyData, ToyTask  # noqa: E402

import reliquary_sandbox_verifiers.scoring_process as scoring_process  # noqa: E402
from reliquary_sandbox_service import scoring_server  # noqa: E402
from reliquary_sandbox_service.episodes.local_runtime import LocalRuntime  # noqa: E402
from reliquary_sandbox_service.task_runtime import BoxSpec  # noqa: E402
from reliquary_sandbox_verifiers.bridge import grading_trace  # noqa: E402
from reliquary_sandbox_verifiers.sandbox_runtime import SandboxRuntime  # noqa: E402
from reliquary_sandbox_verifiers.scoring_process import (  # noqa: E402
    HOOKS,
    run_in_child,
    score_in_child,
)


class WritesWhereItRuns(ToyTask):
    async def setup(self, runtime):
        secret = os.environ.get("RQ_PLAN2B_SECRET", "")
        await runtime.write("/work/setup.pid", f"{os.getpid()} {secret}".encode())

    async def grading_setup(self, runtime):
        await runtime.write("/work/grading_setup.pid", str(os.getpid()).encode())

    async def finalize(self, runtime):
        await runtime.write("/work/finalize.pid", str(os.getpid()).encode())


class FinalizeRaises(ToyTask):
    async def finalize(self, runtime):
        raise ValueError("finalize could not parse the agent's output")


class FinalizeSpins(ToyTask):
    async def finalize(self, runtime):
        while True:  # never awaits: holds the GIL
            pass


@pytest.fixture(autouse=True)
def _no_startup_floor(monkeypatch):
    monkeypatch.setattr(scoring_process, "SCORING_STARTUP_FLOOR_S", 0.0)


def a_box():
    boxes = FakeBoxes([IMAGE])
    boxes.create("s-1", "s-1", BoxSpec(image=IMAGE, workdir="/work", env={}, user=None,
                                       memory_bytes=1, nano_cpus=1, pids=16, disk_bytes=1,
                                       lifetime_s=60))
    inner = LocalRuntime(boxes, "s-1", workdir="/work", env={}, user="0:0",
                         deadline=time.monotonic() + 30, output_limit=1024)
    return boxes, SandboxRuntime(inner)


def a_task(cls=WritesWhereItRuns):
    return cls(ToyData(idx=0, prompt="Write 42.", image="toy/task:1", workdir="/work"))


async def run(hook, cls=WritesWhereItRuns, seconds=20.0):
    boxes, box = a_box()
    outcome = await run_in_child(a_task(cls), box, hook=hook, make_trace=grading_trace,
                                 deadline=time.monotonic() + seconds,
                                 memory_bytes=2 * 1024**3, max_message_bytes=1 << 22)
    return boxes, outcome


@pytest.mark.parametrize("hook", ["setup", "grading_setup", "finalize"])
async def test_a_hook_runs_in_its_own_process_and_reaches_the_box(hook, monkeypatch):
    monkeypatch.setenv("RQ_PLAN2B_SECRET", "a-service-secret")
    scoring_server.stop()  # the next forkserver starts with the secret in our environment
    try:
        boxes, outcome = await run(hook)
    finally:
        scoring_server.stop()
    assert outcome.kind == "done", outcome.detail
    written = boxes.boxes["s-1"]["files"][f"/work/{hook}.pid"].decode().split(" ")
    assert int(written[0]) != os.getpid()
    if hook == "setup":
        assert written[1:] == [""]  # the scoring environment carries nothing of the service's


async def test_score_still_returns_its_summary():
    _, outcome = await run("score")
    assert outcome.kind == "scored"
    assert outcome.summary["rewards"] == {"solved": 0.0}  # no answer in the box


async def test_a_hook_that_raises_is_raised():
    _, outcome = await run("finalize", FinalizeRaises)
    assert outcome.kind == "raised" and "could not parse" in outcome.detail


async def test_a_hook_past_its_deadline_died():
    started = time.monotonic()
    _, outcome = await run("finalize", FinalizeSpins, seconds=2.0)
    assert outcome.kind == "died"
    assert time.monotonic() - started < 15


async def test_an_unknown_hook_is_refused_before_any_process_starts():
    _, box = a_box()
    with pytest.raises(ValueError, match="hook"):
        await run_in_child(a_task(), box, hook="collect", make_trace=grading_trace,
                           deadline=time.monotonic() + 5, memory_bytes=1 << 30,
                           max_message_bytes=1 << 20)


def test_the_hooks_and_the_score_wrapper():
    assert HOOKS == ("setup", "grading_setup", "finalize", "score")
    assert "run_in_child" in (score_in_child.__doc__ or "")


def test_a_result_of_the_other_hooks_kind_is_broken():
    scored = {"result": "scored", "rewards": {"solved": 1.0}, "metrics": {}, "reward": 1.0,
              "reward_repr": "1.0"}
    assert scoring_process._finished({"result": "done"}, {}, "done").kind == "done"
    assert scoring_process._finished(scored, {}, "scored").kind == "scored"
    assert scoring_process._finished({"result": "done"}, {}, "scored").kind == "broken"
    assert scoring_process._finished(scored, {}, "done").kind == "broken"


class ReportsItsHashSeed(ToyTask):
    async def finalize(self, runtime):
        seed = os.environ.get("PYTHONHASHSEED", "")
        await runtime.write("/work/hash.txt", f"{seed} {hash('reliquary')}".encode())


async def test_a_scoring_process_hashes_deterministically():
    scoring_server.stop()  # a forkserver started before the wrapper's change would differ
    try:
        boxes, outcome = await run("finalize", ReportsItsHashSeed)
    finally:
        scoring_server.stop()
    assert outcome.kind == "done", outcome.detail
    expected = subprocess.run([sys.executable, "-c", "print(hash('reliquary'))"],
                              env={"PYTHONHASHSEED": "0"}, capture_output=True, text=True,
                              check=True).stdout.strip()
    assert boxes.boxes["s-1"]["files"]["/work/hash.txt"].decode() == f"0 {expected}"


class _Exited(Exception):
    pass


class _DeepConn:
    """A pipe whose send raises RecursionError (the env's code called the runtime at the
    edge of the recursion limit), then works."""

    def __init__(self):
        self.sends = 0

    def send_bytes(self, message):
        self.sends += 1
        if self.sends == 1:
            raise RecursionError("maximum recursion depth exceeded")

    def recv_bytes(self):
        return scoring_process._frame({"value": None})


def test_a_recursion_error_in_a_runtime_call_is_the_envs_not_our_protocol(monkeypatch):
    def exits(code):
        raise _Exited(code)

    monkeypatch.setattr(scoring_process.os, "_exit", exits)
    inner = scoring_process._RemoteInner(_DeepConn(), name="s-1", env={}, config=None,
                                         max_payload_bytes=1 << 20)
    with pytest.raises(RecursionError):
        inner._call("exists", path="/work")
    # The request may be half sent: the pipe can no longer be trusted, so the next call
    # ends the process as a protocol failure rather than read a reply out of step.
    with pytest.raises(_Exited) as exited:
        inner._call("exists", path="/work")
    assert exited.value.args == (scoring_process.EXIT_PROTOCOL,)
