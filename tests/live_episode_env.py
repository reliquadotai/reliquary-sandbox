"""A real-image env for live gateway tests: the answer is right when
/root/answer.txt holds "42". The image is any digest-pinned image with bash
and coreutils and no Python (ubuntu:22.04 by default), set in
EPISODE_LIVE_IMAGE by the live tests before the env is resolved."""

import os

from reliquary_sandbox.episode_task import GradeResult, SandboxTask


async def prepare(runtime):
    # The image's own sh: preparation runs before any agent code.
    result = await runtime.run(["/bin/sh", "-c", "rm -f /root/answer.txt && echo prepared"], {})
    if result.exit_code:
        raise RuntimeError(result.stderr)
    return result.stdout


async def extract(runtime):
    # A missing answer is the agent's outcome (graded 0.0), never an exception.
    try:
        return await runtime.read("/root/answer.txt", max_bytes=4096)
    except FileNotFoundError:
        return b""


async def grade(runtime, state):
    await runtime.write("/root/answer.txt", state)
    result = await runtime.run(["/bin/sh", "-c", "grep -qx 42 /root/answer.txt"], {})
    return GradeResult(reward=1.0 if result.exit_code == 0 else 0.0,
                       facts={"grep_exit": result.exit_code})


def sandbox_task(split: str, index: int) -> SandboxTask:
    return SandboxTask(image=os.environ["EPISODE_LIVE_IMAGE"], workdir="/root",
                       tools=("bash", "edit"), prepare=prepare, extract=extract, grade=grade,
                       grading_timeout_s=120, publish_state=True)
