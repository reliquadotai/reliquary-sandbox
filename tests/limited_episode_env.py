"""fake_episode_env's task with per-task limits: index 7 wants 0.5 CPU and 3 GiB,
index 8 wants 8 CPUs (more than any test machine gives a box), index 9 an absurd
1e300 CPUs (whose nano-CPU count overflows to infinity)."""

from episode_fixtures import IMAGE
from fake_episode_env import extract, grade, prepare

from reliquary_sandbox.episode_task import SandboxTask, TaskLimits


def sandbox_task(split: str, index: int) -> SandboxTask:
    cpus = {7: 0.5, 8: 8.0, 9: 1e300}[index]
    return SandboxTask(image=IMAGE, workdir="/work", tools=("bash", "edit"), prepare=prepare,
                       extract=extract, grade=grade, publish_state=True,
                       limits=TaskLimits(cpus=cpus, memory_bytes=3 * 1024**3))
