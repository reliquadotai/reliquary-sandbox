"""fake_episode_env's task with controlled egress: index 1 online while grading only,
index 2 online in both phases."""

from episode_fixtures import IMAGE
from fake_episode_env import extract, grade, prepare

from reliquary_sandbox.episode_task import SandboxTask, TaskNetwork


def sandbox_task(split: str, index: int) -> SandboxTask:
    network = TaskNetwork("packages", agent=index == 2)
    return SandboxTask(image=IMAGE, workdir="/work", tools=("bash", "edit"), prepare=prepare,
                       extract=extract, grade=grade, env={"TASK": "1"}, network=network)
