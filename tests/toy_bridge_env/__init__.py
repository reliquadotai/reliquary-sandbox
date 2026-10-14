"""A toy verifiers Taskset in the env norm's shape, for the bridge and conformance tests.

The task asks for "42" in /work/answer.txt. `setup` writes the question; `finalize`
copies the answer into the declared artifact /work/out (index 1's finalize writes
nothing, so its required artifact is missing unless the agent made it); the reward reads
only the restored artifact. No Reliquary import, no network, only `runtime.read` /
`runtime.write`, so it runs on FakeBoxes and on a real box alike. Its
`conformance_cases` speak FakeBoxes' command language (`write <path> <text>`); the live
tests declare real-shell cases for the same taskset (tests/bridge_live_envs.py).
"""

import verifiers.v1 as vf

__all__ = ["ToyTaskset"]

IMAGE_TAG = "toy/task:1"


class ToyData(vf.TaskData):
    finalize_copies: bool = True


class ToyTask(vf.Task[ToyData]):
    async def setup(self, runtime) -> None:
        await runtime.write("/work/question.txt", b"what is six times seven?\n")

    async def finalize(self, runtime) -> None:
        if not self.data.finalize_copies:
            return
        try:
            answer = await runtime.read("/work/answer.txt", max_bytes=64)
        except OSError:
            answer = b""
        await runtime.write("/work/out/answer.txt", answer)

    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        try:
            answer = await runtime.read("/work/out/answer.txt", max_bytes=64)
        except OSError:
            return 0.0
        return 1.0 if answer.strip() == b"42" else 0.0


class ToyConfig(vf.TasksetConfig):
    split: str = "rl"


class ToyTaskset(vf.Taskset[ToyTask, ToyConfig]):
    def load(self):
        if self.config.split != "rl":
            raise ValueError(f"unknown split {self.config.split!r}")
        for idx, copies in enumerate((True, False)):
            yield ToyTask(ToyData(idx=idx, prompt="Write 42 to /work/answer.txt.",
                                  image=IMAGE_TAG, workdir="/work",
                                  artifacts=[vf.Artifact(source="out", required=True)],
                                  finalize_copies=copies),
                          self.config.task)


def conformance_cases(split):
    return [
        {"name": "reference", "index": 0, "expect": 1.0,
         "calls": [["bash", {"command": "write /work/answer.txt 42"}]]},
        {"name": "wrong_answer", "index": 0, "expect": 0.0,
         "calls": [["bash", {"command": "write /work/answer.txt 41"}]]},
        {"name": "answer_in_the_wrong_file", "index": 0, "expect": 0.0,
         "calls": [["bash", {"command": "write /work/answer.bak 42"}]]},
    ]


def reference_calls(split, index):
    """The toy's reference solution (FakeBoxes' command language), task 0 only."""
    if split == "rl" and index == 0:
        return [["bash", {"command": "write /work/answer.txt 42"}]]
    return None
