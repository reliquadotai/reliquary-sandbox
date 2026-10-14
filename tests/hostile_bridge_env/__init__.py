"""A verifiers Taskset whose tasks break the env norm in known ways (or push the bridge's
limits), for the conformance suite's tests on FakeBoxes.

Like the toy env: the answer "42" goes to /work/answer.txt, `finalize` copies it into the
artifact /work/out. Each index's reward differs (`HostileData.mode`):

0 `conversation`: grades the conversation (`trace.last_reply`), never the state;
1 `large_write`: stages 40 MiB in the grading box (as Harbor's `_stage_tests` does), then
  grades the artifact;
2 `large_output`: runs a command with 1 MiB of binary output on both streams, then
  grades the artifact;
3 `strict_json`: parses the artifact as JSON without guarding it (raises on malformed
  bytes);
4 `counter`: `finalize` also writes how many episodes this process finalized; the
  reward needs it odd (not deterministic if finalize ran in a long-lived process; each
  hook runs in a fresh scoring process, so it is always 1);
5 `strict_finalize`: `finalize` parses the agent's answer as JSON without guarding it
  (raises on malformed bytes: extract_failed);
6 `nan_reward`: the reward is NaN when the artifact is not an integer;
7 `agent_box_verdict`: `finalize` runs the check in the agent's box and the reward
  trusts its verdict (a shimmed tool passes it);
8 `shared_state`: `setup` counts the episodes open on the (cached, shared) Task
  instance, `finalize` writes whether it was alone; the reward needs it alone. Each hook
  runs in its own scoring process, so what setup kept on the Task is gone at finalize
  (AttributeError: extract_failed);
9 `always_one`: grades 1 whatever the box holds (no case references it: only the
  split's sample finds it);
10 `hang_finalize`: `finalize` waits for ever when the answer is not a number (the task
  declares a 45 s finalize (the scoring process needs 30 s left) timeout: extract_timeout);
11 `agent_box_tool`: `finalize` runs an image tool in the agent's box and reads the file it
  made (Harbor's `collect` tars there): a shimmed tool makes no file, extract_failed.
"""

import asyncio
import json

import verifiers.v1 as vf

__all__ = ["HostileTaskset"]

IMAGE_TAG = "toy/task:1"
MODES = ("conversation", "large_write", "large_output", "strict_json", "counter",
         "strict_finalize", "nan_reward", "agent_box_verdict", "shared_state", "always_one", "hang_finalize",
         "agent_box_tool")
UNREFERENCED = ("always_one",)
FINALIZED = [0]


class HostileData(vf.TaskData):
    mode: str = "conversation"


class HostileTask(vf.Task[HostileData]):
    async def setup(self, runtime) -> None:
        if self.data.mode == "shared_state":
            self.open_episodes = getattr(self, "open_episodes", 0) + 1

    async def finalize(self, runtime) -> None:
        try:
            answer = await runtime.read("/work/answer.txt", max_bytes=64)
        except OSError:
            answer = b""
        await runtime.write("/work/out/answer.txt", answer)
        mode = self.data.mode
        if mode == "hang_finalize" and not answer.strip().isdigit():
            await asyncio.sleep(3600)
        if mode == "strict_finalize":
            json.loads(answer)
        if mode == "agent_box_verdict":
            result = await runtime.run(["check"], {})
            await runtime.write("/work/out/verdict.txt", b"pass" if result.exit_code == 0
                                else b"fail")
        if mode == "agent_box_tool":
            await runtime.run(["prepare"], {})  # FakeBoxes: writes /work/prepared
            await runtime.read("/work/prepared", max_bytes=8)  # raises when it did not
        if mode == "shared_state":
            alone = self.open_episodes == 1
            self.open_episodes -= 1
            await runtime.write("/work/out/alone.txt", b"1" if alone else b"0")
        if mode == "counter":
            FINALIZED[0] += 1
            await runtime.write("/work/out/episode.txt", str(FINALIZED[0]).encode())

    @vf.reward(weight=1.0)
    async def solved(self, runtime, trace) -> float:
        mode = self.data.mode
        if mode == "conversation":
            return 1.0 if "42" in trace.last_reply else 0.0
        if mode == "always_one":
            return 1.0
        if mode == "agent_box_verdict":
            return 1.0 if await runtime.read("/work/out/verdict.txt", max_bytes=8) == b"pass" else 0.0
        if mode == "large_write":
            await runtime.write("/tmp/tests.tar", b"t" * (40 * 1024**2))
        if mode == "large_output":
            result = await runtime.run(["emit", "ff", str(1 << 20)], {})
            if len(result.stdout) != 1 << 20:
                return 0.0
        answer = await runtime.read("/work/out/answer.txt", max_bytes=64)
        if mode == "strict_json":
            return 1.0 if json.loads(answer) == 42 else 0.0
        if mode == "nan_reward":
            try:
                return 1.0 if int(answer) == 42 else 0.0
            except ValueError:
                return float("nan")
        if mode == "shared_state" and await runtime.read("/work/out/alone.txt",
                                                         max_bytes=8) != b"1":
            return 0.0
        if mode == "counter":
            episode = int(await runtime.read("/work/out/episode.txt", max_bytes=64))
            if episode % 2 == 0:
                return 0.0
        return 1.0 if answer.strip() == b"42" else 0.0


class HostileConfig(vf.TasksetConfig):
    split: str = "rl"


class HostileTaskset(vf.Taskset[HostileTask, HostileConfig]):
    def task_at(self, index):
        """Built once and shared, as a taskset caching its tasks does."""
        if not hasattr(self, "_tasks"):
            self._tasks = list(self.load())
        return self._tasks[index]

    def load(self):
        for idx, mode in enumerate(MODES):
            yield HostileTask(HostileData(idx=idx, prompt="Write 42 to /work/answer.txt.",
                                          image=IMAGE_TAG, workdir="/work",
                                          artifacts=[vf.Artifact(source="out", required=True)],
                                          mode=mode,
                                          timeout=vf.TaskTimeout(finalize=45)
                                          if mode == "hang_finalize" else vf.TaskTimeout()),
                              self.config.task)


def conformance_cases(split):
    solve = [["bash", {"command": "write /work/answer.txt 42"}]]
    wrong = [["bash", {"command": "write /work/answer.txt 41"}]]
    cases = []
    for index, mode in enumerate(MODES):
        if mode in UNREFERENCED:
            continue
        cases.append({"name": f"{mode}_reference", "index": index, "expect": 1.0,
                      "calls": solve})
        cases.append({"name": f"{mode}_wrong", "index": index, "expect": 0.0,
                      "calls": wrong})
    return cases
