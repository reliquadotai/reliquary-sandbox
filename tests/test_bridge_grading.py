"""A verifiers Task served through the bridge: open, extract and grade with the real
Grader over FakeBoxes (no Docker)."""

import asyncio
import json
import os
import posixpath
import resource
import signal
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("verifiers.v1")

import toy_bridge_env  # noqa: E402
import verifiers.v1 as vf  # noqa: E402
from episode_fixtures import IMAGE  # noqa: E402
from fake_boxes import FakeBoxes  # noqa: E402
from toy_bridge_env import ToyData, ToyTask  # noqa: E402
from verifiers.v1.task import TaskTimeout  # noqa: E402

import reliquary_sandbox_verifiers.declaration as declaration  # noqa: E402
import reliquary_sandbox_verifiers.scoring_process as scoring_process  # noqa: E402
from reliquary_sandbox.episode_task import EnvInfraError  # noqa: E402
from reliquary_sandbox_service import scoring_server  # noqa: E402
from reliquary_sandbox_service.episodes import grader as grader_module  # noqa: E402
from reliquary_sandbox_service.episodes.grader import Grader, run_step  # noqa: E402
from reliquary_sandbox_service.task_runtime import BoxSpec  # noqa: E402
from reliquary_sandbox_verifiers.bridge import (  # noqa: E402
    CLEAR_AND_KEEP_WORKDIR,
    STATE_MAGIC,
    HookFailed,
    clearing_commands,
    decode_state,
    encode_state,
    task_from_verifiers,
)
from reliquary_sandbox_verifiers.declaration import (  # noqa: E402
    ArtifactRoot,
    BridgeDefaults,
    TaskDeclarationError,
    setup_output,
)
from reliquary_sandbox_verifiers.image_lock import ImageLock  # noqa: E402

LOCK = ImageLock.of_module(toy_bridge_env)
SETTINGS = SimpleNamespace(episode_grader_output_bytes=1 << 20, episode_grading_attempts=2)
G0 = Grader.pristine_box_id("s-1", 0)


def toy_data(**fields):
    values = dict(idx=0, prompt="Write 42.", image="toy/task:1", workdir="/work",
                  artifacts=[vf.Artifact(source="out")])
    values.update(fields)
    return ToyData(**values)


class Episode:
    """One agent box, prepared by the bridged task; the agent's work is written directly."""

    def __init__(self, task_class=ToyTask, data=None, **bridge):
        self.verifiers_task = task_class(data or toy_data())
        self.task = task_from_verifiers(self.verifiers_task, lock=LOCK, **bridge)
        self.spec = BoxSpec(image=self.task.image, workdir=self.task.workdir,
                            env=dict(self.task.env), user=None, memory_bytes=1, nano_cpus=1,
                            pids=16, disk_bytes=1, lifetime_s=60)
        self.boxes = FakeBoxes([IMAGE])
        self.boxes.create("s-1", "s-1", self.spec)
        self.grader = Grader(SETTINGS, self.boxes)
        runtime = self.grader.runtime("s-1", self.task, self.task.prepare_timeout_s)
        self.setup_output = run_step(lambda: self.task.prepare(runtime), 10)

    def agent_writes(self, path, data):
        self.boxes.make_dirs("s-1", posixpath.dirname(path))
        self.boxes.write_file("s-1", path, data)

    def grade(self):
        return self.grader.grade("s-1", "s-1", self.task, self.spec, env="toy")

    def grading_calls(self):
        return [(name, kwargs) for name, box, kwargs in self.boxes.helper_calls if box == G0]


def test_the_sandbox_task_is_derived_from_the_verifiers_task():
    episode = Episode()
    task = episode.task
    assert task.image == IMAGE and task.workdir == "/work" and task.tools == ("bash", "edit")
    assert task.env == {} and task.publish_state is False
    assert task.prepare_timeout_s == 600.0 and task.grading_timeout_s == 600.0
    assert task.limits.pids == 1024 and task.limits.per_call_timeout_s == 180
    assert task.max_state_bytes > BridgeDefaults().max_artifact_bytes
    assert task.content_sha256 == episode.verifiers_task.hash
    other = task_from_verifiers(ToyTask(toy_data()), lock=LOCK, tools=("bash",),
                                publish_state=True)
    assert other.tools == ("bash",) and other.publish_state is True


@pytest.mark.parametrize("fields, message", [
    ({"image": None}, "no image"),
    ({"image": "toy/task:2"}, "not pinned"),
    ({"workdir": "work"}, "not absolute"),
])
def test_tasks_the_sandbox_cannot_serve_are_refused(fields, message):
    with pytest.raises(TaskDeclarationError, match=message):
        task_from_verifiers(ToyTask(toy_data(**fields)), lock=LOCK)


def test_a_task_without_a_reward_or_with_a_judge_is_refused():
    class NoReward(vf.Task[ToyData]):
        pass

    with pytest.raises(TaskDeclarationError, match="reward"):
        task_from_verifiers(NoReward(toy_data()), lock=LOCK)
    judged = ToyTask(toy_data(), vf.TaskConfig.model_construct(judges=[object()]))
    with pytest.raises(TaskDeclarationError, match="judges"):
        task_from_verifiers(judged, lock=LOCK)


def test_prepare_runs_the_tasks_setup_and_signs_its_content_hash():
    episode = Episode()
    assert episode.boxes.boxes["s-1"]["files"]["/work/question.txt"].startswith(b"what is")
    assert episode.setup_output == setup_output(episode.verifiers_task)


def test_the_reference_answer_grades_one_in_a_pristine_box():
    episode = Episode()
    episode.agent_writes("/work/answer.txt", b"42\n")
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 1.0)
    assert outcome.facts == {"rewards": {"solved": 1.0}, "metrics": {}}
    assert outcome.state.startswith(STATE_MAGIC)
    assert G0 in episode.boxes.removed


@pytest.mark.parametrize("answer", [b"41\n", None])
def test_a_wrong_or_missing_answer_grades_zero(answer):
    episode = Episode()
    if answer is not None:
        episode.agent_writes("/work/answer.txt", answer)
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)


def test_a_required_artifact_the_agent_removed_grades_zero_without_scoring():
    episode = Episode(data=toy_data(finalize_copies=False))
    episode.agent_writes("/work/answer.txt", b"42\n")
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"missing_artifacts": ["/work/out"]}


class Spy(ToyTask):  # scoring runs in its own process: it asks the box, not the fake
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        listed = (await runtime.run(["ls-files"], {})).stdout
        return 1.0 if listed == "/work/out/answer.txt\n" else 0.0


def test_only_the_artifacts_reach_the_grading_box():
    episode = Episode()
    episode.agent_writes("/work/answer.txt", b"42\n")
    episode.agent_writes("/logs/verifier/reward.txt", b"1\n")

    episode.task = task_from_verifiers(Spy(toy_data()), lock=LOCK)
    assert episode.grade().reward == 1.0
    removed = [argv for box, argv, *_ in episode.boxes.exec_log if box == G0 and argv[0] == "rm"]
    assert removed == [["rm", "-rf", "--", "/logs/artifacts"]]


class GradesInItsWorkdir(vf.Task[ToyData]):
    """Harbor's shape: the workdir is itself an optional artifact root; the reward runs a
    command there (Harbor's test.sh) and reads the workdir."""

    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        if (await runtime.run(["true"], {})).exit_code:
            return 0.0  # the grading box can no longer run anything in its workdir
        try:
            return 1.0 if (await runtime.read("/app/hello.txt", max_bytes=64)) == b"hi" else 0.5
        except OSError:
            return 0.5


def test_an_absent_artifact_root_that_is_the_workdir_leaves_a_usable_workdir():
    data = toy_data(workdir="/app", artifacts=[vf.Artifact(source="/app", required=False)])
    episode = Episode(GradesInItsWorkdir, data)
    episode.boxes.exec("s-1", ["rm", "-rf", "--", "/app"], timeout_s=5, output_limit=1024)
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.5), outcome
    cleared = [argv for box, argv, *_ in episode.boxes.exec_log
               if box == G0 and argv[:1] == ["sh"]]
    assert cleared == [["sh", "-c", CLEAR_AND_KEEP_WORKDIR, "/app", "/app", "/logs/artifacts"]]


def test_absent_roots_outside_the_workdir_are_removed_one_by_one():
    assert clearing_commands(["/work/out", "/logs/artifacts"], "/work") == [
        ["rm", "-rf", "--", "/work/out"], ["rm", "-rf", "--", "/logs/artifacts"]]
    assert clearing_commands(["/"], "/app")[0][:3] == ["sh", "-c", CLEAR_AND_KEEP_WORKDIR]
    assert clearing_commands(["/ap"], "/app")[0] == ["rm", "-rf", "--", "/ap"]
    assert clearing_commands([], "/app") == []


def test_every_grading_command_is_followed_by_a_process_stop():
    episode = Episode()
    episode.agent_writes("/work/answer.txt", b"42\n")
    episode.grade()
    names = [name for name, _ in episode.grading_calls()]
    assert names[names.index("run_program") + 1] == "stop_processes"
    assert names[-1] == "stop_processes"


class Forging(ToyTask):
    async def finalize(self, trace, runtime) -> None:
        trace.state.artifacts = {"/work/out": b"forged tar"}


def test_artifacts_verifiers_collect_left_on_the_trace_are_never_transported():
    episode = Episode(Forging)
    outcome = episode.grade()
    assert outcome.facts == {"missing_artifacts": ["/work/out"]}


class EnvBug(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        raise KeyError("the env's own bug")


class OutOfRange(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        return 5.0


class UsesUv(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        await runtime.run_uv_script("print(1)")
        return 1.0


class Hangs(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        await asyncio.sleep(5)
        return 1.0


def test_a_failure_of_the_envs_scoring_code_grades_zero_and_is_counted():
    episode = Episode(EnvBug)
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "grading_failed"}}
    assert episode.grader.env_failure_counts() == {"toy": {"grading_failed": 1}}
    names = [name for name, _ in episode.grading_calls()]
    assert names[-1] == "stop_processes"  # the stop runs even when scoring raised


def test_a_reward_outside_zero_one_grades_zero_with_a_fact():
    outcome = Episode(OutOfRange).grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts["reward_out_of_range"] == "5.0"


def test_a_runtime_member_the_sandbox_does_not_offer_is_aborted():
    outcome = Episode(UsesUv).grade()
    assert outcome.status == "aborted" and "misuse" in outcome.reason


class Breaks(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        await runtime.run(["die"], {})  # the agent's code under test kills the box
        await runtime.run(["true"], {})
        return 1.0


def test_a_grading_box_the_agents_code_broke_grades_zero():
    episode = Episode()
    episode.agent_writes("/work/answer.txt", b"42\n")

    episode.task = task_from_verifiers(Breaks(toy_data()), lock=LOCK)
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "grading_box_failed"}}


def test_scoring_past_its_timeout_grades_zero():
    # 2 s: the finalize, in its own scoring process, gets the same step timeout.
    episode = Episode(Hangs, data=toy_data(timeout=TaskTimeout(scoring=2.0)))
    started = time.monotonic()
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "grading_timeout"}}
    assert time.monotonic() - started < 5  # Hangs sleeps 5 s


ROOTS = (ArtifactRoot("/work/out", True), ArtifactRoot("/logs/artifacts", False))


def test_the_state_header_round_trips():
    state = encode_state(["/work/out"], [], b"archive")
    assert decode_state(state, ROOTS) == (["/work/out"], [], b"archive")
    assert decode_state(encode_state([], ["/work/out"], b""), ROOTS) == ([], ["/work/out"], b"")


@pytest.mark.parametrize("state", [
    b"no header",
    STATE_MAGIC + b'{"present": []',
    STATE_MAGIC + b"[]\n",
    STATE_MAGIC + b'{"present": ["/etc"], "missing_required": []}\n' + b"x",
    STATE_MAGIC + b'{"present": ["/work/out"], "missing_required": ["/work/out"]}\nx',
    STATE_MAGIC + b'{"present": ["/work/out"], "missing_required": []}\n',
    STATE_MAGIC + b'{"present": [], "missing_required": []}\nunexpected',
])
def test_a_malformed_state_is_our_bug(state):
    with pytest.raises(EnvInfraError):
        decode_state(state, ROOTS)


class Weighted(ToyTask):
    @vf.reward(weight=0.25)
    async def solved(self, runtime) -> float:
        return 1.0

    @vf.reward(weight=0.5)
    async def parts(self, runtime) -> dict[str, float]:
        return {"half": 1.0, "other_half": 0.5}

    @vf.metric
    async def noisy(self, runtime) -> float:
        return float("nan")

    @vf.metric
    async def steps(self, runtime) -> float:
        return 3.0


def test_several_weighted_rewards_sum_and_non_finite_metrics_are_dropped():
    outcome = Episode(Weighted).grade()
    assert (outcome.status, outcome.reward) == ("graded", 1.0)  # 0.25 + 0.5 + 0.25
    assert outcome.facts == {"rewards": {"solved": 1.0, "half": 1.0, "other_half": 0.5},
                             "metrics": {"steps": 3.0}}


def test_links_into_the_graders_paths_grade_zero():
    # An image tool (test.sh, cat) would follow them: never "inert" in a grading box.
    episode = Episode()
    episode.agent_writes("/work/answer.txt", b"42\n")
    episode.boxes.make_dirs("s-1", "/work/out")
    links = episode.boxes.boxes["s-1"].setdefault("links", {})
    links["/work/out/.verifier"] = "/logs/verifier"
    links["/work/out/.tests"] = "/tests"
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "state_unreadable"}}
    assert episode.boxes.boxes.get(G0) is None  # removed after grading


class ReadsTheChat(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, trace) -> float:
        return 1.0 if "42" in trace.last_reply else 0.0


def test_a_reward_that_reads_the_conversation_never_sees_one():
    episode = Episode(ReadsTheChat)
    episode.agent_writes("/work/answer.txt", b"42\n")
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)


class BreaksAndSwallows(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        try:
            await runtime.run(["die"], {})
        except Exception:
            pass
        return 1.0


def test_a_reward_read_after_a_fault_the_env_swallowed_is_discarded():
    episode = Episode()
    episode.agent_writes("/work/answer.txt", b"42\n")

    episode.task = task_from_verifiers(BreaksAndSwallows(toy_data()), lock=LOCK)
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "grading_box_failed"}}


class SwallowsUv(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        try:
            await runtime.run_uv_script("print(1)")
        except Exception:
            pass
        return 1.0


def test_an_unsupported_member_the_env_swallowed_still_aborts():
    outcome = Episode(SwallowsUv).grade()
    assert outcome.status == "aborted" and outcome.reward is None


class EnvWithLiterals(ToyData):
    home: str = "${HOME}"
    sub: str = "$(id)"


class LiteralEnvTask(vf.Task[EnvWithLiterals]):
    def runtime_env(self):
        return {"A": self.data.home, "B": self.data.sub}

    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        await runtime.run(["true"], {})
        return 1.0


def test_runtime_env_reaches_the_box_verbatim_without_shell_expansion():
    data = EnvWithLiterals(idx=0, prompt="x", image="toy/task:1", workdir="/work",
                           artifacts=[])
    episode = Episode(LiteralEnvTask, data=data)
    assert episode.task.env == {"A": "${HOME}", "B": "$(id)"}
    assert episode.spec.env == {"A": "${HOME}", "B": "$(id)"}
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 1.0)
    runs = [kw for name, kw in episode.grading_calls() if name == "run_program"]
    assert runs and all(kw["env"] == {"A": "${HOME}", "B": "$(id)"} for kw in runs)


class ProbesALog(ToyTask):
    """Reads a log only when it is small (a probe), then grades the answer."""

    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        try:
            await runtime.read("/work/out/answer.txt", max_bytes=1)
        except OSError:
            pass  # too large for a probe: the env scores on
        answer = await runtime.read("/work/out/answer.txt", max_bytes=64)
        return 1.0 if answer.strip() == b"42" else 0.0


def test_an_oversized_read_the_env_caught_does_not_override_its_reward():
    episode = Episode(ProbesALog)
    episode.agent_writes("/work/answer.txt", b"42\n")
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 1.0)


class Unprotected(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        await runtime.read("/work/out/answer.txt", max_bytes=1)
        return 1.0


def test_an_oversized_read_the_env_let_through_keeps_its_label():
    episode = Episode(Unprotected)
    episode.agent_writes("/work/answer.txt", b"42\n")
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "grading_output_overflow"}}


# -- the agent's links never lead the reward to the grader's own files (C1) ------------

class ComparesToTheReference(ToyTask):
    """The grader's reference is a file of the grading box outside the artifacts."""

    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        await runtime.write("/tests/output", b"the reference\n")
        reference = await runtime.read("/tests/output")
        try:
            output = await runtime.read("/work/out/output")
        except OSError:
            return 0.0
        return 1.0 if output == reference else 0.0


def reference_episode():
    return Episode(ComparesToTheReference, data=toy_data(finalize_copies=False))


def test_the_reference_output_written_by_the_agent_grades_one():
    episode = reference_episode()
    episode.agent_writes("/work/out/output", b"the reference\n")
    assert episode.grade().reward == 1.0


@pytest.mark.parametrize("links", [
    {"/work/out/output": "/tests/output"},  # the artifact is a link to the reference
    {"/work/out/output": "../../tests/output"},  # the same, relative
    {"/work/out": "/tests"},  # the artifact root itself is a link
    {"/work/out/output": "hop", "/work/out/hop": "/tests/output"},  # out in two hops
    {"/work/out/output": "missing"},  # inside, but nothing there in the grading box
], ids=["absolute", "relative", "root_link", "two_hops", "dangling"])
def test_an_artifact_link_leading_out_of_the_roots_is_never_restored(links):
    episode = reference_episode()
    episode.boxes.make_dirs("s-1", "/work/out")
    episode.boxes.boxes["s-1"].setdefault("links", {}).update(links)
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "state_unreadable"}}
    names = [name for name, _ in episode.grading_calls()]
    assert "write_file" not in names  # refused at restore: the reward never ran


def test_an_artifact_link_inside_the_roots_is_kept():
    episode = reference_episode()
    episode.agent_writes("/work/out/real/output", b"the reference\n")
    episode.boxes.boxes["s-1"].setdefault("links", {})["/work/out/output"] = "real/output"
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 1.0)


# -- rewards run concurrently in verifiers ---------------------------------------------

class TwoRewardsRunCommands(vf.Task[ToyData]):
    @vf.reward(weight=0.5)
    async def first(self, runtime) -> float:
        runs = [await runtime.run(["true"], {}) for _ in range(3)]
        return 1.0 if all(r.exit_code == 0 for r in runs) else 0.0

    @vf.reward(weight=0.5)
    async def second(self, runtime) -> float:
        runs = [await runtime.run(["true"], {}) for _ in range(3)]
        listed = await runtime.read("/work/out/answer.txt", max_bytes=64)
        return 1.0 if all(r.exit_code == 0 for r in runs) and listed.strip() == b"42" else 0.0


def test_concurrent_rewards_never_share_a_command_with_a_stop():
    outcomes = []
    for _ in range(2):
        episode = Episode(TwoRewardsRunCommands, data=toy_data(finalize_copies=False))
        episode.agent_writes("/work/out/answer.txt", b"42\n")
        outcomes.append(episode.grade())
        names = [name for name, _ in episode.grading_calls()]
        scoring = names[names.index("tar_extract") + 1:]
        runs = [i for i, name in enumerate(scoring) if name == "run_program"]
        assert len(runs) == 6 and all(scoring[i + 1] == "stop_processes" for i in runs)
    assert [(o.status, o.reward, o.facts) for o in outcomes] == [
        ("graded", 1.0, {"rewards": {"first": 1.0, "second": 1.0}, "metrics": {}})] * 2


# -- the reward code parses the agent's bytes ----------------------------------------

class ParsesJson(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        raw = await runtime.read("/work/out/answer.txt")
        return float(json.loads(raw.decode("utf-8"))["score"])


class PerTest(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> dict[str, float]:
        raw = await runtime.read("/work/out/answer.txt")
        return {name: 1.0 / len(raw.split()) for name in raw.decode().split()}


class ReportsItsOwnFailure(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        raise EnvInfraError("the env blames itself")


@pytest.mark.parametrize("task_class, written", [
    (ParsesJson, b'{"score": 1.0'),  # invalid JSON
    (ParsesJson, b"\xff\xfe{}"),  # not UTF-8
    (ParsesJson, b"[" * 200_000 + b"]" * 200_000),  # nesting past the recursion limit
    (PerTest, b""),  # an empty per-test dict: no reward at all
    (ReportsItsOwnFailure, b"42"),
], ids=["invalid_json", "not_utf8", "deep_nesting", "empty_per_test", "env_infra_error"])
def test_scoring_that_fails_on_the_agents_bytes_grades_zero_never_aborted(task_class, written):
    episode = Episode(task_class, data=toy_data(finalize_copies=False))
    episode.agent_writes("/work/out/answer.txt", written)
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "grading_failed"}}
    assert episode.grader.env_failure_counts() == {"toy": {"grading_failed": 1}}


def test_a_per_test_reward_that_ran_grades_normally():
    episode = Episode(PerTest, data=toy_data(finalize_copies=False))
    episode.agent_writes("/work/out/answer.txt", b"a b")
    assert episode.grade().reward == 1.0


# -- scoring runs in a bounded process -----------------------------------------------

class Spins(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        end = time.monotonic() + 8
        while time.monotonic() < end:  # never awaits: holds the GIL in-process
            pass
        return 1.0


class Hoards(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        blob = bytearray(256 * 1024**2)
        return 1.0 if blob else 1.0


def test_cpu_bound_scoring_is_cut_at_the_deadline():
    episode = Episode(Spins, data=toy_data(timeout=TaskTimeout(scoring=1.0)))
    started = time.monotonic()
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "grading_timeout"}}
    assert time.monotonic() - started < 5


def test_scoring_past_its_memory_limit_grades_zero():
    episode = Episode(Hoards, defaults=BridgeDefaults(scoring_memory_bytes=64 * 1024**2))
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "grading_timeout"}}


def test_extract_checks_the_artifacts_with_the_helper_not_the_images_test():
    episode = Episode()
    episode.agent_writes("/work/answer.txt", b"42\n")
    assert episode.grade().reward == 1.0
    assert not [argv for box, argv, *_ in episode.boxes.exec_log
                if box == "s-1" and argv[0] == "test"]
    assert ("file_kind", "s-1") in [(name, box) for name, box, _ in episode.boxes.helper_calls]


class ProbesMissing(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        try:
            await runtime.read("/work/out/none.txt")
        except FileNotFoundError:  # the parent's error crosses the pipe with its errno
            return 0.5
        return 0.0


def test_a_runtime_error_reaches_the_envs_code_as_its_builtin_type():
    assert Episode(ProbesMissing).grade().reward == 0.5


class SpeaksGarbage(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        runtime.inner._call(self.data.prompt, path=5)
        return 1.0


@pytest.mark.parametrize("op", ["docker_exec", "read"])
def test_a_scoring_process_that_breaks_the_protocol_grades_zero(op, caplog):
    episode = Episode(SpeaksGarbage, data=toy_data(prompt=op))
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "grading_failed"}}
    assert f"sent a bad call ({op})" in caplog.text  # refused before reaching the box


class StallsMidMessage(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        runtime.inner.conn._send((1000).to_bytes(4, "big", signed=True) + b"x" * 10)
        time.sleep(8)  # the parent is blocked reading the rest of the message
        return 1.0


def test_a_child_that_stalls_the_parent_is_killed_at_the_deadline():
    episode = Episode(StallsMidMessage, data=toy_data(timeout=TaskTimeout(scoring=1.0)))
    started = time.monotonic()
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "grading_timeout"}}
    assert time.monotonic() - started < 5


class SendsAHugeHeader(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        header = b"[" * 200_000 + b"]" * 200_000
        runtime.inner.conn.send_bytes(len(header).to_bytes(4, "big") + header)
        return 1.0


def test_a_header_nested_past_the_recursion_limit_is_a_broken_protocol(caplog):
    outcome = Episode(SendsAHugeHeader).grade()
    assert outcome.facts == {"_grader": {"agent_fault": "grading_failed"}}
    assert "sent a malformed call" in caplog.text


# -- the scoring process starts from the forkserver, not a fork of the service (R2-2) --

def open_fds() -> set[int]:
    found = set()
    for fd in range(4096):
        try:
            os.fstat(fd)
        except OSError:
            continue
        found.add(fd)
    return found


class ListsItsFds(ToyTask):
    """Scores 1 only with no fd but stdio, its pipe and its own event loop's, and with
    its CPU and memory limits set."""

    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        loop = asyncio.get_running_loop()
        allowed = {0, 1, 2, runtime.inner.conn.fileno(), loop._selector.fileno(),
                   loop._ssock.fileno(), loop._csock.fileno()}
        memory = resource.getrlimit(resource.RLIMIT_AS)
        cpu = resource.getrlimit(resource.RLIMIT_CPU)
        limited = (memory[0] == memory[1] != resource.RLIM_INFINITY
                   and cpu[0] != resource.RLIM_INFINITY)
        return 1.0 if open_fds() <= allowed and limited else 0.0

    @vf.metric
    async def extra_fds(self, runtime) -> float:
        loop = asyncio.get_running_loop()
        allowed = {0, 1, 2, runtime.inner.conn.fileno(), loop._selector.fileno(),
                   loop._ssock.fileno(), loop._csock.fileno()}
        return float(len(open_fds() - allowed))


def test_the_scoring_process_holds_no_fd_but_its_pipe_and_keeps_its_limits(tmp_path):
    held = open(tmp_path / "held", "wb")  # a parent fd (as the Docker socket would be)
    try:
        outcome = Episode(ListsItsFds).grade()
    finally:
        held.close()
    assert (outcome.status, outcome.reward) == ("graded", 1.0)
    assert outcome.facts["metrics"] == {"extra_fds": 0.0}


HELD = threading.Lock()


class TakesAModuleLock(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        with HELD:  # a fork of the service would copy it held, by a thread not copied
            return 1.0


def test_scoring_works_while_another_thread_of_the_service_holds_a_lock():
    episode = Episode(TakesAModuleLock, data=toy_data(timeout=TaskTimeout(scoring=5.0)))
    taken, release = threading.Event(), threading.Event()

    def hold():
        with HELD:
            taken.set()
            release.wait(30)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    taken.wait(5)
    try:
        outcome = episode.grade()
    finally:
        release.set()
        holder.join()
    assert (outcome.status, outcome.reward) == ("graded", 1.0)


def test_a_task_that_cannot_be_pickled_is_refused_at_declaration():
    class Local(ToyTask):
        pass

    with pytest.raises(TaskDeclarationError, match="pickled"):
        task_from_verifiers(Local(toy_data()), lock=LOCK)


# -- who is to blame when the scoring process gives no result (R2-3) -------------------

def ghost_task_class(monkeypatch):
    """A task class whose module exists only in this process: its import fails in the
    scoring process before it is ready."""
    module = types.ModuleType("ghost_env_only_in_the_service")
    ghost = type("GhostTask", (ToyTask,), {"__module__": module.__name__})
    module.GhostTask = ghost
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return ghost


def test_a_scoring_process_whose_import_fails_is_ours(monkeypatch, caplog):
    ghost = ghost_task_class(monkeypatch)
    with pytest.raises(EnvInfraError, match="before it was ready"):
        Episode(ghost)  # its setup: the open fails, ours
    episode = Episode()
    episode.task = task_from_verifiers(ghost(toy_data()), lock=LOCK)
    episode.agent_writes("/work/answer.txt", b"42\n")
    outcome = episode.grade()
    assert outcome.status == "aborted" and outcome.reward is None
    assert "before it was ready" in caplog.text


def test_a_reply_the_scoring_process_cannot_read_is_ours(monkeypatch, caplog):
    """The parent detects its own broken reply before sending it: ours, never the
    child's EXIT_PROTOCOL (which is graded 0 after `ready`)."""
    episode = Episode()
    episode.agent_writes("/work/answer.txt", b"42\n")
    monkeypatch.setattr(scoring_process, "_reply", lambda header, payload=b"": b"\0\0\0\x09{")
    outcome = episode.grade()
    assert outcome.status == "aborted" and outcome.reward is None
    assert "is one the scoring process cannot read" in caplog.text


@pytest.mark.parametrize("reply, problem", [
    (scoring_process._frame({"value": None}), None),
    (scoring_process._frame({"error": {"type": "OSError"}}), None),
    (scoring_process._frame({"error": "x"}), "a malformed error"),
    (scoring_process._frame({}), "no value"),
    (b"\0\0\0\x09{", "truncated or oversized header"),
    (scoring_process._frame({"value": {"exit_code": 0, "stdout_bytes": 2,
                                       "stderr_bytes": 0}}, b"ok"), None),
    (scoring_process._frame({"value": {"exit_code": 0, "stdout_bytes": 3,
                                       "stderr_bytes": 0}}, b"ok"), "a malformed run reply"),
    (scoring_process._frame({"value": {"exit_code": "0", "stdout_bytes": 0,
                                       "stderr_bytes": 0}}), "a malformed run reply"),
])
def test_the_parent_checks_its_replies_as_the_child_reads_them(reply, problem):
    op = "run" if b"exit_code" in reply else "exists"
    assert scoring_process._reply_problem(op, reply) == problem


def test_a_forkserver_that_cannot_start_is_ours(monkeypatch, caplog):
    def broken():
        raise OSError("the forkserver is gone")

    episode = Episode()
    episode.agent_writes("/work/answer.txt", b"42\n")
    monkeypatch.setattr(scoring_server, "start", broken)
    outcome = episode.grade()
    assert outcome.status == "aborted" and "could not start" in caplog.text


def test_the_tmpdir_is_the_slots_before_any_scoring_process_starts(monkeypatch):
    """As root, a scoring process that never reaches its own chown (start failed, killed
    before ready) leaves a TMPDIR the sweeper, as the slot's uid, can still empty."""
    import shutil

    calls = []

    def broken():
        raise OSError("the forkserver is gone")

    def sweep(path, identity, pool, slot):
        calls.append(("sweep", path))
        pool.release(slot)
        shutil.rmtree(path)

    episode = Episode()
    episode.agent_writes("/work/answer.txt", b"42\n")
    monkeypatch.setattr(scoring_process.os, "geteuid", lambda: 0)
    monkeypatch.setattr(scoring_process.os, "chown",
                        lambda path, uid, gid: calls.append(("chown", path, uid, gid)))
    monkeypatch.setattr(scoring_process, "_sweep_slot", sweep)
    monkeypatch.setattr(scoring_server, "start", broken)
    assert episode.grade().status == "aborted"
    assert calls and len(calls) % 2 == 0  # one pair per grading attempt
    for (_, path, uid, gid), swept in zip(calls[::2], calls[1::2]):
        assert uid == gid and scoring_server.SLOTS.base <= uid and swept == ("sweep", path)


class ExitsOnTheAgentsBytes(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        raise SystemExit(2)  # an argparse-style parser of the agent's output


class CrashesOnTheAgentsBytes(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        os.kill(os.getpid(), signal.SIGSEGV)  # a native parser crashing
        return 1.0


@pytest.mark.parametrize("task_class, fault", [
    (ExitsOnTheAgentsBytes, "grading_failed"),
    (CrashesOnTheAgentsBytes, "grading_timeout"),
], ids=["system_exit", "native_crash"])
def test_a_scoring_failure_after_it_started_is_graded_zero_never_aborted(task_class, fault):
    outcome = Episode(task_class).grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": fault}}


# -- the audit hook: no code or process from the agent's bytes ------------------------

FORGED = {"result": "scored", "rewards": {"solved": 1.0}, "metrics": {}, "reward": 1.0,
          "reward_repr": "1.0"}
FORGE = "conn.send_bytes(frame(forged))"
"""What a parser exploited into running the agent's code would do: send a result of
its own on the scoring process's pipe (the parent takes the first result it reads)."""


def _forge_scope(runtime):
    return {"conn": runtime.inner.conn, "frame": scoring_process._frame, "forged": FORGED}


class ForgesThroughExec(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        exec(FORGE, _forge_scope(runtime))
        return 0.0


class ForgesThroughEval(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        eval(FORGE, _forge_scope(runtime))
        return 0.0


class ForgesThroughCompiledCode(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        import ast

        exec(compile(ast.parse(FORGE), "<agent>", "exec"), _forge_scope(runtime))
        return 0.0


class ForgesAndSwallowsTheRefusal(ToyTask):
    """An env that catches everything (`except Exception`) and returns a reward anyway:
    the refusal is remembered, the result is `raised` whatever the env returns."""

    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        try:
            exec(FORGE, _forge_scope(runtime))
        except Exception:
            pass
        return 1.0


class _Gadget:
    def __reduce__(self):
        return (os.system, ("true",))


class UnpicklesTheAgentsBytes(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        import pickle

        pickle.loads(pickle.dumps(_Gadget()))
        return 1.0


class UnmarshalsTheAgentsBytes(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        import marshal

        marshal.loads(marshal.dumps(1))
        return 1.0


class CallsCtypes(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        import ctypes

        ctypes.CDLL(None).getpid()
        return 1.0


class StartsAProcess(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        import subprocess

        subprocess.run(["true"], check=False)
        return 1.0


class Forks(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        if os.fork() == 0:
            os._exit(0)
        return 1.0


class RunsAShell(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        os.system("true")
        return 1.0


@pytest.mark.parametrize("task_class", [
    ForgesThroughExec, ForgesThroughEval, ForgesThroughCompiledCode,
    ForgesAndSwallowsTheRefusal, UnpicklesTheAgentsBytes, UnmarshalsTheAgentsBytes,
    CallsCtypes, StartsAProcess, Forks, RunsAShell,
], ids=["exec", "eval", "compile_ast", "swallowed", "pickle", "marshal", "ctypes",
        "subprocess", "fork", "os_system"])
def test_a_refused_event_in_the_scoring_process_grades_zero(task_class, caplog):
    episode = Episode(task_class)
    episode.agent_writes("/work/answer.txt", b"42\n")
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "grading_failed"}}
    assert "PermissionError" in caplog.text and "refuse" in caplog.text


class ImportsAndParsesWhileScoring(ToyTask):
    """What a reward legitimately does once ready: import a module not loaded yet
    (its `.pyc` or `.py`, a dataclass and a namedtuple at its module level), read a
    test file's tree (`ast.parse`), resolve a string annotation (`typing.ForwardRef`),
    parse JSON and a regex."""

    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        import ast
        import re
        import typing

        sys.modules.pop("lazy_scoring_module", None)
        import lazy_scoring_module

        ast.parse("def test_x():\n    assert True\n")
        assert typing.get_type_hints(lazy_scoring_module.Annotated) == {"score": float}
        json.loads('{"a": [1, 2]}')
        answer = (await runtime.read("/work/out/answer.txt")).decode()
        assert re.fullmatch(r"\d+\n", answer)
        return lazy_scoring_module.verdict(answer + " done").score


def test_imports_and_parsing_still_work_once_the_scoring_process_is_ready():
    episode = Episode(ImportsAndParsesWhileScoring)
    episode.agent_writes("/work/answer.txt", b"42\n")
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 1.0)


class ExitsWithOurCode(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        os._exit(int(self.data.prompt))


@pytest.mark.parametrize("code", [scoring_process.EXIT_OURS, scoring_process.EXIT_PROTOCOL])
def test_our_exit_codes_after_ready_grade_zero_never_abort(code):
    """After `ready` the child runs code the agent's bytes steer: it can exit with any
    code, ours included; only what the parent detects itself is ours."""
    episode = Episode(ExitsWithOurCode, data=toy_data(prompt=str(code)))
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "grading_timeout"}}


# -- fix round 3 ------------------------------------------------------------------------

FLOOR_S = getattr(scoring_process, "SCORING_STARTUP_FLOOR_S", None)


@pytest.fixture(autouse=True)
def _no_startup_floor(monkeypatch):
    """Most tests score with deadlines of a few seconds, below the startup floor; the
    floor's own tests put it back."""
    monkeypatch.setattr(scoring_process, "SCORING_STARTUP_FLOOR_S", 0.0, raising=False)
    monkeypatch.setattr(declaration, "MIN_GRADING_TIMEOUT_S", 0.0)


class EmitsOutput(ToyTask):
    """Runs FakeBoxes' `emit <hex byte> <count>` (stdout and stderr) and scores 1 when
    both streams come back whole."""

    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        byte, count = self.data.prompt.split()
        result = await runtime.run(["emit", byte, count], {})
        expected = bytes.fromhex(byte).decode("utf-8", "replace") * int(count)
        return 1.0 if result.stdout == expected and result.stderr == expected else 0.0


@pytest.mark.parametrize("prompt, output_bytes", [
    ("ff 1048576", 1 << 20),  # each byte a U+FFFD: 6 MiB per stream as JSON escapes
    ("41 5242880", 8 << 20),  # 5 MiB of ASCII per stream
], ids=["binary_1MiB", "ascii_5MiB"])
def test_a_commands_large_output_is_graded_never_aborted(prompt, output_bytes):
    episode = Episode(EmitsOutput, data=toy_data(prompt=prompt))
    episode.grader = Grader(SimpleNamespace(episode_grader_output_bytes=output_bytes,
                                            episode_grading_attempts=2), episode.boxes)
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 1.0)


class WritesALargeFile(ToyTask):
    """Writes `prompt` bytes (a tests tar, as Harbor's `_stage_tests` does)."""

    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        await runtime.write("/tmp/tests.tar", b"t" * int(self.data.prompt))
        return 1.0


class WritesALargeFileCarefully(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        try:
            await runtime.write("/tmp/tests.tar", b"t" * int(self.data.prompt))
        except OSError:
            return 0.5  # the child is alive and its code goes on
        return 1.0


def grading_box_files(episode):
    """The grading box's files, kept when the grader removes it."""
    kept, remove = {}, episode.boxes.remove

    def keeping(box_id, *args, **kwargs):
        if box_id in episode.boxes.boxes:
            kept[box_id] = dict(episode.boxes.boxes[box_id]["files"])
        return remove(box_id, *args, **kwargs)

    episode.boxes.remove = keeping
    return kept


def test_an_envs_write_larger_than_the_state_is_served():
    episode = Episode(WritesALargeFile, data=toy_data(prompt=str(40 * 1024**2)))
    kept = grading_box_files(episode)
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 1.0)
    assert len(kept[G0]["/tmp/tests.tar"]) == 40 * 1024**2


def test_a_write_past_its_cap_fails_in_the_envs_code_never_kills_it():
    small = BridgeDefaults(max_artifact_bytes=1 << 20)  # write cap: about 4.3 MiB
    episode = Episode(WritesALargeFile, data=toy_data(prompt=str(5 << 20)), defaults=small)
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "grading_failed"}}
    assert episode.grader.env_failure_counts() == {"toy": {"grading_failed": 1}}
    careful = Episode(WritesALargeFileCarefully, data=toy_data(prompt=str(5 << 20)),
                      defaults=small)
    assert careful.grade().reward == 0.5


SECRET = "reliquary-test-secret-7c1e"
THREAD_VARS = {"OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"}
ALLOWED_ENV = {"PATH", "LANG", "HOME", "TMPDIR", "PYTHONHASHSEED"} | THREAD_VARS


class ReportsItsCredentials(ToyTask):
    """Scores 1 only with a non-root uid, no secret, a minimal environment, a private
    TMPDIR and no_new_privs; writes its TMPDIR to the box for the test."""

    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        tmp = os.environ.get("TMPDIR", "")
        await runtime.write("/tmp/scoring-tmpdir", tmp.encode())
        with open("/proc/self/environ", "rb") as raw:
            initial = raw.read()
        with open("/proc/self/status") as status:
            no_new_privs = "NoNewPrivs:\t1" in status.read()
        info = os.stat(tmp) if tmp and os.path.isdir(tmp) else None
        ok = (os.getuid() != 0 and os.geteuid() != 0
              and set(os.environ) <= ALLOWED_ENV and "PATH" in os.environ
              and os.environ.get("HOME") == "/nonexistent"
              and SECRET not in str(dict(os.environ)) and SECRET.encode() not in initial
              and info is not None and info.st_uid == os.getuid()
              and info.st_mode & 0o777 == 0o700 and no_new_privs)
        return 1.0 if ok else 0.0


@pytest.mark.skipif(os.geteuid() == 0, reason="as root the dropped process is not dumpable: its /proc/self files are root's")
def test_the_scoring_process_runs_without_the_services_secrets_or_root():
    scoring_server.stop()  # the next forkserver starts with the secret in its environment
    os.environ["RELIQUARY_TEST_SECRET"] = SECRET
    try:
        episode = Episode(ReportsItsCredentials)
        kept = grading_box_files(episode)
        outcome = episode.grade()
    finally:
        del os.environ["RELIQUARY_TEST_SECRET"]
        scoring_server.stop()
    assert (outcome.status, outcome.reward) == ("graded", 1.0)
    tmp = kept[G0]["/tmp/scoring-tmpdir"].decode()
    assert tmp and not os.path.exists(tmp)  # removed once the scoring ended


class ReadsARootOnlyFile(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        try:
            with open(self.data.prompt, "rb"):
                return 0.0
        except PermissionError:
            return 1.0 if os.getuid() >= scoring_server.SLOTS.base else 0.0


@pytest.mark.skipif(os.geteuid() != 0, reason="the privilege drop needs root")
def test_as_root_the_scoring_process_cannot_open_a_root_only_path(tmp_path):
    secret = tmp_path / "root-only"
    secret.write_bytes(b"x")
    secret.chmod(0o600)
    os.chmod(tmp_path, 0o755)
    assert Episode(ReadsARootOnlyFile, data=toy_data(prompt=str(secret))).grade().reward == 1.0


class FakeOs:
    """Records the privilege calls `_drop_privileges` makes."""

    def __init__(self, euid, honour=True):
        self.uid = self.euid = euid
        self.gid = self.egid = euid
        self.groups = [0, 4]
        self.honour = honour
        self.calls = []

    def geteuid(self):
        return self.euid

    def getuid(self):
        return self.uid

    def getgid(self):
        return self.gid

    def getegid(self):
        return self.egid

    def getgroups(self):
        return list(self.groups)

    def setgroups(self, groups):
        self.calls.append(("setgroups", list(groups)))
        if self.honour:
            self.groups = list(groups)

    def setgid(self, gid):
        self.calls.append(("setgid", gid))
        if self.honour:
            self.gid = self.egid = gid

    def setuid(self, uid):
        self.calls.append(("setuid", uid))
        if self.honour:
            self.uid = self.euid = uid

    def chown(self, path, uid, gid):
        self.calls.append(("chown", path, uid, gid))

    def kill(self, pid, sig):
        """Never a real signal: recorded; no process is left after `alive_rounds`."""
        self.calls.append(("kill", pid, sig, self.euid))
        self.alive_rounds = getattr(self, "alive_rounds", 1) - 1
        if self.alive_rounds < 0:
            raise ProcessLookupError(3, "No such process")


def test_as_root_the_privileges_are_dropped_groups_first():
    ops = FakeOs(0)
    scoring_process._drop_privileges(65534, 65533, ops)
    assert ops.calls == [("setgroups", []), ("setgid", 65533), ("setuid", 65534)]


def test_a_drop_that_did_not_take_is_refused():
    with pytest.raises(PermissionError):
        scoring_process._drop_privileges(65534, 65534, FakeOs(0, honour=False))


def test_as_a_normal_user_nothing_is_changed():
    ops = FakeOs(1000)
    scoring_process._drop_privileges(65534, 65534, ops)
    assert ops.calls == []


def test_the_cpu_limit_adds_what_was_already_spent():
    assert scoring_process._cpu_limit(seconds_left=10.2, spent=3.5, hard=None) == (14, 15)
    assert scoring_process._cpu_limit(seconds_left=10.2, spent=3.5, hard=12) == (12, 12)


class BytesPath(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        await runtime.read(b"/work/out/answer.txt")
        return 1.0


def test_an_argument_the_protocol_cannot_encode_fails_in_the_envs_code():
    outcome = Episode(BytesPath).grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "grading_failed"}}


class KillsItsForkserver(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        parent = os.getppid()
        os.kill(parent, signal.SIGKILL)  # the forkserver (a host OOM kill)
        # Sending a signal does not wait for exit. Observe reparenting before this
        # child exits, so the controller sees a forkserver that has actually died.
        deadline = time.monotonic() + 5
        while os.getppid() == parent:
            if time.monotonic() >= deadline:
                raise TimeoutError("the forkserver did not exit")
            time.sleep(0.001)
        os.kill(os.getpid(), signal.SIGKILL)
        return 1.0


@pytest.mark.skipif(os.geteuid() == 0, reason="as root a slot uid cannot signal the root forkserver (the attack is impossible)")
@pytest.mark.parametrize("_run", range(10))
def test_a_forkserver_dying_while_scoring_is_ours(caplog, _run):
    outcome = Episode(KillsItsForkserver).grade()
    assert outcome.status == "aborted" and outcome.reward is None
    assert "forkserver" in caplog.text


class Unpickles:
    """Runs `action` when unpickled in the scoring process, before it is ready."""

    def __init__(self, action):
        self.action = action

    def __reduce__(self):
        return (_before_ready, (self.action,))


def _before_ready(action):
    import multiprocessing

    if multiprocessing.current_process().name == "reliquary-scoring":
        if action == "oom":
            os.kill(os.getpid(), signal.SIGKILL)
        time.sleep(30)
    return None


class SlowToStart(ToyTask):
    def __reduce__(self):
        return (_rebuilt, (type(self).__bases__[0], self.data, Unpickles(self.data.prompt)))


def _rebuilt(cls, data, _marker):
    return cls(data)


@pytest.mark.parametrize("action", ["oom", "stall"])
def test_a_scoring_process_killed_before_it_is_ready_is_ours(action, caplog, monkeypatch):
    monkeypatch.setattr(scoring_process, "SCORING_STARTUP_FLOOR_S", 4.0)
    episode = Episode()  # opened by a task that starts; extract and grade by SlowToStart
    episode.task = task_from_verifiers(
        SlowToStart(toy_data(prompt=action, timeout=TaskTimeout(scoring=10.0))), lock=LOCK,
        defaults=BridgeDefaults(scoring_ready_s=4.0))  # ready within 4 s
    episode.agent_writes("/work/answer.txt", b"42\n")
    outcome = episode.grade()
    assert outcome.status == "aborted" and outcome.reward is None
    assert "before it was ready" in caplog.text


def test_no_scoring_process_starts_with_less_than_the_floor_left(monkeypatch):
    episode = Episode(data=toy_data(timeout=TaskTimeout(scoring=20.0)))
    episode.agent_writes("/work/answer.txt", b"42\n")
    state = episode.grader.extract("s-1", episode.task, env="toy")  # floor off: extracted
    assert isinstance(state, bytes)
    monkeypatch.setattr(scoring_process, "SCORING_STARTUP_FLOOR_S", FLOOR_S)

    def refused():
        raise AssertionError("a scoring process was started")

    monkeypatch.setattr(scoring_server, "start", refused)
    outcome = episode.grade()  # the finalize is the first hook with 20 s: never started
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "extract_timeout"}}
    monkeypatch.setattr(episode.grader, "extract", lambda *args, **kwargs: state)
    outcome = episode.grade()  # the reward step alone
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "grading_timeout"}}


def test_with_more_than_the_floor_left_the_scoring_process_starts(monkeypatch):
    assert FLOOR_S == 30
    monkeypatch.setattr(scoring_process, "SCORING_STARTUP_FLOOR_S", FLOOR_S)
    episode = Episode(data=toy_data(timeout=TaskTimeout(scoring=60.0)))
    episode.agent_writes("/work/answer.txt", b"42\n")
    assert episode.grade().reward == 1.0


# -- fix round 4 ------------------------------------------------------------------------

class Lingers(ToyTask):
    """Scores after `prompt` seconds (so two grades overlap)."""

    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        await asyncio.sleep(float(self.data.prompt))
        return 1.0


def record_scoring(monkeypatch, pool):
    """Uses `pool` for the slots and logs each scoring process's start (its uid), each
    reap and each release, in order: only the reward step's (`score`), not the setup and
    finalize hooks' processes, which take slots too."""
    log = []
    monkeypatch.setattr(scoring_server, "SLOTS", pool)
    original = scoring_server.CONTEXT.Process
    scoring_slots = set()

    def process(*args, **kwargs):
        made = original(*args, **kwargs)
        if kwargs.get("name") == "reliquary-scoring" and kwargs["args"][-1] == "score":
            identity = next(a for a in kwargs["args"]
                            if isinstance(a, scoring_process.ScoringIdentity))
            log.append(("start", identity.uid, identity.gid))
            scoring_slots.add(identity.uid - pool.base)
            made.reliquary_scores = True
        return made

    reap, release = scoring_process._Child.reap, pool.release

    def reaped(self):
        code = reap(self)
        if getattr(self.process, "reliquary_scores", False):
            log.append(("reap",))
        return code

    def released(slot):
        if slot in scoring_slots:
            scoring_slots.discard(slot)
            log.append(("release", slot))
        return release(slot)

    monkeypatch.setattr(scoring_server.CONTEXT, "Process", process)
    monkeypatch.setattr(scoring_process._Child, "reap", reaped)
    monkeypatch.setattr(pool, "release", released)
    return log


def grade_together(*episodes):
    results = [None] * len(episodes)

    def grade(index):
        results[index] = episodes[index].grade()

    threads = [threading.Thread(target=grade, args=(i,)) for i in range(len(episodes))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    return results


def lingering_episode(seconds="1.5"):
    episode = Episode(Lingers, data=toy_data(prompt=seconds,
                                             timeout=TaskTimeout(scoring=20.0)))
    return episode


def test_concurrent_scoring_processes_get_different_uids(monkeypatch):
    pool = scoring_server.ScoringSlots(base=300000, count=4)
    log = record_scoring(monkeypatch, pool)
    outcomes = grade_together(lingering_episode(), lingering_episode())
    assert [o.reward for o in outcomes] == [1.0, 1.0]
    starts = [entry for entry in log if entry[0] == "start"]
    assert len(starts) == 2 and starts[0][1] != starts[1][1]
    assert all(300000 <= uid < 300004 and gid == uid for _, uid, gid in starts)
    assert [entry[0] for entry in log[:2]] == ["start", "start"]  # they overlapped
    assert pool.free() == 4


def test_a_slot_is_reused_only_after_its_scoring_process_was_reaped(monkeypatch):
    pool = scoring_server.ScoringSlots(base=300000, count=1)
    log = record_scoring(monkeypatch, pool)
    outcomes = grade_together(lingering_episode("0.5"), lingering_episode("0.5"))
    assert [o.reward for o in outcomes] == [1.0, 1.0]  # the second waited for the slot
    assert log == [("start", 300000, 300000), ("reap",), ("release", 0),
                   ("start", 300000, 300000), ("reap",), ("release", 0)]
    assert pool.free() == 1


def test_slots_are_handed_out_once_until_released():
    pool = scoring_server.ScoringSlots(base=300000, count=2)
    first, second = pool.acquire(), pool.acquire()
    assert {first, second} == {0, 1} and pool.acquire() is None
    assert pool.uid(second) == 300000 + second
    pool.release(first)
    assert pool.acquire() == first
    with pytest.raises(ValueError):
        pool.release(7)


def test_as_root_the_scoring_process_takes_its_slots_uid_and_owns_its_tmpdir():
    ops = FakeOs(0)
    identity = scoring_process.ScoringIdentity(uid=200003, gid=200003)
    scoring_process._take_identity(identity, "/scoring/reliquary-scoring-x", ops)
    assert ops.calls == [("chown", "/scoring/reliquary-scoring-x", 200003, 200003),
                         ("setgroups", []), ("setgid", 200003), ("setuid", 200003)]
    user = FakeOs(1000)
    scoring_process._take_identity(identity, "/tmp/x", user)
    assert user.calls == []


# Nothing of a slot's uid outlives its grade.

class RemovedPaths(list):
    def __call__(self, path):
        self.append(path)


def test_the_sweeper_kills_every_process_of_the_slots_uid_before_removing_its_tmpdir():
    ops = FakeOs(0)
    ops.alive_rounds = 2
    removed = RemovedPaths()
    identity = scoring_process.ScoringIdentity(uid=200001, gid=200001)
    scoring_process._sweep_as_slot("/scoring/t", identity, ops, removed, lambda _s: None,
                                   lambda uid: ops.calls.append(("shm", uid)))
    assert ops.calls == [("setgroups", []), ("setgid", 200001), ("setuid", 200001),
                         ("kill", -1, signal.SIGKILL, 200001),
                         ("kill", -1, signal.SIGKILL, 200001),
                         ("kill", -1, signal.SIGKILL, 200001),
                         ("shm", 200001)]
    assert removed == ["/scoring/t"]


@pytest.mark.parametrize("ops, uid", [
    (FakeOs(1000), 200001),            # the service's own user: kill(-1) would hit it
    (FakeOs(0, honour=False), 200001),  # the drop did not take
    (FakeOs(0), 0),                     # never as root
], ids=["not_root", "drop_failed", "root_slot"])
def test_the_sweeper_never_signals_but_as_a_dedicated_slot_uid(ops, uid):
    removed = RemovedPaths()
    identity = scoring_process.ScoringIdentity(uid=uid, gid=uid)
    with pytest.raises(PermissionError):
        scoring_process._sweep_as_slot("/scoring/t", identity, ops, removed, lambda _s: None,
                                       lambda uid: ops.calls.append(("shm", uid)))
    assert not [call for call in ops.calls if call[0] in ("kill", "shm")] and removed == []


class FakeResource:
    """resource's constants and reads, with setrlimit recorded."""

    def __init__(self):
        self.set = {}

    def __getattr__(self, name):
        return getattr(resource, name)

    def getrlimit(self, which):
        return (resource.RLIM_INFINITY, resource.RLIM_INFINITY)

    def setrlimit(self, which, limits):
        self.set[which] = limits


@pytest.mark.parametrize("dedicated", [True, False])
def test_the_limits_cap_processes_on_a_dedicated_uid_and_every_file(dedicated):
    fake = FakeResource()
    identity = scoring_process.ScoringIdentity(uid=200001, gid=200001, nproc=64,
                                               file_bytes=1 << 20)
    scoring_process._limit(time.monotonic() + 10, 1 << 30, identity, dedicated, fake)
    assert fake.set[resource.RLIMIT_FSIZE] == (1 << 20, 1 << 20)
    assert fake.set.get(resource.RLIMIT_NPROC) == ((64, 64) if dedicated else None)
    assert resource.RLIMIT_CPU in fake.set and resource.RLIMIT_AS in fake.set


class LeavesAGrandchild(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        import subprocess

        child = subprocess.Popen(["sleep", "300"], start_new_session=True)
        await runtime.write("/tmp/grandchild", str(child.pid).encode())
        return 1.0


def processes_of(uids):
    found = []
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                with open(f"/proc/{entry}/status") as status:
                    line = next(x for x in status if x.startswith("Uid:"))
            except (OSError, StopIteration):
                continue
            if int(line.split()[1]) in uids:
                found.append(int(entry))
    return found


@pytest.mark.skipif(os.geteuid() != 0, reason="slot uids and the sweeper need root")
def test_as_root_no_process_of_the_slots_uid_survives_the_grade():
    episode = Episode(LeavesAGrandchild)
    kept = grading_box_files(episode)
    assert episode.grade().reward == 1.0
    pid = int(kept[G0]["/tmp/grandchild"])
    pool = scoring_server.SLOTS
    assert processes_of(range(pool.base, pool.base + pool.count)) == []
    assert not os.path.exists(f"/proc/{pid}") or "Z" in open(f"/proc/{pid}/stat").read().split()[2]


# The forkserver never had the service's secrets.

class ScansItsMemory(ToyTask):
    """`prompt` = "<first half> <second half> [plant]": scores 1 when no writable mapping
    of the process holds the two halves joined (they never are in its own objects)."""

    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        first, second, *plant = self.data.prompt.split()
        head, tail = first.encode(), second.encode()
        planted = bytearray(head) + bytearray(tail) if plant else None  # control
        with open("/proc/self/maps") as maps:
            regions = [line.split() for line in maps]
        found = 0
        with open("/proc/self/mem", "rb", buffering=0) as memory:
            for fields in regions:
                if not fields[1].startswith("rw"):
                    continue
                start, end = (int(x, 16) for x in fields[0].split("-"))
                overlap = len(head) + len(tail)
                while start < end:  # 16 MiB at a time, overlapping by a needle
                    try:
                        memory.seek(start)
                        chunk = memory.read(min(end - start, 16 << 20))
                    except (OSError, OverflowError, ValueError):
                        break
                    at = chunk.find(head)
                    while at >= 0:
                        if chunk[at + len(head):at + overlap] == tail:
                            found += 1
                        at = chunk.find(head, at + 1)
                    if len(chunk) <= overlap or start + len(chunk) >= end:
                        break
                    start += len(chunk) - overlap
                    del chunk
        del planted
        return 0.0 if found else 1.0


@pytest.mark.skipif(os.geteuid() == 0, reason="as root the dropped process is not dumpable: it cannot read its /proc/self/mem")
def test_no_copy_of_a_service_secret_is_in_the_scoring_process_memory(monkeypatch):
    import secrets

    first, second = secrets.token_hex(24), secrets.token_hex(24)
    scoring_server.stop()  # the next forkserver starts while the service holds the secret
    monkeypatch.setenv("RELIQUARY_TEST_HEAP_SECRET", first + second)
    monkeypatch.setattr(scoring_server, "EXTRA_PRELOAD", [])
    try:
        # A preloaded module that keeps what its environment held at import.
        scoring_server.start(preload=["caches_its_environment"])
        clean = Episode(ScansItsMemory, data=toy_data(prompt=f"{first} {second}")).grade()
        control = Episode(ScansItsMemory,
                          data=toy_data(prompt=f"{first} {second} plant")).grade()
        with open(f"/proc/{scoring_server.server_pid()}/environ", "rb") as raw:
            server_env = dict(item.split(b"=", 1) for item in raw.read().split(b"\0") if item)
    finally:
        scoring_server.stop()
    assert set(server_env) <= {b"PATH", b"LANG", b"PYTHONHASHSEED"} | {
        name.encode() for name in THREAD_VARS}
    assert (control.status, control.reward) == ("graded", 0.0)  # the scan finds a copy
    assert (clean.status, clean.reward) == ("graded", 1.0)


def test_the_forkserver_runs_through_the_wrapper_or_not_at_all(monkeypatch, tmp_path):
    scoring_server.stop()
    monkeypatch.setattr(scoring_server, "WRAPPER", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="wrapper"):
        scoring_server.start()
    assert scoring_server.server_pid() is None


# A slow import is not ours while it stays within the ready limit.

def test_a_slow_import_within_the_ready_limit_still_grades(monkeypatch):
    import slow_import_env

    monkeypatch.setattr(scoring_process, "SCORING_STARTUP_FLOOR_S", 2.0)  # < the import
    episode = Episode(slow_import_env.SlowImportTask,
                      data=toy_data(timeout=TaskTimeout(scoring=30.0)),
                      defaults=BridgeDefaults(scoring_ready_s=10.0))
    episode.agent_writes("/work/answer.txt", b"42\n")
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 1.0)


def test_an_env_module_the_forkserver_preloaded_is_not_imported_again(monkeypatch):
    import slow_import_env

    assert slow_import_env.SLOW_IMPORT_S > 2.0
    scoring_server.stop()
    monkeypatch.setattr(scoring_server, "EXTRA_PRELOAD", [])
    try:
        scoring_server.start(preload=["slow_import_env"])
        episode = Episode(slow_import_env.SlowImportTask,
                          data=toy_data(timeout=TaskTimeout(scoring=30.0)),
                          defaults=BridgeDefaults(scoring_ready_s=2.0))
        episode.agent_writes("/work/answer.txt", b"42\n")
        outcome = episode.grade()
    finally:
        scoring_server.stop()
    assert (outcome.status, outcome.reward) == ("graded", 1.0)


def test_the_ready_limit_defaults_to_two_minutes_and_the_floor_stays():
    assert BridgeDefaults().scoring_ready_s == 120.0
    assert FLOOR_S == 30


# After ready, only our own send is ours.

class StopsTheLoop(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        asyncio.get_running_loop().stop()  # asyncio.run raises past `_score`
        await asyncio.sleep(1)
        return 1.0


def test_anything_escaping_the_scoring_run_after_ready_is_graded_zero():
    outcome = Episode(StopsTheLoop).grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "grading_failed"}}


# A controller that cannot drop privileges does not score.

@pytest.mark.skipif(os.geteuid() == 0, reason="checks the unprivileged controller")
def test_an_unprivileged_controller_refuses_to_score_unless_allowed(monkeypatch, caplog):
    episode = Episode()  # opened while allowed
    monkeypatch.delenv("RELIQUARY_SCORING_UNPRIVILEGED_OK")
    with pytest.raises(PermissionError, match="RELIQUARY_SCORING_UNPRIVILEGED_OK"):
        scoring_server.start()
    episode.agent_writes("/work/answer.txt", b"42\n")
    outcome = episode.grade()
    assert outcome.status == "aborted" and outcome.reward is None
    assert "RELIQUARY_SCORING_UNPRIVILEGED_OK" in caplog.text


# A scoring TMPDIR is bounded per file; a failed sweep is retried.

class FillsItsTmpdir(ToyTask):
    """Writes `prompt` bytes to a file in its TMPDIR; 0.5 when the write is refused
    with EFBIG and the code goes on (a `careful` prompt), else lets it through."""

    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        import errno
        import tempfile

        size, *careful = self.data.prompt.split()
        try:
            with tempfile.NamedTemporaryFile() as handle:
                handle.write(b"x" * int(size))
                handle.flush()
        except OSError as exc:
            if careful and exc.errno == errno.EFBIG:
                return 0.5
            raise
        return 1.0


def test_a_scoring_file_past_its_size_cap_fails_in_the_envs_code():
    small = BridgeDefaults(scoring_file_bytes=1 << 20)
    careful = Episode(FillsItsTmpdir, data=toy_data(prompt=f"{2 << 20} careful"),
                      defaults=small)
    assert careful.grade().reward == 0.5  # the process lives on (no SIGXFSZ)
    outcome = Episode(FillsItsTmpdir, data=toy_data(prompt=str(2 << 20)),
                      defaults=small).grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "grading_failed"}}
    assert Episode(FillsItsTmpdir, data=toy_data(prompt=str(1 << 19)),
                   defaults=small).grade().reward == 1.0


def test_a_sweep_that_failed_keeps_its_slot_and_is_retried_at_the_next(monkeypatch):
    pool = scoring_server.ScoringSlots(base=300000, count=2)
    monkeypatch.setattr(scoring_process, "_PENDING", [])
    first, second = pool.acquire(), pool.acquire()
    identity = scoring_process.ScoringIdentity(uid=300000, gid=300000)
    attempts = []

    def sweeper(path, ident):
        attempts.append(path)
        return len(attempts) > 1  # the first sweep times out

    scoring_process._sweep_slot("/scoring/a", identity, pool, first, sweeper)
    assert pool.free() == 0 and len(scoring_process._PENDING) == 1
    scoring_process._sweep_slot("/scoring/b", identity, pool, second, sweeper)
    assert attempts == ["/scoring/a", "/scoring/a", "/scoring/b"]
    assert pool.free() == 2 and scoring_process._PENDING == []


# Round 5. A1: a TMPDIR the env's code locked (chmod 000) is still removed.

def test_the_sweeper_removes_a_tmpdir_whose_subdirectories_the_env_locked(tmp_path):
    top = tmp_path / "reliquary-scoring-x"
    locked, read_only = top / "locked" / "deeper", top / "read-only"
    locked.mkdir(parents=True)
    read_only.mkdir()
    (locked / "f").write_bytes(b"x")
    (read_only / "g").write_bytes(b"y")
    os.symlink("/", top / "link")
    os.chmod(locked, 0o000)
    os.chmod(top / "locked", 0o000)
    os.chmod(read_only, 0o500)
    try:
        scoring_process._remove_tree(str(top))
        assert not os.path.lexists(top)
    finally:
        for path in (top / "locked", locked, read_only):
            if os.path.lexists(path):
                os.chmod(path, 0o700)
    import inspect

    sweep = inspect.signature(scoring_process._sweep_as_slot).parameters
    assert sweep["rmtree"].default is scoring_process._empty_dir


def test_the_sweeper_empties_the_tmpdir_and_keeps_it_for_the_controller(tmp_path):
    """As the slot's uid the sweeper cannot unlink an entry of /scoring (root's, 0755): it
    empties the TMPDIR, locked subdirectories and links included, and leaves it."""
    top = tmp_path / "reliquary-scoring-y"
    (top / "locked" / "deeper").mkdir(parents=True)
    (top / "locked" / "deeper" / "f").write_bytes(b"x")
    (top / "file").write_bytes(b"y")
    os.symlink("/", top / "link")
    os.chmod(top / "locked" / "deeper", 0o000)
    os.chmod(top / "locked", 0o000)
    os.chmod(top, 0o500)
    try:
        scoring_process._empty_dir(str(top))
        assert top.is_dir() and os.listdir(top) == []
    finally:
        for path in (top, top / "locked", top / "locked" / "deeper"):
            if os.path.lexists(path):
                os.chmod(path, 0o700)


def test_the_controller_removes_the_tmpdir_the_sweeper_emptied_as_root(tmp_path, monkeypatch):
    top = tmp_path / "reliquary-scoring-z"
    (top / "sub").mkdir(parents=True)

    class Sweeper:  # the sweeper process, run inline: it empties, never removes, the dir
        def __init__(self, target, args, **_kw):
            self.args, self.exitcode = args, None

        def start(self):
            scoring_process._empty_dir(self.args[0])
            self.exitcode = 0

        def join(self, *_a):
            pass

        def close(self):
            pass

    monkeypatch.setattr(scoring_process.os, "geteuid", lambda: 0)
    monkeypatch.setattr(scoring_server.CONTEXT, "Process", Sweeper)
    identity = scoring_process.ScoringIdentity(uid=200001, gid=200001)
    assert scoring_process._run_sweeper(str(top), identity) is True
    assert not os.path.lexists(top)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the modes that lock the tree")
def test_a_tree_the_sweeper_cannot_unlock_is_an_error(tmp_path, monkeypatch):
    top = tmp_path / "t"
    (top / "locked").mkdir(parents=True)
    os.chmod(top / "locked", 0o000)
    monkeypatch.setattr(scoring_process.os, "chmod", lambda *_a, **_k: None)  # cannot unlock
    try:
        with pytest.raises(PermissionError):
            scoring_process._remove_tree(str(top))
    finally:
        monkeypatch.undo()
        os.chmod(top / "locked", 0o700)


# A2 (c): the sweeper removes what its uid left in /dev/shm, nothing else.

def test_the_sweeper_removes_only_its_uids_files_in_dev_shm(tmp_path):
    owners = {"mine": 200001, "sem.mine": 200001, "dir-mine": 200001,
              "theirs": 200002, "root": 0, "inner": 200001}
    for name in owners:
        if name.startswith("dir-"):
            (tmp_path / name / "inner").mkdir(parents=True)
        else:
            (tmp_path / name).write_bytes(b"x")

    def lstat(path):
        info = os.lstat(path)
        fields = list(info)
        fields[4] = owners[os.path.basename(path)]  # st_uid
        return os.stat_result(fields)

    scoring_process._sweep_shm(200001, root=str(tmp_path), lstat=lstat)
    assert sorted(os.listdir(tmp_path)) == ["root", "theirs"]
    scoring_process._sweep_shm(200001, root=str(tmp_path / "missing"), lstat=lstat)
    assert scoring_process.SHM_ROOT == "/dev/shm"


def test_a_foreign_unreadable_entry_inside_the_slots_shm_dir_is_logged_not_fatal(
        tmp_path, caplog, monkeypatch):
    owners = {"mine": 200001, "foreign": 200002}
    (tmp_path / "mine" / "foreign").mkdir(parents=True)
    (tmp_path / "mine" / "foreign" / "inner").write_bytes(b"x")
    os.chmod(tmp_path / "mine" / "foreign", 0o000)

    def lstat(path):
        info = os.lstat(path)
        fields = list(info)
        fields[4] = owners[os.path.basename(path)]  # st_uid
        return os.stat_result(fields)

    real_chmod = os.chmod
    monkeypatch.setattr(scoring_process.os, "chmod", lambda *_a, **_k: None)  # not ours: no unlock
    try:
        with caplog.at_level("WARNING"):
            scoring_process._sweep_shm(200001, root=str(tmp_path), lstat=lstat)  # no raise
        assert "foreign" in caplog.text
    finally:
        real_chmod(tmp_path / "mine" / "foreign", 0o700)


# B4: one grade retries at most one pending sweep.

def test_a_grade_retries_at_most_one_pending_sweep(monkeypatch):
    pool = scoring_server.ScoringSlots(base=300000, count=4)
    identity = scoring_process.ScoringIdentity(uid=300000, gid=300000)
    slots = [pool.acquire() for _ in range(4)]
    monkeypatch.setattr(scoring_process, "_PENDING",
                        [(f"/scoring/{n}", identity, pool, slots[n]) for n in range(3)])
    attempts = []

    def sweeper(path, ident):
        attempts.append(path)
        return True

    scoring_process._sweep_slot("/scoring/now", identity, pool, slots[3], sweeper)
    assert attempts == ["/scoring/0", "/scoring/now"]
    assert [item[0] for item in scoring_process._PENDING] == ["/scoring/1", "/scoring/2"]
    assert pool.free() == 2


# B5: the scoring process's math libraries run on one thread.

class ReportsItsThreads(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        return 1.0 if all(os.environ.get(name) == "1" for name in THREAD_VARS) else 0.0


def test_the_scoring_process_runs_its_math_libraries_on_one_thread():
    scoring_server.stop()  # a forkserver started through the wrapper now
    try:
        outcome = Episode(ReportsItsThreads).grade()
        with open(f"/proc/{scoring_server.server_pid()}/environ", "rb") as raw:
            server_env = dict(item.split(b"=", 1) for item in raw.read().split(b"\0") if item)
    finally:
        scoring_server.stop()
    assert (outcome.status, outcome.reward) == ("graded", 1.0)
    assert all(server_env.get(name.encode()) == b"1" for name in THREAD_VARS)


# B6: the host's OOM killer takes a scoring process first.

class ReportsItsOomScore(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        with open("/proc/self/oom_score_adj") as raw:
            return 1.0 if raw.read().strip() == "1000" else 0.0


def test_the_scoring_process_is_the_first_the_oom_killer_takes():
    assert Episode(ReportsItsOomScore).grade().reward == 1.0


def test_an_oom_score_that_cannot_be_set_is_logged_not_fatal(tmp_path, caplog):
    scoring_process._prefer_for_oom_kill(str(tmp_path))  # a directory: EISDIR
    assert "oom_score_adj" in caplog.text
    target = tmp_path / "oom_score_adj"
    target.write_text("0")
    scoring_process._prefer_for_oom_kill(str(target))
    assert target.read_text() == "1000"


# B1: an env module the forkserver cannot preload stops the service's start.

def test_the_preload_check_imports_the_env_modules_as_the_forkserver_would(monkeypatch):
    scoring_server.check_preload(["toy_bridge_env"])
    with pytest.raises(RuntimeError, match="no_such_env_module"):
        scoring_server.check_preload(["toy_bridge_env", "no_such_env_module"])
    # Importable in the service, which holds the credential; not in the forkserver.
    monkeypatch.setenv("RELIQUARY_TEST_IMPORT_CREDENTIAL", "token")
    import needs_a_credential_env  # noqa: F401

    with pytest.raises(RuntimeError, match="needs_a_credential_env.*credential"):
        scoring_server.check_preload(["needs_a_credential_env"])


# The scoring process has no network. Verified live by the hostile env's live run
# (as root).

def test_the_scoring_process_has_no_network_but_unix_sockets():
    # Imported here, never at module level: its import turns the audit hook off in the
    # scoring processes that unpickle its tasks (see its docstring).
    from seccomp_probes import NET_PROBE, ProbesTheNetwork

    episode = Episode(ProbesTheNetwork)
    kept = grading_box_files(episode)
    outcome = episode.grade()
    seen = json.loads(kept[G0][NET_PROBE])
    assert seen["AF_INET"] == "EPERM" and seen["AF_INET6"] == "EPERM"
    assert seen["AF_NETLINK"] == "EPERM" and seen["AF_PACKET"] == "EPERM"
    assert seen["AF_UNIX"] == "ok" and seen["io_uring"] == "EPERM"
    assert seen["seccomp"] == "2"  # a filter, inherited by anything the reward starts
    assert (outcome.status, outcome.reward) == ("graded", 1.0)


def test_the_scoring_process_cannot_reach_into_another_process():
    from seccomp_probes import NET_PROBE, ProbesProcessReach

    episode = Episode(ProbesProcessReach)
    kept = grading_box_files(episode)
    outcome = episode.grade()
    seen = json.loads(kept[G0][NET_PROBE])
    assert seen == {name: "EPERM" for name in
                    ("ptrace", "process_vm_readv", "process_vm_writev", "pidfd_getfd")}
    assert (outcome.status, outcome.reward) == ("graded", 1.0)


class ConnectsToTheHost(ToyTask):
    """`prompt` = "<tcp port> <unix socket path>": scores 1 only when a TCP connection to
    the host's 127.0.0.1 is refused at socket() while a Unix socket on the host answers."""

    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        import socket

        port, path = self.data.prompt.split()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
                tcp.settimeout(5)
                tcp.connect(("127.0.0.1", int(port)))
            return 0.0  # reached the host
        except PermissionError:
            pass
        except OSError:
            return 0.0  # a socket opened: the filter did not apply
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as unix:
            unix.settimeout(5)
            unix.connect(path)
            return 1.0 if unix.recv(2) == b"ok" else 0.0


def test_a_reward_cannot_reach_the_host_over_tcp_but_unix_sockets_work():
    """Through the real forkserver and slot path (as root: the slot's uid, so the Unix
    socket lives where any uid can reach it, not in pytest's 0700 directory)."""
    import shutil
    import socket
    import tempfile

    tcp = socket.create_server(("127.0.0.1", 0))
    directory = tempfile.mkdtemp(prefix="unix-probe-")
    os.chmod(directory, 0o777)
    path = os.path.join(directory, "s")
    unix = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    unix.bind(path)
    os.chmod(path, 0o777)
    unix.listen()
    tcp.settimeout(0.2)
    reached = []

    def serve():
        connection, _ = unix.accept()
        connection.sendall(b"ok")
        connection.close()

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    try:
        prompt = f"{tcp.getsockname()[1]} {path}"
        outcome = Episode(ConnectsToTheHost, data=toy_data(prompt=prompt)).grade()
        try:
            reached.append(tcp.accept())
        except TimeoutError:
            pass
        assert (outcome.status, outcome.reward) == ("graded", 1.0), outcome
        assert reached == []
    finally:
        server.join(5)
        tcp.close()
        unix.close()
        shutil.rmtree(directory, ignore_errors=True)


@pytest.mark.parametrize("machine", ["riscv64", "i686", "armv7l", "s390x"])
def test_the_network_filter_refuses_an_architecture_it_does_not_know(machine):
    with pytest.raises(OSError, match="architecture"):
        scoring_process._network_filter(machine)


def run_filter(program, nr, arch, arg0=0):
    """A classic-BPF interpreter for the few instructions the filter uses."""
    data = {0: nr, 4: arch, 16: arg0}
    pc, acc = 0, 0
    while True:
        code, jt, jf, k = program[pc]
        if code == 0x20:
            acc = data[k]
        elif code in (0x15, 0x35):
            taken = acc == k if code == 0x15 else acc >= k
            pc += jt if taken else jf
        elif code == 0x06:
            return k
        else:
            raise AssertionError(f"unexpected instruction {code:#x}")
        pc += 1


@pytest.mark.parametrize("machine", ["x86_64", "aarch64"])
def test_the_network_filter_allows_only_unix_sockets_on_its_own_abi(machine):
    import socket

    program = scoring_process._network_filter(machine)
    arch, nr_socket, nr_io_uring, x32 = scoring_process.SECCOMP_ARCHES[machine]
    allow = scoring_process.SECCOMP_RET_ALLOW
    deny = scoring_process.SECCOMP_RET_ERRNO | 1
    kill = scoring_process.SECCOMP_RET_KILL_PROCESS
    for family in (socket.AF_INET, socket.AF_INET6, socket.AF_NETLINK, socket.AF_PACKET,
                   socket.AF_VSOCK, 0xFFFFFFFF):
        assert run_filter(program, nr_socket, arch, family) == deny
    assert run_filter(program, nr_socket, arch, socket.AF_UNIX) == allow
    assert run_filter(program, nr_io_uring, arch) == deny
    for nr in scoring_process.DENIED_SYSCALLS[machine]:
        assert run_filter(program, nr, arch) == deny
    assert run_filter(program, 0, arch, socket.AF_INET) == allow  # any other syscall
    for foreign in (0x40000003, 0x40000028, 0xC00000B7 ^ 0xC000003E ^ arch):  # i386, arm
        assert run_filter(program, nr_socket, foreign, socket.AF_UNIX) == kill
    assert run_filter(program, nr_socket | 0x40000000, arch, socket.AF_UNIX) == (
        kill if x32 else allow)  # aarch64: an unknown syscall (ENOSYS)


def test_a_scoring_process_without_its_network_filter_refuses_to_score(monkeypatch):
    """The filter cannot be installed (here: an architecture it does not know): the
    scoring process exits before `ready`, so the grade is ours (aborted), never scored."""
    episode = Episode(ToyTask)  # opened before the forkserver loses its filter
    scoring_server.stop()
    monkeypatch.setattr(scoring_server, "EXTRA_PRELOAD", [])
    try:
        scoring_server.start(preload=["foreign_arch_preload"])
        episode.agent_writes("/work/answer.txt", b"42\n")
        outcome = episode.grade()
    finally:
        scoring_server.stop()
    assert outcome.status == "aborted" and outcome.reward is None


# A pipe that cannot be made leaks neither the slot nor the TMPDIR.

def test_a_pipe_that_cannot_be_made_is_ours_and_frees_the_slot_and_tmpdir(monkeypatch,
                                                                          tmp_path):
    def no_pipe(duplex=True):
        raise OSError(24, "Too many open files")

    episode = Episode()
    monkeypatch.setattr(scoring_server, "TMP_ROOT", str(tmp_path))
    monkeypatch.setattr(scoring_server.CONTEXT, "Pipe", no_pipe, raising=False)
    free = scoring_server.SLOTS.free()
    episode.agent_writes("/work/answer.txt", b"42\n")
    outcome = episode.grade()
    assert outcome.status == "aborted" and outcome.reward is None
    assert scoring_server.SLOTS.free() == free
    assert list(tmp_path.iterdir()) == []


# -- every hook in a scoring process; grading_setup before the restore ---------------


class HooksReportTheirProcess(ToyTask):
    async def setup(self, runtime):
        await runtime.write("/work/setup.pid", str(os.getpid()).encode())
        await super().setup(runtime)

    async def finalize(self, runtime):
        await runtime.write("/work/finalize.pid", str(os.getpid()).encode())
        await super().finalize(runtime)


def test_setup_and_finalize_run_in_scoring_processes_not_in_the_controller():
    episode = Episode(HooksReportTheirProcess)
    episode.agent_writes("/work/answer.txt", b"42\n")
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 1.0)
    files = episode.boxes.boxes["s-1"]["files"]
    assert int(files["/work/setup.pid"]) != os.getpid()
    assert int(files["/work/finalize.pid"]) != os.getpid()


class SetupRaises(ToyTask):
    async def setup(self, runtime):
        raise RuntimeError("the box is not what the task says")


def test_a_setup_that_fails_fails_the_open():
    with pytest.raises(HookFailed, match="setup"):
        Episode(SetupRaises)


class FinalizeRaises(ToyTask):
    async def finalize(self, runtime):
        raise ValueError("finalize could not parse the agent's output")


def test_a_finalize_that_raises_grades_zero_extract_failed():
    episode = Episode(FinalizeRaises)
    episode.agent_writes("/work/answer.txt", b"42\n")
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "extract_failed"}}
    assert episode.grader.env_failure_counts() == {"toy": {"extract_failed": 1}}


class FinalizeSpins(ToyTask):
    async def finalize(self, runtime):
        while True:
            pass


def test_a_finalize_past_its_timeout_grades_zero_extract_timeout(monkeypatch):
    episode = Episode(FinalizeSpins, data=toy_data(timeout=TaskTimeout(finalize=3.0,
                                                                       scoring=3.0)))
    longer = grader_module.run_step  # the step timer 60 s past the hook's deadline: only
    monkeypatch.setattr(grader_module, "run_step",  # the hook's own bounds can stop it
                        lambda factory, timeout_s: longer(factory, timeout_s + 60))
    started = time.monotonic()
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "extract_timeout"}}
    assert time.monotonic() - started < 30


class GradesFromThePreparedImage(ToyTask):
    async def grading_setup(self, runtime):
        try:
            await runtime.read("/work/out/answer.txt", max_bytes=64)
            seen = b"restored-too-early"
        except OSError:
            seen = b"clean"
        await runtime.write("/work/prepared.txt", seen)

    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        try:
            prepared = await runtime.read("/work/prepared.txt", max_bytes=64)
            answer = await runtime.read("/work/out/answer.txt", max_bytes=64)
        except OSError:
            return 0.0
        return 1.0 if prepared == b"clean" and answer.strip() == b"42" else 0.0


def test_grading_setup_prepares_the_pristine_box_before_the_restore():
    episode = Episode(GradesFromThePreparedImage)
    episode.agent_writes("/work/answer.txt", b"42\n")
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 1.0)


class GradingSetupRaises(ToyTask):
    async def grading_setup(self, runtime):
        raise RuntimeError("the image is not what the corpus says")


def test_a_grading_setup_failure_is_ours_never_the_agents():
    episode = Episode(GradingSetupRaises)
    episode.agent_writes("/work/answer.txt", b"42\n")
    outcome = episode.grade()
    assert outcome.status == "aborted"
    assert "env_infra_error" in episode.grader.env_failure_counts()["toy"]


class GradingSetupSpins(ToyTask):
    async def grading_setup(self, runtime):
        while True:
            pass


def test_a_grading_setup_past_its_timeout_is_ours():
    episode = Episode(GradingSetupSpins, data=toy_data(timeout=TaskTimeout(scoring=10.0)))
    episode.agent_writes("/work/answer.txt", b"42\n")
    assert episode.grade().status == "aborted"


class GradingSetupSleeps(ToyTask):
    """Sleeps `prompt` seconds (no CPU: only a deadline stops it)."""

    async def grading_setup(self, runtime):
        time.sleep(float(self.data.prompt))


def test_a_grading_setup_past_its_reserved_time_is_ours_even_within_the_step():
    # A 10 s step (floor 0 here): grading_setup gets 10 - 0 - 5 = 5 s, so a 7.5 s sleep is
    # killed (ours); with the step's own deadline it would end in time and be scored.
    episode = Episode(GradingSetupSleeps,
                      data=toy_data(prompt="7.5", timeout=TaskTimeout(scoring=10.0)))
    episode.agent_writes("/work/answer.txt", b"42\n")
    outcome = episode.grade()
    assert outcome.status == "aborted" and outcome.reward is None
    assert "env_infra_error" in episode.grader.env_failure_counts()["toy"]


def test_a_grading_setup_within_its_reserved_time_is_scored():
    episode = Episode(GradingSetupSleeps,
                      data=toy_data(prompt="0.5", timeout=TaskTimeout(scoring=10.0)))
    episode.agent_writes("/work/answer.txt", b"42\n")
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 1.0)


def test_a_grading_setup_without_time_for_two_scoring_processes_is_refused(monkeypatch):
    monkeypatch.setattr(scoring_process, "SCORING_STARTUP_FLOOR_S", FLOOR_S)
    monkeypatch.setattr(declaration, "MIN_GRADING_TIMEOUT_S", 35.0)
    short = toy_data(timeout=TaskTimeout(scoring=2 * FLOOR_S + 9))
    with pytest.raises(TaskDeclarationError, match="grading_setup"):
        task_from_verifiers(GradingSetupSleeps(short), lock=LOCK)
    task_from_verifiers(ToyTask(short), lock=LOCK)  # no grading_setup: one process
    task_from_verifiers(GradingSetupSleeps(toy_data(timeout=TaskTimeout(
        scoring=2 * FLOOR_S + 10))), lock=LOCK)


class FinalizeLeavesAWriter(ToyTask):
    """finalize copies the answer, then its command leaves a process that rewrites it."""

    async def finalize(self, runtime):
        await super().finalize(runtime)
        await runtime.run(["start-writer"], {})


def test_a_process_finalize_started_is_stopped_before_the_archive():
    episode = Episode(FinalizeLeavesAWriter)
    boxes = episode.boxes
    exec_, stop, tar = boxes.exec, boxes.stop_processes, boxes.tar_create

    def exec_with_writer(box_id, argv, **kwargs):
        if argv == ["start-writer"]:
            boxes.boxes[box_id]["writer"] = True
            return exec_(box_id, ["true"], **kwargs)
        return exec_(box_id, argv, **kwargs)

    def stop_with_writer(box_id, **kwargs):
        boxes.boxes[box_id].pop("writer", None)
        return stop(box_id, **kwargs)

    def tar_after_the_writer(box_id, members, max_bytes, **kwargs):
        if boxes.boxes[box_id].get("writer"):  # still running: it rewrote the artifact
            boxes.boxes[box_id]["files"]["/work/out/answer.txt"] = b"41\n"
        return tar(box_id, members, max_bytes, **kwargs)

    boxes.exec, boxes.stop_processes, boxes.tar_create = (
        exec_with_writer, stop_with_writer, tar_after_the_writer)
    episode.agent_writes("/work/answer.txt", b"42\n")
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 1.0)


def test_the_bridge_version_is_two():
    assert json.loads(setup_output(ToyTask(toy_data())))["bridge"] == 2


# -- the runtime failing to start a command (runsc's generic start failure) --------

class RunsAProgram(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        await runtime.run(["true"], {})  # the program under test
        return await ToyTask.solved(self, runtime)


def solved_episode(task_class=RunsAProgram):
    episode = Episode(task_class)
    episode.agent_writes("/work/answer.txt", b"42\n")
    return episode


def test_a_grading_attempt_the_runtime_failed_is_run_again_in_a_fresh_box():
    episode = solved_episode()
    episode.boxes.fail_start[("g-", "true")] = 1
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 1.0)
    assert Grader.pristine_box_id("s-1", 1) in episode.boxes.removed  # the second attempt


def test_the_runtime_failing_every_grading_attempt_keeps_the_old_zero_marked():
    # Never a void: the agent's code ran in that box and might have caused it.
    episode = solved_episode()
    episode.boxes.fail_start[("g-", "true")] = 99
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "grading_box_failed",
                                         "runtime_exec_failed": True}}
    assert episode.boxes.fail_start[("g-", "true")] == 97  # one per attempt


def test_no_time_for_a_retry_keeps_the_zero_not_a_void():
    episode = solved_episode()
    episode.boxes.fail_start[("g-", "true")] = 99
    episode.grader.clock = lambda: 0.0
    # A whole first attempt fits, a retry would not (it needs RETRY_MARGIN_S on top).
    deadline = episode.task.grading_timeout_s + grader_module.RETRY_MARGIN_S / 2
    outcome = episode.grader.grade("s-1", "s-1", episode.task, episode.spec, env="toy",
                                   deadline=deadline)
    assert outcome.status == "graded" and outcome.reward == 0.0
    assert outcome.facts["_grader"]["runtime_exec_failed"] is True


def test_a_grading_box_the_agents_code_killed_is_not_run_again():
    episode = solved_episode(Breaks)
    outcome = episode.grade()
    assert outcome.facts == {"_grader": {"agent_fault": "grading_box_failed"}}
    assert Grader.pristine_box_id("s-1", 1) not in episode.boxes.removed


def test_the_runtime_failing_to_stop_the_agents_processes_is_box_failed_with_its_reason():
    from reliquary_sandbox_service.task_runtime import BoxFault, RuntimeExecFailed

    episode = solved_episode()
    episode.boxes.fail_stop = RuntimeExecFailed("Stopping the agent's processes: ...")
    outcome = episode.grade()
    assert outcome.status == "box_failed"
    assert outcome.reason == grader_module.START_FAILED_REASON
    episode = solved_episode()
    episode.boxes.fail_stop = BoxFault("Stopping the agent's processes: the box is running")
    assert episode.grade().reason == "the agent's box is unusable"


def test_a_start_failure_then_an_infra_failure_keeps_the_zero_not_a_void():
    # The attempt that hit the runtime start failure decides: never `aborted`.
    episode = solved_episode()
    episode.boxes.fail_start[("g-", "true")] = 1
    episode.boxes.broken.add(Grader.pristine_box_id("s-1", 1))  # ours, on the retry
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts == {"_grader": {"agent_fault": "grading_box_failed",
                                         "runtime_exec_failed": True}}


def test_a_box_fault_naming_the_runtime_failure_is_the_agents_and_not_run_again():
    # The grader goes by the runtime's classification, never by text in a message.
    from reliquary_sandbox_service.task_runtime import BoxFault

    episode = solved_episode()
    real = episode.boxes.exec

    def exec_(box_id, argv, **kwargs):
        if box_id.startswith("g-") and argv[:1] == ["true"]:
            raise BoxFault("OCI runtime exec failed: /usr/local/bin/runsc did not terminate "
                           "successfully: exit status 128")
        return real(box_id, argv, **kwargs)

    episode.boxes.exec = exec_
    outcome = episode.grade()
    assert outcome.facts == {"_grader": {"agent_fault": "grading_box_failed"}}
    assert Grader.pristine_box_id("s-1", 1) not in episode.boxes.removed


class WritesThenRuns(ToyTask):
    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        await runtime.run(["prepare"], {})  # the code under test writes /work/prepared
        await runtime.run(["true"], {})
        return 1.0


def test_the_zero_kept_after_start_failures_records_the_canaries():
    # A conformance episode: a canary that ran before the failure is still seen.
    episode = solved_episode(WritesThenRuns)
    episode.grader.canaries["s-1"] = ("/work/prepared", "/never")
    episode.boxes.fail_start[("g-", "true")] = 99
    outcome = episode.grade()
    assert (outcome.status, outcome.reward) == ("graded", 0.0)
    assert outcome.facts["_grader"] == {
        "agent_fault": "grading_box_failed", "runtime_exec_failed": True,
        "canary_checked": 2, "canary_executed": ["/work/prepared"]}
