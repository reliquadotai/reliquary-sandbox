"""The env conformance suite, run on the toy env through a gateway over FakeBoxes."""

import io
import json
import tarfile
from dataclasses import replace

import pytest

pytest.importorskip("verifiers.v1")

import hostile_bridge_env  # noqa: E402
import toy_bridge_env  # noqa: E402

from reliquary_sandbox_service.episodes import conformance  # noqa: E402
from reliquary_sandbox_service.episodes.conformance import (  # noqa: E402
    BUILTIN,
    ENV,
    SAMPLE_SIZE,
    CaseRun,
    ConformanceCase,
    ConformanceError,
    builtin_cases,
    check_case,
    env_cases,
    plan_cases,
    run_conformance,
    suite_problems,
)
from reliquary_sandbox_service.episodes.registry import bridged_env_package_of  # noqa: E402
from reliquary_sandbox_service.episodes.testing import CONFORMANCE_DIRECTIVE, fake_gateway  # noqa: E402
from reliquary_sandbox_verifiers import bridge  # noqa: E402
from reliquary_sandbox_verifiers.loading import VerifiersEnv  # noqa: E402

TOY = "verifiers:toy-bridge-env==unpackaged"
HOSTILE = "verifiers:hostile-bridge-env==unpackaged"
GW = {"episode_allow_unpackaged_verifiers": True}
GRADED_1 = CaseRun("graded", 1.0, None, True)
GRADED_0 = CaseRun("graded", 0.0, None, True)
BUILTIN_NAMES = ["untouched", "empty_artifacts", "symlinked_artifacts", "links_into_grader",
                 "planted_reward_files", "background_reward_writer", "shimmed_tools"]
ON_REFERENCE = ["malformed_empty", "malformed_not_utf8", "malformed_invalid_json",
                "malformed_hostile", "reference_with_links_into_grader",
                "reference_with_oversized_output", "reference_with_large_artifact"]
SALT = "salt-0"
TOY_SEED = conformance.sample_seed("toy-bridge-env", "rl", SALT)


def toy_env():
    return VerifiersEnv(TOY, allow_unpackaged=True)


def lying_cases(split):
    """An operator's cases whose 'reference' is a wrong answer: the suite must fail it."""
    return [{"name": "not_a_solution", "index": 0, "expect": 1.0,
             "calls": [["bash", {"command": "write /work/answer.txt 41"}]]},
            {"name": "wrong", "index": 0, "expect": 0.0,
             "calls": [["bash", {"command": "write /work/answer.txt 40"}]]}]


@pytest.fixture
def scoring_children(monkeypatch):
    """Every grade's scoring process start (the bridge's `run_in_child` with the reward
    step's hook, `score`), counted; the setup and finalize hooks' processes are not."""
    started = []
    real = bridge.run_in_child

    async def counted(task, *args, **kwargs):
        if kwargs.get("hook") == "score":
            started.append(task.data.idx)
        return await real(task, *args, **kwargs)

    monkeypatch.setattr(bridge, "run_in_child", counted)
    return started


def test_the_env_declares_its_cases_as_plain_data():
    cases = env_cases(toy_bridge_env, "rl")
    assert [(c.name, c.index, c.expect, c.kind, c.source) for c in cases] == [
        ("reference", 0, 1.0, "reference", ENV), ("wrong_answer", 0, 0.0, "attack", ENV),
        ("answer_in_the_wrong_file", 0, 0.0, "attack", ENV)]
    assert cases[0].calls == (("bash", {"command": "write /work/answer.txt 42"}),)


def test_an_operator_entry_replaces_the_envs_cases():
    cases = env_cases(toy_bridge_env, "rl", "test_conformance:lying_cases")
    assert [c.name for c in cases] == ["not_a_solution", "wrong"]


@pytest.mark.parametrize("raw", [
    {"name": "x", "index": 0, "expect": 1.0},
    {"name": "", "index": 0, "expect": 1.0, "calls": []},
    {"name": "x", "index": -1, "expect": 1.0, "calls": []},
    {"name": "x", "index": True, "expect": 1.0, "calls": []},
    {"name": "x", "index": 0, "expect": 0.5, "calls": []},
    {"name": "x", "index": 0, "expect": True, "calls": []},
    {"name": "x", "index": 0, "expect": 1.0, "calls": "bash"},
    {"name": "x", "index": 0, "expect": 1.0, "calls": [["python", {}]]},
    {"name": "x", "index": 0, "expect": 1.0, "calls": [[["bash"], {}]]},
    {"name": "x", "index": 0, "expect": 1.0, "calls": [["bash", "ls"]]},
    {"name": "x", "index": 0, "expect": 1.0, "calls": [], "extra": 1},
    {"name": "untouched@0", "index": 0, "expect": 0.0, "calls": []},
    "not a case",
])
def test_malformed_cases_are_refused(raw, monkeypatch):
    monkeypatch.setattr(toy_bridge_env, "conformance_cases", lambda split: [raw])
    with pytest.raises(ConformanceError):
        env_cases(toy_bridge_env, "rl")


def test_missing_or_duplicate_cases_are_refused(monkeypatch):
    with pytest.raises(ConformanceError, match="conformance_cases"):
        env_cases(json, "rl")
    with pytest.raises(ConformanceError, match="cannot load"):
        env_cases(toy_bridge_env, "rl", "no_such_module:cases")
    with pytest.raises(ConformanceError, match="callable"):
        env_cases(toy_bridge_env, "rl", "toy_bridge_env:IMAGE_TAG")
    twice = {"name": "x", "index": 0, "expect": 0.0, "calls": []}
    monkeypatch.setattr(toy_bridge_env, "conformance_cases", lambda split: [twice, twice])
    with pytest.raises(ConformanceError, match="unique"):
        env_cases(toy_bridge_env, "rl")
    monkeypatch.setattr(toy_bridge_env, "conformance_cases", lambda split: twice)
    with pytest.raises(ConformanceError, match="list"):
        env_cases(toy_bridge_env, "rl")


def test_builtin_attacks_target_the_tasks_artifact_roots():
    cases = builtin_cases(["/work/out", "/logs/artifacts"], "rl", 3)
    assert [c.name for c in cases] == [f"{name}@3" for name in BUILTIN_NAMES]
    assert all(c.expect == 0.0 and c.source == BUILTIN and c.index == 3 for c in cases)
    assert cases[0].calls == ()
    assert cases[1].calls == (("bash", {"command": "rm -rf -- /work/out /logs/artifacts"}),)
    assert "ln -s / /work/out" in cases[2].calls[0][1]["command"]
    assert "ln -s /logs/verifier /work/out/.verifier" in cases[3].calls[0][1]["command"]
    assert "ln -s /tests /logs/artifacts/.tests" in cases[3].calls[0][1]["command"]


def test_cases_built_on_the_reference_corrupt_what_it_wrote():
    reference = ConformanceCase("ref", "rl", 3, 1.0, (("bash", {"command": "solve"}),))
    cases = builtin_cases(["/work/out"], "rl", 3, reference=reference, hostile_bytes=1000,
                          tag="t0")
    assert [c.name for c in cases] == [f"{name}@3" for name in BUILTIN_NAMES + ON_REFERENCE]
    by_name = {c.name: c for c in cases}
    for kind in ("empty", "not_utf8", "invalid_json", "hostile"):
        mark, solve, corrupt = by_name[f"malformed_{kind}@3"].calls
        assert mark[1]["command"].startswith(CONFORMANCE_DIRECTIVE + "mark\n")
        assert mark[1]["command"].endswith(" && sleep 1")  # busybox: whole-second mtimes
        assert solve == ("bash", {"command": "solve"})
        assert corrupt[1]["command"].startswith(f"{CONFORMANCE_DIRECTIVE}corrupt {kind} 1000\n")
        assert "-newer /tmp/.t0m" in corrupt[1]["command"]
        assert " /work/out -xdev" in corrupt[1]["command"]
    linked = by_name["reference_with_links_into_grader@3"]
    assert linked.calls[0] == solve and "ln -s /tests /work/out/.tests" in linked.calls[1][1]["command"]
    loud = by_name["reference_with_oversized_output@3"]
    assert loud.expect == 1.0 and loud.kind == "reference" and loud.calls[0] == solve
    large = by_name["reference_with_large_artifact@3"]
    assert large.expect == 1.0 and large.calls[0] == solve
    assert large.calls[1][1]["command"].startswith(
        f"{CONFORMANCE_DIRECTIVE}fill {conformance.LARGE_ARTIFACT_BYTES} .t0f /work/out\n")
    assert all(c.source == BUILTIN for c in cases)


def _bin(tmp_path, tools):
    """A PATH directory holding only `tools` (links to the host's)."""
    import os
    import shutil

    directory = tmp_path / "bin"
    directory.mkdir(exist_ok=True)
    for tool in tools:
        os.symlink(shutil.which(tool), directory / tool)
    return str(directory)


def test_a_script_missing_a_tool_exits_4_and_changes_nothing(tmp_path):
    import subprocess

    victim = tmp_path / "answer.txt"
    victim.write_text("42\n")
    script = conformance._corrupt_script(
        "hostile", [str(tmp_path)], 100, "/dev/null").replace("find / ", f"find {tmp_path} ")
    path = _bin(tmp_path, ["find", "sort", "wc", "tr"])  # no head
    done = subprocess.run(["/bin/sh", "-c", script], env={"PATH": path}, timeout=30,
                          capture_output=True)
    assert done.returncode == 4 and b"head is missing" in done.stderr
    assert victim.read_text() == "42\n"


def test_the_fill_script_writes_inside_the_directory_roots(tmp_path):
    import subprocess

    out, absent, single = tmp_path / "out", tmp_path / "absent", tmp_path / "file.txt"
    out.mkdir()
    single.write_text("solo")
    script = conformance._fill_script([str(out), str(absent), str(single)], 1000, ".fillme")
    script = script.replace(conformance.IMPLICIT_ROOT, str(absent))
    subprocess.run(["/bin/sh", "-c", script], check=True, timeout=30)
    assert (out / ".fillme").read_bytes() == b"x" * 500
    assert (absent / ".fillme").read_bytes() == b"x" * 500  # the implicit root
    assert single.read_text() == "solo"
    none = conformance._fill_script([str(tmp_path / "nothing")], 10, ".fillme")
    assert subprocess.run(["/bin/sh", "-c", none], timeout=30).returncode == 3


def test_the_shim_script_makes_every_tool_and_python_pass(tmp_path):
    """The real script, under /bin/sh, on a PATH and a root of its own (never the host's)."""
    import os
    import shutil
    import subprocess

    tools = tmp_path / "tools"
    tools.mkdir()
    shutil.copy(shutil.which("false"), tools / "pytest")
    os.symlink(shutil.which("false"), tools / "check")
    root = tmp_path / "root"
    (root / "usr" / "lib" / "python3" / "site-packages").mkdir(parents=True)
    kit = tmp_path / "kit"
    script = conformance._shim_script("k")
    for old, new in (("dirs=$PATH\n", f"dirs={tools}\n"), ("find / ", f"find {root} "),
                     (" ".join(conformance.shim_kits("k")), str(kit))):
        assert old in script
        script = script.replace(old, new)
    subprocess.run(["/bin/sh", "-c", script], check=True, timeout=60)
    for tool in ("pytest", "check"):
        done = subprocess.run([str(tools / tool)], capture_output=True, timeout=30)
        assert (done.returncode, done.stdout) == (0, b"pass\n")
    assert os.path.realpath(shutil.which("false")) != str(tools / "check")  # host untouched
    assert subprocess.run([shutil.which("false")], timeout=30).returncode == 1
    site = root / "usr" / "lib" / "python3" / "site-packages"
    done = subprocess.run([shutil.which("python3"), "-c", "raise SystemExit(3)"],
                          env={"PYTHONPATH": str(site)}, capture_output=True, timeout=30)
    assert (done.returncode, done.stdout) == (0, b"pass\n")


def test_the_sample_is_the_whole_split_or_32_seeded_indices():
    assert conformance.sample_indices(7, 5) == [0, 1, 2, 3, 4]
    assert conformance.sample_indices(7, 32) == list(range(32))
    drawn = conformance.sample_indices(7, 10_000)
    assert len(drawn) == 32 == len(set(drawn)) and drawn == sorted(drawn)
    assert drawn == conformance.sample_indices(7, 10_000)
    assert drawn != conformance.sample_indices(8, 10_000)
    assert conformance.sample_seed("a", "rl") == conformance.sample_seed("a", "rl")
    assert conformance.sample_seed("a", "rl") != conformance.sample_seed("a", "eval")


def test_the_corruption_script_rewrites_only_files_newer_than_the_mark(tmp_path):
    """The real script, under /bin/sh, on a directory standing in for `/`."""
    import os
    import subprocess

    root = tmp_path / "root"
    (root / "work" / "out").mkdir(parents=True)
    (root / "work" / "old.txt").write_text("old")
    mark = tmp_path / "mark"
    mark.write_text("")
    os.utime(mark, (1_000_000_000, 1_000_000_000))
    os.utime(root / "work" / "old.txt", (999_999_000, 999_999_000))
    (root / "work" / "answer.txt").write_text("42\n")
    (root / "work" / "out" / "answer.txt").write_text("42\n")
    for kind, expected in (("not_utf8", b"\xff\xfe\x80\xc0"),
                           ("invalid_json", b'{"reward": 1.0, "passed": [tru'),
                           ("hostile", b"[" * 50), ("empty", b"")):
        mark.write_text("")
        os.utime(mark, (1_000_000_000, 1_000_000_000))
        script = conformance._corrupt_script(kind, [str(root / "work" / "out")], 100, str(mark))
        script = script.replace("find / ", f"find {root} ")
        subprocess.run(["/bin/sh", "-c", script], check=True, timeout=30)
        assert not mark.exists()  # the mark does not outlive the case
        assert (root / "work" / "answer.txt").read_bytes() == expected
        assert (root / "work" / "out" / "answer.txt").read_bytes() == expected
        assert (root / "work" / "old.txt").read_text() == "old"
    os.utime(root / "work" / "answer.txt", (999_999_000, 999_999_000))
    os.utime(root / "work" / "out" / "answer.txt", (999_999_000, 999_999_000))
    mark.write_text("")
    os.utime(mark, (1_000_000_000, 1_000_000_000))
    script = conformance._corrupt_script("empty", [], 100, str(mark)).replace(
        "find / ", f"find {root} ")
    assert subprocess.run(["/bin/sh", "-c", script], timeout=30).returncode == 3


def test_the_plan_adds_builtin_cases_for_every_referenced_task_and_the_sample():
    cases = plan_cases(toy_env(), "rl", salt=SALT)
    assert [c.name for c in cases][:3] == ["reference", "wrong_answer", "answer_in_the_wrong_file"]
    assert [c.name for c in cases[3:]] == [f"{n}@0" for n in BUILTIN_NAMES + ON_REFERENCE] + [
        "untouched@1", "empty_artifacts@1"]  # the toy's split has 2 tasks: all sampled
    assert {c.sample_seed for c in cases[-2:]} == {TOY_SEED}
    assert {c.sample_salt for c in cases[-2:]} == {SALT}
    assert {c.sample_seed for c in cases[:-2]} == {None}
    assert len(plan_cases(toy_env(), "rl", builtin=False)) == 3
    corrupt = next(c for c in cases if c.name == "malformed_hostile@0").calls[-1][1]["command"]
    assert corrupt.startswith(f"{CONFORMANCE_DIRECTIVE}corrupt hostile {31 * 1024**2}\n")


def test_a_case_outside_the_split_is_refused(monkeypatch):
    far = {"name": "far", "index": 2, "expect": 1.0, "calls": [["bash", {"command": "x"}]]}
    monkeypatch.setattr(toy_bridge_env, "conformance_cases", lambda split: [far])
    with pytest.raises(ConformanceError, match="outside the split"):
        plan_cases(toy_env(), "rl", builtin=False)


def test_a_case_passes_only_with_its_expected_verified_deterministic_reward():
    reference = ConformanceCase("ref", "rl", 0, 1.0, ())
    assert check_case(reference, [GRADED_1, GRADED_1]) == ()
    assert check_case(reference, [GRADED_1, GRADED_0]) == (
        "run 1: reward 0.0, expected 1.0", "not deterministic: graded/1.0, graded/0.0")
    aborted = CaseRun("aborted", None, "grading failed", True)
    assert check_case(reference, [aborted]) == ("run 0: ended aborted (grading failed)",)
    unverified = CaseRun("graded", 1.0, None, False)
    assert check_case(reference, [unverified]) == ("run 0: the transcript does not verify",)


def test_a_zero_the_envs_scoring_caused_or_a_slow_run_fails_the_case():
    attack = ConformanceCase("atk", "rl", 0, 0.0, ())
    failed = CaseRun("graded", 0.0, None, True, "grading_failed")
    assert check_case(attack, [failed]) == (
        "run 0: graded 0 by grading_failed: the env's scoring raised on the agent's state",)
    for label in ("extract_failed", "invalid_grade_result", "reward_out_of_range"):
        run = CaseRun("graded", 0.0, None, True, label)
        assert check_case(attack, [run])[0].startswith(f"run 0: graded 0 by {label}: ")
    timed_out = CaseRun("graded", 0.0, None, True, "grading_timeout")
    assert check_case(attack, [timed_out])[0].startswith("run 0: graded 0 by grading_timeout")
    sandbox_verdict = CaseRun("graded", 0.0, None, True, "state_unreadable")
    assert check_case(attack, [sandbox_verdict, sandbox_verdict]) == ()
    assert check_case(attack, [sandbox_verdict, GRADED_0]) == (
        "not deterministic: graded/0.0/state_unreadable, graded/0.0",)
    slow = CaseRun("graded", 0.0, None, True, None, 12.5)
    assert check_case(attack, [slow], budget_s=10) == (
        "run 0: took 12.5 s, over the 10 s budget",)
    assert check_case(attack, [slow], budget_s=None) == ()


def test_the_suite_requires_a_reference_and_a_declared_attack_on_its_task():
    call = (("bash", {"command": "x"}),)
    reference = ConformanceCase("ref", "rl", 0, 1.0, ())
    attack = ConformanceCase("atk", "rl", 0, 0.0, call)
    builtin = ConformanceCase("untouched@0", "rl", 0, 0.0, call, BUILTIN)
    builtin_reference = ConformanceCase("loud@0", "rl", 0, 1.0, (), BUILTIN)
    assert suite_problems([reference, attack]) == ()
    assert suite_problems([attack, builtin_reference]) == (
        "no reference solution (a case expecting 1.0)",)
    assert suite_problems([reference, builtin]) == (
        "no declared attack (a case of the env expecting 0.0, with calls)",)
    idle = ConformanceCase("idle", "rl", 0, 0.0, ())
    assert suite_problems([reference, idle]) == (
        "no declared attack (a case of the env expecting 0.0, with calls)",)
    other = ConformanceCase("ref1", "rl", 1, 1.0, ())
    assert suite_problems([reference, attack, other]) == (
        "task 1 has a reference but no declared attack with calls",)


async def test_the_report_records_the_image_lock_it_was_given(tmp_path):
    env = toy_env()
    cases = env_cases(toy_bridge_env, "rl")[:1]
    with fake_gateway(tmp_path, envs={"toy": TOY}, **GW) as (gateway, _boxes):
        report = await run_conformance(gateway, env="toy", env_package="toy", cases=cases,
                                       image_lock_sha256=env.lock.sha256())
        unbound = await run_conformance(gateway, env="toy", env_package="toy", cases=cases)
    assert report.image_lock_sha256 == env.lock.sha256()
    assert report.to_json()["image_lock_sha256"] == env.lock.sha256()
    assert unbound.to_json()["image_lock_sha256"] is None


async def test_fewer_than_two_runs_cannot_pass_the_gate(tmp_path):
    cases = env_cases(toy_bridge_env, "rl")
    with fake_gateway(tmp_path, envs={"toy": TOY}, **GW) as (gateway, _boxes):
        report = await run_conformance(gateway, env="toy", env_package="toy", cases=cases,
                                       repeat=1)
    assert all(result.ok for result in report.results)
    assert report.problems == ("repeat 1: determinism needs 2 runs at the same time",)
    assert not report.passed


async def test_the_toy_env_passes_the_suite(tmp_path, scoring_children):
    env = toy_env()
    with fake_gateway(tmp_path, envs={"toy": TOY}, **GW) as (gateway, boxes):
        report = await run_conformance(gateway, env="toy",
                                       env_package=bridged_env_package_of(env.module),
                                       cases=plan_cases(env, "rl", salt=SALT))
    assert report.passed, json.dumps(report.to_json(), indent=1)
    assert all(len(result.runs) == 2 for result in report.results)
    runs = {result.case.name: result.runs for result in report.results}
    for kind in ("empty", "not_utf8", "invalid_json", "hostile"):
        # graded through the scoring process, never aborted, no label: the env scored 0
        assert {(r.status, r.reward, r.fault) for r in runs[f"malformed_{kind}@0"]} == {
            ("graded", 0.0, None)}
    assert {r.fault for r in runs["reference_with_links_into_grader@0"]} == {"state_unreadable"}
    assert {r.reward for r in runs["reference_with_oversized_output@0"]} == {1.0}
    # about the state cap inside the roots: archived, restored, then scored to 1
    assert {(r.status, r.reward, r.fault) for r in runs["reference_with_large_artifact@0"]} == {
        ("graded", 1.0, None)}
    restored = [member.size for _box, _dir, data in boxes.extracted
                for member in tarfile.open(fileobj=io.BytesIO(data)).getmembers()
                if member.size == 31 * 1024**2]
    assert restored == [31 * 1024**2] * 2, ([d for _b, d, _t in boxes.extracted], restored)
    assert {r.reward for r in runs["shimmed_tools@0"]} == {0.0}
    assert {r.reward for r in runs["untouched@1"]} == {0.0}
    ones = sum(run.reward == 1.0 for result in report.results for run in result.runs)
    assert ones == 6 and len(scoring_children) >= ones
    document = json.loads(json.dumps(report.to_json()))
    assert document["passed"] is True and document["env_package"].endswith(".vfb2")
    assert document["sample_seed"] == TOY_SEED and document["sample_salt"] == SALT


async def test_a_reference_that_does_not_solve_the_task_fails_the_suite(tmp_path):
    env = toy_env()
    cases = plan_cases(env, "rl", entry="test_conformance:lying_cases", builtin=False)
    with fake_gateway(tmp_path, envs={"toy": TOY}, **GW) as (gateway, _boxes):
        report = await run_conformance(gateway, env="toy", env_package="toy", cases=cases,
                                       repeat=1)
    assert not report.passed
    assert report.results[0].problems == ("run 0: reward 0.0, expected 1.0",)
    assert report.results[1].ok


async def test_a_run_over_its_time_budget_fails_the_suite(tmp_path):
    cases = env_cases(toy_bridge_env, "rl")[:1]
    with fake_gateway(tmp_path, envs={"toy": TOY}, **GW) as (gateway, _boxes):
        report = await run_conformance(gateway, env="toy", env_package="toy", cases=cases,
                                       repeat=1, run_budget_s=0.001)
    assert not report.passed
    assert report.results[0].problems[0].endswith("over the 0.001 s budget")


async def test_envs_that_break_the_norm_fail_the_suite(tmp_path, scoring_children):
    env = VerifiersEnv(HOSTILE, allow_unpackaged=True)
    assert env.module is hostile_bridge_env
    declared = {case.name: case for case in env_cases(env.module, "rl")}
    plan = {case.name: case for case in plan_cases(env, "rl", salt=SALT)}
    cases = [declared[name] for name in (
        "conversation_reference", "large_write_reference", "large_output_reference",
        "strict_json_reference", "counter_reference", "strict_finalize_reference",
        "nan_reward_reference", "agent_box_verdict_reference", "shared_state_reference")]
    for index, mode in ((3, "strict_json"), (5, "strict_finalize"), (6, "nan_reward")):
        cases += [case for case in builtin_cases(["/work/out", "/logs/artifacts"], "rl", index,
                                                 reference=declared[f"{mode}_reference"],
                                                 hostile_bytes=1 << 20)
                  if case.name.startswith("malformed_")]
    # index 9 has no case of its own: only the split's sample covers it
    cases += [plan["shimmed_tools@7"], plan["untouched@9"], plan["empty_artifacts@9"],
              declared["agent_box_tool_reference"], plan["shimmed_tools@11"],
              plan["malformed_empty@11"]]
    with fake_gateway(tmp_path, envs={"hostile": HOSTILE}, **GW) as (gateway, _boxes):
        report = await run_conformance(gateway, env="hostile", env_package="hostile",
                                       cases=cases)
    results = {result.case.name: result for result in report.results}
    assert not report.passed
    # a reward that reads the conversation sees none: its reference fails
    assert results["conversation_reference"].problems == (
        "run 0: reward 0.0, expected 1.0", "run 1: reward 0.0, expected 1.0")
    # the env's own large writes and large command outputs are served
    assert results["large_write_reference"].ok, results["large_write_reference"]
    assert results["large_output_reference"].ok, results["large_output_reference"]
    for mode in ("strict_json", "strict_finalize", "nan_reward", "agent_box_verdict"):
        assert results[f"{mode}_reference"].ok, results[f"{mode}_reference"]
    # an env whose own code fails on malformed bytes is graded 0 (never aborted) but fails
    for index, label, what in (
            (3, "grading_failed", "the env's scoring raised on the agent's state"),
            (5, "extract_failed", "the env's extract (finalize) raised on the agent's state"),
            (6, "reward_out_of_range", "the env's reward is not finite in [0, 1]")):
        for kind in ("empty", "not_utf8", "invalid_json", "hostile"):
            result = results[f"malformed_{kind}@{index}"]
            assert [(r.status, r.reward, r.fault) for r in result.runs] == [
                ("graded", 0.0, label)] * 2, result
            assert result.problems == tuple(f"run {i}: graded 0 by {label}: {what}"
                                            for i in (0, 1))
    # a verdict computed in the agent's box trusts the agent's tools
    assert results["shimmed_tools@7"].problems == (
        "run 0: reward 1.0, expected 0.0", "run 1: reward 1.0, expected 0.0")
    # an extract that runs the agent's own (shimmed) tools fails: the agent's own 0, accepted
    # on shimmed_tools only; elsewhere extract_failed stays the env's fault
    assert results["agent_box_tool_reference"].ok, results["agent_box_tool_reference"]
    shimmed = results["shimmed_tools@11"]
    assert [(r.status, r.reward, r.fault) for r in shimmed.runs] == [
        ("graded", 0.0, "extract_failed")] * 2 and shimmed.ok, shimmed
    assert results["malformed_empty@11"].ok  # its extract does not fail there
    # a task no case references, graded 1 for doing nothing: the sample finds it
    assert results["untouched@9"].problems == (
        "run 0: reward 1.0, expected 0.0", "run 1: reward 1.0, expected 0.0")
    assert results["untouched@9"].case.sample_seed == conformance.sample_seed(
        "hostile-bridge-env", "rl", SALT)
    assert not results["empty_artifacts@9"].ok
    # per-process state in the env no longer leaks between episodes: every hook runs in a
    # fresh scoring process (bridge v2), so the counter is 1 in each
    counter = results["counter_reference"]
    assert [run.reward for run in counter.runs] == [1.0, 1.0] and counter.ok, counter
    # ... and state kept on the Task between setup and finalize is dropped with setup's
    # process: finalize cannot read it (graded 0 extract_failed, the suite fails it)
    shared = results["shared_state_reference"]
    assert [(r.status, r.reward, r.fault) for r in shared.runs] == [
        ("graded", 0.0, "extract_failed")] * 2, shared
    assert not shared.ok
    assert scoring_children


def test_the_hostile_env_declares_an_attack_on_every_referenced_task():
    env = VerifiersEnv(HOSTILE, allow_unpackaged=True)
    assert suite_problems(plan_cases(env, "rl", salt=SALT)) == ()


def test_a_fresh_salt_changes_the_sample_and_a_given_salt_reproduces_it():
    assert conformance.sample_seed("a", "rl", "s1") == conformance.sample_seed("a", "rl", "s1")
    assert conformance.sample_seed("a", "rl", "s1") != conformance.sample_seed("a", "rl", "s2")
    assert conformance.sample_seed("a", "rl", "s1") != conformance.sample_seed("a", "rl")
    seeds = [conformance.sample_seed("a", "rl", f"s{i}") for i in range(2)]
    assert (conformance.sample_indices(seeds[0], 10_000)
            != conformance.sample_indices(seeds[1], 10_000))
    env = VerifiersEnv(HOSTILE, allow_unpackaged=True)  # 11 tasks: whole split sampled
    fresh = {case.sample_salt for case in plan_cases(env, "rl") + plan_cases(env, "rl")
             if case.sample_salt}
    assert len(fresh) == 2 and all(len(salt) >= 16 for salt in fresh)  # one per plan, random
    cases = plan_cases(env, "rl", salt="fixed")
    assert {c.sample_salt for c in cases if c.sample_seed is not None} == {"fixed"}


def test_no_fixed_name_appears_in_what_the_box_sees():
    reference = ConformanceCase("ref", "rl", 3, 1.0, (("bash", {"command": "solve"}),))

    def seen(tag=None):
        cases = builtin_cases(["/work/out"], "rl", 3, reference=reference, tag=tag)
        return "\n".join(call[1]["command"] for case in cases for call in case.calls)

    first, second = seen(), seen()
    assert "reliquary-conformance-" not in first and "conformance-mark" not in first
    assert first != second  # a fresh tag per build
    assert seen("t0") == seen("t0")
    assert "/tmp/.t0m" in seen("t0") and "/tmp/.t0m" not in first


def test_the_planner_gives_each_plan_its_own_names():
    env = toy_env()
    marks = []
    for _ in range(2):
        cases = plan_cases(env, "rl", salt=SALT)
        mark = next(c for c in cases if c.name == "malformed_empty@0").calls[0][1]["command"]
        marks.append(mark)
    assert marks[0] != marks[1]


def test_builtin_cases_count_every_grading_fault_as_the_envs_not_the_attacks():
    builtin = ConformanceCase("b", "rl", 0, 0.0, (), BUILTIN)
    declared = ConformanceCase("d", "rl", 0, 0.0, (), ENV)
    for label in ("extract_timeout", "grading_output_overflow", "grading_box_failed"):
        run = CaseRun("graded", 0.0, None, True, label)
        assert check_case(builtin, [run])[0].startswith(f"run 0: graded 0 by {label}: ")
        assert check_case(declared, [run]) == ()  # legal for a declared attack
    for label in conformance.ENV_FAULTS:  # the shared faults count for both
        run = CaseRun("graded", 0.0, None, True, label)
        assert check_case(declared, [run]) != ()


def test_an_extract_broken_by_the_agents_own_shims_is_accepted_on_shimmed_tools_only():
    failed = CaseRun("graded", 0.0, None, True, "extract_failed")
    shimmed = ConformanceCase("shimmed_tools@0", "rl", 0, 0.0, (), BUILTIN)
    assert check_case(shimmed, [failed, failed]) == ()
    assert check_case(shimmed, [GRADED_1])[0] == "run 0: reward 1.0, expected 0.0"
    for name in ("malformed_empty@0", "untouched@0", "empty_artifacts@0"):
        other = ConformanceCase(name, "rl", 0, 0.0, (), BUILTIN)
        assert check_case(other, [failed])[0].startswith("run 0: graded 0 by extract_failed")
    declared = ConformanceCase("shimmed_tools@0", "rl", 0, 0.0, (), ENV)  # the env's own name
    assert check_case(declared, [failed]) != ()
    timeout = CaseRun("graded", 0.0, None, True, "grading_failed")
    assert check_case(shimmed, [timeout]) != ()  # only extract_failed is the agent's there


async def test_a_finalize_that_hangs_on_a_malformed_answer_fails_the_env(tmp_path):
    env = VerifiersEnv(HOSTILE, allow_unpackaged=True)
    declared = {case.name: case for case in env_cases(env.module, "rl")}
    cases = [declared["hang_finalize_reference"], *(
        case for case in builtin_cases(["/work/out", "/logs/artifacts"], "rl", 10,
                                       reference=declared["hang_finalize_reference"],
                                       hostile_bytes=1 << 20)
        if case.name == "malformed_empty@10")]
    with fake_gateway(tmp_path, envs={"hostile": HOSTILE}, **GW) as (gateway, _boxes):
        report = await run_conformance(gateway, env="hostile", env_package="hostile",
                                       cases=cases)
    results = {result.case.name: result for result in report.results}
    assert results["hang_finalize_reference"].ok
    malformed = results["malformed_empty@10"]
    assert [r.fault for r in malformed.runs] == ["extract_timeout"] * 2, malformed
    assert malformed.problems[0].startswith("run 0: graded 0 by extract_timeout: ")
    assert not report.passed


def test_a_given_tag_names_the_box_files_and_a_fresh_one_differs():
    env = toy_env()
    one = plan_cases(env, "rl", salt=SALT, tag="cafe0123")
    assert one == plan_cases(env, "rl", salt=SALT, tag="cafe0123")
    assert one != plan_cases(env, "rl", salt=SALT)
    assert any("cafe0123" in json.dumps(case.calls) for case in one)


def test_a_reduced_sample_draws_fewer_sampled_cases():
    bridged = toy_env()
    full = plan_cases(bridged, "rl", salt="s", tag="t")
    none = plan_cases(bridged, "rl", salt="s", tag="t", sample_size=0)
    assert any(case.sample_seed is not None for case in full)
    assert not any(case.sample_seed is not None for case in none)
    assert [c.name for c in none] == [c.name for c in full if c.sample_seed is None]


@pytest.mark.parametrize("size", [-1, SAMPLE_SIZE + 1, True, 1.5])
def test_a_sample_size_outside_the_range_is_refused(size):
    with pytest.raises(ConformanceError, match="sample"):
        plan_cases(toy_env(), "rl", sample_size=size)


# -- canary markers (`absent`) -----------------------------------------------

CANARY = "/tmp/.canary-0"


def reference_case(**extra):
    base = {"name": "ref", "index": 0, "expect": 1.0,
            "calls": [["bash", {"command": "write /work/answer.txt 42"}]]}
    return {**base, **extra}


_declared_cases: list = []


def _declared(split):
    return list(_declared_cases)


@pytest.fixture
def declared():
    _declared_cases.clear()
    yield _declared_cases
    _declared_cases.clear()


def test_absent_is_optional_and_parsed_into_the_case(declared):
    declared.extend([reference_case(), reference_case(name="canary", absent=[CANARY, "/a/b"])])
    plain, canary = env_cases(toy_bridge_env, "rl", entry="test_conformance:_declared")
    assert plain.absent == () and canary.absent == (CANARY, "/a/b")


@pytest.mark.parametrize("absent", [
    "/tmp/x", [1], ["relative/x"], [""], ["/a/../etc/x"], ["/a/\0x"],
    [f"/tmp/m{i}" for i in range(9)], None])
def test_a_malformed_absent_is_refused(declared, absent):
    declared.append(reference_case(absent=absent))
    with pytest.raises(ConformanceError, match="absent"):
        env_cases(toy_bridge_env, "rl", entry="test_conformance:_declared")


def test_eight_absent_paths_are_accepted(declared):
    declared.append(reference_case(absent=[f"/tmp/m{i}" for i in range(8)]))
    assert len(env_cases(toy_bridge_env, "rl", entry="test_conformance:_declared")[0].absent) == 8


def test_check_case_fails_an_executed_or_unchecked_canary():
    case = ConformanceCase("c", "rl", 0, 1.0, (), ENV, absent=(CANARY,))
    clean = CaseRun("graded", 1.0, None, True, canary_checked=1)
    assert check_case(case, [clean, clean]) == ()
    hit = CaseRun("graded", 1.0, None, True, canary_checked=1, canary_executed=(CANARY,))
    assert check_case(case, [hit, hit])[0] == f"run 0: canary_executed:{CANARY}"
    unchecked = CaseRun("graded", 1.0, None, True)
    assert "not checked" in check_case(case, [unchecked, unchecked])[0]
    broke = CaseRun("graded", 1.0, None, True, canary_probe_failed="BoxFault")
    assert check_case(case, [broke, broke])[0] == "run 0: canary_probe_failed:BoxFault"
    no_absent = ConformanceCase("c", "rl", 0, 1.0, (), ENV)
    assert check_case(no_absent, [unchecked, unchecked]) == ()


def plant_in_grading_boxes(boxes):
    """What the grader running agent-placed code would do: the marker appears in the
    pristine grading box."""
    real = boxes.create

    def create(box_id, episode_id, spec):
        real(box_id, episode_id, spec)
        if box_id.startswith("g-"):
            boxes.boxes[box_id]["files"][CANARY] = b""

    boxes.create = create


async def canary_run(tmp_path, absent, plant):
    case = ConformanceCase("ref", "rl", 0, 1.0,
                           (("bash", {"command": "write /work/answer.txt 42"}),), ENV,
                           absent=tuple(absent))
    with fake_gateway(tmp_path, envs={"toy": TOY}, **GW) as (gateway, boxes):
        if plant:
            plant_in_grading_boxes(boxes)
        report = await run_conformance(gateway, env="toy", env_package="toy", cases=[case])
    return report.results[0], boxes


async def test_a_canary_that_exists_after_grading_fails_the_case_whatever_the_reward(tmp_path):
    result, _ = await canary_run(tmp_path, [CANARY], plant=True)
    assert [run.reward for run in result.runs] == [1.0, 1.0]
    assert [run.canary_executed for run in result.runs] == [(CANARY,)] * 2
    assert result.problems[0] == f"run 0: canary_executed:{CANARY}"
    assert not result.ok


async def test_an_absent_canary_passes_and_was_checked(tmp_path):
    result, _ = await canary_run(tmp_path, [CANARY, "/tmp/.other"], plant=False)
    assert result.ok, result.problems
    assert [run.canary_checked for run in result.runs] == [2, 2]


async def test_production_grading_runs_no_probe_without_the_suite(tmp_path, monkeypatch):
    from reliquary_sandbox_service.episodes import grader as grader_module
    from reliquary_sandbox_service.episodes.local_gateway import run_scripted_episode

    probes = []
    monkeypatch.setattr(grader_module.Grader, "_with_canaries",
                        lambda self, *a, **k: probes.append(a))
    with fake_gateway(tmp_path, envs={"toy": TOY}, **GW) as (gateway, boxes):
        plant_in_grading_boxes(boxes)
        episode = await run_scripted_episode(
            gateway, env="toy", split="rl", index=0,
            calls=[("bash", {"command": "write /work/answer.txt 42"})])
        assert gateway.app.state.service["episodes"].grader.canaries == {}
    assert probes == []
    assert episode.final["reward"] == 1.0
    assert "canary_checked" not in json.dumps(episode.final)
    assert not any(call[0] == "file_kind" and call[1].startswith("g-")
                   for call in boxes.helper_calls)


# -- runs the host failed on its side -------------------------------------------------

def failing_on_the_host(boxes, method, times, fault=None):
    """The first `times` calls of `boxes.<method>` on an agent box raise `fault` (the
    runtime failing to start a command, by default)."""
    from reliquary_sandbox_service.task_runtime import RuntimeExecFailed

    real = getattr(boxes, method)
    left = [times]

    def failing(box_id, *args, **kwargs):
        if not box_id.startswith("g-") and left[0]:
            left[0] -= 1
            raise fault or RuntimeExecFailed(f"{method}: the runtime failed to start the "
                                             "command; the box is running")
        return real(box_id, *args, **kwargs)

    setattr(boxes, method, failing)
    return left


SOLVE = ConformanceCase("ref", "rl", 0, 1.0,
                        (("bash", {"command": "write /work/answer.txt 42"}),), ENV)


@pytest.mark.parametrize("method,why", [
    ("stop_processes", "box_failed: the runtime failed to start a command in the agent's box"),
    ("write_file", "the episode could not be opened")])
async def test_a_run_the_host_failed_is_run_again_and_recorded(tmp_path, method, why):
    with fake_gateway(tmp_path, envs={"toy": TOY}, **GW) as (gateway, boxes):
        failing_on_the_host(boxes, method, 1)
        report = await run_conformance(gateway, env="toy", env_package="toy", cases=[SOLVE])
    result = report.results[0]
    assert result.ok, result.problems
    retried = [why for run in result.runs for why in run.infra_retries]
    assert len(retried) == 1 and retried[0].startswith(why)
    assert report.to_json()["results"][0]["runs"][0]["infra_retries"] is not None


async def test_the_grading_zero_kept_after_runtime_failures_is_run_again(tmp_path):
    from reliquary_sandbox_service.task_runtime import RuntimeExecFailed

    with fake_gateway(tmp_path, envs={"toy": TOY}, **GW) as (gateway, boxes):
        real, left = boxes.read_file, [2]  # both grading attempts of one run

        def read_file(box_id, *args, **kwargs):
            if box_id.startswith("g-") and left[0]:
                left[0] -= 1
                raise RuntimeExecFailed("Reading a file: the runtime failed to start it")
            return real(box_id, *args, **kwargs)

        boxes.read_file = read_file
        # One run: the two failures hit both of its grading attempts.
        report = await run_conformance(gateway, env="toy", env_package="toy", cases=[SOLVE],
                                       repeat=1)
    result = report.results[0]
    assert result.ok, result.problems
    assert [why for run in result.runs for why in run.infra_retries] == [
        "graded 0 by grading_box_failed: the runtime failed to start a command on every "
        "grading attempt"]


async def test_a_case_the_host_keeps_failing_stops_the_suite_never_passes(tmp_path):
    with fake_gateway(tmp_path, envs={"toy": TOY}, **GW) as (gateway, boxes):
        failing_on_the_host(boxes, "stop_processes", 99)
        with pytest.raises(conformance.ConformanceInfraError, match="3 times"):
            await run_conformance(gateway, env="toy", env_package="toy", cases=[SOLVE])


async def test_the_agents_own_box_failure_is_never_run_again(tmp_path):
    from reliquary_sandbox_service.task_runtime import BoxFault

    with fake_gateway(tmp_path, envs={"toy": TOY}, **GW) as (gateway, boxes):
        left = failing_on_the_host(boxes, "stop_processes", 99,
                                   BoxFault("Stopping the agent's processes: the box is running"))
        report = await run_conformance(gateway, env="toy", env_package="toy", cases=[SOLVE])
    result = report.results[0]
    assert not result.ok and left[0] == 97  # one per run, none again
    assert result.problems[0] == "run 0: ended box_failed (the agent's box is unusable)"
    assert all(run.infra_retries == () for run in result.runs)


def test_a_kept_zero_whose_canary_ran_is_the_cases_failure_not_the_hosts():
    kept = CaseRun("graded", 0.0, None, True, "grading_box_failed", canary_checked=1,
                   runtime_exec_failed=True)
    assert conformance._host_failure(kept) is not None  # run again
    hit = replace(kept, canary_executed=(CANARY,))
    assert conformance._host_failure(hit) is None  # judged: canary_executed fails the case
    problems = check_case(ConformanceCase("c", "rl", 0, 1.0, (), ENV, absent=(CANARY,)),
                          [hit, hit])
    assert f"run 0: canary_executed:{CANARY}" in problems


async def test_the_other_runs_of_a_case_finish_before_a_host_failure_stops_the_suite(
        monkeypatch):
    import asyncio

    finished = []

    async def run_retrying(gateway, env, case, budgets):
        if not finished and not getattr(run_retrying, "failed", False):
            run_retrying.failed = True
            raise conformance.ConformanceInfraError("ref: the host failed a run 3 times")
        await asyncio.sleep(0.05)  # an episode still open when the first one gave up
        finished.append(True)
        return CaseRun("graded", 1.0, None, True)

    monkeypatch.setattr(conformance, "_run_retrying", run_retrying)
    with pytest.raises(conformance.ConformanceInfraError):
        await run_conformance(None, env="toy", env_package="toy", cases=[SOLVE], repeat=2)
    assert finished == [True]


# -- re-gradings and in-place retries are recorded even when the case passes -----------

async def test_a_passing_case_records_the_fresh_box_regrading_and_helper_retries(tmp_path):
    from reliquary_sandbox_service.task_runtime import RuntimeExecFailed

    with fake_gateway(tmp_path, envs={"toy": TOY}, **GW) as (gateway, boxes):
        real, left = boxes.read_file, [1]  # the first grading attempt of one run only

        def read_file(box_id, *args, **kwargs):
            if box_id.startswith("g-") and left[0]:
                left[0] -= 1
                raise RuntimeExecFailed("Reading a file: the runtime failed to start it")
            return real(box_id, *args, **kwargs)

        boxes.read_file = read_file
        pending = {"n": 2}  # in-place retries the runtime would have counted in the agent's box

        def take(box_id):
            taken, pending["n"] = (pending["n"], 0) if not box_id.startswith("g-") else (0, pending["n"])
            return taken

        boxes.take_helper_retries = take
        report = await run_conformance(gateway, env="toy", env_package="toy", cases=[SOLVE],
                                       repeat=1)
    result = report.results[0]
    assert result.ok, result.problems
    assert [(r.regraded, r.helper_retries) for r in result.runs] == [(1, 2)]
    assert (report.regraded, report.helper_retries) == (1, 2)
    document = report.to_json()
    assert (document["regraded"], document["helper_retries"]) == (1, 2)
    assert document["results"][0]["runs"][0]["regraded"] == 1


async def test_a_clean_run_records_no_retries(tmp_path):
    with fake_gateway(tmp_path, envs={"toy": TOY}, **GW) as (gateway, boxes):
        report = await run_conformance(gateway, env="toy", env_package="toy", cases=[SOLVE])
        assert gateway.app.state.service["episodes"].grader.retry_log == {}
    assert report.results[0].ok and (report.regraded, report.helper_retries) == (0, 0)


# -- the env's package imports nothing that turns data into code ----------------------

def _package(tmp_path, files):
    import types

    root = tmp_path / "some_env"
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    module = types.ModuleType("some_env")
    module.__path__ = [str(root)]
    return module


def test_the_toy_envs_have_no_unsafe_import():
    assert conformance.unsafe_imports(toy_bridge_env) == ()
    assert conformance.unsafe_imports(hostile_bridge_env) == ()


def test_unsafe_imports_are_found_at_any_depth_of_the_package(tmp_path):
    module = _package(tmp_path, {
        "__init__.py": "import json\nimport yaml\n\ndef f(t):\n    return yaml.safe_load(t)\n",
        "grading.py": ("def reward(text):\n    from math_verify import parse\n"
                       "    return parse(text)\n"),
        "sub/deep.py": ("import pickle as p\nimport sympy.parsing\nfrom latex2sympy2 import x\n"
                        "import yaml as y\nfrom yaml import FullLoader, safe_load\n"
                        "y.load('a', Loader=y.Loader)\ny.unsafe_load('a')\n"),
        "broken.py": "def (:\n",
        "data.txt": "import pickle\n",
    })
    assert conformance.unsafe_imports(module) == (
        "some_env/broken.py: cannot be read (SyntaxError)",
        "some_env/grading.py:2: imports from math_verify",
        "some_env/sub/deep.py:1: imports pickle",
        "some_env/sub/deep.py:2: imports sympy.parsing",
        "some_env/sub/deep.py:3: imports from latex2sympy2",
        "some_env/sub/deep.py:5: imports yaml.FullLoader",
        "some_env/sub/deep.py:6: uses yaml.Loader",
        "some_env/sub/deep.py:6: uses yaml.load",
        "some_env/sub/deep.py:7: uses yaml.unsafe_load",
    )


def test_a_single_module_env_is_read_alone(tmp_path):
    import types

    path = tmp_path / "single_env.py"
    path.write_text("import dill\n")
    module = types.ModuleType("single_env")
    module.__file__ = str(path)
    assert conformance.unsafe_imports(module) == ("single_env.py:1: imports dill",)


async def test_an_unsafe_import_fails_the_suite_as_requiring_review(tmp_path):
    cases = env_cases(toy_bridge_env, "rl")[:1]
    with fake_gateway(tmp_path, envs={"toy": TOY}, **GW) as (gateway, _boxes):
        report = await run_conformance(gateway, env="toy", env_package="toy", cases=cases,
                                       unsafe=("toy/grading.py:3: imports sympy",))
    assert not report.passed
    assert report.unsafe_imports == ("toy/grading.py:3: imports sympy",)
    assert report.to_json()["unsafe_imports"] == ["toy/grading.py:3: imports sympy"]
    assert any(p.startswith("requires review: the env's package toy/grading.py:3: imports "
                            "sympy") for p in report.problems)
