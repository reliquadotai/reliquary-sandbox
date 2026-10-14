"""The reference sweep: an env's reference solutions over many tasks, through the bridge."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("verifiers.v1")

import toy_bridge_env  # noqa: E402

from reliquary_sandbox_service.episodes import reference_sweep  # noqa: E402
from reliquary_sandbox_service.episodes.reference_sweep import main  # noqa: E402

TOY = "verifiers:toy-bridge-env==unpackaged"


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_the_toy_reference_grades_one_and_a_task_without_one_is_skipped(tmp_path):
    out = tmp_path / "sweep.jsonl"
    assert main(["--entry", TOY, "--fake", "--split", "rl", "--start", "0", "--count", "2",
                 "--out", str(out)]) == 0
    got = [(r["index"], r["status"], r["reward"]) for r in lines(out)]
    assert got == [(0, "graded", 1.0), (1, "skipped", None)]
    assert all(r["entry"] == TOY and r["split"] == "rl" for r in lines(out))
    assert all(r["fake"] and not r["provenance"]["gate"] for r in lines(out))


def test_a_reference_that_does_not_grade_one_exits_one(tmp_path, monkeypatch):
    monkeypatch.setattr(toy_bridge_env, "reference_calls",
                        lambda split, index: [["bash", {"command": "write /work/answer.txt 41"}]])
    out = tmp_path / "sweep.jsonl"
    assert main(["--entry", TOY, "--fake", "--indices", "0", "--out", str(out)]) == 1
    assert lines(out)[0]["reward"] == 0.0


def test_an_env_without_reference_calls_cannot_run(tmp_path, monkeypatch):
    monkeypatch.delattr(toy_bridge_env, "reference_calls")
    assert main(["--entry", TOY, "--fake", "--indices", "0",
                 "--out", str(tmp_path / "x.jsonl")]) == 2


def test_a_crash_is_could_not_run(tmp_path, monkeypatch):
    def boom(split, index):
        raise RuntimeError("boom")

    monkeypatch.setattr(toy_bridge_env, "reference_calls", boom)
    assert main(["--entry", TOY, "--fake", "--indices", "0",
                 "--out", str(tmp_path / "x.jsonl")]) == 2


def test_the_hosts_own_docker_socket_is_refused(tmp_path, capsys):
    assert main(["--entry", TOY, "--docker-host", "unix:///var/run/docker.sock",
                 "--busybox", "/bin/true", "--indices", "0",
                 "--out", str(tmp_path / "x.jsonl")]) == 2
    assert "isolated test daemon" in capsys.readouterr().err


def test_a_real_run_refuses_an_unpackaged_env(tmp_path, monkeypatch, capsys):
    ran = []

    async def spy(*args, **kwargs):
        ran.append(True)

    monkeypatch.setattr(reference_sweep, "_run", spy)
    assert main(["--entry", TOY, "--docker-host", "unix:///run/test-docker.sock",
                 "--busybox", "/bin/true", "--indices", "0",
                 "--out", str(tmp_path / "x.jsonl")]) == 2
    assert not ran
    assert "unpackaged" in capsys.readouterr().err
    assert not (tmp_path / "x.jsonl").exists()


def test_a_sweep_where_every_task_is_skipped_exits_one(tmp_path):
    out = tmp_path / "sweep.jsonl"
    assert main(["--entry", TOY, "--fake", "--indices", "1", "--out", str(out)]) == 1
    [line] = lines(out)
    assert line["status"] == "skipped" and line["verified"] is False
    assert (line["ran"], line["skipped"]) == (0, 1)


def test_reward_one_that_the_verification_refuses_exits_one(tmp_path, monkeypatch):
    async def unverified(gateway, **kwargs):
        return SimpleNamespace(final={"status": "graded", "reward": 1.0, "reason": None},
                               verification=SimpleNamespace(ok=False))

    monkeypatch.setattr(reference_sweep, "run_scripted_episode", unverified)
    out = tmp_path / "sweep.jsonl"
    assert main(["--entry", TOY, "--fake", "--indices", "0", "--out", str(out)]) == 1
    assert lines(out)[0]["reward"] == 1.0 and lines(out)[0]["verified"] is False


def test_a_crash_at_task_n_keeps_the_lines_before_it(tmp_path, monkeypatch):
    real = reference_sweep.run_scripted_episode
    calls = []

    async def second_one_crashes(gateway, **kwargs):
        calls.append(kwargs["index"])
        if len(calls) == 2:
            raise RuntimeError("box lost")
        return await real(gateway, **kwargs)

    monkeypatch.setattr(toy_bridge_env, "reference_calls",
                        lambda split, index: [["bash", {"command": "write /work/answer.txt 42"}]])
    monkeypatch.setattr(reference_sweep, "run_scripted_episode", second_one_crashes)
    out = tmp_path / "sweep.jsonl"
    assert main(["--entry", TOY, "--fake", "--indices", "0,1", "--out", str(out)]) == 2
    assert [r["index"] for r in lines(out)] == [0]


def test_each_line_records_the_version_the_time_and_the_counts(tmp_path):
    out = tmp_path / "sweep.jsonl"
    main(["--entry", TOY, "--fake", "--start", "0", "--out", str(out)])
    [line] = lines(out)
    assert line["sandbox_version"] and isinstance(line["finished_at"], int)
    assert (line["ran"], line["skipped"]) == (1, 0)


def test_count_defaults_to_one_and_is_refused_with_indices(tmp_path):
    out = tmp_path / "sweep.jsonl"
    assert main(["--entry", TOY, "--fake", "--start", "0", "--out", str(out)]) == 0
    assert len(lines(out)) == 1
    other = tmp_path / "other.jsonl"
    assert main(["--entry", TOY, "--fake", "--indices", "0", "--count", "2",
                 "--out", str(other)]) == 2
    assert not other.exists()


def test_indices_and_a_range_are_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        main(["--entry", TOY, "--fake", "--indices", "0", "--start", "0", "--count", "1",
              "--out", str(tmp_path / "x.jsonl")])
