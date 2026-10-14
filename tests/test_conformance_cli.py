"""`reliquary-sandbox-conformance --fake` on the toy env (no Docker)."""

import json

import pytest

pytest.importorskip("verifiers.v1")

from reliquary_sandbox_service.episodes.conformance_cli import main, parser, provenance  # noqa: E402

TOY = "verifiers:toy-bridge-env==unpackaged"


def lying_cases(split):
    return [{"name": "not_a_solution", "index": 0, "expect": 1.0,
             "calls": [["bash", {"command": "write /work/answer.txt 41"}]]},
            {"name": "wrong", "index": 0, "expect": 0.0, "calls": []}]


def test_the_toy_env_passes_and_the_report_is_written(tmp_path, capsys, monkeypatch):
    from reliquary_sandbox_service.episodes import conformance_cli

    monkeypatch.setattr(conformance_cli, "_euid", lambda: 1000)  # as on a dev machine
    out = tmp_path / "report.json"
    assert main(["--entry", TOY, "--fake", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["passed"] is True and report["entry"] == TOY and report["repeat"] == 2
    assert report["env_package"] == "toy_bridge_env==unpackaged.vfb2"
    assert len(report["results"]) == 19 and report["fake"] is True
    assert capsys.readouterr().err.strip().endswith(
        "PASSED (FAKE, unprivileged scoring without slot uids: not a gate)")
    assert report["gate"] is False and report["gate_reasons"] == ["--fake", "unprivileged"]
    assert report["euid"] == 1000 and report["scoring_slot_uids"] is False
    assert report["allow_unquoted_disk"] is False and report["cpu_source"] == "docker"
    assert report["docker_host"] is None and report["busybox_sha256"] is None
    assert len(report["sample_salt_used"]) >= 16 and len(report["file_tag"]) >= 8


def test_the_report_binds_the_image_lock_the_env_was_served_with(tmp_path):
    import importlib.resources

    from reliquary_sandbox_verifiers.image_lock import LOCK_FILE, ImageLock

    out = tmp_path / "report.json"
    assert main(["--entry", TOY, "--fake", "--out", str(out)]) == 0
    lock = ImageLock.parse((importlib.resources.files("toy_bridge_env") / LOCK_FILE).read_bytes())
    report = json.loads(out.read_text())
    assert report["image_lock_sha256"] == lock.sha256()
    assert len(report["image_lock_sha256"]) == 64


def test_a_failing_env_exits_one(tmp_path, capsys):
    out = tmp_path / "report.json"
    assert main(["--entry", TOY, "--fake", "--cases", "test_conformance_cli:lying_cases",
                 "--no-builtin", "--repeat", "1", "--out", str(out)]) == 1
    assert json.loads(out.read_text())["passed"] is False
    assert "not_a_solution (reference): FAIL" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["--entry", "fake_episode_env:sandbox_task", "--fake"],
    ["--entry", "verifiers:toy-bridge-env==9.9", "--fake"],
    ["--entry", TOY, "--fake", "--options", "[]"],
    ["--entry", TOY, "--fake", "--split", "eval"],
    ["--entry", TOY],
])
def test_an_env_the_suite_cannot_run_exits_two(argv, capsys):
    assert main(argv) == 2
    assert capsys.readouterr().err.startswith("conformance:")


def test_a_real_run_refuses_an_unpackaged_taskset_and_never_builds_fakes(tmp_path, capsys):
    busybox = tmp_path / "busybox"
    busybox.write_bytes(b"x")
    assert main(["--entry", TOY, "--docker-host", "unix:///nonexistent",
                 "--busybox", str(busybox)]) == 2
    assert capsys.readouterr().err.startswith("conformance:")


def test_a_missing_busybox_exits_two(tmp_path, capsys):
    assert main(["--entry", TOY, "--docker-host", "unix:///nonexistent",
                 "--busybox", str(tmp_path / "absent")]) == 2
    assert "is not a file" in capsys.readouterr().err


def test_an_unexpected_failure_exits_two_not_one(monkeypatch, capsys):
    from reliquary_sandbox_service.episodes import conformance_cli

    async def broken(*args, **kwargs):
        raise RuntimeError("docker fell over")

    monkeypatch.setattr(conformance_cli, "_run", broken)
    assert main(["--entry", TOY, "--fake"]) == 2
    assert "docker fell over" in capsys.readouterr().err


def test_a_keyboard_interrupt_passes_through(monkeypatch):
    from reliquary_sandbox_service.episodes import conformance_cli

    async def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(conformance_cli, "_run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        main(["--entry", TOY, "--fake"])


def test_repeat_zero_exits_two(capsys):
    assert main(["--entry", TOY, "--fake", "--repeat", "0"]) == 2
    assert capsys.readouterr().err.startswith("conformance:")


def test_the_same_salt_and_tag_plan_the_same_cases(monkeypatch, tmp_path):
    from reliquary_sandbox_service.episodes import conformance_cli

    planned = []
    real = conformance_cli.plan_cases

    def spy(*args, **kwargs):
        cases = real(*args, **kwargs)
        planned.append((kwargs, cases))
        return cases

    monkeypatch.setattr(conformance_cli, "plan_cases", spy)
    for name in ("a.json", "b.json"):
        assert main(["--entry", TOY, "--fake", "--sample-salt", "fixed", "--file-tag", "abcdef12",
                     "--out", str(tmp_path / name)]) == 0
    assert planned[0][0]["salt"] == "fixed" and planned[0][0]["tag"] == "abcdef12"
    assert planned[0][1] == planned[1][1]
    one, two = (json.loads((tmp_path / n).read_text()) for n in ("a.json", "b.json"))
    assert one["sample_salt_used"] == "fixed" and one["file_tag"] == "abcdef12"
    assert [r["case"] for r in one["results"]] == [r["case"] for r in two["results"]]


def test_a_fresh_salt_and_tag_are_drawn_and_recorded(tmp_path):
    docs = []
    for name in ("a.json", "b.json"):
        assert main(["--entry", TOY, "--fake", "--out", str(tmp_path / name)]) == 0
        docs.append(json.loads((tmp_path / name).read_text()))
    assert docs[0]["sample_salt_used"] != docs[1]["sample_salt_used"]
    assert docs[0]["file_tag"] != docs[1]["file_tag"]


@pytest.mark.parametrize("host", ["unix:///var/run/docker.sock", "unix:///run/docker.sock",
                                  "unix:///var/run//docker.sock"])
def test_a_real_run_refuses_the_hosts_own_daemon(host, tmp_path, capsys):
    busybox = tmp_path / "busybox"
    busybox.write_bytes(b"x")
    assert main(["--entry", TOY, "--docker-host", host, "--busybox", str(busybox)]) == 2
    assert "isolated" in capsys.readouterr().err


def test_the_isolated_test_daemon_is_not_refused(tmp_path, capsys):
    busybox = tmp_path / "busybox"
    busybox.write_bytes(b"x")
    assert main(["--entry", TOY, "--docker-host", "unix:///run/reliquary-xfs-docker.sock",
                 "--busybox", str(busybox)]) == 2
    assert "isolated" not in capsys.readouterr().err


def test_the_gate_flag_and_the_provenance_of_a_real_run():
    import argparse

    from reliquary_sandbox_service.episodes.conformance_cli import provenance

    args = argparse.Namespace(fake=False, allow_unquoted_disk=True, cpu_source="cgroupfs",
                              docker_host="unix:///run/reliquary-xfs-docker.sock",
                              busybox=None)
    doc = provenance(args, busybox_sha256="ab" * 32, euid=0)
    assert doc["gate"] is False and doc["gate_reasons"] == ["--allow-unquoted-disk"]
    assert doc["allow_unquoted_disk"] is True and doc["cpu_source"] == "cgroupfs"
    assert doc["docker_host"] == "unix:///run/reliquary-xfs-docker.sock"
    assert doc["busybox_sha256"] == "ab" * 32
    args.allow_unquoted_disk = False
    doc = provenance(args, busybox_sha256="ab" * 32, euid=0)
    assert doc["gate"] is True and doc["gate_reasons"] == []
    assert doc["euid"] == 0 and doc["scoring_slot_uids"] is True


def test_a_run_whose_scoring_kept_the_callers_uid_is_not_a_gate(monkeypatch):
    """Only as root do the scoring processes take their slot uids (the isolation the
    gate attests); unprivileged (RELIQUARY_SCORING_UNPRIVILEGED_OK) they keep the uid of
    the run, so its verdict says so and is not a gate."""
    import argparse

    from reliquary_sandbox_service.episodes.conformance_cli import provenance

    monkeypatch.setenv("RELIQUARY_SCORING_UNPRIVILEGED_OK", "1")
    args = argparse.Namespace(fake=False, allow_unquoted_disk=False, cpu_source="docker",
                              docker_host="unix:///run/reliquary-xfs-docker.sock", busybox=None)
    doc = provenance(args, busybox_sha256="ab" * 32, euid=1000)
    assert doc["gate"] is False and doc["gate_reasons"] == ["unprivileged"]
    assert doc["euid"] == 1000 and doc["scoring_slot_uids"] is False
    assert doc["scoring_unprivileged_ok"] is True


def test_a_reduced_sample_is_never_a_gate():
    args = parser().parse_args(["--entry", TOY, "--sample-size", "0"])
    document = provenance(args, busybox_sha256="x", euid=0)
    assert document["gate"] is False and "sample" in document["gate_reasons"]
    full = provenance(parser().parse_args(["--entry", TOY]), busybox_sha256="x", euid=0)
    assert "sample" not in full["gate_reasons"]


def test_a_run_without_the_builtin_attacks_is_never_a_gate():
    args = parser().parse_args(["--entry", TOY, "--no-builtin"])
    document = provenance(args, busybox_sha256="x", euid=0)
    assert document["gate"] is False and "--no-builtin" in document["gate_reasons"]


def test_the_report_records_the_sample_size(tmp_path):
    out = tmp_path / "report.json"
    assert main(["--entry", TOY, "--fake", "--sample-size", "0", "--out", str(out)]) == 0
    document = json.loads(out.read_text())
    assert document["sample_size"] == 0 and document["gate"] is False


def test_a_case_the_host_keeps_failing_exits_two(monkeypatch, capsys):
    from reliquary_sandbox_service.episodes import conformance_cli
    from reliquary_sandbox_service.episodes.conformance import ConformanceInfraError

    async def failing(*args, **kwargs):
        raise ConformanceInfraError("ref: the host failed a run on its side 3 times")

    monkeypatch.setattr(conformance_cli, "_run", failing)
    assert main(["--entry", TOY, "--fake"]) == 2
    assert "the host failed a run" in capsys.readouterr().err
