"""The bridge and the conformance suite on real gVisor boxes. Opt-in, test box only
(docs/deployment.md, "Episode hosts: testing"); never on a production host.

    RUN_EPISODE_LIVE=1                 enable
    DOCKER_HOST=unix:///run/reliquary-xfs-docker.sock   (scripts/test-xfs-docker.sh env)
    EPISODE_LIVE_XFS=1                 that daemon has the xfs project quota
    EPISODE_LIVE_DOCKER_PIDFILE=...    its pidfile
    EPISODE_LIVE_CPU_SOURCE=cgroupfs   read box CPU from cgroupfs instead of Docker stats
    EPISODE_LIVE_BUSYBOX=/usr/bin/busybox
    EPISODE_LIVE_IMAGE=repo@sha256:... default: the daemon's ubuntu:22.04, by its RepoDigest
    EPISODE_LIVE_METRICS=path          append each report as a JSON line

Run sequentially (`-p no:randomly`), one live process at a time, as root (the scoring
processes then run as their slots' uids, as in production). The envs are written to a
world-readable directory under the system temporary directory (pytest's own is 0700 for
root, which the slot uids could not read). The first test of this file runs everywhere:
it loads the live envs through the production gate, without Docker.
"""

import contextlib
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("verifiers.v1")

from bridge_live_envs import DISTS, write_live_envs  # noqa: E402

from reliquary_sandbox_service.episodes.conformance import (  # noqa: E402
    plan_cases,
    run_conformance,
)
from reliquary_sandbox_service.episodes.conformance_cli import main  # noqa: E402
from reliquary_sandbox_service.episodes.local_gateway import local_gateway  # noqa: E402
from reliquary_sandbox_service.episodes.registry import TaskRegistry  # noqa: E402
from reliquary_sandbox_service.scoring_server import DEFAULT_UID_BASE  # noqa: E402

needs_live = pytest.mark.skipif(os.getenv("RUN_EPISODE_LIVE") != "1",
                                reason="set RUN_EPISODE_LIVE=1 on the episode test host")

XFS = os.getenv("EPISODE_LIVE_XFS") == "1"
CPU_SOURCE = os.getenv("EPISODE_LIVE_CPU_SOURCE", "docker")
BUSYBOX = Path(os.getenv("EPISODE_LIVE_BUSYBOX", "/usr/bin/busybox"))
PIDFILE = Path(os.getenv("EPISODE_LIVE_DOCKER_PIDFILE", "/var/run/docker.pid"))
METRICS = os.getenv("EPISODE_LIVE_METRICS")
IDENTITY = re.compile(r"[a-z-]+==0\.0\.1\+g[0-9a-f]{16}\.vfb\d+")


def record(name, document):
    line = json.dumps({"check": name, **document}, sort_keys=True)
    print("LIVE", line)
    if METRICS:
        with open(METRICS, "a") as out:
            out.write(line + "\n")


def _forget_modules():
    for name in list(sys.modules):
        if name.split(".", 1)[0] in DISTS:
            del sys.modules[name]
    importlib.invalidate_caches()


@contextlib.contextmanager
def live_envs(image):
    """The live env packages in a fresh world-readable directory on sys.path."""
    root = Path(tempfile.mkdtemp(prefix="bridge-live-envs-"))
    try:
        entries = write_live_envs(root, image)
        sys.path.insert(0, str(root))
        _forget_modules()
        try:
            yield entries
        finally:
            sys.path.remove(str(root))
            _forget_modules()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_the_live_envs_load_through_the_production_gate():
    from verifiers.v1.tasksets.harbor import HarborTask

    image = "registry.example/task@sha256:" + "b" * 64
    with live_envs(image) as entries:
        registry = TaskRegistry(entries)  # unpackaged tasksets refused, as in production
        for env in entries:
            assert IDENTITY.fullmatch(registry.versions()[env]), registry.versions()[env]
            assert registry.resolve(env, "rl", 0).task.image == image
            names = {case.name for case in plan_cases(registry.bridged(env), "rl")}
            assert "reference" in names and "shimmed_tools@0" in names
        assert type(registry.bridged("hello").task("rl", 0)) is HarborTask


def test_the_live_envs_are_readable_by_the_scoring_uids(tmp_path):
    write_live_envs(tmp_path / "envs", "registry.example/task@sha256:" + "b" * 64)
    for path in (tmp_path / "envs").rglob("*"):
        assert path.stat().st_mode & 0o777 == (0o755 if path.is_dir() else 0o644), path


@pytest.fixture(scope="module")
def image():
    if os.getenv("EPISODE_LIVE_IMAGE"):
        return os.environ["EPISODE_LIVE_IMAGE"]
    import docker

    client = docker.from_env(timeout=120)
    try:
        found = client.images.get("ubuntu:22.04")
    finally:
        client.close()
    return next(d for d in found.attrs["RepoDigests"] if d.startswith("ubuntu@"))


@pytest.fixture(scope="module")
def envs(image):
    with live_envs(image) as entries:
        yield entries


def gateway_settings():
    return dict(episode_helper_busybox=BUSYBOX,
                episode_helper_busybox_sha256=hashlib.sha256(BUSYBOX.read_bytes()).hexdigest(),
                episode_docker_pidfile=PIDFILE, episode_cpu_source=CPU_SOURCE,
                episode_allow_unquoted_disk=not XFS)


@pytest.mark.live
@needs_live
@pytest.mark.skipif(os.geteuid() != 0, reason="root is required to start a dedicated scoring uid")
def test_the_live_envs_import_as_a_scoring_uid(envs):
    """The interpreter, installed dependencies and env data must survive the uid drop.

    Loading as root alone cannot prove that the scoring process can read its files.
    A fresh interpreter also checks imports the forkserver may already have cached.
    """
    code = """
import json, os, sys
from pathlib import Path
assert os.getuid() == os.geteuid() == int(sys.argv[3])
assert os.getgid() == os.getegid() == int(sys.argv[3])
assert os.getgroups() == []
sys.path[:] = json.loads(sys.argv[1])
from reliquary_sandbox_service.episodes.registry import TaskRegistry
registry = TaskRegistry(json.loads(sys.argv[2]))
for env in registry.versions():
    registry.resolve(env, "rl", 0)
    task = registry.bridged(env).task("rl", 0)
    if hasattr(task.data, "task_dir"):
        assert (Path(task.data.task_dir) / "tests" / "test.sh").read_bytes()
"""
    with tempfile.TemporaryDirectory(prefix="scoring-imports-") as tmpdir:
        os.chown(tmpdir, DEFAULT_UID_BASE, DEFAULT_UID_BASE)
        result = subprocess.run(
            [sys.executable, "-c", code, json.dumps(sys.path), json.dumps(envs), str(DEFAULT_UID_BASE)],
            user=DEFAULT_UID_BASE, group=DEFAULT_UID_BASE, extra_groups=[],
            cwd=tmpdir, env={"PATH": f"{Path(sys.executable).parent}:/usr/local/bin:/usr/bin:/bin",
                             "LANG": "C.UTF-8", "HOME": "/nonexistent", "TMPDIR": tmpdir,
                             "PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1",
                             "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.live
@needs_live
@pytest.mark.parametrize("env", ["toy", "hello"])
async def test_the_env_passes_the_conformance_suite_on_gvisor(tmp_path, envs, image, env):
    registry = TaskRegistry(envs)
    cases = plan_cases(registry.bridged(env), "rl")
    with local_gateway(tmp_path / "gateway", envs=envs, images=[image],
                       **gateway_settings()) as gateway:
        report = await run_conformance(gateway, env=env, env_package=registry.versions()[env],
                                       cases=cases, repeat=2)
    record(f"conformance-{env}", report.to_json())
    assert report.passed, json.dumps(report.to_json(), indent=1)


@pytest.mark.live
@needs_live
def test_the_cli_runs_the_suite_on_real_boxes(tmp_path, envs):
    out = tmp_path / "report.json"
    argv = ["--entry", envs["toy"], "--repeat", "2", "--out", str(out),
            "--docker-host", os.environ["DOCKER_HOST"], "--busybox", str(BUSYBOX),
            "--docker-pidfile", str(PIDFILE), "--cpu-source", CPU_SOURCE]
    if not XFS:
        argv.append("--allow-unquoted-disk")
    assert main(argv) == 0
    report = json.loads(out.read_text())
    record("conformance-cli", {"passed": report["passed"], "cases": len(report["results"]),
                               "gate": report["gate"],
                               "busybox_sha256": report["busybox_sha256"]})
    assert report["passed"] and report["fake"] is False and report["gate"] is XFS
