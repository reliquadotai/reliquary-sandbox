"""`reliquary-sandbox-conformance`: run the env conformance suite on a local gateway.

    reliquary-sandbox-conformance --entry 'verifiers:reliquary-swe==0.3.0' --split rl \
        --docker-host unix:///run/reliquary-xfs-docker.sock --busybox /usr/bin/busybox \
        --out conformance.json

Real boxes need Docker with runsc, the task images pulled, and the helper busybox: run it
on the test box only (docs/deployment.md, "Episode hosts: testing"), never on a production
host. `--fake` runs the same suite over FakeBoxes (no Docker) and also lets an unpackaged
taskset load: a smoke test of the cases and of the env's loading, not a gate (the report
says `fake: true`). A real run never loads an unpackaged taskset and never imports the
fakes. Exit status: 0 passed, 1 failed, 2 not runnable.

The suite starts a forkserver, so this module keeps its entry behind the `__main__` guard.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import posixpath
import secrets
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .conformance import SAMPLE_SIZE, ConformanceReport, plan_cases, run_conformance
from .local_gateway import local_gateway
from .registry import TaskRegistry

ENV = "conformance-env"
HOST_DAEMON_SOCKETS = ("/var/run/docker.sock", "/run/docker.sock")
"""The host's own Docker socket: the attacks must never run on it."""
UNPRIVILEGED_OK = "RELIQUARY_SCORING_UNPRIVILEGED_OK"  # scoring_server.UNPRIVILEGED_OK
_euid = os.geteuid
"""The run's effective uid (patched in tests: os.geteuid itself also drives scoring)."""


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reliquary-sandbox-conformance",
                                description="Run the env conformance suite on a local gateway.")
    p.add_argument("--entry", required=True, help="verifiers:<taskset-id>==<version>")
    p.add_argument("--options", default="{}", help="the env's episode_env_options, as JSON")
    p.add_argument("--split", default="rl")
    p.add_argument("--cases", default=None,
                   help="module:callable returning the cases (default: the env's own)")
    p.add_argument("--no-builtin", action="store_true", help="skip the built-in attacks")
    p.add_argument("--repeat", type=int, default=2, help="runs per case (determinism)")
    p.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    p.add_argument("--fake", action="store_true",
                   help="FakeBoxes instead of Docker, unpackaged tasksets allowed (smoke test)")
    p.add_argument("--docker-host", default=None, help="the test daemon's socket URL")
    p.add_argument("--docker-pidfile", type=Path, default=None)
    p.add_argument("--busybox", type=Path, default=None, help="the static helper busybox")
    p.add_argument("--cpu-source", choices=("docker", "cgroupfs"), default="docker")
    p.add_argument("--allow-unquoted-disk", action="store_true",
                   help="a daemon whose storage has no xfs project quota (test hosts)")
    p.add_argument("--sample-size", type=int, default=SAMPLE_SIZE,
                   help="tasks of the split's sample (default 32; fewer is not a gate)")
    p.add_argument("--sample-salt", default=None,
                   help="salt of the split sample (default: random, recorded in the report)")
    p.add_argument("--file-tag", default=None,
                   help="tag naming the files the cases leave in the box (default: random)")
    return p


def _is_host_daemon(url: str) -> bool:
    if not url.startswith("unix://"):
        return False
    return posixpath.normpath(url[len("unix://"):]) in HOST_DAEMON_SOCKETS


def provenance(args: argparse.Namespace, *, busybox_sha256: str | None,
               euid: int | None = None) -> dict[str, Any]:
    """What the run was configured with, and whether its verdict can stand as a gate.
    Only a run as root scores in processes dropped to their slot uids (a scoring process
    that cannot drop is refused, so euid 0 means the slot uids were used): any other run
    (RELIQUARY_SCORING_UNPRIVILEGED_OK) kept the caller's uid and is not a gate."""
    euid = _euid() if euid is None else euid
    reasons = []
    if args.fake:
        reasons.append("--fake")
    if args.allow_unquoted_disk:
        reasons.append("--allow-unquoted-disk")
    if euid != 0:
        reasons.append("unprivileged")
    if getattr(args, "sample_size", SAMPLE_SIZE) < SAMPLE_SIZE:
        reasons.append("sample")
    if getattr(args, "no_builtin", False):
        reasons.append("--no-builtin")
    return {"allow_unquoted_disk": args.allow_unquoted_disk, "cpu_source": args.cpu_source,
            "docker_host": args.docker_host, "busybox_sha256": busybox_sha256,
            "euid": euid, "scoring_slot_uids": euid == 0,
            "scoring_unprivileged_ok": os.environ.get(UNPRIVILEGED_OK) == "1",
            "gate": not reasons, "gate_reasons": reasons}


def _gateway_settings(args: argparse.Namespace, workdir: Path) -> dict[str, Any]:
    if args.fake:
        helper = workdir / "busybox"
        helper.write_bytes(b"not really busybox")
        return dict(episode_helper_busybox=helper, episode_allow_unquoted_disk=True,
                    episode_allow_unpackaged_verifiers=True,
                    episode_helper_busybox_sha256=hashlib.sha256(helper.read_bytes()).hexdigest())
    settings: dict[str, Any] = dict(
        docker_host=args.docker_host, episode_helper_busybox=args.busybox,
        episode_helper_busybox_sha256=hashlib.sha256(args.busybox.read_bytes()).hexdigest(),
        episode_cpu_source=args.cpu_source,
        episode_allow_unquoted_disk=args.allow_unquoted_disk)
    if args.docker_pidfile is not None:
        settings["episode_docker_pidfile"] = args.docker_pidfile
    return settings


async def _run(args, registry, cases, images, options, workdir: Path) -> ConformanceReport:
    task_runtime = None
    if args.fake:
        from .testing import FakeBoxes

        task_runtime = FakeBoxes(images)
    env_options = {ENV: options} if options else {}
    with local_gateway(workdir / "gateway", envs={ENV: args.entry}, images=images,
                       task_runtime=task_runtime, episode_env_options=env_options,
                       **_gateway_settings(args, workdir)) as gateway:
        return await run_conformance(gateway, env=ENV, env_package=registry.versions()[ENV],
                                     cases=cases, repeat=args.repeat,
                                     image_lock_sha256=registry.bridged(ENV).lock.sha256())


def _could_not_run(exc: BaseException) -> int:
    print(f"conformance: {exc}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.fake and (args.docker_host is None or args.busybox is None):
        print("conformance: real boxes need --docker-host and --busybox (or --fake)",
              file=sys.stderr)
        return 2
    if not args.fake and _is_host_daemon(args.docker_host):
        print("conformance: the attacks must run on an isolated test daemon "
              "(scripts/test-xfs-docker.sh), not the host's own Docker socket", file=sys.stderr)
        return 2
    salt = args.sample_salt if args.sample_salt is not None else secrets.token_hex(8)
    tag = args.file_tag if args.file_tag is not None else secrets.token_hex(4)
    try:
        options = json.loads(args.options)
        if not isinstance(options, dict):
            raise ValueError("--options must be a JSON object")
        if not args.fake and not args.busybox.is_file():
            raise ValueError(f"--busybox {args.busybox} is not a file")
        busybox_sha256 = (None if args.fake
                          else hashlib.sha256(args.busybox.read_bytes()).hexdigest())
        registry = TaskRegistry({ENV: args.entry}, {ENV: options} if options else {},
                                allow_unpackaged=args.fake)
        bridged = registry.bridged(ENV)
        if bridged is None:
            raise ValueError("the suite runs verifiers envs: --entry verifiers:<id>==<version>")
        cases = plan_cases(bridged, args.split, entry=args.cases, builtin=not args.no_builtin,
                           salt=salt, tag=tag, sample_size=args.sample_size)
        images = sorted({registry.resolve(ENV, args.split, index).task.image
                         for index in {case.index for case in cases}})
        with tempfile.TemporaryDirectory(prefix="conformance-") as workdir:
            report = asyncio.run(_run(args, registry, cases, images, options, Path(workdir)))
    except Exception as exc:  # anything that stops the run is "could not run", never "fails"
        return _could_not_run(exc)
    document = {**report.to_json(), **provenance(args, busybox_sha256=busybox_sha256),
                "entry": args.entry, "options": options, "split": args.split,
                "repeat": args.repeat, "sample_size": args.sample_size, "fake": args.fake, "sample_salt_used": salt,
                "file_tag": tag,
                "sandbox_version": importlib.metadata.version("reliquary-sandbox"),
                "finished_at": int(time.time())}
    text = json.dumps(document, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.write_text(text + "\n")
    else:
        print(text)
    for result in report.results:
        status = "ok" if result.ok else "FAIL: " + "; ".join(result.problems)
        retried = [why for run in result.runs for why in run.infra_retries]
        if retried:
            status += f" (run again after a host failure: {'; '.join(retried)})"
        again = (sum(run.regraded for run in result.runs),
                 sum(run.helper_retries for run in result.runs))
        if any(again):
            status += f" (fresh-box re-gradings {again[0]}, in-place helper retries {again[1]})"
        print(f"{result.case.name} ({result.case.kind}): {status}", file=sys.stderr)
    for problem in report.problems:
        print(f"suite: {problem}", file=sys.stderr)
    if report.regraded or report.helper_retries:
        print(f"suite: the runtime failed to start a command: {report.regraded} fresh-box "
              f"re-gradings, {report.helper_retries} in-place helper retries", file=sys.stderr)
    verdict = "PASSED" if report.passed else "FAILED"
    if not document["gate"]:
        why = {"--fake": "FAKE", "--allow-unquoted-disk": "unquoted disk",
               "unprivileged": "unprivileged scoring without slot uids",
               "sample": "a reduced sample", "--no-builtin": "no built-in attacks"}
        verdict += f" ({', '.join(why[r] for r in document['gate_reasons'])}: not a gate)"
    print(verdict, file=sys.stderr)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
