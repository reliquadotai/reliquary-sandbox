"""`reliquary-sandbox-reference-sweep`: grade an env's reference solutions over many task
indices on a local gateway (a qualification measurement).

The env declares them as plain data in its top-level module,
`reference_calls(split, index) -> [[tool, arguments], ...] | None` (None: no reference for
that task; it is skipped). Each task runs once, one after the other, and one JSON line per
task goes to `--out`, carrying the run's provenance (as the conformance report does). It
measures how many references grade 1 through the bridge on real boxes (an env change such
as a new grading uid must not turn honest solutions into 0); it is not the conformance
gate. Exit status: 0 every swept reference graded 1 (at least one was swept), 1 otherwise,
2 not runnable (any crash; the lines of the tasks done before it stay in `--out`). Real boxes: the test box only, against the isolated daemon
(docs/deployment.md, "Episode hosts: testing"); a real run never loads an unpackaged
taskset. `--fake` runs over FakeBoxes: a smoke test.

The run starts a forkserver, so this module keeps its entry behind the `__main__` guard.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .conformance import ConformanceCase, ConformanceError, _case, _fault
from .conformance_cli import _gateway_settings, _is_host_daemon, provenance
from .local_gateway import local_gateway, run_scripted_episode
from .registry import TaskRegistry

__all__ = ["REFERENCES_ATTRIBUTE", "SweepResult", "main", "reference_case", "sweep"]

REFERENCES_ATTRIBUTE = "reference_calls"
ENV = "reference-sweep-env"


@dataclass(frozen=True)
class SweepResult:
    index: int
    status: str
    """The final record's status, or `skipped` (no reference for the task)."""
    reward: float | None
    reason: str | None
    fault: str | None
    verified: bool
    seconds: float


def reference_case(module: Any, split: str, index: int) -> ConformanceCase | None:
    function = getattr(module, REFERENCES_ATTRIBUTE, None)
    if not callable(function):
        raise ConformanceError(f"{module.__name__} defines no "
                               f"{REFERENCES_ATTRIBUTE}(split, index)")
    calls = function(split, index)
    if calls is None:
        return None
    return _case({"name": "reference", "index": index, "expect": 1.0, "calls": calls}, split)


async def sweep(gateway: Any, *, env: str, module: Any, split: str,
                indices: list[int],
                on_result: Callable[[SweepResult], None] | None = None) -> list[SweepResult]:
    """`on_result` sees each result as soon as the task is done (a crash at task N keeps
    the results before it)."""
    results = []

    def add(result: SweepResult) -> None:
        results.append(result)
        if on_result is not None:
            on_result(result)

    for index in indices:
        case = reference_case(module, split, index)
        if case is None:
            add(SweepResult(index, "skipped", None, "no reference", None, False, 0.0))
            continue
        started = time.monotonic()
        episode = await run_scripted_episode(gateway, env=env, split=split, index=index,
                                             calls=case.calls)
        final = episode.final
        add(SweepResult(index, final["status"], final["reward"], final["reason"],
                        _fault(final), episode.verification.ok, time.monotonic() - started))
    return results


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reliquary-sandbox-reference-sweep",
                                description="Grade an env's reference solutions over many "
                                            "tasks on a local gateway.")
    p.add_argument("--entry", required=True, help="verifiers:<taskset-id>==<version>")
    p.add_argument("--options", default="{}", help="the env's episode_env_options, as JSON")
    p.add_argument("--split", default="rl")
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--indices", help="comma-separated task indices")
    which.add_argument("--start", type=int, help="first index of a range (with --count)")
    p.add_argument("--count", type=int, default=None,
                   help="tasks of the range (with --start; default 1)")
    p.add_argument("--out", type=Path, required=True, help="JSON lines, one per task")
    p.add_argument("--fake", action="store_true", help="FakeBoxes (smoke test)")
    p.add_argument("--docker-host", default=None)
    p.add_argument("--docker-pidfile", type=Path, default=None)
    p.add_argument("--busybox", type=Path, default=None)
    p.add_argument("--cpu-source", choices=("docker", "cgroupfs"), default="docker")
    p.add_argument("--allow-unquoted-disk", action="store_true")
    return p


def _indices(args: argparse.Namespace) -> list[int]:
    if args.indices is not None:
        if args.count is not None:
            raise ValueError("--count goes with --start, not with --indices")
        indices = [int(item) for item in args.indices.split(",") if item.strip()]
    else:
        count = 1 if args.count is None else args.count
        if count < 1:
            raise ValueError("--count must be at least 1")
        indices = list(range(args.start, args.start + count))
    if not indices or any(index < 0 for index in indices):
        raise ValueError("no task index, or a negative one")
    return indices


async def _run(args, bridged, indices, images, options, workdir: Path, on_result=None):
    task_runtime = None
    if args.fake:
        from .testing import FakeBoxes

        task_runtime = FakeBoxes(images)
    env_options = {ENV: options} if options else {}
    with local_gateway(workdir / "gateway", envs={ENV: args.entry}, images=images,
                       task_runtime=task_runtime, episode_env_options=env_options,
                       **_gateway_settings(args, workdir)) as gateway:
        return await sweep(gateway, env=ENV, module=bridged.module, split=args.split,
                           indices=indices, on_result=on_result)


def _execute(args: argparse.Namespace) -> int:
    if not args.fake and (args.docker_host is None or args.busybox is None):
        print("reference-sweep: real boxes need --docker-host and --busybox (or --fake)",
              file=sys.stderr)
        return 2
    if not args.fake and _is_host_daemon(args.docker_host):
        print("reference-sweep: run it on an isolated test daemon, not the host's own "
              "Docker socket", file=sys.stderr)
        return 2
    indices = _indices(args)
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
        raise ValueError("the sweep runs verifiers envs: --entry verifiers:<id>==<version>")
    swept = [i for i in indices if reference_case(bridged.module, args.split, i) is not None]
    images = sorted({registry.resolve(ENV, args.split, i).task.image for i in swept})
    meta = {"entry": args.entry, "options": options, "split": args.split, "fake": args.fake,
            "ran": len(swept), "skipped": len(indices) - len(swept),
            "sandbox_version": importlib.metadata.version("reliquary-sandbox"),
            "provenance": provenance(args, busybox_sha256=busybox_sha256)}
    results: list[SweepResult] = []
    with args.out.open("w") as out:

        def record(result: SweepResult) -> None:
            results.append(result)
            line = {**meta, **asdict(result), "finished_at": int(time.time())}
            out.write(json.dumps(line, sort_keys=True) + "\n")
            out.flush()

        with tempfile.TemporaryDirectory(prefix="reference-sweep-") as workdir:
            if swept:
                asyncio.run(_run(args, bridged, indices, images, options, Path(workdir),
                                 record))
            else:  # nothing to grade: no gateway (it refuses an empty image list)
                asyncio.run(sweep(None, env=ENV, module=bridged.module, split=args.split,
                                  indices=indices, on_result=record))
    ran = [r for r in results if r.status != "skipped"]
    good = [r for r in ran if r.status == "graded" and r.reward == 1.0 and r.verified]
    print(f"{len(good)}/{len(ran)} references graded 1 "
          f"({len(results) - len(ran)} tasks without a reference)", file=sys.stderr)
    return 0 if ran and len(good) == len(ran) else 1


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return _execute(args)
    except Exception as exc:  # anything that stops the run is "could not run", never "fails"
        print(f"reference-sweep: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
