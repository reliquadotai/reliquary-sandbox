"""Env packages for the live bridge tests, written at test time so their image lock pins
the image the test daemon actually has (`write_live_envs(root, image)`):

* `toy_bridge_live`: the toy taskset (tests/toy_bridge_env) with cases in real shell;
* `harbor_bridge_live`: verifiers' own, unmodified `HarborTask` (the Harbor pattern:
  `tests/test.sh` writes `/logs/verifier/reward.txt`) on a local task directory. verifiers'
  `HarborTaskset.load` needs the Harbor CLI and network to download Hub packages, which our
  hosts do not have, so only the loader is ours; the Task class and its grading are
  verifiers'.

Each package gets a dist-info whose RECORD lists its files, as a wheel install would: the
gateway and the CLI then load them through the same gate as a production env (an
installed distribution at the pinned version), never as unpackaged test modules. Every
file is world-readable and every directory world-traversable: as root, the scoring
processes import the env and read Harbor's `tests/` as their slot's uid.
"""

import json
import os
from pathlib import Path

from reliquary_sandbox_verifiers.image_lock import LOCK_FILE

VERSION = "0.0.1"

TOY_LIVE = '''
from toy_bridge_env import ToyTaskset

__all__ = ["ToyTaskset"]


def conformance_cases(split):
    return [
        {"name": "reference", "index": 0, "expect": 1.0,
         "calls": [["bash", {"command": "printf '42\\\\n' > /work/answer.txt"}]]},
        {"name": "wrong_answer", "index": 0, "expect": 0.0,
         "calls": [["bash", {"command": "printf '41\\\\n' > /work/answer.txt"}]]},
        {"name": "answer_in_the_wrong_file", "index": 0, "expect": 0.0,
         "calls": [["bash", {"command": "printf '42\\\\n' > /work/answer.bak"}]]},
        {"name": "artifact_planted_before_finalize", "index": 0, "expect": 0.0,
         "calls": [["bash", {"command": "mkdir -p /work/out && printf '42\\\\n' > /work/out/answer.txt"}]]},
    ]
'''

HARBOR_LIVE = '''
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborData, HarborTask

__all__ = ["HelloTaskset"]

TASK_DIR = Path(__file__).parent / "hello"


class HelloConfig(vf.TasksetConfig):
    split: str = "rl"


class HelloTaskset(vf.Taskset[HarborTask, HelloConfig]):
    def load(self):
        yield HarborTask(HarborData(
            idx=0, name="hello", prompt="Write 'Hello, world!' to /app/hello.txt.",
            image="toy/task:1", workdir="/app", task_dir=str(TASK_DIR),
            artifacts=[vf.Artifact(source="/app", required=False)]), self.config.task)


def conformance_cases(split):
    return [
        {"name": "reference", "index": 0, "expect": 1.0,
         "calls": [["bash", {"command": "printf 'Hello, world!\\\\n' > /app/hello.txt"}]]},
        {"name": "wrong_text", "index": 0, "expect": 0.0,
         "calls": [["bash", {"command": "printf 'Hello\\\\n' > /app/hello.txt"}]]},
        {"name": "forged_reward_in_the_artifact", "index": 0, "expect": 0.0,
         "calls": [["bash", {"command": "mkdir -p /app/logs && echo 1 > /app/logs/reward.txt"}]]},
    ]
'''

TEST_SH = """#!/bin/bash
mkdir -p /logs/verifier
if [ "$(cat /app/hello.txt 2>/dev/null)" = "Hello, world!" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""

DISTS = {"toy_bridge_live": "toy-bridge-live", "harbor_bridge_live": "harbor-bridge-live"}


def _package(root: Path, name: str, source: str, image: str) -> None:
    package = root / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(source)
    (package / LOCK_FILE).write_text(json.dumps({"version": 1, "images": {"toy/task:1": image}}))


def _dist_info(root: Path, name: str) -> None:
    """A wheel-style dist-info whose RECORD lists every file of the package."""
    info = root / f"{name}-{VERSION}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {DISTS[name]}\nVersion: {VERSION}\n")
    (info / "top_level.txt").write_text(name + "\n")
    files = sorted(p.relative_to(root).as_posix() for p in (root / name).rglob("*")
                   if p.is_file())
    files += [f"{info.name}/METADATA", f"{info.name}/top_level.txt", f"{info.name}/RECORD"]
    (info / "RECORD").write_text("".join(f"{f},,\n" for f in files))


def _world_readable(root: Path) -> None:
    root.chmod(0o755)
    for directory, names, files in os.walk(root):
        for name in names:
            Path(directory, name).chmod(0o755)
        for name in files:
            Path(directory, name).chmod(0o644)


def write_live_envs(root: Path, image: str) -> dict[str, str]:
    """Write both packages, installed as distributions, under `root` (put it on
    sys.path); returns env name -> entry."""
    _package(root, "toy_bridge_live", TOY_LIVE, image)
    _package(root, "harbor_bridge_live", HARBOR_LIVE, image)
    tests = root / "harbor_bridge_live" / "hello" / "tests"
    tests.mkdir(parents=True)
    (tests / "test.sh").write_text(TEST_SH)
    (root / "harbor_bridge_live" / "hello" / "instruction.md").write_text("Write hello.\n")
    for name in DISTS:
        _dist_info(root, name)
    _world_readable(root)
    return {"toy": f"verifiers:toy-bridge-live=={VERSION}",
            "hello": f"verifiers:harbor-bridge-live=={VERSION}"}
