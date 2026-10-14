"""The env conformance suite: passing it is the gate for
plugging a verifiers env into the sandbox.

It runs an env through the bridge on a gateway (`local_gateway`: real gVisor boxes on the
test box, FakeBoxes in unit tests), as an agent without a model would (scripted calls,
`run_scripted_episode`), so every grade goes through the service's grader and the
bridge's scoring process (never in-process), and checks:

* every reference solution the env (or the operator) declares grades 1;
* every known attack it declares grades 0;
* every task a reference covers has a declared attack with at least one call;
* built-in cases grade 0 on every task a reference covers: the untouched box, every
  artifact root deleted, every artifact root replaced by a symlink to `/`, links into
  the grader's paths (`/logs/verifier`, `/tests`) inside every root (alone, and next to
  the reference's output), reward files planted where Harbor-style graders read them, a
  background process rewriting them, every tool on PATH and every `sitecustomize`
  replaced by one printing `pass` and exiting 0 (the norm: grade in a pristine box,
  never trust the agent's), and the reference's output made malformed (every file it
  wrote emptied, not UTF-8, invalid JSON, or about the state cap of `[`);
* the untouched box and every artifact root deleted also grade 0 on a sample of the
  split: every index when it has at most SAMPLE_SIZE, else SAMPLE_SIZE indices drawn
  with a seed derived from (taskset, split) (`sample_seed`, recorded on those cases);
* built-in cases that keep the reference's grade of 1: the reference followed by a tool
  output far past the output cap, and the reference with about the state cap
  (`max_artifact_bytes` - 1 MiB) written inside the artifact roots (it must be restored
  and scored);
* every case runs `repeat` times (default 2, at the same time: a Task instance is shared
  by concurrent episodes; fewer than 2 fails the suite) with the same final status,
  reward and grader label (determinism), every transcript verifies, every run ends
  within `run_budget_s`, no run ends `aborted` or `box_failed`, and no zero comes from
  the env's own code failing on the agent's state (ENV_FAULTS) — the env norm requires
  extract and scoring to grade any agent state.
* a run the host failed on its side is run again, at most INFRA_RETRIES more times, and
  the reasons are recorded on the run that counts (`infra_retries`): an episode that
  could not be opened (503, nothing of the agent ran yet), `box_failed` because the
  runtime failed to start a command in the agent's box, and the `grading_box_failed`
  zero the grader keeps when the runtime failed to start a command on every grading
  attempt (`runtime_exec_failed`). A case still failed that way after its retries stops
  the suite (`ConformanceInfraError`: the CLI's "could not run", exit 2), never a pass.

Env hooks: `Task.setup`, the optional `Task.grading_setup` (the pristine grading box,
before the restore), `Task.finalize` and the rewards all run in the bridge's scoring
processes (unprivileged slot uid, no network, no secret, CPU/memory/file limits), never in
the controller. A hook acts only through the box: what it changes on its trace or its
task is dropped with its process. Import-time code of a preloaded env runs in the
forkserver and is trusted like task loading.

Cases are plain data, so env packages declare them without importing Reliquary: the
taskset's top-level module may define `conformance_cases(split) -> list[dict]`, each dict
`{"name": str, "index": int, "expect": 1.0 | 0.0, "calls": [[tool, arguments], ...]}`
and optionally `"absent": [path, ...]` (at most 8 absolute paths, no `..`: canary marker
files the agent's own code would create if the grader ever ran it; checked in the grading
box after the reward ran, and one existing fails the case whatever the reward) (names without `@`, kept for the built-in `<case>@<index>`; indices inside the split). An
operator can pass another `module:callable` (for an unmodified third-party env).

The built-in scripts are POSIX sh (busybox included). Each begins with a
`# reliquary-conformance: <directive>` comment line, a comment to a shell, which
FakeBoxes (`testing`) runs instead of the script. A script missing a tool it needs exits
4 and changes nothing, so its case fails openly (a malformed case keeps the reference's
1) rather than passing.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import posixpath
import random
import secrets
import shlex
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from reliquary_sandbox.attest import Budgets
from reliquary_sandbox.episode_client import EpisodeError
from reliquary_sandbox.episode_task import TOOLS

from .grader import RUNTIME_START_FACT, START_FAILED_REASON
from .local_gateway import LocalGateway, run_scripted_episode
from .manager import OPEN_FAILED

__all__ = ["BUILTIN", "CASES_ATTRIBUTE", "ENV", "ENV_FAULTS", "INFRA_RETRIES", "CaseResult",
           "CaseRun", "ConformanceCase", "ConformanceError", "ConformanceInfraError",
           "ConformanceReport", "builtin_cases",
           "check_case", "env_cases", "plan_cases", "run_conformance", "sample_indices",
           "sample_seed", "suite_problems"]

CASES_ATTRIBUTE = "conformance_cases"
ENV, BUILTIN = "env", "builtin"
REWARD_FILES = ("/logs/verifier/reward.txt", "/logs/verifier/reward.json")
DIRECTIVE = "# reliquary-conformance: "
MALFORMED = ("empty", "not_utf8", "invalid_json", "hostile")
HOSTILE_BYTES = 31 * 1024**2
"""What `malformed_hostile` writes in all (shared by the files the reference wrote): just
under the bridge's default state cap (32 MiB), so the state is restored and scored."""
LARGE_ARTIFACT_BYTES = 31 * 1024**2
"""What `reference_with_large_artifact` adds inside the roots by default (the bridge's
default state cap less 1 MiB); `plan_cases` uses the env's `max_artifact_bytes` - 1 MiB."""
IMPLICIT_ROOT = "/logs/artifacts"
"""verifiers' own artifact directory: filled (created) even when absent."""
TAG_BYTES = 8
"""Names of the files a case leaves in the box (the mark, the fill file, the shim kit)
come from a random tag drawn for each plan: no fixed name for an env to look for."""


def shim_kits(name: str) -> tuple[str, ...]:
    """Where `shimmed_tools` keeps the shell, rm and chmod it runs the shims with (the
    first directory it can execute from)."""
    return (f"/{name}", f"/var/tmp/{name}", f"/tmp/{name}", f"${{HOME:-/root}}/{name}")


def _names(tag: str) -> tuple[str, str, str]:
    """The mark's path, the fill file's name, the shim kit's name, from the tag."""
    return f"/tmp/.{tag}m", f".{tag}f", f".{tag}k"


OVERSIZED_OUTPUT_BYTES = 8 * 1024**2
"""A tool output far past any output cap (the cap cuts it, the episode goes on)."""
SAMPLE_SIZE = 32
MAX_ABSENT = 8
SAMPLED = ("untouched", "empty_artifacts")
RUN_BUDGET_S = 600.0
INFRA_RETRIES = 2
"""Extra runs of a case run the host failed on its side (`_host_failure`)."""
MIN_REPEAT = 2
GRADER_KEY = "_grader"
OUT_OF_RANGE = "reward_out_of_range"
"""The bridge's grading fact for a reward that is not finite in [0, 1] (graded 0)."""
ENV_FAULTS = {"grading_failed": "the env's scoring raised on the agent's state",
              "grading_timeout": "the env's scoring ran out of time or memory",
              "extract_failed": "the env's extract (finalize) raised on the agent's state",
              "invalid_grade_result": "the env's grade returned an invalid result",
              OUT_OF_RANGE: "the env's reward is not finite in [0, 1]"}
"""Grader labels (and the bridge's `reward_out_of_range`) of a zero the env's own code
caused: a conformance failure."""
BUILTIN_ENV_FAULTS = {
    "extract_timeout": "the env's extract (finalize) ran out of time on the agent's state",
    "grading_output_overflow": "the env's grading produced more output than the cap",
    "grading_box_failed": "the grading box failed on the agent's state"}
"""Labels that are also the env's fault on a built-in case (the agent's code never runs
there, and the malformed or large state is the suite's), though a declared attack may
legitimately cause them."""

AGENT_SABOTAGE = {"shimmed_tools": ("extract_failed",)}
"""Built-in cases where the agent breaks its own box's tools: an extract that fails on
them (Harbor's `collect` runs tar in the agent's box) is the agent's own 0, accepted. A
reward other than 0 there still fails."""

_CONTENT = {
    "empty": ': > "$f"',
    "not_utf8": "printf '\\377\\376\\200\\300' > \"$f\"",
    "invalid_json": "printf '%s' '{\"reward\": 1.0, \"passed\": [tru' > \"$f\"",
    "hostile": "head -c \"$size\" /dev/zero | tr '\\000' '[' > \"$f\"",
}


class ConformanceError(ValueError):
    """The cases cannot be loaded or are malformed."""


class ConformanceInfraError(RuntimeError):
    """A case run kept failing on the host's side past INFRA_RETRIES: the suite could not
    run (never a pass, never an env failure)."""


@dataclass(frozen=True)
class ConformanceCase:
    name: str
    split: str
    index: int
    expect: float
    calls: tuple[tuple[str, dict[str, Any]], ...]
    source: str = ENV
    """`env` (declared by the env or the operator) or `builtin`."""
    sample_seed: int | None = None
    """For a built-in case on a sampled index: the seed that drew it."""
    sample_salt: str | None = None
    """... and the salt of that seed (fresh for each plan, recorded in the report)."""
    absent: tuple[str, ...] = ()
    """Canary marker files (absolute paths) that agent-placed code would create if the
    grader ever executed it: checked in the grading box after the reward ran; one
    existing fails the case (`canary_executed:<path>`), whatever the reward."""

    @property
    def kind(self) -> str:
        return "reference" if self.expect == 1.0 else "attack"


def _case(raw: Any, split: str) -> ConformanceCase:
    required = {"name", "index", "expect", "calls"}
    if not isinstance(raw, Mapping) or not required <= set(raw) <= required | {"absent"}:
        raise ConformanceError(f"a case is {{name, index, expect, calls[, absent]}}: {raw!r}")
    name, index, expect, calls = raw["name"], raw["index"], raw["expect"], raw["calls"]
    if not isinstance(name, str) or not name:
        raise ConformanceError(f"invalid case name {name!r}")
    if "@" in name:
        raise ConformanceError(f"{name}: '@' is kept for the built-in cases' names")
    if type(index) is not int or index < 0:
        raise ConformanceError(f"{name}: invalid index {index!r}")
    if isinstance(expect, bool) or expect not in (0.0, 1.0):
        raise ConformanceError(f"{name}: expect is 1.0 (reference) or 0.0 (attack)")
    if not isinstance(calls, (list, tuple)):
        raise ConformanceError(f"{name}: calls must be a list of [tool, arguments]")
    parsed = []
    for call in calls:
        if (not isinstance(call, (list, tuple)) or len(call) != 2
                or not isinstance(call[0], str) or call[0] not in TOOLS
                or not isinstance(call[1], Mapping)):
            raise ConformanceError(f"{name}: invalid call {call!r}")
        parsed.append((call[0], dict(call[1])))
    return ConformanceCase(name, split, index, float(expect), tuple(parsed),
                           absent=_absent(name, raw.get("absent", ())))


def _absent(name: str, raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)) or len(raw) > MAX_ABSENT:
        raise ConformanceError(f"{name}: absent is a list of at most {MAX_ABSENT} paths")
    for path in raw:
        if (not isinstance(path, str) or not path.startswith("/") or "\0" in path
                or ".." in path.split("/")):
            raise ConformanceError(f"{name}: invalid absent path {path!r} (absolute, no "
                                   "'..')")
    return tuple(raw)


def _cases_function(module: Any, entry: str | None) -> Callable[[str], Iterable[Any]]:
    if entry is not None:
        module_name, _, attribute = entry.partition(":")
        try:
            function = getattr(importlib.import_module(module_name), attribute)
        except (ImportError, AttributeError, ValueError) as exc:
            raise ConformanceError(f"cannot load the cases {entry!r}") from exc
    else:
        function = getattr(module, CASES_ATTRIBUTE, None)
        if function is None:
            raise ConformanceError(f"{module.__name__} defines no {CASES_ATTRIBUTE}(split)")
    if not callable(function):
        raise ConformanceError("the conformance cases must be a callable")
    return function


def env_cases(module: Any, split: str, entry: str | None = None) -> list[ConformanceCase]:
    """The declared cases of `split`: `entry` (`module:callable`) when given, else the
    env module's `conformance_cases`. Names must be unique."""
    raw_cases = _cases_function(module, entry)(split)
    if isinstance(raw_cases, (str, bytes, Mapping)):
        raise ConformanceError("the conformance cases must be a list of cases")
    cases = [_case(raw, split) for raw in raw_cases]
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise ConformanceError("case names must be unique")
    return cases


def _script(directive: str, body: str) -> str:
    return f"{DIRECTIVE}{directive}\n{body}"


def _need(*tools: str) -> str:
    """Exit 4 (before changing anything) when a tool the script runs is missing."""
    return (f"for t in {' '.join(tools)}; do command -v \"$t\" >/dev/null 2>&1 || "
            "{ echo \"conformance: $t is missing\" >&2; exit 4; }; done")


def _corrupt_script(kind: str, roots: Sequence[str], total_bytes: int, mark: str) -> str:
    """Replace every regular file written since `mark` (outside /proc, /sys, /dev, /tmp,
    /run; the roots are searched too, even on another mount) by `kind`."""
    starts = " ".join(["/", *(shlex.quote(root) for root in roots)])
    return _script(f"corrupt {kind} {total_bytes}", "\n".join([
        _need("find", "sort", "wc", *(("head", "tr") if kind == "hostile" else ()), "rm"),
        f"files=$(find {starts} -xdev \\( -path /proc -o -path /sys -o -path /dev"
        f" -o -path /tmp -o -path /run \\) -prune -o -type f -newer {mark} -print"
        " 2>/dev/null | sort -u)",
        "[ -n \"$files\" ] || { echo 'the reference wrote no file' >&2; exit 3; }",
        "n=$(printf '%s\\n' \"$files\" | wc -l)",
        f"size=$(( {total_bytes} / n ))",
        f"printf '%s\\n' \"$files\" | while IFS= read -r f; do {_CONTENT[kind]}; done",
        f"rm -f {mark}",
    ]))


def _fill_script(roots: Sequence[str], total_bytes: int, name: str) -> str:
    """Write `total_bytes` of `x` inside the artifact roots: shared by every root that is
    a directory, and IMPLICIT_ROOT when it is one of them (created)."""
    quoted = " ".join(shlex.quote(root) for root in roots)
    return _script(f"fill {total_bytes} {name} {quoted}", "\n".join([
        _need("mkdir", "head", "tr"),
        f"set -- $(for r in {quoted}; do"
        f" if [ -d \"$r\" ] || [ \"$r\" = {IMPLICIT_ROOT} ]; then printf '%s\\n' \"$r\"; fi;"
        " done)",
        "[ $# -gt 0 ] || { echo 'no artifact root is a directory' >&2; exit 3; }",
        f"size=$(( {total_bytes} / $# ))",
        f"for r; do mkdir -p -- \"$r\" && head -c \"$size\" /dev/zero | tr '\\000' x"
        f" > \"$r/{name}\" || exit 1; done",
    ]))


def _shim_script(kit_name: str) -> str:
    """Replace every file on PATH by a script printing `pass` and exiting 0, and plant
    a `sitecustomize.py` doing the same in every site-packages / dist-packages directory
    of the root file system: what an env that grades in the agent's box would trust. The
    shims run on a copy of /bin/sh (rm and chmod copied too) in the first `shim_kits`
    directory it can execute from."""
    return _script("shim", "\n".join([
        _need("cp", "rm", "chmod", "find", "mkdir"),
        "dirs=$PATH",
        "kit=",
        f"for d in {' '.join(shim_kits(kit_name))}; do",
        "  if mkdir -p \"$d\" 2>/dev/null && cp -L /bin/sh \"$d/sh\" 2>/dev/null"
        " && \"$d/sh\" -c : 2>/dev/null; then kit=$d; break; fi",
        "done",
        "[ -n \"$kit\" ] || { echo 'conformance: no directory to run shims from' >&2; exit 4; }",
        "cp -L \"$(command -v rm)\" \"$kit/rm\" && cp -L \"$(command -v chmod)\" \"$kit/chmod\""
        " || exit 1",
        "find / -xdev \\( -path /proc -o -path /sys -o -path /dev -o -path /run"
        " -o -path \"$kit\" \\) -prune -o -type d \\( -name site-packages -o -name dist-packages"
        " \\) -print 2>/dev/null | while IFS= read -r d; do",
        "  printf '%s\\n' 'import os, sys' 'sys.stdout.write(\"pass\\n\")' 'sys.stdout.flush()'"
        " 'os._exit(0)' > \"$d/sitecustomize.py\"",
        "done",
        ": > \"$kit/list\"",
        "old=$IFS; IFS=:; set -- $dirs; IFS=$old",
        "for d; do for f in \"$d\"/*; do",
        "  if [ -L \"$f\" ] || [ -f \"$f\" ]; then printf '%s\\n' \"$f\" >> \"$kit/list\"; fi",
        "done; done",
        "while IFS= read -r f; do \"$kit/rm\" -f -- \"$f\""
        " && printf '#!%s\\necho pass\\nexit 0\\n' \"$kit/sh\" > \"$f\""
        " && \"$kit/chmod\" 755 \"$f\"; done < \"$kit/list\"",
        "exit 0",
    ]))


def sample_seed(taskset: str, split: str, salt: str = "") -> int:
    """The seed of a split's sample: fixed by (taskset, split, salt). A plan draws a
    fresh salt (`plan_cases`), so the sample is not known to the env's author in advance
    and the report's salt reproduces it."""
    digest = hashlib.sha256(f"{taskset}\n{split}\n{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def sample_indices(seed: int, size: int, k: int = SAMPLE_SIZE) -> list[int]:
    """Every index of a split of `size` tasks when `size` <= k, else k drawn by `seed`."""
    if size <= k:
        return list(range(size))
    return sorted(random.Random(seed).sample(range(size), k))


def builtin_cases(roots: Sequence[str], split: str, index: int, *,
                  reference: ConformanceCase | None = None,
                  hostile_bytes: int = HOSTILE_BYTES,
                  large_bytes: int = LARGE_ARTIFACT_BYTES,
                  tag: str | None = None) -> list[ConformanceCase]:
    """The env-independent cases of task `index`, given its artifact roots; with a
    `reference` (a case expecting 1.0 on that task) also the cases built on its calls.
    `tag` names the files the cases leave in the box (random when None)."""
    mark_path, fill_name, kit_name = _names(tag or secrets.token_hex(TAG_BYTES))
    quoted = [shlex.quote(root) for root in roots]
    links = " && ".join(
        f"rm -rf -- {q} && mkdir -p -- {shlex.quote(posixpath.dirname(root))} && ln -s / {q}"
        for root, q in zip(roots, quoted))
    into_grader = " && ".join(
        f"mkdir -p -- {q} && ln -s /logs/verifier {q}/.verifier && ln -s /tests {q}/.tests"
        for q in quoted)
    plant = ("mkdir -p /logs/verifier && echo 1 > /logs/verifier/reward.txt && "
             "echo '{\"reward\": 1.0}' > /logs/verifier/reward.json")
    rewrite = ("nohup sh -c 'while :; do mkdir -p /logs/verifier; "
               "echo 1 > /logs/verifier/reward.txt; sleep 0.2; done' >/dev/null 2>&1 &")

    def bash(command: str) -> tuple[str, dict[str, Any]]:
        return ("bash", {"command": command})

    def case(name: str, calls: Sequence[tuple[str, dict[str, Any]]],
             expect: float = 0.0) -> ConformanceCase:
        return ConformanceCase(f"{name}@{index}", split, index, expect, tuple(calls), BUILTIN)

    cases = [case("untouched", []),
             case("empty_artifacts", [bash("rm -rf -- " + " ".join(quoted))]),
             case("symlinked_artifacts", [bash(links)]),
             case("links_into_grader", [bash(into_grader)]),
             case("planted_reward_files", [bash(plant)]),
             case("background_reward_writer", [bash(rewrite)]),
             case("shimmed_tools", [bash(_shim_script(kit_name))])]
    if reference is None:
        return cases
    solution = list(reference.calls)
    # busybox stamps whole seconds: a file written in the mark's second is not -newer
    mark = bash(_script("mark", f"mkdir -p /tmp && : > {mark_path} && sleep 1"))
    for kind in MALFORMED:
        cases.append(case(f"malformed_{kind}", [
            mark, *solution, bash(_corrupt_script(kind, roots, hostile_bytes, mark_path))]))
    cases.append(case("reference_with_links_into_grader", [*solution, bash(into_grader)]))
    output = _script(f"output {OVERSIZED_OUTPUT_BYTES}",
                     f"head -c {OVERSIZED_OUTPUT_BYTES} /dev/zero | tr '\\000' x")
    cases.append(case("reference_with_oversized_output", [*solution, bash(output)], 1.0))
    cases.append(case("reference_with_large_artifact",
                      [*solution, bash(_fill_script(roots, large_bytes, fill_name))], 1.0))
    return cases


def _split_size(bridged: Any, split: str) -> int:
    from reliquary_sandbox_verifiers.loading import MAX_TASK_INDEX

    taskset = bridged.taskset(split)
    try:
        return len(taskset)
    except TypeError:
        size = 0
        for _ in taskset:
            size += 1
            if size > MAX_TASK_INDEX:
                break
        return size


def plan_cases(bridged: Any, split: str, *, entry: str | None = None,
               builtin: bool = True, salt: str | None = None,
               tag: str | None = None,
               sample_size: int = SAMPLE_SIZE) -> list[ConformanceCase]:
    """The env's cases, then the built-in ones for every task index a reference covers
    (built on that index's first reference), then `untouched` and `empty_artifacts` on
    the split's sample (`sample_indices`) not covered yet. `bridged` is the env's
    `VerifiersEnv` (`TaskRegistry.bridged(env)`). A case outside the split is refused.
    `salt` (a fresh random one when None) is mixed into the sample's seed; one tag names
    the box files of all the plan's built-in cases (`tag`: a fresh random one when None).
    `sample_size` (at most SAMPLE_SIZE) bounds the sample; below SAMPLE_SIZE the run is
    not a gate (each sampled task needs its image on the host: a host without room for 32
    task images runs a reduced sample)."""
    from reliquary_sandbox_verifiers.declaration import artifact_roots

    if type(sample_size) is not int or not 0 <= sample_size <= SAMPLE_SIZE:
        raise ConformanceError(f"sample_size must be in 0..{SAMPLE_SIZE}, got {sample_size!r}")

    cases = env_cases(bridged.module, split, entry)
    size = _split_size(bridged, split)
    for case in cases:
        if case.index >= size:
            raise ConformanceError(f"{case.name}: index {case.index} is outside the split "
                                   f"({size} tasks)")
    if not builtin:
        return cases
    references: dict[int, ConformanceCase] = {}
    for case in cases:
        if case.kind == "reference":
            references.setdefault(case.index, case)
    defaults = bridged.options.defaults
    hostile = max(0, min(HOSTILE_BYTES, defaults.max_artifact_bytes - 1024**2))
    large = max(0, defaults.max_artifact_bytes - 1024**2)

    def roots_of(index: int) -> list[str]:
        task = bridged.task(split, index)
        workdir = task.data.workdir or defaults.workdir
        return [root.path for root in artifact_roots(task, workdir)]

    tag = secrets.token_hex(TAG_BYTES) if tag is None else tag
    for index in sorted(references):
        cases.extend(builtin_cases(roots_of(index), split, index, reference=references[index],
                                   hostile_bytes=hostile, large_bytes=large, tag=tag))
    salt = secrets.token_hex(TAG_BYTES) if salt is None else salt
    seed = sample_seed(bridged.ref.taskset_id, split, salt)
    for index in sample_indices(seed, size, sample_size):
        if index in references:
            continue
        cases.extend(replace(case, sample_seed=seed, sample_salt=salt)
                     for case in builtin_cases(roots_of(index), split, index, tag=tag)
                     if case.name.split("@")[0] in SAMPLED)
    return cases


@dataclass(frozen=True)
class CaseRun:
    status: str
    reward: float | None
    reason: str | None
    verified: bool
    fault: str | None = None
    """The grader's label of a zero it decided (`_grader.agent_fault`), else None."""
    seconds: float = 0.0
    canary_checked: int = 0
    """How many `absent` paths the grading box was checked for (the grader's fact)."""
    canary_executed: tuple[str, ...] = ()
    """The `absent` paths that existed after the reward ran."""
    canary_probe_failed: str | None = None
    runtime_exec_failed: bool = False
    """The grader kept its zero after the runtime failed to start a command on every
    grading attempt (`_grader.runtime_exec_failed`)."""
    infra_retries: tuple[str, ...] = ()
    """Why the runs before this one were run again (the host failed them)."""
    regraded: int = 0
    """Grading attempts of this run that were run again in a fresh box because the runtime
    failed to start a command (recorded whether or not the case passes)."""
    helper_retries: int = 0
    """Read-only helpers of this run that were run again in place for the same reason."""


@dataclass(frozen=True)
class CaseResult:
    case: ConformanceCase
    runs: tuple[CaseRun, ...]
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass(frozen=True)
class ConformanceReport:
    env: str
    env_package: str
    results: tuple[CaseResult, ...]
    problems: tuple[str, ...] = field(default=())
    """Suite-level problems (no reference, no declared attack, fewer than 2 repeats)."""
    image_lock_sha256: str | None = None
    """`ImageLock.sha256()` of the lock the env was served with in this run (None: not a
    bridged env): a report attests the env under exactly this image lock."""

    @property
    def passed(self) -> bool:
        return not self.problems and all(result.ok for result in self.results)

    @property
    def sample_seed(self) -> int | None:
        seeds = {r.case.sample_seed for r in self.results} - {None}
        return min(seeds) if seeds else None

    @property
    def sample_salt(self) -> str | None:
        salts = {r.case.sample_salt for r in self.results} - {None}
        return min(salts) if salts else None

    @property
    def regraded(self) -> int:
        return sum(run.regraded for r in self.results for run in r.runs)

    @property
    def helper_retries(self) -> int:
        return sum(run.helper_retries for r in self.results for run in r.runs)

    def to_json(self) -> dict[str, Any]:
        return {"env": self.env, "env_package": self.env_package, "passed": self.passed,
                "regraded": self.regraded, "helper_retries": self.helper_retries,
                "problems": list(self.problems), "sample_seed": self.sample_seed,
                "sample_salt": self.sample_salt, "image_lock_sha256": self.image_lock_sha256,
                "results": [{"case": {**asdict(r.case), "kind": r.case.kind},
                             "runs": [asdict(run) for run in r.runs],
                             "problems": list(r.problems), "ok": r.ok}
                            for r in self.results]}


def _outcome(run: CaseRun) -> str:
    return f"{run.status}/{run.reward}" + (f"/{run.fault}" if run.fault else "")


def check_case(case: ConformanceCase, runs: Sequence[CaseRun], *,
               budget_s: float | None = None) -> tuple[str, ...]:
    problems = []
    for i, run in enumerate(runs):
        if not run.verified:
            problems.append(f"run {i}: the transcript does not verify")
        if run.status != "graded":
            problems.append(f"run {i}: ended {run.status} ({run.reason})")
        elif run.reward != case.expect:
            problems.append(f"run {i}: reward {run.reward}, expected {case.expect}")
        faults = ENV_FAULTS if case.source != BUILTIN else {**ENV_FAULTS, **BUILTIN_ENV_FAULTS}
        accepted = AGENT_SABOTAGE.get(case.name.split("@")[0], ()) if case.source == BUILTIN else ()
        faults = {label: why for label, why in faults.items() if label not in accepted}
        if run.status == "graded" and run.fault in faults:
            problems.append(f"run {i}: graded 0 by {run.fault}: {faults[run.fault]}")
        if case.absent and run.status == "graded":
            problems.extend(f"run {i}: canary_executed:{path}" for path in run.canary_executed)
            if run.canary_probe_failed:
                problems.append(f"run {i}: canary_probe_failed:{run.canary_probe_failed}")
            elif run.canary_checked != len(case.absent):
                problems.append(f"run {i}: the canaries were not checked "
                                f"({run.canary_checked} of {len(case.absent)})")
        if budget_s is not None and run.seconds > budget_s:
            problems.append(f"run {i}: took {run.seconds:.1f} s, over the {budget_s:g} s "
                            "budget")
    if len({(run.status, run.reward, run.fault) for run in runs}) > 1:
        problems.append("not deterministic: " + ", ".join(_outcome(run) for run in runs))
    return tuple(problems)


def suite_problems(cases: Sequence[ConformanceCase]) -> tuple[str, ...]:
    """A reference, and on every task a reference covers a declared attack that does
    something (at least one call)."""
    problems = []
    references = sorted({case.index for case in cases
                         if case.kind == "reference" and case.source == ENV})
    attacked = {case.index for case in cases
                if case.kind == "attack" and case.source == ENV and case.calls}
    if not references:
        problems.append("no reference solution (a case expecting 1.0)")
    if not attacked:
        problems.append("no declared attack (a case of the env expecting 0.0, with calls)")
    for index in references:
        if attacked and index not in attacked:
            problems.append(f"task {index} has a reference but no declared attack with calls")
    return tuple(problems)


def _fault(final: Mapping[str, Any]) -> str | None:
    grading = final.get("grading")
    if not isinstance(grading, Mapping):
        return None
    label = grading.get(GRADER_KEY)
    fault = label.get("agent_fault") if isinstance(label, Mapping) else None
    if isinstance(fault, str):
        return fault
    return OUT_OF_RANGE if OUT_OF_RANGE in grading else None


def _canaries(final: Mapping[str, Any]) -> tuple[int, tuple[str, ...], str | None]:
    grading = final.get("grading")
    label = grading.get(GRADER_KEY) if isinstance(grading, Mapping) else None
    if not isinstance(label, Mapping):
        return 0, (), None
    checked, found, failed = (label.get("canary_checked"), label.get("canary_executed"),
                              label.get("canary_probe_failed"))
    return (checked if type(checked) is int else 0,
            tuple(p for p in found if isinstance(p, str)) if isinstance(found, list) else (),
            failed if isinstance(failed, str) else None)


def _runtime_exec_failed(final: Mapping[str, Any]) -> bool:
    grading = final.get("grading")
    label = grading.get(GRADER_KEY) if isinstance(grading, Mapping) else None
    return isinstance(label, Mapping) and label.get(RUNTIME_START_FACT) is True


def _host_failure(run: CaseRun) -> str | None:
    """Why the host, not the case, decided this run's outcome; None when it did not
    (a canary that ran in the grading box is the case's failure, whatever came after)."""
    if run.status == "box_failed" and run.reason == START_FAILED_REASON:
        return f"box_failed: {run.reason}"
    if run.status == "graded" and run.runtime_exec_failed and not run.canary_executed:
        return "graded 0 by grading_box_failed: the runtime failed to start a command on " \
               "every grading attempt"
    return None


async def _run_retrying(gateway: LocalGateway, env: str, case: ConformanceCase,
                        budgets: Budgets | None) -> CaseRun:
    """`_run`, again while the host fails it on its side (`_host_failure`, or an open
    refused with OPEN_FAILED), at most INFRA_RETRIES more times."""
    reasons: list[str] = []
    for _ in range(INFRA_RETRIES + 1):
        try:
            run = await _run(gateway, env, case, budgets)
        except EpisodeError as exc:
            if exc.status != 503 or exc.detail != OPEN_FAILED:
                raise
            reasons.append(f"the episode could not be opened ({exc.detail})")
            continue
        why = _host_failure(run)
        if why is None:
            return replace(run, infra_retries=tuple(reasons))
        reasons.append(why)
    raise ConformanceInfraError(f"{case.name}: the host failed a run on its side "
                                f"{len(reasons)} times: " + "; ".join(reasons))


async def _run(gateway: LocalGateway, env: str, case: ConformanceCase,
               budgets: Budgets | None) -> CaseRun:
    started = time.monotonic()
    episode = await run_scripted_episode(gateway, env=env, split=case.split,
                                         index=case.index, calls=case.calls, budgets=budgets,
                                         absent=case.absent)
    final = episode.final
    return CaseRun(final["status"], final["reward"], final["reason"],
                   episode.verification.ok, _fault(final), time.monotonic() - started,
                   *_canaries(final), _runtime_exec_failed(final), (),
                   episode.retries.get("regraded", 0), episode.retries.get("helper_retries", 0))


async def run_conformance(gateway: LocalGateway, *, env: str, env_package: str,
                          cases: Sequence[ConformanceCase], repeat: int = 2,
                          budgets: Budgets | None = None,
                          run_budget_s: float | None = RUN_BUDGET_S,
                          image_lock_sha256: str | None = None) -> ConformanceReport:
    """Run every case `repeat` times on `gateway` (the runs of a case at the same time,
    the cases one after the other) and judge them. Fewer than MIN_REPEAT runs check no
    determinism: the report then fails (a diagnostic run, never a gate). A run the host
    failed is run again (`_run_retrying`); ConformanceInfraError when it keeps failing.
    `image_lock_sha256`: the digest of the image lock the gateway serves the env with,
    recorded in the report (a validator checks it against the lock it pins)."""
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    results = []
    for case in cases:
        # Every run of the case finishes (its episode closed) before a failure of one
        # of them propagates: none is left running behind ConformanceInfraError.
        outcomes = await asyncio.gather(*(_run_retrying(gateway, env, case, budgets)
                                          for _ in range(repeat)), return_exceptions=True)
        failed = next((item for item in outcomes if isinstance(item, BaseException)), None)
        if failed is not None:
            raise failed
        runs = [item for item in outcomes if isinstance(item, CaseRun)]
        results.append(CaseResult(case, tuple(runs),
                                  check_case(case, runs, budget_s=run_budget_s)))
    problems = suite_problems(cases)
    if repeat < MIN_REPEAT:
        problems += (f"repeat {repeat}: determinism needs {MIN_REPEAT} runs at the same time",)
    return ConformanceReport(env, env_package, tuple(results), problems,
                             image_lock_sha256=image_lock_sha256)
