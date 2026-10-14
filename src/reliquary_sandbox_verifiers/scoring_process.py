"""Run a verifiers Task hook (setup, grading_setup, finalize) or `Task.score` in a
separate scoring process (the bridge's prepare, extract and grade steps).

The reward code parses bytes the agent wrote (up to the artifact bound), so it must not
run unbounded in the service: a regex or a JSON parse that never awaits would hold the
GIL past the step deadline, where `asyncio.timeout` cannot cut it. `run_in_child`
starts a child from the service's forkserver (`reliquary_sandbox_service.scoring_server`:
never a fork of the multi-threaded service, so no lock another thread held is copied
into it, and no fd but the ones handed over). The task is pickled by the parent (our
own bytes, never the child's). The child

* closes every fd it inherited except its pipe and stdio (the forkserver's own, a
  resource tracker's, the parent sentinel);
* replaces its environment by PATH, LANG, HOME=/nonexistent, a private TMPDIR and the
  wrapper's thread counts (OPENBLAS/OMP/MKL_NUM_THREADS=1) and PYTHONHASHSEED=0 (the
  forkserver starts with it: `hash()` is the same on every validator and after every
  restart, so an env's set iteration order is too) (and zeroes its initial
  environment block; the forkserver itself was started with an empty environment, so
  no secret of the service is anywhere in its memory), sets its oom_score_adj to 1000
  (best effort: the host's OOM killer takes it before the service), and, when the
  service runs as root, drops to its slot's `ScoringIdentity` (no supplementary group,
  uid = gid = `base + slot`, one slot per concurrent scoring process: no Docker socket,
  no state directory, no signal to another episode's scoring process), then sets
  no_new_privs and installs a seccomp filter (`_deny_network`): `socket` of any family
  but AF_UNIX and io_uring_setup fail with EPERM, a syscall of another ABI kills it (no
  network: the env's code cannot reach the host's or the internet's services; it talks
  to the parent over its already-open pipe). Without the filter it refuses to score;
* unpickles the task (the env's module is normally preloaded by the forkserver; else it
  is imported here, as that user) and builds the trace, then sets RLIMIT_CPU (the CPU
  it already spent plus the seconds left before the step deadline, +1 hard), RLIMIT_AS
  (what it maps plus `memory_bytes`; soft = hard, so the env's code cannot raise it
  back), RLIMIT_FSIZE (`file_bytes`: a file past it fails with EFBIG, SIGXFSZ ignored)
  and, on its own uid, RLIMIT_NPROC (`nproc`);
* installs an audit hook (`_install_audit_guard`, `sys.addaudithook`: it cannot be
  removed) that refuses, from then on, what turns the agent's bytes into code or a
  process: `exec`, `compile` (but `ast.parse`, which only builds a tree, and
  `typing.ForwardRef` resolving a string annotation of the env's source), `code.__new__`,
  `marshal.loads` / `marshal.load`, `pickle.find_class` (so no unpickling of anything
  but builtin containers and scalars), every `ctypes.*` event, `subprocess.Popen`,
  `os.system`, `os.exec*`, `os.posix_spawn*`, `os.spawn*`, `os.fork` and `os.forkpty`.
  An import still loads its files: `exec`, `compile`, `marshal.loads`, `code.__new__`
  and `ctypes.*` are let through while importlib runs a module of the env's
  installation (`dataclasses`, `namedtuple` and the like at module level included); a
  process, `pickle.find_class` and string code outside an import never are. A refusal
  raises PermissionError where it happened and is remembered: whatever the env's code
  then does (catches it, returns a reward), the hook's result is `raised` (graded 0);
* says `ready`, then runs the hook (`Task.score`, or `setup` / `grading_setup` /
  `finalize` through verifiers' `invoke`) on its own event loop;

and a watchdog thread in the parent SIGKILLs it at the step deadline, whatever the
parent's event loop is doing. After it ended (reaped), a sweeper process (as root:
from the forkserver, dropped to the slot's uid) kills every process of that uid
(`kill(-1, SIGKILL)`: what the env's code left running), removes the TMPDIR (a
directory the env's code locked is given back 0700 first) and the uid's entries in
/dev/shm; only then is the slot released. A sweep that failed keeps its slot and is
retried at a later grade (one pending sweep per grade).

The child never touches a box. Its runtime is a `SandboxRuntime` over `_RemoteInner`,
which sends each call (run, read, write, archive, restore_archive, exists, alive,
stop_processes, prepare_*, and the unsupported members) over a pipe to the parent; the
parent serves it on its own `SandboxRuntime` (stop after each run, serialised calls,
reads that never leave the restored roots, restores whose links stay inside them, faults
recorded for the grader). Nothing the child sends is unpickled: messages are a 4-byte
length, a JSON header and raw bytes, validated per operation, bounded by
`max_message_bytes` (`max_payload_bytes` for the bytes of a write or a restore: a larger
one raises OSError EFBIG in the env's code, before it is sent). Replies carry file
contents and a command's stdout/stderr as raw bytes after the header (their lengths in
it), never JSON-escaped: no header bound trips on a command's output. An error of a parent call is handed back as the nearest builtin
exception (an `OSError` keeps its errno), tagged with the call's id, so the env's
`except OSError` works and the parent can tell which of its own exceptions the env let
through. The parent checks every reply it sends against what the child reads
(`_reply_problem`): a reply of ours the child could not read ends the scoring as ours
before it is sent.

Who is to blame when no result comes back (`ScoringOutcome.kind`):

* `done` (setup, grading_setup, finalize): the hook returned (a `scored` result from one
  of them, or `done` from `score`, is `broken`). A hook acts only through the box: what
  it changes on its trace or its task stays in the scoring process and is dropped;
* none started: less than SCORING_STARTUP_FLOOR_S was left (the restore of the agent's
  state used the time): `died`, graded 0 `grading_timeout`;
* `died` (graded 0 `grading_timeout`: the agent's bytes drove the env's code there):
  after `ready`, killed by the watchdog at the deadline, past RLIMIT_CPU
  (SIGXCPU/SIGKILL) or RLIMIT_AS (MemoryError), or any other end that is not ours (a
  crash in the env's native code, the env's code exiting), EXIT_OURS and EXIT_PROTOCOL
  included: after `ready` the child runs code the agent's bytes steer, which can exit
  with any code, so only what the parent itself detects is ours;
* `raised` (graded 0 `grading_failed`): after `ready`, anything that escapes the
  scoring's run, not only what `Task.score` (or the hook) raises (the env's code stopping the event
  loop, SystemExit), and any event the audit hook refused;
* `ours` (`aborted`): the process could not be started (pickling the task, the
  forkserver, its wrapper, an unprivileged controller, no free slot, its TMPDIR), or
  ended before `ready` for any reason (an import of the env's module failed, our setup
  failed, the host killed it), or was not ready within its ready limit
  (`ready_s`, 120 s by default: an import too slow), or the forkserver it came from
  died while it scored (a host OOM kill), or the parent was about to send a reply the
  child could not read. A broken message the CHILD sends stays `broken` (graded 0):
  after `ready` the child runs code the agent's bytes steer.

Residual (code execution in the scoring process): the audit hook takes the usual ways
from data to code away from a parser (eval/exec of a string, unpickling, marshal,
ctypes, a process), not a deserializer that calls arbitrary callables with arguments
it builds (PyYAML's unsafe loaders: `python/object/apply` reaches, for example,
`os.write` on the pipe, which no audit event covers). Such code can still forge its
result (the result is the child's word). The conformance suite therefore refuses an env
package that imports math_verify, sympy, latex2sympy2, pickle (or dill, cloudpickle,
shelve, marshal) or uses an unsafe PyYAML loader (`conformance.unsafe_imports`). What
is enforced whatever runs: isolation (no secret, no root, no network, no other
episode), and no exit code of the child after `ready` voids its episode.
"""

from __future__ import annotations

import ast
import asyncio
import builtins
import ctypes
import errno
import json
import logging
import math
import os
import pickle
import resource
import shutil
import signal
import socket
import stat
import sys
import tempfile
import threading
import time
import typing
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from multiprocessing.connection import Connection
from types import SimpleNamespace
from typing import Any

from verifiers.v1.utils.decorators import invoke

from .sandbox_runtime import SandboxRuntime

__all__ = ["HOOKS", "SCORING_STARTUP_FLOOR_S", "ScoringIdentity", "ScoringOutcome",
           "run_in_child", "score_in_child"]

LOG = logging.getLogger(__name__)

MAX_TEXT = 2000
"""Exception and repr texts crossing the pipe are cut to this many characters."""
MAX_HEADER_BYTES = 4 * 1024 * 1024
"""A message's JSON header (arguments, a command's output, the result) is at most this;
file contents travel as the raw payload after it."""
CALL_ID = "reliquary_call_id"
"""Attribute of an exception the child raised for a failed parent call."""
EXIT_OURS = 70
"""The child's exit code when our own code failed in it (setup, its event loop)."""
EXIT_PROTOCOL = 71
"""The child's exit code when it could not read a reply of the parent (our protocol)."""
EXIT_MEMORY = 72
"""The child's exit code past its memory limit (a MemoryError reached our code)."""
LIMIT_SIGNALS = (-signal.SIGKILL, -signal.SIGXCPU)
"""Exit codes of a child ended by RLIMIT_CPU (SIGXCPU soft, SIGKILL hard) or the watchdog."""
SCORING_STARTUP_FLOOR_S = 30.0
"""No scoring process is started with less than this left before the step deadline: the
time went to the restore of the agent's state, so the episode is graded 0
`grading_timeout`. With more left, a process killed before it is `ready` (it was still
importing the env) is ours (`aborted`)."""
DEFAULT_READY_S = 120.0
"""A started scoring process must say `ready` within this (BridgeDefaults.scoring_ready_s):
the import of the env's module when the forkserver did not preload it."""
READY_MARGIN_S = 1.0
"""A child not ready this long before the step deadline is ours (decided before the
grader's own step timeout can cut the wait and grade it 0)."""
HOOKS = ("setup", "grading_setup", "finalize", "score")
"""What a scoring process may run: the env's own code, never in the controller."""
DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2
SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ERRNO = 0x00050000
SECCOMP_RET_ALLOW = 0x7FFF0000
X32_SYSCALL_BIT = 0x40000000
SECCOMP_ARCHES = {
    # machine: (AUDIT_ARCH_*, __NR_socket, __NR_io_uring_setup, has the x32 ABI)
    "x86_64": (0xC000003E, 41, 425, True),
    "aarch64": (0xC00000B7, 198, 425, False),
}
"""The architectures the network filter knows. Neither has `socketcall` in its native
ABI (only the 32-bit compat ABIs, i386 and arm, do: their syscalls carry another audit
arch and kill the process), so `socket` is the only way to a new socket besides
`socketpair` (AF_UNIX only on Linux) and io_uring's IORING_OP_SOCKET (io_uring_setup
refused). x86_64's x32 ABI shares its audit arch: syscalls with X32_SYSCALL_BIT kill."""
DENIED_SYSCALLS = {
    # machine: (ptrace, process_vm_readv, process_vm_writev, pidfd_getfd)
    "x86_64": (101, 310, 311, 438),
    "aarch64": (117, 270, 271, 438),
}
"""Syscalls refused with EPERM besides io_uring_setup and `socket`: they reach into
another process of the same uid (its memory, its file descriptors)."""
MACHINE = os.uname().machine
"""The architecture the network filter is built for (a test forkserver overrides it)."""
SWEEP_TIMEOUT_S = 30.0
SHM_ROOT = "/dev/shm"
"""POSIX shared memory and semaphores (the sweeper removes its slot uid's entries)."""
OOM_SCORE_ADJ = "/proc/self/oom_score_adj"
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
"""Kept from the wrapper's environment (each "1"): math libraries run on one thread."""
KILL_ROUNDS = 50
"""The sweeper signals its uid's processes until none is left, at most this many times."""
REFUSED_EVENTS = frozenset({
    "exec", "compile", "code.__new__", "marshal.loads", "marshal.load", "pickle.find_class",
    "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
    "os.forkpty"})
"""Audit events a scoring process refuses once it is about to be ready (besides every
`ctypes.*` event): from the agent's bytes to code (string source, code objects,
marshal, unpickling a global, foreign calls) or to a process."""
REFUSED_PREFIX = "ctypes."
IMPORT_EVENTS = frozenset({"exec", "compile", "code.__new__", "marshal.loads"})
"""Refused events an import needs to load the env's files (a `.pyc`, a `.py`, the
`exec` of the module, `dataclasses` and `namedtuple` at module level): let through
while importlib runs, along with `ctypes.*` (a module that loads its library)."""
IMPORT_FRAMES = ("<frozen importlib._bootstrap>", "<frozen importlib._bootstrap_external>",
                 "<frozen zipimport>")
"""The filenames of importlib's own frames (frozen: no file an env's code can replace)."""
_AST_FILE = ast.__file__
_TYPING_FILE = typing.__file__
TYPING_CODE = frozenset({"__init__", "_evaluate"})
"""`typing.ForwardRef` compiles (`__init__`) and evaluates (`_evaluate`) a string
annotation of the env's source: let through (pydantic resolves a model's postponed
annotations the first time it builds one, as the terminal env's `grading_setup` does)."""
_REFUSED: list[str] = []
"""The events the audit hook refused in this process (the first decides the result)."""


@dataclass(frozen=True)
class ScoringIdentity:
    """Who the scoring process runs as. When the service runs as root (it must then
    hold CAP_SETUID, CAP_SETGID and CAP_CHOWN), the child drops to `uid`:`gid` with no
    supplementary group; otherwise it keeps the service's user. Either way its
    environment is only PATH, LANG, HOME=/nonexistent and TMPDIR (a private directory
    under `tmp_root`, removed after the scoring), and it runs with no_new_privs.
    `run_in_child` sets uid and gid from the slot it takes (scoring_server.SLOTS)."""

    uid: int | None = None
    gid: int | None = None
    tmp_root: str | None = None
    """Where the private TMPDIRs are made (None: scoring_server.TMP_ROOT, else the
    service's temporary directory)."""
    nproc: int = 64
    """RLIMIT_NPROC on the slot's own uid (processes and threads of that uid)."""
    file_bytes: int | None = 64 * 1024**2
    """RLIMIT_FSIZE: the largest file the scoring process may write (EFBIG past it)."""


@dataclass
class ScoringOutcome:
    kind: str
    """`done` (a setup, grading_setup or finalize hook returned), `scored` (summary
    set), `raised` (Task.score or the hook raised in the child), `died` (the
    child ended without a result because of the scoring: killed at the deadline, past
    its CPU or memory limit, crashed after `ready`), `broken` (the child broke the
    protocol) or `ours` (our side failed: see the module docstring)."""
    summary: dict[str, Any] | None = None
    call_errors: list[BaseException] = field(default_factory=list)
    """`raised`: the parent-side exceptions of the calls the child's exception chains from."""
    detail: str = ""


# -- framing --------------------------------------------------------------------------

def _path_text(value: Any) -> str:
    """JSON for a path-like argument; anything else (bytes included) is a TypeError."""
    text = os.fspath(value) if isinstance(value, os.PathLike) else value
    if not isinstance(text, str):
        raise TypeError(f"{type(value).__name__} cannot cross to the sandbox: use str")
    return text


def _frame(header: dict[str, Any], payload: bytes = b"") -> bytes:
    """A message. TypeError when the header holds what JSON cannot carry (a bytes path)."""
    try:
        raw = json.dumps(header, separators=(",", ":"), default=_path_text).encode()
    except ValueError as exc:  # a circular argument
        raise TypeError(f"an argument cannot cross to the sandbox: {exc}") from None
    return len(raw).to_bytes(4, "big") + raw + payload


def _unframe(message: bytes) -> tuple[dict[str, Any], bytes]:
    """(header, payload); ValueError for anything malformed (a header over
    MAX_HEADER_BYTES, nesting past the recursion limit included)."""
    if len(message) < 4:
        raise ValueError("short message")
    size = int.from_bytes(message[:4], "big")
    if size > len(message) - 4 or size > MAX_HEADER_BYTES:
        raise ValueError("truncated or oversized header")
    try:
        header = json.loads(message[4:4 + size])
    except RecursionError as exc:
        raise ValueError("the header nests too deeply") from exc
    if not isinstance(header, dict):
        raise ValueError("the header is not an object")
    return header, message[4 + size:]


def _text(value: Any) -> str:
    try:
        return str(value)[:MAX_TEXT]
    except Exception:
        return "<unprintable>"


# -- the child ------------------------------------------------------------------------

def _rebuild(error: dict[str, Any], call_id: int) -> Exception:
    """The parent's exception as the nearest builtin (OSError keeps its errno)."""
    cls = getattr(builtins, str(error.get("type")), None)
    if not (isinstance(cls, type) and issubclass(cls, Exception)):
        cls = RuntimeError
    message, number = _text(error.get("message")), error.get("errno")
    try:
        if isinstance(number, int) and issubclass(cls, OSError):
            exc: Exception = OSError(number, message)  # ENOENT -> FileNotFoundError, ...
        else:
            exc = cls(message)
    except Exception:
        exc = RuntimeError(message)
    setattr(exc, CALL_ID, call_id)
    return exc


class _RemoteInner:
    """The child's TaskRuntime: every call is served by the parent (blocking, so the
    child's rewards reach the box one call at a time)."""

    def __init__(self, conn: Connection, *, name: Any, env: dict[str, str], config: Any,
                 max_payload_bytes: int):
        self.conn = conn
        self.name = name
        self.env = env
        self.config = config
        self.max_payload_bytes = max_payload_bytes
        self._last_id = 0
        self._out_of_step = False

    def _call(self, op: str, payload: bytes = b"", **args: Any) -> tuple[Any, bytes]:
        """One call served by the parent. Arguments the protocol cannot carry (a bytes
        path) and a payload past `max_payload_bytes` raise in the env's code (TypeError,
        OSError EFBIG): its failure, graded 0 if it lets them through. A reply the child
        cannot read, or a pipe that broke, is our protocol failing: the child exits with
        EXIT_PROTOCOL at once (the env's code cannot catch it), so the episode is
        `aborted`, never graded on it. A MemoryError or RecursionError raised during the
        exchange is the env's code's (a reply past its memory limit, a call at the edge of
        its recursion): raised to it; the pipe may then be out of step, so any later call
        exits with EXIT_PROTOCOL."""
        if self._out_of_step:
            os._exit(EXIT_PROTOCOL)
        if len(payload) > self.max_payload_bytes:
            raise OSError(errno.EFBIG, f"{op} of {len(payload)} bytes: the scoring process "
                                       f"sends at most {self.max_payload_bytes}")
        self._last_id += 1
        call_id = self._last_id
        message = _frame({"op": op, "id": call_id, "args": args}, payload)
        try:
            self.conn.send_bytes(message)
            del message
            header, data = _unframe(self.conn.recv_bytes())
            if "error" in header:
                error = header["error"]
                if not isinstance(error, dict):
                    raise ValueError("a malformed error")
                return_value: Any = None
            else:
                return_value = header["value"]
        except (MemoryError, RecursionError):
            self._out_of_step = True  # the env's, not the protocol's (see above)
            raise
        except Exception:
            os._exit(EXIT_PROTOCOL)
        if "error" in header:
            raise _rebuild(error, call_id)
        return return_value, data

    async def run(self, argv: list[str], env: dict[str, str]) -> Any:
        value, data = self._call("run", argv=list(argv), env=dict(env))
        try:  # the streams travel as raw UTF-8 after the header, their lengths in it
            out, err = value["stdout_bytes"], value["stderr_bytes"]
            if not (isinstance(value["exit_code"], int) and _int(out) and _int(err)
                    and out + err == len(data)):
                raise ValueError("a malformed run reply")
            stdout = data[:out].decode("utf-8", "surrogatepass")
            stderr = data[out:].decode("utf-8", "surrogatepass")
        except (MemoryError, RecursionError):
            raise
        except Exception:
            os._exit(EXIT_PROTOCOL)
        return SimpleNamespace(exit_code=value["exit_code"], stdout=stdout, stderr=stderr)

    async def read(self, path: str, max_bytes: int | None = None) -> bytes:
        return self._call("read", path=path, max_bytes=max_bytes)[1]

    async def write(self, path: str, data: bytes) -> None:
        self._call("write", bytes(data), path=path)

    async def exists(self, path: str) -> bool:
        return bool(self._call("exists", path=path)[0])

    async def alive(self) -> bool:
        return bool(self._call("alive")[0])

    async def prepare_setup(self) -> None:
        self._call("prepare_setup")

    async def prepare_execution(self, routes: list[str] | None) -> None:
        self._call("prepare_execution", routes=None if routes is None else list(routes))

    async def archive(self, paths: list[str], max_bytes: int) -> bytes:
        return self._call("archive", paths=list(paths), max_bytes=max_bytes)[1]

    async def restore_archive(self, data: bytes, roots: list[str], *,
                              contained_links: bool = True) -> None:
        self._call("restore_archive", bytes(data), roots=list(roots))  # always contained

    async def stop_processes(self) -> None:
        self._call("stop_processes")

    async def prepare_uv_script(self, *args: Any, **kwargs: Any) -> Any:
        self._call("prepare_uv_script")

    async def run_uv_script(self, *args: Any, **kwargs: Any) -> Any:
        self._call("run_uv_script")

    async def open_process(self, *args: Any, **kwargs: Any) -> Any:
        self._call("open_process")

    async def run_background(self, *args: Any, **kwargs: Any) -> Any:
        self._call("run_background")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _summary(trace: Any) -> dict[str, Any]:
    rewards = {_text(name): None if r is None else _number(r.score)
               for name, r in trace.rewards.items()}
    metrics = {_text(name): _number(value) for name, value in trace.metrics.items()}
    total = trace.reward
    return {"result": "scored", "rewards": rewards, "metrics": metrics,
            "reward": _number(total), "reward_repr": _text(repr(total))}


def _chain(exc: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    item: BaseException | None = exc
    while item is not None and not any(item is s for s in seen):
        seen.append(item)
        item = item.__cause__ or item.__context__
    return seen


async def _run_hook(task: Any, trace: Any, conn: Connection, view: dict[str, Any],
                    hook: str) -> dict[str, Any]:
    box = SandboxRuntime(_RemoteInner(conn, **view))  # the parent stops after each run
    try:
        if hook == "score":
            await task.score(trace, box)
            return _summary(trace)
        await invoke(getattr(task, hook), {"trace": trace, "runtime": box})
        return {"result": "done"}
    except BaseException as exc:  # SystemExit from the env's parser included
        chain = _chain(exc)
        if any(isinstance(item, MemoryError) for item in chain):
            raise MemoryError from None  # past the memory limit: no result
        calls = sorted({getattr(item, CALL_ID) for item in chain
                        if isinstance(getattr(item, CALL_ID, None), int)})
        return {"result": "raised", "type": type(exc).__name__, "message": _text(exc),
                "calls": calls}


def _close_inherited_fds(keep: int) -> None:
    """Close every fd but stdio and `keep` (the pipe). Objects of the forkserver that
    held one are pointed away from it, so nothing writes to a reused number later."""
    try:
        from multiprocessing import forkserver, resource_tracker

        forkserver._forkserver._forkserver_alive_fd = None
        forkserver._forkserver._inherited_fds = None
        resource_tracker._resource_tracker._fd = None  # a fresh tracker if one is needed
    except Exception:
        pass
    top = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    if top == resource.RLIM_INFINITY or top > 1 << 20:
        top = 1 << 20
    os.closerange(3, keep)
    os.closerange(keep + 1, max(keep + 1, top))


def _cpu_limit(seconds_left: float, spent: float, hard: int | None) -> tuple[int, int]:
    """RLIMIT_CPU (soft, hard): RLIMIT_CPU counts the CPU the process already spent
    (unpickling, imports), so the limit is that plus the seconds left (+1 hard),
    capped by an existing hard limit (None: none)."""
    soft = max(1, math.ceil(spent + max(0.0, seconds_left)))
    if hard is None:
        return soft, soft + 1
    return min(soft, hard), min(soft + 1, hard)


def _capped(res: Any, which: int, value: int) -> tuple[int, int]:
    """(value, value), never above an existing hard limit."""
    _, hard = res.getrlimit(which)
    if hard != res.RLIM_INFINITY:
        value = min(value, hard)
    return value, value


def _limit(deadline: float, memory_bytes: int, identity: ScoringIdentity, dedicated: bool,
           res: Any = resource) -> None:
    """CPU, address space and file size; processes only on a `dedicated` uid (the
    slot's: RLIMIT_NPROC counts every process of the real uid, so on the service's own
    user it would count the service's)."""
    usage = res.getrusage(res.RUSAGE_SELF)
    _, hard = res.getrlimit(res.RLIMIT_CPU)
    res.setrlimit(res.RLIMIT_CPU, _cpu_limit(
        deadline - time.monotonic(), usage.ru_utime + usage.ru_stime,
        None if hard == res.RLIM_INFINITY else hard))
    with open("/proc/self/statm") as statm:
        mapped = int(statm.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    res.setrlimit(res.RLIMIT_AS, _capped(res, res.RLIMIT_AS, mapped + memory_bytes))
    if identity.file_bytes is not None:
        res.setrlimit(res.RLIMIT_FSIZE, _capped(res, res.RLIMIT_FSIZE, identity.file_bytes))
    if dedicated:
        res.setrlimit(res.RLIMIT_NPROC, _capped(res, res.RLIMIT_NPROC, identity.nproc))


def _scrub_environment(tmpdir: str) -> None:
    """os.environ (and the C environment) becomes PATH, LANG, HOME=/nonexistent,
    TMPDIR and what the wrapper set (PYTHONHASHSEED, the thread counts); the process's initial environment block (what /proc/self/environ shows,
    the service's secrets included) is zeroed through /proc/self/mem."""
    kept = {"PATH": os.environ.get("PATH") or DEFAULT_PATH, "HOME": "/nonexistent",
            "TMPDIR": tmpdir}
    for name in ("LANG", "PYTHONHASHSEED", *THREAD_VARS):  # what the wrapper set
        if name in os.environ:
            kept[name] = os.environ[name]
    os.environ.clear()  # unsetenv for each
    os.environ.update(kept)
    tempfile.tempdir = None
    with open("/proc/self/stat", "rb") as stat:
        fields = stat.read().rsplit(b")", 1)[1].split()
    start, end = int(fields[47]), int(fields[48])  # env_start, env_end (fields 50, 51)
    if 0 < start < end:
        with open("/proc/self/mem", "r+b", buffering=0) as memory:
            memory.seek(start)
            memory.write(bytes(end - start))


def _drop_privileges(uid: int, gid: int, ops: Any = os) -> None:
    """As root: no supplementary group, then `gid`, then `uid` (real, effective and
    saved), checked afterwards (PermissionError if any id is still another). Not root:
    nothing to change (the service already runs unprivileged: tests, a user install)."""
    if ops.geteuid() != 0:
        return
    ops.setgroups([])
    ops.setgid(gid)
    ops.setuid(uid)
    if (ops.getuid(), ops.geteuid(), ops.getgid(), ops.getegid(), ops.getgroups()) != (
            uid, uid, gid, gid, []):
        raise PermissionError(f"the scoring process could not drop to {uid}:{gid}")


def _no_new_privs() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")


def _network_filter(machine: str) -> list[tuple[int, int, int, int]]:
    """The seccomp BPF program (code, jt, jf, k) of a scoring process: a syscall of
    another audit arch (or x32) kills it, io_uring_setup, ptrace, process_vm_readv/writev,
    pidfd_getfd and `socket` of any family but AF_UNIX fail with EPERM, everything else is allowed. OSError for an architecture it
    does not know (fail closed: no scoring without the filter)."""
    if machine not in SECCOMP_ARCHES:
        raise OSError(errno.ENOSYS, f"no network filter for the architecture {machine!r}: "
                                    "the scoring process refuses to score")
    arch, nr_socket, nr_io_uring_setup, x32 = SECCOMP_ARCHES[machine]
    ld, jeq, jge, ret = 0x20, 0x15, 0x35, 0x06  # BPF_LD|W|ABS, JMP|JEQ|K, JMP|JGE|K, RET|K
    # seccomp_data: nr at 0, arch at 4, args[0] at 16 (its low 32 bits: little endian;
    # the kernel reads the family as an int)
    body: list[tuple[int, Any, Any, int]] = [(ld, 0, 0, 4), (jeq, 0, "kill", arch),
                                             (ld, 0, 0, 0)]
    if x32:
        body.append((jge, "kill", 0, X32_SYSCALL_BIT))
    body += [(jeq, "deny", 0, nr) for nr in (nr_io_uring_setup, *DENIED_SYSCALLS[machine])]
    body += [(jeq, 0, "allow", nr_socket),
             (ld, 0, 0, 16), (jeq, "allow", "deny", socket.AF_UNIX)]
    labels = {"allow": len(body), "deny": len(body) + 1, "kill": len(body) + 2}
    tail = [(ret, 0, 0, SECCOMP_RET_ALLOW), (ret, 0, 0, SECCOMP_RET_ERRNO | errno.EPERM),
            (ret, 0, 0, SECCOMP_RET_KILL_PROCESS)]

    def offset(pc: int, target: Any) -> int:
        return target if isinstance(target, int) else labels[target] - pc - 1

    return [(code, offset(pc, jt), offset(pc, jf), k)
            for pc, (code, jt, jf, k) in enumerate(body)] + tail


class _SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte),
                ("k", ctypes.c_uint32)]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_SockFilter))]


def _deny_network() -> None:
    """Install `_network_filter(MACHINE)` on this process (after no_new_privs, which an
    unprivileged filter requires; inherited by every thread and process it starts) and
    check it took: a socket of AF_INET must then fail with EPERM. Refused (OSError)
    unless the process has a single thread (a prctl filter covers the calling thread
    only), runs a 64-bit interpreter, and the kernel accepts the filter."""
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        raise OSError(errno.ENOSYS, "the network filter needs a 64-bit interpreter")
    program = _network_filter(MACHINE)
    if len(os.listdir("/proc/self/task")) != 1:
        raise OSError(errno.EBUSY, "the network filter is installed before any thread")
    filters = (_SockFilter * len(program))(*(_SockFilter(*item) for item in program))
    fprog = _SockFprog(len(program), filters)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(ctypes.c_int(PR_SET_SECCOMP), ctypes.c_ulong(SECCOMP_MODE_FILTER),
                  ctypes.byref(fprog), ctypes.c_ulong(0), ctypes.c_ulong(0)) != 0:
        code = ctypes.get_errno()
        raise OSError(code, f"prctl(PR_SET_SECCOMP) failed: {os.strerror(code)}")
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except PermissionError:
        return
    probe.close()
    raise OSError(errno.EPERM, "the network filter did not take: AF_INET still opens")


def _take_identity(identity: ScoringIdentity, tmpdir: str, ops: Any = os) -> bool:
    """As root: the private TMPDIR to the slot's uid, then the drop to it (True: the
    process now runs on its own uid). Not root: nothing (False)."""
    if ops.geteuid() != 0:
        return False
    ops.chown(tmpdir, identity.uid, identity.gid)
    _drop_privileges(identity.uid, identity.gid, ops)
    return True


def _prefer_for_oom_kill(path: str = OOM_SCORE_ADJ) -> None:
    """oom_score_adj 1000: under host memory pressure the kernel kills a scoring process
    before the service. Before the drop (after it the process is not dumpable and its
    /proc files are root's). Best effort: a failure is logged."""
    try:
        with open(path, "w") as score:
            score.write("1000")
    except OSError as exc:
        LOG.warning("the scoring process could not set its oom_score_adj: %s", exc)


def _unprivileged(identity: ScoringIdentity, tmpdir: str) -> bool:
    """Before anything of the env runs: the environment, the OOM score, the private
    TMPDIR's owner, the uid, no_new_privs and the network filter (no socket but AF_UNIX:
    its pipe to the parent is already open). Whether it runs on its slot's own uid.
    Any failure, the filter's included, is before `ready`: ours, never scored."""
    _scrub_environment(tmpdir)
    _prefer_for_oom_kill()
    dedicated = _take_identity(identity, tmpdir)
    _no_new_privs()
    try:
        _deny_network()
    except BaseException as exc:
        LOG.error("the scoring process refuses to score without its network filter: %s", exc)
        raise
    return dedicated


def _importing(frame: Any) -> bool:
    """Whether importlib is running below `frame` (an import of the env's files)."""
    while frame is not None:
        if frame.f_code.co_filename in IMPORT_FRAMES:
            return True
        frame = frame.f_back
    return False


def _audit_guard(event: str, args: tuple[Any, ...]) -> None:
    """The scoring process's audit hook (see the module docstring): PermissionError for a
    refused event, remembered in _REFUSED."""
    if event not in REFUSED_EVENTS and not event.startswith(REFUSED_PREFIX):
        return
    caller = sys._getframe(1)
    if event == "compile" and caller.f_code.co_filename == _AST_FILE \
            and caller.f_code.co_name == "parse":
        return  # ast.parse: a tree, never code (compiling the tree again is refused)
    if (event in ("compile", "exec") and caller.f_code.co_filename == _TYPING_FILE
            and caller.f_code.co_name in TYPING_CODE):
        return  # an annotation of the env's source, never the agent's bytes
    if (event in IMPORT_EVENTS or event.startswith(REFUSED_PREFIX)) and _importing(caller):
        return
    _REFUSED.append(event)
    raise PermissionError(f"the scoring process refuses {event!r}: no code or process "
                          "from what the agent wrote")


def _install_audit_guard() -> None:
    """The audit hook, for the rest of this process (it cannot be removed)."""
    sys.addaudithook(_audit_guard)


def _refused_result(result: dict[str, Any]) -> dict[str, Any]:
    """`raised` when the audit hook refused anything, whatever the hook returned."""
    if not _REFUSED:
        return result
    return {"result": "raised", "type": "PermissionError",
            "message": _text(f"the scoring process refused {_REFUSED[0]!r}"), "calls": []}


def _child_main(conn: Connection, task_bytes: bytes, make_trace: Callable[[Any], Any],
                view: dict[str, Any], deadline: float, memory_bytes: int,
                identity: ScoringIdentity, tmpdir: str, hook: str) -> None:
    """Never returns: the child leaves with os._exit (no atexit, no inherited cleanup).
    Before `ready` every failure is ours (EXIT_OURS). After it, a failure of our own
    send of the result exits EXIT_OURS (the parent grades it `died`: it cannot tell it
    from the env's code exiting so), a MemoryError is the scoring's (EXIT_MEMORY) and
    anything else escaping the scoring's run is the scoring's exit (a `raised` result:
    graded 0; so is any event the audit hook refused). It drops its privileges before
    it imports the env's module (the task's unpickling): the env's modules and task
    files must be readable by `identity` (deployment)."""
    try:
        _close_inherited_fds(conn.fileno())
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)  # past RLIMIT_FSIZE: EFBIG, no kill
        dedicated = _unprivileged(identity, tmpdir)
        task = pickle.loads(task_bytes)
        trace = make_trace(task)
        _limit(deadline, memory_bytes, identity, dedicated)
        _install_audit_guard()  # the last of our setup: nothing of the agent's ran yet
        conn.send_bytes(_frame({"ready": True}))
    except BaseException:
        os._exit(EXIT_OURS)
    try:
        result = asyncio.run(_run_hook(task, trace, conn, view, hook))
    except MemoryError:
        os._exit(EXIT_MEMORY)
    except BaseException as exc:
        # The env's code in a task of its own (verifiers gathers the rewards):
        # SystemExit, KeyboardInterrupt, or the loop stopped under `_run_hook`.
        try:
            result = {"result": "raised", "type": type(exc).__name__,
                      "message": _text(exc), "calls": []}
        except BaseException:
            result = {"result": "raised", "type": "BaseException", "message": "", "calls": []}
    try:
        conn.send_bytes(_frame(_refused_result(result)))
    except MemoryError:
        os._exit(EXIT_MEMORY)
    except BaseException:
        os._exit(EXIT_OURS)
    os._exit(0)


# -- the parent -----------------------------------------------------------------------

class _Child:
    """The scoring process: killed by the watchdog or at the end, joined exactly once."""

    def __init__(self, process: Any):
        self.process = process
        self._lock = threading.Lock()
        self._joined = False
        self.killed_at_deadline = False
        self.exitcode: int | None = None

    def kill_at_deadline(self) -> None:
        with self._lock:
            if not self._joined and self.process.exitcode is None:
                self.killed_at_deadline = True
                self.process.kill()

    def reap(self) -> int | None:
        """SIGKILL if still running, join, release; the exit code."""
        with self._lock:
            if not self._joined:
                if self.process.exitcode is None:
                    self.process.kill()
                self.process.join()
                self._joined = True
                self.exitcode = self.process.exitcode
                self.process.close()
            return self.exitcode


def _str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _str_dict(value: Any) -> bool:
    return isinstance(value, dict) and all(isinstance(k, str) and isinstance(v, str)
                                           for k, v in value.items())


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


ARGS: dict[str, dict[str, Callable[[Any], bool]]] = {
    "run": {"argv": _str_list, "env": _str_dict},
    "read": {"path": lambda v: isinstance(v, str), "max_bytes": lambda v: v is None or _int(v)},
    "write": {"path": lambda v: isinstance(v, str)},
    "exists": {"path": lambda v: isinstance(v, str)},
    "alive": {},
    "prepare_setup": {},
    "prepare_execution": {"routes": lambda v: v is None or _str_list(v)},
    "archive": {"paths": _str_list, "max_bytes": _int},
    "restore_archive": {"roots": _str_list},
    "stop_processes": {},
    "prepare_uv_script": {},
    "run_uv_script": {},
    "open_process": {},
    "run_background": {},
}
"""Every operation the child may ask for, with a check per argument (all required)."""


async def _dispatch(box: SandboxRuntime, op: str, args: dict[str, Any],
                    payload: bytes) -> tuple[Any, bytes]:
    if op == "run":
        result = await box.run(args["argv"], args["env"])
        out = str(result.stdout).encode("utf-8", "surrogatepass")
        err = str(result.stderr).encode("utf-8", "surrogatepass")
        return {"exit_code": result.exit_code, "stdout_bytes": len(out),
                "stderr_bytes": len(err)}, out + err
    if op == "read":
        return None, bytes(await box.read(args["path"], args["max_bytes"]))
    if op == "write":
        await box.write(args["path"], payload)
    elif op == "exists":
        return await box.exists(args["path"]), b""
    elif op == "alive":
        return await box.alive(), b""
    elif op == "prepare_setup":
        await box.prepare_setup()
    elif op == "prepare_execution":
        await box.prepare_execution(args["routes"])
    elif op == "archive":
        return None, bytes(await box.archive(args["paths"], args["max_bytes"]))
    elif op == "restore_archive":
        await box.restore_archive(payload, args["roots"])
    elif op == "stop_processes":
        await box.stop_processes()
    elif op == "prepare_uv_script":
        await box.prepare_uv_script(None)
    elif op == "run_uv_script":
        await box.run_uv_script(None)
    elif op == "open_process":
        await box.open_process([], {})
    else:  # run_background
        await box.run_background([], {}, "")
    return None, b""


def _describe(exc: BaseException) -> dict[str, Any]:
    base = next((c for c in type(exc).__mro__ if c.__module__ == "builtins"
                 and issubclass(c, Exception)), Exception)
    number = getattr(exc, "errno", None) if isinstance(exc, OSError) else None
    return {"type": base.__name__, "errno": number if isinstance(number, int) else None,
            "message": _text(exc)}


def _valid_summary(header: dict[str, Any]) -> bool:
    def numbers(table: Any, allow_none: bool) -> bool:
        return isinstance(table, dict) and all(
            isinstance(v, float) or (allow_none and v is None) for v in table.values())

    return (numbers(header.get("rewards"), True) and numbers(header.get("metrics"), True)
            and (header.get("reward") is None or isinstance(header.get("reward"), float))
            and isinstance(header.get("reward_repr"), str))


def _finished(header: dict[str, Any], errors: dict[int, BaseException],
              expected: str) -> ScoringOutcome:
    """The child's result. `expected` is the kind its hook returns (`scored` for score,
    `done` for the other hooks): the other kind is a broken result."""
    if expected == "done" and header.get("result") == "done" and set(header) == {"result"}:
        return ScoringOutcome("done")
    if expected == "scored" and header.get("result") == "scored" and _valid_summary(header):
        return ScoringOutcome("scored", summary=header)
    calls = header.get("calls")
    if header.get("result") == "raised" and isinstance(calls, list) and all(
            _int(call) for call in calls):
        return ScoringOutcome("raised", call_errors=[errors[c] for c in calls if c in errors],
                              detail=f"{_text(header.get('type'))}: {_text(header.get('message'))}")
    return ScoringOutcome("broken", detail="the scoring process sent a malformed result")


async def _serve(conn: Connection, box: SandboxRuntime, max_message_bytes: int,
                 errors: dict[int, BaseException], state: dict[str, bool],
                 ready_by: float, expected: str) -> ScoringOutcome:
    """Serve the child's calls until its result. `state["ready"]` is set once the child
    said `ready`; before that it runs only our code, so anything else it sends is ours,
    and so is a child not ready by `ready_by` (monotonic; before the step deadline)."""
    loop = asyncio.get_running_loop()
    readable = asyncio.Event()
    loop.add_reader(conn.fileno(), readable.set)
    try:
        while True:
            readable.clear()
            if not conn.poll(0):
                if state["ready"]:
                    await readable.wait()
                else:
                    try:
                        await asyncio.wait_for(readable.wait(), ready_by - time.monotonic())
                    except TimeoutError:
                        return ScoringOutcome("ours", detail="the scoring process was killed "
                                                             "before it was ready (not ready "
                                                             "within its ready limit)")
                    continue
            try:
                message = conn.recv_bytes(max_message_bytes)
            except (EOFError, OSError) as exc:  # killed, crashed, or an oversized message
                return ScoringOutcome("died", detail=_text(exc) or type(exc).__name__)
            try:
                header, payload = _unframe(message)
            except ValueError:
                header, payload = None, b""
            if not state["ready"]:
                if header != {"ready": True}:
                    return ScoringOutcome("ours", detail="the scoring process sent something "
                                                         "else than ready")
                state["ready"] = True
                continue
            if header is None:
                return ScoringOutcome("broken", detail="the scoring process sent a malformed call")
            if "result" in header:
                return _finished(header, errors, expected)
            op, call_id, args = header.get("op"), header.get("id"), header.get("args")
            checks = ARGS.get(op) if isinstance(op, str) else None
            if (checks is None or not _int(call_id) or call_id in errors
                    or not isinstance(args, dict) or set(args) != set(checks)
                    or not all(check(args[name]) for name, check in checks.items())):
                return ScoringOutcome("broken", detail=f"the scoring process sent a bad call "
                                                       f"({_text(op)})")
            try:
                value, data = await _dispatch(box, op, args, payload)
                reply = _reply({"value": value}, data)
            except Exception as exc:  # handed to the env's code, kept for the verdict
                errors[call_id] = exc
                reply = _reply({"error": _describe(exc)})
            problem = _reply_problem(op, reply)
            if problem is not None:  # the child would exit EXIT_PROTOCOL: ours, not its
                return ScoringOutcome("ours", detail=f"our reply to {op} is one the scoring "
                                                     f"process cannot read ({problem})")
            try:
                conn.send_bytes(reply)
            except OSError as exc:
                return ScoringOutcome("died", detail=_text(exc))
    finally:
        loop.remove_reader(conn.fileno())


def _reply_problem(op: str, reply: bytes) -> str | None:
    """Why the child could not read `reply` to `op` (`_RemoteInner._call` and `run`
    exit EXIT_PROTOCOL on it), or None. The parent checks its own replies so that a
    protocol failure of ours is never the child's exit code to report."""
    try:
        header, data = _unframe(reply)
    except ValueError as exc:
        return _text(exc)
    if "error" in header:
        return None if isinstance(header["error"], dict) else "a malformed error"
    if "value" not in header:
        return "no value"
    value = header["value"]
    if op == "run" and not (
            isinstance(value, dict) and isinstance(value.get("exit_code"), int)
            and _int(value.get("stdout_bytes")) and _int(value.get("stderr_bytes"))
            and value["stdout_bytes"] + value["stderr_bytes"] == len(data)):
        return "a malformed run reply"
    return None


def _reply(header: dict[str, Any], payload: bytes = b"") -> bytes:
    """A reply to the child (one function, so a test can break our side of the protocol)."""
    return _frame(header, payload)


def _blame(outcome: ScoringOutcome, child: _Child, ready: bool,
           server_alive: bool = True) -> ScoringOutcome:
    """A child that ended without a result: the scoring's (`died`) or ours (`ours`).
    Ours: the forkserver it came from died (a host OOM kill took both; the agent's code
    runs unprivileged and cannot signal it), or it ended before `ready` whatever the
    cause (it ran only our code and the env's import; it started with at least
    SCORING_STARTUP_FLOOR_S left). After `ready` every end without a result is the
    scoring's, EXIT_OURS and EXIT_PROTOCOL included: code the agent's bytes steer can
    exit with them, and a protocol failure of ours is detected by the parent itself
    (`_serve` returns `ours` before it sends a reply the child could not read)."""
    if outcome.kind != "died":
        return outcome
    code = child.exitcode
    if not server_alive:
        return ScoringOutcome("ours", detail=f"the forkserver died while it scored (exit code "
                                             f"{code})")
    if not ready:
        return ScoringOutcome("ours", detail=f"the scoring process ended before it was ready "
                                             f"(exit code {code}, killed at the deadline: "
                                             f"{child.killed_at_deadline}: an import or our "
                                             "setup failed or was too slow, or the host "
                                             "killed it)")
    if child.killed_at_deadline or code in LIMIT_SIGNALS or code == EXIT_MEMORY:
        return ScoringOutcome("died", detail=f"{outcome.detail}; exit code {code}, killed at "
                                             f"the deadline: {child.killed_at_deadline}")
    return ScoringOutcome("died", detail=f"{outcome.detail}; exit code {code}")


def _remove_tree(path: str, rmtree: Callable[..., Any] = shutil.rmtree) -> None:
    """rmtree, as the owner of the tree: an entry it cannot list or remove (the env's
    code made a directory `chmod 000` or read-only) has its parent, and itself when it is
    a directory, given back 0700, and is retried once; a second failure raises (the
    sweep failed: the slot stays taken)."""
    retried: set[str] = set()

    def unlock(func: Callable[..., Any], failed: str, exc: BaseException) -> None:
        if isinstance(exc, FileNotFoundError):
            return
        if failed in retried:
            raise exc
        retried.add(failed)
        is_dir = os.path.isdir(failed) and not os.path.islink(failed)
        for target in (os.path.dirname(failed), failed) if is_dir else (os.path.dirname(failed),):
            try:
                os.chmod(target, 0o700)
            except OSError:
                pass  # not ours (the scoring root's parent): the retry says
        if is_dir:
            remove(failed)
        else:
            func(failed)

    def remove(target: str) -> None:
        if sys.version_info >= (3, 12):
            rmtree(target, onexc=unlock)
        else:  # pragma: no cover - Python 3.11
            rmtree(target, onerror=lambda func, failed, info: unlock(func, failed, info[1]))

    remove(path)


def _empty_dir(path: str, remove_tree: Callable[[str], Any] = _remove_tree) -> None:
    """Remove everything inside `path` (the slot's TMPDIR), keeping `path` itself: as the
    slot's uid the sweeper cannot unlink an entry of the scoring root (/scoring is root's,
    0755, not sticky); the controller removes the empty TMPDIR after the sweep."""
    os.chmod(path, 0o700)  # ours: the env's code may have locked it
    for name in os.listdir(path):
        entry = os.path.join(path, name)
        if os.path.isdir(entry) and not os.path.islink(entry):
            remove_tree(entry)
        else:
            try:
                os.unlink(entry)
            except FileNotFoundError:
                pass


def _sweep_shm(uid: int, root: str = SHM_ROOT,
               lstat: Callable[[str], os.stat_result] = os.lstat) -> None:
    """Remove the entries of `root` (/dev/shm: POSIX shared memory, semaphores) that
    `uid` owns. Run as that uid (the directory is sticky: only an owner removes)."""
    try:
        names = os.listdir(root)
    except FileNotFoundError:
        return
    for name in names:
        entry = os.path.join(root, name)
        try:
            info = lstat(entry)
        except FileNotFoundError:
            continue
        if info.st_uid != uid:
            continue
        if stat.S_ISDIR(info.st_mode):
            _remove_shm_dir(entry, uid, lstat)
        else:
            try:
                os.unlink(entry)
            except FileNotFoundError:
                pass


def _remove_shm_dir(path: str, uid: int, lstat: Callable[[str], os.stat_result]) -> None:
    """Remove a directory of the slot's uid from /dev/shm. Anything inside it that the
    uid does not own (another slot's entry, nested there) is logged and skipped, never an
    error: only a failure on the slot's own entries keeps the slot taken."""
    foreign: list[str] = []

    def walk(directory: str) -> None:
        os.chmod(directory, 0o700)  # ours: the env may have locked it
        for name in os.listdir(directory):
            child = os.path.join(directory, name)
            try:
                info = lstat(child)
            except FileNotFoundError:
                continue
            if info.st_uid != uid:
                foreign.append(child)
            elif stat.S_ISDIR(info.st_mode):
                walk(child)
            else:
                os.unlink(child)
        try:
            os.rmdir(directory)
        except FileNotFoundError:
            pass
        except OSError:
            if not any(f.startswith(directory + os.sep) for f in foreign):
                raise

    walk(path)
    if foreign:
        LOG.warning("shm sweep: skipped %d entries not owned by uid %d: %s",
                    len(foreign), uid, foreign[:5])


def _sweep_as_slot(path: str, identity: ScoringIdentity, ops: Any = os,
                   rmtree: Callable[[str], Any] = _empty_dir,
                   sleep: Callable[[float], Any] = time.sleep,
                   sweep_shm: Callable[[int], Any] = _sweep_shm) -> None:
    """In the sweeper process (root): drop to the slot's uid, kill every process of that
    uid (`kill(-1)`: all the caller may signal, i.e. its own uid's), then empty the
    TMPDIR (the controller removes the directory itself: `_run_sweeper`) and remove the
    uid's entries in /dev/shm (both tried; the first failure raises).
    PermissionError, nothing signalled, unless it runs as root and lands on the slot's
    own uid: `kill(-1)` as any other user (the service's, a test's) would hit that
    user's processes."""
    uid = identity.uid
    if ops.geteuid() != 0 or not isinstance(uid, int) or uid <= 0:
        raise PermissionError("the sweeper signals only as a dedicated slot uid, from root")
    _drop_privileges(uid, identity.gid, ops)  # verified: real, effective and saved ids
    for _ in range(KILL_ROUNDS):
        try:
            ops.kill(-1, signal.SIGKILL)
        except ProcessLookupError:
            break  # none left
        sleep(0.02)
    failures: list[BaseException] = []
    for step in (lambda: rmtree(path), lambda: sweep_shm(uid)):
        try:
            step()
        except Exception as exc:
            failures.append(exc)
    if failures:
        raise failures[0]


def _sweep(path: str, identity: ScoringIdentity) -> None:
    """The sweeper process's target (from the forkserver): exit 0 once swept."""
    code = 1
    try:
        _sweep_as_slot(path, identity)
        code = 0
    finally:
        os._exit(code)


def _run_sweeper(path: str, identity: ScoringIdentity) -> bool:
    """Whether the slot is clean: as root, a sweeper process (see `_sweep_as_slot`; the
    service's root may also lack CAP_DAC_OVERRIDE to enter what the scoring process
    made) that ended in time and left no TMPDIR; otherwise the TMPDIR removed (the
    service's own user cannot be swept by uid: tests, a user install)."""
    from reliquary_sandbox_service import scoring_server

    try:
        if os.geteuid() == 0:
            sweeper = scoring_server.CONTEXT.Process(target=_sweep, args=(path, identity),
                                                     name="reliquary-scoring-sweep",
                                                     daemon=True)
            sweeper.start()
            sweeper.join(SWEEP_TIMEOUT_S)
            if sweeper.exitcode is None:
                sweeper.kill()
                sweeper.join()
            swept = sweeper.exitcode == 0
            sweeper.close()
            if swept and os.path.lexists(path):
                os.rmdir(path)  # emptied by the sweeper; an entry of the root's directory
        else:
            shutil.rmtree(path, ignore_errors=True)
            swept = True
        return swept and not os.path.lexists(path)
    except Exception:
        LOG.warning("the sweep of the scoring TMPDIR %s failed", path, exc_info=True)
        return False


_PENDING: list[tuple[str, ScoringIdentity, Any, int]] = []
"""Sweeps that failed (TMPDIR, identity, pool, slot): their slots stay taken."""
_PENDING_LOCK = threading.Lock()


def _sweep_slot(path: str, identity: ScoringIdentity, pool: Any, slot: int,
                sweeper: Callable[[str, ScoringIdentity], bool] = _run_sweeper) -> None:
    """After the reap: the oldest sweep that failed before is retried (one at most: a
    grade's overrun stays bounded at two sweeps), then this one; each slot is released
    only once its sweep succeeded (never handed to another episode with a process of its
    uid alive). Logged; never changes the outcome."""
    with _PENDING_LOCK:
        work = [_PENDING.pop(0)] if _PENDING else []
    work.append((path, identity, pool, slot))
    for item in work:
        if sweeper(item[0], item[1]):
            item[2].release(item[3])
            continue
        LOG.warning("the scoring slot uid %s (TMPDIR %s) could not be swept: it stays in "
                    "use and is retried at the next sweep", item[1].uid, item[0])
        with _PENDING_LOCK:
            _PENDING.append(item)


async def _take_slot(pool: Any, deadline: float) -> int | None:
    """A free slot; None when none freed up while at least SCORING_STARTUP_FLOOR_S was
    left (more episodes score at once than there are slots: the deployment's)."""
    while True:
        slot = pool.acquire()
        if slot is not None:
            return slot
        if deadline - time.monotonic() <= SCORING_STARTUP_FLOOR_S + 0.05:
            return None
        await asyncio.sleep(0.05)


async def run_in_child(task: Any, box: SandboxRuntime, *, hook: str,
                       make_trace: Callable[[Any], Any], deadline: float, memory_bytes: int,
                       max_message_bytes: int, max_payload_bytes: int | None = None,
                       identity: ScoringIdentity = ScoringIdentity(),
                       ready_s: float = DEFAULT_READY_S) -> ScoringOutcome:
    """Run `hook` of `task` (one of HOOKS: `task.score(make_trace(task), runtime)`, or
    `setup` / `grading_setup` / `finalize` through verifiers' `invoke`) in a scoring
    process started from the service's forkserver, its runtime calls served on `box`.
    ValueError for a hook not in HOOKS, before any process starts. `deadline` is on
    `time.monotonic()` (the same clock in every process). `max_message_bytes` bounds a
    message of the child without a payload; `max_payload_bytes` (default the same) the
    bytes of a `write` or `restore_archive` it sends. With less than
    SCORING_STARTUP_FLOOR_S left no process is started (`died`: grading_timeout); one
    that started must be ready within `ready_s` (and READY_MARGIN_S before the
    deadline), else it is ours. It runs on a slot of scoring_server.SLOTS (its uid),
    released after the reap and the sweep."""
    if hook not in HOOKS:
        raise ValueError(f"unknown hook {hook!r}: one of {HOOKS}")
    from reliquary_sandbox_service import scoring_server

    left = deadline - time.monotonic()
    if left < SCORING_STARTUP_FLOOR_S:
        return ScoringOutcome("died", detail=f"{max(0.0, left):.1f} s left before the step "
                                             f"deadline, less than the "
                                             f"{SCORING_STARTUP_FLOOR_S:g} s a scoring process "
                                             "needs to start: the restore used the time")
    payload_cap = max_message_bytes if max_payload_bytes is None else max_payload_bytes
    receive_cap = max(max_message_bytes, payload_cap + MAX_HEADER_BYTES + 4)
    view = {"name": box.name, "env": dict(box.env),
            "config": SimpleNamespace(type=box.config.type, workdir=box.config.workdir),
            "max_payload_bytes": payload_cap}
    try:
        task_bytes = pickle.dumps(task)
    except Exception as exc:
        return ScoringOutcome("ours", detail=f"the task cannot be pickled: {_text(exc)}")
    pool = scoring_server.SLOTS
    slot = await _take_slot(pool, deadline)
    if slot is None:
        return ScoringOutcome("ours", detail=f"no free scoring slot (all {pool.count} in use: "
                                             "more episodes grade at once than there are "
                                             "slots)")
    uid = pool.uid(slot)
    identity = replace(identity, uid=uid, gid=uid,
                       tmp_root=identity.tmp_root or scoring_server.TMP_ROOT)
    try:
        tmpdir = tempfile.mkdtemp(prefix="reliquary-scoring-", dir=identity.tmp_root)
    except OSError as exc:
        pool.release(slot)
        return ScoringOutcome("ours", detail=f"no private TMPDIR: {_text(exc)}")
    if os.geteuid() == 0:
        # The slot's from the start: a scoring process that never reached its own chown
        # (start failed, killed before ready) still leaves a TMPDIR its sweeper can empty.
        try:
            os.chown(tmpdir, uid, uid)
        except OSError as exc:
            os.rmdir(tmpdir)
            pool.release(slot)
            return ScoringOutcome("ours", detail=f"no private TMPDIR: {_text(exc)}")
    try:
        parent_conn, child_conn = scoring_server.CONTEXT.Pipe(duplex=True)
    except Exception as exc:  # nothing ran as the slot's uid: the slot and TMPDIR go back
        try:
            os.rmdir(tmpdir)
        except OSError:
            LOG.warning("the scoring TMPDIR %s could not be removed", tmpdir, exc_info=True)
        pool.release(slot)
        return ScoringOutcome("ours", detail=f"no pipe to a scoring process: "
                                             f"{type(exc).__name__}: {_text(exc)}")
    try:
        process = scoring_server.CONTEXT.Process(
            target=_child_main, name="reliquary-scoring", daemon=True,
            args=(child_conn, task_bytes, make_trace, view, deadline, memory_bytes,
                  identity, tmpdir, hook))
        scoring_server.start()
        process.start()
        server = scoring_server.server_pid()
    except Exception as exc:
        parent_conn.close()
        _sweep_slot(tmpdir, identity, pool, slot)
        return ScoringOutcome("ours", detail=f"the scoring process could not start: "
                                             f"{type(exc).__name__}: {_text(exc)}")
    finally:
        child_conn.close()
    child = _Child(process)
    # Not ready within its limit (an import too slow, a host stalling it): ours, decided
    # before the step deadline, so never mistaken for the scoring's timeout.
    started = time.monotonic()
    ready_by = min(deadline if SCORING_STARTUP_FLOOR_S <= 0 else deadline - READY_MARGIN_S,
                   started + ready_s)
    watchdog = threading.Timer(max(0.0, deadline - time.monotonic()), child.kill_at_deadline)
    watchdog.daemon = True
    watchdog.start()
    errors: dict[int, BaseException] = {}
    state = {"ready": False}
    outcome = ScoringOutcome("died", detail="cancelled")
    server_alive = True
    try:
        outcome = await _serve(parent_conn, box, receive_cap, errors, state, ready_by,
                               "scored" if hook == "score" else "done")
    finally:
        watchdog.cancel()
        child.reap()
        parent_conn.close()
        server_alive = scoring_server.alive(server)  # before the sweeper restarts it
        _sweep_slot(tmpdir, identity, pool, slot)  # kills what the env left, then releases
    return _blame(outcome, child, state["ready"], server_alive)


async def score_in_child(task: Any, box: SandboxRuntime, **kwargs: Any) -> ScoringOutcome:
    """`run_in_child(task, box, hook="score", **kwargs)`: the reward step."""
    return await run_in_child(task, box, hook="score", **kwargs)
