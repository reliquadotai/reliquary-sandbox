"""The forkserver the verifiers bridge starts its scoring processes from.

An `os.fork` of the service (uvicorn, the episode pools and Docker clients run threads)
copies into the child every lock another thread holds at that instant, never to be
released there (an honest episode would hang until its deadline), and every open fd,
the Docker socket included. Scoring processes are forked by a multiprocessing
forkserver instead: a separate single-threaded process, started with exec, that holds
only the fds multiprocessing hands it. `start()` runs it at service start, before any
thread, with the bridge's scoring module and the configured env modules preloaded, so
a scoring process imports nothing heavy. The forkserver skips a preload that fails
(an ImportError there is ignored): `check_preload()` imports the same modules the same
way (the wrapper, its empty environment, the service's sys.path) at the service's start,
which is refused when one fails (an env must read no credential at import time); it is idempotent and also called before
each scoring process, so a forkserver that died is started again (fork+exec: safe from
a threaded process). It lives as long as the service.

The forkserver is executed through WRAPPER (`scripts/scoring-python`, installed in the
image), which starts Python with an empty environment: no secret of the service is ever
in its memory, so none is in a scoring process forked from it. A missing wrapper fails
the start (every grade `aborted`), never a forkserver with the service's environment.

Each scoring process runs as the uid of a slot (SLOTS: `base + slot`, one per
concurrent scoring process), when the service runs as root. A service that does not
run as root cannot give them their own uid: it refuses to start the forkserver (every
verifiers grade `aborted`) unless RELIQUARY_SCORING_UNPRIVILEGED_OK=1 (tests).
"""

from __future__ import annotations

import collections
import json
import multiprocessing
import os
import subprocess
import sys
import threading
from collections.abc import Iterable
from multiprocessing import forkserver

__all__ = ["CONTEXT", "PRELOAD", "SLOTS", "ScoringSlots", "alive", "check_preload", "configure",
           "server_pid", "start", "stop"]

PRELOAD = ("reliquary_sandbox_verifiers.scoring_process",)
EXTRA_PRELOAD: list[str] = []
"""The env modules preloaded besides PRELOAD (`start(preload=...)`; kept for restarts)."""
CONTEXT = multiprocessing.get_context("forkserver")
DEFAULT_WRAPPER = "/usr/local/libexec/reliquary-scoring-python"
WRAPPER = os.environ.get("RELIQUARY_SANDBOX_SCORING_WRAPPER") or DEFAULT_WRAPPER
"""The executable the forkserver starts through (Settings.scoring_wrapper)."""
UNPRIVILEGED_OK = "RELIQUARY_SCORING_UNPRIVILEGED_OK"
DEFAULT_UID_BASE = 200000
DEFAULT_SLOTS = 64
PRELOAD_CHECK_TIMEOUT_S = 300.0
TMP_ROOT: str | None = None
"""Where the scoring TMPDIRs are made (Settings.scoring_tmp_root; None: the service's
temporary directory)."""


class ScoringSlots:
    """The uids of concurrent scoring processes: slot `i` runs as `base + i` (its gid
    too). A slot is handed out once until released, which happens only after its
    process was reaped and every process of its uid killed; the least recently released
    is handed out first. Thread-safe (grades run on several threads)."""

    def __init__(self, base: int, count: int):
        if base < 1000 or count < 1:
            raise ValueError("the scoring uid range starts at 1000 or above, with a slot")
        self.base, self.count = base, count
        self._lock = threading.Lock()
        self._free = collections.deque(range(count))

    def acquire(self) -> int | None:
        """A free slot, or None when every one is in use."""
        with self._lock:
            return self._free.popleft() if self._free else None

    def release(self, slot: int) -> None:
        with self._lock:
            if not 0 <= slot < self.count or slot in self._free:
                raise ValueError(f"slot {slot} is not in use")
            self._free.append(slot)

    def uid(self, slot: int) -> int:
        return self.base + slot

    def free(self) -> int:
        with self._lock:
            return len(self._free)


SLOTS = ScoringSlots(DEFAULT_UID_BASE, DEFAULT_SLOTS)


def configure(*, wrapper: str | os.PathLike[str] | None = None, uid_base: int | None = None,
              slots: int | None = None, tmp_root: str | os.PathLike[str] | None = None) -> None:
    """The service's settings, at start (before any grade)."""
    global WRAPPER, SLOTS, TMP_ROOT
    if wrapper is not None:
        WRAPPER = os.fspath(wrapper)
    if (uid_base, slots) != (None, None):
        SLOTS = ScoringSlots(SLOTS.base if uid_base is None else uid_base,
                             SLOTS.count if slots is None else slots)
    TMP_ROOT = None if tmp_root is None else os.fspath(tmp_root)


def _check_privileges() -> None:
    if os.geteuid() != 0 and os.environ.get(UNPRIVILEGED_OK) != "1":
        raise PermissionError(
            "the controller does not run as root, so its scoring processes cannot run as "
            "their own uids: verifiers grades are refused (aborted). Run the controller as "
            f"root with CAP_SETUID, CAP_SETGID and CAP_CHOWN, or set {UNPRIVILEGED_OK}=1 "
            "for tests only")


def start(preload: Iterable[str] = ()) -> None:
    """Start the forkserver if it is not running, through WRAPPER, preloading PRELOAD
    and the env modules given now or before. PermissionError when the controller is
    unprivileged (see the module docstring), FileNotFoundError without the wrapper."""
    _check_privileges()
    for module in preload:
        if module not in EXTRA_PRELOAD:
            EXTRA_PRELOAD.append(module)
    if not (os.path.isfile(WRAPPER) and os.access(WRAPPER, os.X_OK)):
        raise FileNotFoundError(f"the scoring wrapper {WRAPPER} is missing or not executable: "
                                "the forkserver never starts with the service's environment")
    CONTEXT.set_executable(WRAPPER)
    CONTEXT.set_forkserver_preload(list(PRELOAD) + EXTRA_PRELOAD)
    forkserver.ensure_running()


_IMPORTS = ("import importlib, json, sys\n"
            "sys.path[:] = json.loads(sys.argv[1])\n"
            "for name in sys.argv[2:]:\n"
            "    importlib.import_module(name)\n")


def check_preload(modules: Iterable[str], timeout: float = PRELOAD_CHECK_TIMEOUT_S) -> None:
    """Import PRELOAD and `modules` as the forkserver does (through WRAPPER: an empty
    environment, the service's sys.path), in a process of their own. RuntimeError, with
    the import's error, when one fails: the forkserver would skip it silently and every
    scoring process would import it as its slot's uid, or fail in it."""
    names = list(PRELOAD) + [name for name in modules if name not in PRELOAD]
    try:
        done = subprocess.run([WRAPPER, "-c", _IMPORTS, json.dumps(sys.path), *names],
                              stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout,
                              check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"the scoring forkserver's preload of {names} could not be "
                           f"checked: {exc}") from exc
    if done.returncode != 0:
        error = done.stderr.decode("utf-8", "replace").strip()[-4000:]
        raise RuntimeError(
            f"the scoring forkserver cannot preload the env modules {names} (exit code "
            f"{done.returncode}); the controller does not start. They are imported with an "
            "empty environment: an env must read no credential (no variable of the "
            f"controller) at import time.\n{error}")


def server_pid() -> int | None:
    """The running forkserver's pid (None if none was started)."""
    return forkserver._forkserver._forkserver_pid


def alive(pid: int | None) -> bool:
    """Whether the forkserver `pid` (our child) still runs. Never reaps it (WNOWAIT):
    `start()` still sees it died and starts another."""
    if pid is None:
        return False
    try:
        return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None
    except ChildProcessError:
        return False


def stop() -> None:
    """Stop the forkserver if it runs (tests; the service's ends with the service)."""
    forkserver._forkserver._stop()
