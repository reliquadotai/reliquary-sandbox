"""Task images as-is under gVisor: writable root capped by an xfs project quota,
network none, per-call timeouts that kill only the command's process group.

Separate from `DockerRuntime`, whose guests are read-only, uid-1000 and
Python-based. Boxes carry their own ownership label, so the tenant
reconciler never deletes them and vice versa.

Fault attribution (this is what decides whether an episode is voided). After
the box has started, its state is entirely the agent's doing; only failures of
our own side void an episode:

* `InfraFault` (ours): a Docker API exception (connection error, daemon 5xx
  without OCI text, NotFound for a box we think exists); `create`/`start`
  failing before the box first ran; or a box that is not running because of
  the host, detected as (i) the daemon having started after the box did
  (a daemon restart; read from the docker pidfile) or (ii) a non-empty
  `State.Error` (a runtime/Sentry failure). The episode manager maps it to the signed
  final status `aborted` (void, the task may be re-run).
* `BoxFault` (the agent): everything else once the box has run: a non-zero or
  126/127 helper exit, OCI start text in the stream, any helper timeout,
  `OOMKilled` (the box's own memory limit), PID 1 exiting with an empty
  `State.Error`, pids exhaustion. The episode manager maps it to `box_failed` (not paid,
  engagement consumed, not voided).
* `RuntimeExecFailed` (a `BoxFault`, so handled like one unless a caller asks):
  the runtime's own generic start failure (`OCI runtime exec failed: ...runsc did
  not terminate successfully: exit status N`, no reason from runsc) on a box that
  still runs, is not at its pids limit and runs a helper right after. Measured on a
  loaded 2-vCPU host under runsc: 13 in a few hours, on fresh grading boxes and
  helpers alike, while the agent's own failures (killed init, OOM, a fork bomb, a
  deleted workdir) come with a stopped box, a full pids count or runsc's reason.
  Read-only helpers are run again (RUNTIME_START_RETRIES, a ghost of one only
  reads); the grader re-runs a grading attempt in a fresh box on it and falls back
  to the old verdict when every attempt hits it. Nothing else is re-run: whether
  the failed exec started its process cannot be told from Docker's side.
  Only Docker's own report counts (`_START_FAILED`): exit 126/128, the stream
  exactly the message, no frame marker; text a helper or program echoes never does.
  Residual: whether every runsc exec failure an agent can provoke from inside a
  live box (a workdir it made unreadable, a sentry whose memory-backed rootfs it
  filled; its processes still alive or already stopped) carries runsc's reason is
  not established. If one does not, the agent gets `box_failed` instead of the
  env's 0 at extract (still unpaid, the seed consumed) and one more grading run in a
  fresh box, ending in the same kept 0 if it fails again; never a void.
* A rare host slowdown misattributed to a miner is accepted: the validator
  caps voids per hotkey anyway. `pids_exhausted` is a recorded fact in
  the fault message, not a decision input.
* Helpers cannot be broken by the agent: they run from a controller-owned
  busybox whose whole directory (`/.reliquary`, containing only the binary)
  is bind-mounted read-only, from `/`, as a numeric user. Renaming or
  replacing it fails (measured live: EBUSY on rename, EROFS, ETXTBSY).
* The box user (`uid:gid`) is resolved exactly once, in `create`, before any
  agent code runs, and is never looked up again inside a box the agent has
  touched. The episode manager persists `box_user(box_id)` with the episode and
  hand it back with `restore_user` after a controller restart.
* An `OOMKilled` box counts against the agent only when its own cgroup limit
  was hit (`memory.events` `oom` > 0, cgroupfs source); when the cgroup is not
  readable (it is gone once the box stops) we still blame the agent, a
  documented residual: a host-wide OOM killing a box is misattributed.
* An exception of a kind we do not expect on this path (not Docker/OS errors)
  is our own bug: it is logged and classified `InfraFault`.
* A box that stopped with an empty `State.Error` and no daemon restart is the
  agent's, whatever exception reported it. Preflight refuses episodes when the
  daemon start time cannot be read (pidfile and /proc must be visible).

The runsc version check reads `runsc --version` from the runtime path in
`docker info`, so that path must be visible to the controller (a controller in
a container needs the host's runsc mounted at the same path).

Process-group isolation is best effort: a command that escapes its group
(`setsid`, a daemon) is bounded by the pids/memory/cpu limits and removed with
the box, and the CPU it burns is charged to whichever call is measured next.
"""

from __future__ import annotations

import errno
import hashlib
import logging
import math
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import docker
from docker.errors import APIError, DockerException, ImageNotFound, NotFound
from docker.utils.socket import STDERR, STDOUT

from .episodes.exec_transport import ExecStream, unix_transport_for
from .runtime import RuntimeOperationError, RuntimeUnavailable

_log = logging.getLogger(__name__)


class InfraFault(RuntimeOperationError):
    """Our side failed (daemon, image, creation): the episode is aborted."""


class BoxFault(RuntimeOperationError):
    """The agent's box is unusable because of what the agent did."""


class RuntimeExecFailed(BoxFault):
    """The runtime failed to start a command (its generic start failure, no reason
    given) in a box that still runs, below its pids limit, and that runs a helper
    right after: most likely the host's, not the agent's (module docstring). A
    BoxFault, so every caller that does not look for it keeps the agent's verdict."""


class InfraUnavailable(InfraFault, RuntimeUnavailable):
    """Host cannot serve episodes at all (preflight, image, create)."""


class HelperBudgetTimeout(Exception):
    """A helper ran out of the caller's time budget (a per-call timeout or a step
    deadline shorter than the helper's own bound). The caller's step timed out;
    the box is not blamed, so this is never classified (not a BoxFault). Not an
    OSError/TimeoutError, so callers' file-error handlers never swallow it."""


class NotRegularFile(ValueError):
    """`read_file` on something that is not a regular file (FIFO, device)."""


def _cut_by_budget(fixed: float, timeout_s: float | None) -> bool:
    """True when the caller's budget, not the helper's own bound, limits it."""
    return timeout_s is not None and timeout_s < fixed


def _bounded(fixed: float, timeout_s: float | None) -> float:
    """A helper's own timeout, cut to the caller's remaining deadline."""
    return fixed if timeout_s is None else min(fixed, timeout_s)


# Docker's default capability set minus NET_RAW, MKNOD, AUDIT_WRITE and
# SETFCAP. Task images run as their own user (often root) and expect
# chown/chmod/setuid to work; gVisor, not capabilities, is the isolation
# boundary here.
TASK_CAPABILITIES = ["CHOWN", "DAC_OVERRIDE", "FOWNER", "FSETID", "KILL", "SETGID",
                     "SETUID", "SETPCAP", "NET_BIND_SERVICE", "SYS_CHROOT"]

# Every helper exec (runner, kill, type check, file I/O, user lookup) runs from
# this controller-owned, read-only static busybox, bind-mounted when the box is
# created, with workdir "/" and a numeric user. The image's own sh, head, cat or
# stat are never used, so an agent cannot break our plumbing by replacing them.
# Only the agent's command runs under the image's own /bin/bash.
HELPER_ROOT = "/.reliquary"
HELPER = HELPER_ROOT + "/bin/busybox"


def _script(text: str) -> str:
    return text.replace("@B@", HELPER).replace("@TOOLS@", TOOLS_DIR)


# Every helper whose output we parse prints it between markers carrying a
# per-exec random nonce (its first argument), so a lost or cut output is told
# apart from an empty one and content cannot forge the end. Output was lost by
# docker-py's exec stream (fixed in episodes/exec_transport.py); the markers
# stay as defence in depth. Idempotent helpers are re-run when a marker is
# missing; an env program is not re-run. An output still lost (or a re-run that
# fails) is classified like any failed operation: on a box that still runs it
# is the agent's (it killed or starved the helper), on a stopped box the usual
# rules decide (a runtime error or daemon restart is ours).
OUTPUT_ATTEMPTS = 4


def frame_markers(nonce: str) -> tuple[bytes, bytes]:
    return f"<RQ{{{nonce}{{".encode(), f"}}{nonce}}}RQ>".encode()


def _new_nonce() -> str:
    return secrets.token_hex(8)


def _framed(text: str) -> str:
    return _script("n=$1; shift; printf '<RQ{%s{' \"$n\"; ( " + text + " ); s=$?; "
                   "printf '}%s}RQ>' \"$n\"; exit $s")


# On the box's /dev/shm tmpfs, outside the disk quota: a full disk must not
# stop tool calls (or their timeouts) from working. Created at box creation as
# root, mode 1777, so one caller's uid/umask cannot lock the others out.
TOOLS_DIR = "/dev/shm/.reliquary-tools"
MAKE_TOOLS_DIR = _script('@B@ mkdir -p -m 1777 "$1" && @B@ chmod 1777 "$1"')

REFUSED_MARKER = "RELIQUARY-RUNNER-REFUSED"
RUNNER_DRAIN_S = 5.0
"""After a timed-out call's group is killed, how long the stream is still read:
the runner's `head` buffers (stdio), so what the command printed before the
timeout reaches us only when the runner drains (about 1.5 s) and exits. A
timed-out call returns after about timeout + kill + up to this much."""
TRAILER = "RQ-END"
"""The runner's very last write on its own stdout: `RQ-END <call id> <status>`.
The command never holds that descriptor (its output goes through FIFOs and the
runner's stdout is not inherited), so the trailer is the final bytes of a
runner that finished; the controller requires it and its status."""

# Runs one bash tool call as its own process group, under busybox ash.
# Arguments: $1 per-stream output limit, $2 call id, $3 command, $4 workdir.
# The command runs under the image's /bin/bash, in its own session (setsid),
# so its process group id is its pid.
#
# * A placeholder `<id>.<uid>.pgid` is created first, so "absent" means
#   "finished". The real group id replaces it, then the runner looks for the
#   tombstone `<id>.killed`: a kill that arrived before the group id existed
#   leaves one, and the call is then killed instead of run. The tombstone is
#   also checked before anything is launched. The uid in the name lets the
#   killer check that the file is owned by the runner's user.
# * If any of that cannot be recorded (full disk, tampered directory) the
#   runner prints a refusal marker on the real stderr and exits 125: a command
#   we could not stop never runs.
# * stdout and stderr go through FIFOs, `cat`, `head -c LIMIT+1`, then a drain,
#   so a flood is capped inside the box while the command still runs to the end
#   and the controller can tell exactly whether it was truncated. The runner
#   holds the output FIFOs open (O_RDWR) until the command is gone, so the
#   command never blocks on open. A final open/close of the intermediate FIFOs
#   guarantees `head` sees EOF even if its `cat` was killed before it opened.
# * Once the command is reaped the cappers get ~1.5 s to drain, then their
#   FIFO readers (`cat`, not the agent's processes) are killed so `head` sees
#   EOF and flushes what it holds: a background service or a setsid escapee
#   that keeps the pipes open never keeps the call, or runner, alive.
# * A workdir that no longer exists is the agent's normal `cd` error, exit 1.
# * The runner's own diagnostics go to /dev/null; the command's stderr is kept.
# * Its last action is the trailer `RQ-END <id> <status>` on its own stdout.
BASH_RUNNER = _script(r"""
exec 3>&2 2>/dev/null
B=@B@; d=@TOOLS@; id=$2; n=$(( $1 + 1 )); me=$($B id -u)
pg="$d/$id.$me.pgid"
refuse() { printf 'RELIQUARY-RUNNER-REFUSED %s\n' "$id" >&3; exit 125; }
cleanup() { $B rm -f "$pg" "$d/$id.out" "$d/$id.err" "$d/$id.o2" "$d/$id.e2" "$d/$id.d1" "$d/$id.d2" "$d/$id.killed"; }
$B mkdir -p "$d" || refuse
true > "$pg" || refuse
if [ -e "$d/$id.killed" ]; then cleanup; printf 'RQ-END %s %s' "$id" 137; exit 137; fi
$B mkfifo "$d/$id.out" "$d/$id.err" "$d/$id.o2" "$d/$id.e2" || refuse
( exec 4<>"$d/$id.out" 5<>"$d/$id.err" ) || refuse
exec 4<>"$d/$id.out" 5<>"$d/$id.err"
$B cat <"$d/$id.out" >"$d/$id.o2" 4>&- 5>&- &
r1=$!
{ $B head -c "$n"; $B cat >/dev/null; : > "$d/$id.d1"; } <"$d/$id.o2" 4>&- 5>&- &
c1=$!
$B cat <"$d/$id.err" >"$d/$id.e2" 4>&- 5>&- &
r2=$!
{ $B head -c "$n"; $B cat >/dev/null; : > "$d/$id.d2"; } <"$d/$id.e2" >&3 4>&- 5>&- &
c2=$!
(
  cd -- "$4" 2>/dev/null || { echo "bash: cd: $4: No such file or directory" >&2; exit 1; }
  exec $B setsid /bin/bash -c "$3"
) </dev/null >"$d/$id.out" 2>"$d/$id.err" 3>&- 4>&- 5>&- &
child=$!
printf '%s' "$child" > "$pg" || { $B kill -s KILL -- "-$child"; wait "$child"; refuse; }
[ -e "$d/$id.killed" ] && $B kill -s KILL -- "-$child"
wait "$child"
status=$?
exec 4>&- 5>&-
i=0
while [ "$i" -lt 15 ] && { [ ! -e "$d/$id.d1" ] || [ ! -e "$d/$id.d2" ]; }; do
  $B sleep 0.1; i=$((i+1))
done
$B kill -s KILL "$r1" "$r2" 2>/dev/null
( exec 8<>"$d/$id.o2" 9<>"$d/$id.e2" )
wait "$c1" "$c2"
cleanup
printf 'RQ-END %s %s' "$id" "$status"
exit "$status"
""")

# Busybox sh, run from `/` as the call's own numeric user: it can only signal
# what that uid can signal. Tombstone first, then look for the group id:
# whichever of runner and killer comes second sees the other. Exit 0 when the
# call is over or never got a group yet (the tombstone stops it), 1 when the
# group exists but could not be killed. The pgid file must be a regular file
# (never a symlink or FIFO), owned by the uid in its name, holding all digits
# greater than 1. Once a recorded group is dead, the call's leftovers go.
KILL_GROUP = _script(
    'B=@B@; d=@TOOLS@; $B mkdir -p -m 1777 "$d" 2>/dev/null; true > "$d/$1.killed" || exit 1; '
    'for f in "$d/$1".*.pgid; do '
    '[ -e "$f" ] || [ -L "$f" ] || continue; '
    '[ -L "$f" ] && exit 1; [ -f "$f" ] || exit 1; '
    'u="${f%.pgid}"; u="${u##*.}"; [ "$($B stat -c %u "$f")" = "$u" ] || exit 1; '
    'read p < "$f"; [ -n "$p" ] || continue; '
    'case "$p" in *[!0-9]*) exit 1;; esac; [ "$p" -gt 1 ] || exit 1; '
    'if $B kill -s KILL -- "-$p" 2>/dev/null || ! $B kill -0 -- "-$p" 2>/dev/null; then '
    '$B rm -f "$d/$1.killed" "$d/$1.out" "$d/$1.err" "$d/$1.o2" "$d/$1.e2" "$d/$1.d1" "$d/$1.d2" "$f"; exit 0; fi; '
    'exit 1; done; exit 0'
)

# Exit 3 missing, 4 directory, 5 not a regular file (FIFOs would hang).
READ_FILE = _framed('[ -e "$1" ] || exit 3; [ -d "$1" ] && exit 4; [ -f "$1" ] || exit 5; '
                    'exec @B@ head -c "$2" -- "$1"')
WRITE_FILE = _script('exec @B@ cat > "$1"')
MAKE_DIRS = _script('exec @B@ mkdir -p -- "$1"')
# Never `-h`: symlinks are archived as symlinks, never followed out of the paths.
TAR_CREATE = _framed('cd / && exec @B@ tar -cf - -- "$@"')
# `-o`: never restore owners. Setuid/setgid bits are stripped by the caller,
# which only ever hands this a tar it rebuilt itself.
TAR_EXTRACT = _script('@B@ mkdir -p -- "$1" && cd -- "$1" && exec @B@ tar -xof -')
# Remove a restore root before extraction, without following it: exit 3 when
# the root itself is a symlink (an image/env problem, never written through).
CLEAR_ROOT = _script('[ -L "$1" ] && exit 3; exec @B@ rm -rf -- "$1"')
# Remove a restored root whose links were refused (`contained_links`), the
# root itself included when it was restored as a link (rm never follows it).
REMOVE_RESTORED = _script('exec @B@ rm -rf -- "$1"')
# The canonical path of the nearest existing ancestor of $1 (symlinks
# followed; busybox resolves a dangling link to its target).
RESOLVE_EXISTING = _framed(
    'p="$1"; while [ ! -e "$p" ] && [ ! -L "$p" ]; do p=$(@B@ dirname -- "$p"); done; '
    'exec @B@ readlink -f -- "$p"')
# Per argument: the canonical path its links lead to (`readlink -f`) then a
# NUL, or an empty line then a NUL when nothing exists there (dangling, a loop,
# unreadable). A path never holds a NUL, so each record is `<path>\n\0`.
PATH_MAX = 4096
RESOLVE_LINKS = _framed(
    'for p in "$@"; do if [ -e "$p" ]; then @B@ readlink -f -- "$p" || exit 1; '
    "else echo; fi; printf '\\0'; done")
# An env hook's program (`LocalRuntime.run`), its stdout framed the same way
# (nonce first). Not re-run when its output is lost (a program may have side
# effects): that is an InfraFault. busybox-static prefers its own applets for
# any bare name (`ls`, `grep`), so the name is looked up on PATH here and the
# image's file run by its path.
RUN_PROGRAM = _script(
    'n=$1; c=$2; shift 2; '
    'case $c in */*) p=$c ;; *) p=; o=$IFS; IFS=:; '
    'for d in $PATH; do [ -n "$d" ] || d=.; '
    'if [ -f "$d/$c" ] && [ -x "$d/$c" ]; then p=$d/$c; break; fi; done; IFS=$o ;; esac; '
    "printf '<RQ{%s{' \"$n\"; "
    'if [ -z "$p" ]; then echo "$c: command not found" >&2; s=127; else "$p" "$@"; s=$?; fi; '
    "printf '}%s}RQ>' \"$n\"; exit $s")
FRAMED_SCRIPTS = (READ_FILE, TAR_CREATE, RESOLVE_EXISTING, RESOLVE_LINKS, RUN_PROGRAM)
# Before extract, as root from `/`: SIGKILL every process of the box except
# PID 1 (docker-init), this shell, and the entrypoint `sleep` (the oldest child
# of PID 1: anything the agent left behind and that PID 1 adopted is younger).
# Zombies are skipped (PID 1 reaps them). Repeated until a round finds nothing
# to kill (a fork bomb respawns), at most 20 rounds; exit 3 if processes remain. Field 22 of /proc/<pid>/stat
# (starttime) is ${20} once the fields after the command name are split.
STOP_PROCESSES = _script(
    'B=@B@; round=0; while [ "$round" -lt 20 ]; do round=$((round+1)); '
    'keep=; oldest=; '
    'for d in /proc/[0-9]*; do read -r s < "$d/stat" 2>/dev/null || continue; '
    'set -- ${s##*) }; [ "$2" = 1 ] || continue; '
    'if [ -z "$oldest" ] || [ "${20}" -lt "$oldest" ]; then oldest=${20}; keep=${d#/proc/}; fi; '
    'done; '
    'killed=0; '
    'for d in /proc/[0-9]*; do p=${d#/proc/}; '
    'if [ "$p" = 1 ] || [ "$p" = "$$" ] || [ "$p" = "$keep" ]; then continue; fi; '
    'read -r s < "$d/stat" 2>/dev/null || continue; set -- ${s##*) }; '
    '[ "$1" = Z ] && continue; '
    '$B kill -s KILL "$p" 2>/dev/null && killed=1; done; '
    '[ "$killed" = 0 ] && exit 0; $B sleep 0.05; done; exit 3')
MIN_HELPER_VERSION = (1, 36)
IMAGE_INVENTORY_TTL_S = 45.0
"""How long the local image inventory is used before one `images.list()` refreshes it."""
IMAGE_INVENTORY_RETRY_S = 5.0
"""After a failed listing, the last inventory is used this long before the next try."""
_DEFAULT_REGISTRY = ("docker.io/library/", "index.docker.io/library/", "docker.io/", "index.docker.io/")


def _image_key(reference: str) -> str:
    """`repo@sha256:<hex>` in the form Docker lists it: no default-registry prefix and no
    tag next to the digest, so an approved image and a listed RepoDigest compare equal."""
    name, sep, digest = reference.partition("@")
    for prefix in _DEFAULT_REGISTRY:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if sep and ":" in name.rsplit("/", 1)[-1]:
        name = name.rsplit(":", 1)[0]
    return name + sep + digest


# exit 0 regular file, 3 exists but is not a regular file, 4 absent.
FILE_KIND = ('if [ -f "$1" ]; then exit 0; '
             'elif [ -e "$1" ] || [ -L "$1" ]; then exit 3; else exit 4; fi')

_BOX_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
# Container-runtime text that must never become an agent's signed call output:
# a line carrying it is replaced by SCRUBBED (the agent may print it too).
_RUNTIME_TEXT = re.compile(rb"OCI runtime|urpc method|containerManager\.")
SCRUBBED = b"[runtime output removed]"


# The runtime's generic start failure, exactly as Docker reports it: the shim's
# `runsc exec` exited non-zero and runsc gave no reason (a reason, e.g. `chdir: no
# such file or directory`, replaces the `...did not terminate successfully` text).
# Docker's exec-start handler writes the error on the exec's stdout stream as
# `"%v\r\n"` and nothing else (moby daemon/server/router/container/exec.go,
# postContainerExecStart; api/server/router/... before v29), and sets the exec's exit
# code: 128 (`exitUnknown`, daemon/errors.go setExitCodeFromError, Docker >= 24) or
# the deferred default 126 (daemon/exec.go ContainerExecStart; Docker <= 23, whose
# translateContainerdStartErr sets no code for it). The whole stream must be that
# message (anchored at both ends, nothing on stderr) with one of those codes, and a
# framed helper's stream must lack its open marker: an agent-controlled name echoed
# by a helper (tar's `can't open '<name>'`, on stderr, prefixed) or a program's own
# output (behind the marker) can never pass for it, and the helpers (the
# read-only-mounted busybox) cannot exit 126 or 128.
# The runtime's path is `.../runsc`, or, when the daemon configures `runtimeArgs`
# for it, the wrapper dockerd generates: `<data-root>/runtimes/runsc.<base32 hash>`
# (seen live: `.../runtimes/runsc.14BCX5...====`).
_START_FAILED = re.compile(
    rb"OCI runtime exec failed: (?:\S*/)?runsc(?:\.[A-Za-z0-9=_-]+)? did not terminate successfully: "
    rb"exit status \d{1,3}(?:\r\n)?")
START_FAILED_EXITS = (126, 128)
RUNTIME_START_RETRIES = 2
"""Extra runs of a read-only helper whose exec the runtime failed to start."""
RUNTIME_START_PAUSE_S = 1.0


def _start_failed(result: ExecResult, opened: bytes | None = None) -> bool:
    """The exec failed with the runtime's generic start failure, as Docker reports it
    (`_START_FAILED`); `opened`: the framed helper's open marker, which must be absent."""
    return (not result.timed_out and result.exit_code in START_FAILED_EXITS
            and not result.stderr
            and (opened is None or not result.stdout.startswith(opened))
            and _START_FAILED.fullmatch(result.stdout) is not None)


def _start_failed_cause(cause: BaseException | None) -> bool:
    """A Docker API error whose explanation is exactly the generic start failure."""
    explanation = getattr(cause, "explanation", None) if isinstance(cause, APIError) else None
    if isinstance(explanation, bytes):
        explanation = explanation.decode("utf-8", "replace")
    return (isinstance(explanation, str)
            and _START_FAILED.fullmatch(explanation.strip().encode()) is not None)


def _scrub(data: bytes) -> bytes:
    if not _RUNTIME_TEXT.search(data):
        return data
    return re.sub(rb"[^\n]*(?:OCI runtime|urpc method|containerManager\.)[^\n]*", SCRUBBED, data)


_BOX_HOSTNAME = "sandbox"
_ETC_FILES = {
    "/etc/hosts": f"127.0.0.1 localhost\n::1 localhost ip6-localhost ip6-loopback\n"
                  f"127.0.0.1 {_BOX_HOSTNAME}\n",
    "/etc/hostname": _BOX_HOSTNAME + "\n",
    "/etc/resolv.conf": "",
}


@dataclass(frozen=True)
class BoxSpec:
    image: str
    workdir: str
    env: dict[str, str]
    user: str | None
    memory_bytes: int
    nano_cpus: int
    pids: int
    disk_bytes: int
    lifetime_s: int


@dataclass(frozen=True)
class ExecResult:
    stdout: bytes
    stderr: bytes
    exit_code: int
    timed_out: bool
    truncated: bool


class TaskBoxRuntime:
    LABEL = "reliquary.episode.managed"
    BOX_LABEL = "reliquary.episode.box"
    EPISODE_LABEL = "reliquary.episode.id"
    LABEL_NAMESPACE = "reliquary.sandbox.namespace"

    def __init__(self, settings: Any, *, client: Any, namespace: str,
                 exec_transport: Any = None):
        self.settings = settings
        self.client = client
        self.namespace = namespace
        # Exec streams go through our own upgrade on the daemon's Unix socket
        # (episodes/exec_transport.py): docker-py's loses a first frame that
        # arrives with the response headers.
        self.exec_transport = exec_transport or unix_transport_for(client)
        self._connections: dict[str, Any] = {}
        self._cpu_last: dict[str, int] = {}
        self._box_users: dict[str, tuple[str | None, str]] = {}
        self._helper_retries: dict[str, int] = {}
        """box id -> read-only helpers run again in place (`_exec_read_only`) since last taken."""
        self._probing = threading.local()  # classify_failure never probes from a probe
        # Image inventory (`_image_inventory`): the approved images present locally.
        self._clock = time.monotonic
        self._inventory: frozenset[str] | None = None
        self._inventory_at: float | None = None
        self._inventory_due = 0.0
        self._inventory_generation = 0
        """Bumped by `invalidate_images`: a listing that started before an invalidation does not
        hold the inventory for a whole TTL (it may predate the pull or removal)."""
        self._inventory_refresh = threading.Lock()      # one listing at a time
        self._inventory_state_lock = threading.Lock()   # guards the four fields above

    # -- host checks ---------------------------------------------------------

    def images_present(self) -> list[str]:
        """Approved images present locally, in their approved order (from the inventory)."""
        inventory = self._image_inventory()
        return [image for image in self.settings.episode_images if image in inventory]

    def image_present(self, image: str) -> bool:
        """Whether one approved image is present: a set lookup in the inventory."""
        return image in self._image_inventory()

    def invalidate_images(self) -> None:
        """Make the next lookup list the images again (after a pull or a removal, or when
        Docker says an image the inventory holds is gone). The last inventory is kept
        until that listing succeeds."""
        with self._inventory_state_lock:
            self._inventory_due = 0.0
            self._inventory_generation += 1

    def image_inventory_age_s(self) -> float | None:
        """Seconds since the inventory was last listed successfully (None: never)."""
        listed = self._inventory_at
        return None if listed is None else max(0.0, self._clock() - listed)

    def _image_inventory(self) -> frozenset[str]:
        """The approved images Docker lists locally, refreshed at most every
        `IMAGE_INVENTORY_TTL_S` with ONE `images.list()` call (never a call per image).
        Blocking: callers run on worker threads (maintenance or episode pool), never on
        the event loop. While one thread lists, the others keep the last inventory; a
        failed listing keeps it too (logged with its age) and is retried after
        `IMAGE_INVENTORY_RETRY_S`."""
        inventory = self._inventory
        if inventory is not None and self._clock() < self._inventory_due:
            return inventory
        if not self._inventory_refresh.acquire(blocking=inventory is None):
            return inventory  # another thread is listing; serve the last inventory meanwhile
        try:
            if self._inventory is not None and self._clock() < self._inventory_due:
                return self._inventory
            approved = {_image_key(image): image for image in self.settings.episode_images}
            with self._inventory_state_lock:
                generation = self._inventory_generation
            try:
                listed = self.client.images.list()
                local = set()
                for item in listed:
                    attrs = getattr(item, "attrs", None) or {}
                    for name in attrs.get("RepoDigests") or ():
                        local.add(_image_key(name))
                fresh = frozenset(approved[key] for key in local & approved.keys())
            except Exception:
                age = self.image_inventory_age_s()
                _log.warning("listing the local images failed; keeping the last inventory (%s)",
                             "never listed" if age is None else f"{age:.0f} s old", exc_info=True)
                with self._inventory_state_lock:
                    self._inventory_due = self._clock() + IMAGE_INVENTORY_RETRY_S
                    if self._inventory is None:
                        self._inventory = frozenset()
                return self._inventory
            with self._inventory_state_lock:
                now = self._clock()
                self._inventory, self._inventory_at = fresh, now
                # Invalidated while listing: serve this listing, but list again on the next lookup.
                self._inventory_due = (now + IMAGE_INVENTORY_TTL_S
                                       if self._inventory_generation == generation else 0.0)
            return fresh
        finally:
            self._inventory_refresh.release()

    def _runsc_version(self, path: str) -> str:
        output = subprocess.run([path, "--version"], capture_output=True, text=True,
                                timeout=10, check=True).stdout
        match = re.search(r"release-\S+", output)
        if not match:
            raise ValueError("unrecognised runsc version output")
        return match.group(0)

    def preflight(self, *, probe: bool = True) -> dict[str, Any]:
        """Refuse to serve episodes on a host that cannot enforce their limits."""
        if self.exec_transport is None:
            raise InfraUnavailable("Episode boxes need the Docker daemon on a Unix socket")
        try:
            info = self.client.info()
        except Exception as exc:
            raise InfraUnavailable("Docker is unavailable") from exc
        if info.get("OSType") != "linux":
            raise InfraUnavailable("A Linux Docker daemon is required")
        runtime = self.settings.docker_runtime
        if self.settings.require_gvisor and runtime != "runsc":
            raise InfraUnavailable("gVisor runsc isolation is required")
        configured = info.get("Runtimes", {}).get(runtime)
        if configured is None:
            raise InfraUnavailable("The configured container runtime is unavailable")
        runsc_version = self._check_runsc(configured) if runtime == "runsc" else None
        helper = self._check_helper()
        try:
            version = self._helper_version(helper)
        except Exception as exc:
            raise InfraUnavailable("The helper busybox version could not be read") from exc
        _log.info("helper busybox %s", ".".join(map(str, version)))
        if version[:2] < MIN_HELPER_VERSION:
            raise InfraUnavailable(f"The helper busybox must be >= "
                                   f"{'.'.join(map(str, MIN_HELPER_VERSION))}")
        if self._daemon_started_at() is None:
            raise InfraUnavailable("The Docker daemon start time is unreadable "
                                   "(episode_docker_pidfile and /proc must be visible)")
        present = self.images_present()
        if not present:
            raise InfraUnavailable("No approved episode image is present locally")
        quota = "disabled"
        if not self.settings.episode_allow_unquoted_disk:
            status = {tuple(item) for item in info.get("DriverStatus") or []}
            if info.get("Driver") != "overlay2" or ("Backing Filesystem", "xfs") not in status:
                raise InfraUnavailable(
                    "Episode disks need Docker's overlay2 driver on xfs mounted with pquota"
                )
            if probe:
                self._probe_quota(present[0])
            quota = "xfs-pquota"
        host_cpus = int(info.get("NCPU") or os.cpu_count() or 0)
        # Each episode may run its agent box and a grading box at once.
        reserved = getattr(self.settings, "episode_cpu_millis_total", None)
        if reserved is not None:
            # Agent boxes (before the deliberate overcommit) plus the grading pool.
            wanted = (reserved + (getattr(self.settings, "episode_grading_cpu_millis_total", None) or 0)) / 1000
            what = "episode_cpu_millis_total + episode_grading_cpu_millis_total"
        else:
            wanted = (getattr(self.settings, "episode_capacity", 1) * 2
                      * getattr(self.settings, "episode_nano_cpus", 0) / 1e9)
            what = "episode capacity x 2 boxes x episode_nano_cpus"
        overcommitted = wanted > host_cpus
        if overcommitted:
            _log.warning("%s asks for %.1f CPUs; the host has %d: CPU-bound calls and grading will "
                         "be slower than their budgets assume", what, wanted, host_cpus)
        return {"runtime": runtime, "images": present, "disk_quota": quota,
                "runsc_version": runsc_version, "busybox_version": ".".join(map(str, version)),
                "host_cpus": host_cpus, "cpu_overcommitted": overcommitted}

    def _check_helper(self) -> Path:
        path = getattr(self.settings, "episode_helper_busybox", None)
        pinned = getattr(self.settings, "episode_helper_busybox_sha256", None)
        if not path or not Path(path).is_file():
            raise InfraUnavailable("The helper busybox is missing")
        if not os.access(path, os.X_OK):
            raise InfraUnavailable("The helper busybox is not executable")
        if not pinned or hashlib.sha256(Path(path).read_bytes()).hexdigest() != pinned:
            raise InfraUnavailable("The helper busybox does not match its pinned sha256")
        return Path(path)

    @staticmethod
    def _helper_version(path: Path) -> tuple[int, ...]:
        """`BusyBox vX.Y.Z` from the binary's own banner (run on the host)."""
        result = subprocess.run([str(path)], capture_output=True, text=True, timeout=10)
        match = re.search(r"BusyBox v(\d+)\.(\d+)(?:\.(\d+))?", result.stdout + result.stderr)
        if not match:
            raise ValueError("unrecognised busybox banner")
        return tuple(int(part) for part in match.groups() if part is not None)

    def _check_runsc(self, configured: dict[str, Any]) -> str | None:
        """The pinned runsc version, once checked (None when no version is pinned)."""
        args = [str(arg) for arg in configured.get("runtimeArgs") or []]
        if any(arg.lstrip("-").startswith("ignore-cgroups") for arg in args):
            raise InfraUnavailable("runsc must not run with --ignore-cgroups")
        if not self.settings.episode_allow_unquoted_disk:
            # Docker's XFS project quota owns the writable layer. Keep writes
            # on that layer rather than adding a separate runsc overlay.
            overlays = [arg for arg in args if arg.startswith(("--overlay", "-overlay"))]
            if not overlays or any(arg != "--overlay2=none" for arg in overlays):
                raise InfraUnavailable("Quoted episode disks require runsc --overlay2=none")
        wanted = getattr(self.settings, "episode_runsc_version", "")
        if not wanted:
            return None
        try:
            found = self._runsc_version(configured.get("path") or "runsc")
        except Exception as exc:
            raise InfraUnavailable("The runsc version could not be determined") from exc
        if found != wanted:
            raise InfraUnavailable(f"runsc {found} is installed, {wanted} is required")
        return found

    def _probe_quota(self, image: str) -> None:
        # Docker refuses `--storage-opt size` at create time unless the backing
        # xfs is mounted with pquota; creating (not starting) a box proves it.
        container = None
        try:
            container = self.client.containers.create(
                image=self.client.images.get(image).id,
                entrypoint=["true"], command=[], network_mode="none",
                runtime=self.settings.docker_runtime,
                storage_opt={"size": str(64 * 1024**2)},
                labels={self.LABEL: "probe", self.LABEL_NAMESPACE: self.namespace},
            )
        except APIError as exc:
            raise InfraUnavailable("Docker cannot enforce per-box disk quotas") from exc
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    # -- lifecycle -----------------------------------------------------------

    def _name(self, box_id: str) -> str:
        if not _BOX_ID.fullmatch(box_id):
            raise ValueError("Invalid box identifier")
        return "episode-" + self.namespace + "-" + box_id

    def _helper_dir(self) -> Path:
        """Host directory holding only `bin/busybox` (a verified copy), mounted
        read-only as a whole so the binary cannot be renamed out from under us."""
        original = self._check_helper()
        directory = Path(self.settings.state_dir) / "episode-helper"
        target = directory / "bin" / "busybox"
        pinned = self.settings.episode_helper_busybox_sha256
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != pinned:
            (directory / "bin").mkdir(parents=True, exist_ok=True)
            # Unique temp name, then an atomic rename: concurrent creates are safe.
            partial = target.with_name(f".busybox.{os.getpid()}.{secrets.token_hex(6)}")
            try:
                shutil.copyfile(original, partial)
                partial.chmod(0o755)
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
        if hashlib.sha256(target.read_bytes()).hexdigest() != pinned:
            raise InfraUnavailable("The helper busybox copy does not match its pinned sha256")
        directory.chmod(0o755)
        (directory / "bin").chmod(0o755)
        return directory

    def _etc_dir(self, box_id: str) -> Path:
        return Path(self.settings.state_dir) / "episode-etc" / self._name(box_id)

    def _etc_mounts(self, box_id: str) -> list[Any]:
        """Read-only controller-owned /etc/{hosts,hostname,resolv.conf}: Docker
        would otherwise bind-mount writable copies from outside the box's disk
        quota, an unmetered place for the agent to write."""
        directory = self._etc_dir(box_id)
        directory.mkdir(parents=True, exist_ok=True, mode=0o755)
        mounts = []
        for target, content in _ETC_FILES.items():
            source = directory / Path(target).name
            source.write_text(content)
            source.chmod(0o644)
            mounts.append(docker.types.Mount(target=target, source=str(source), type="bind",
                                             read_only=True))
        mounts.append(docker.types.Mount(target=HELPER_ROOT, source=str(self._helper_dir()),
                                         type="bind", read_only=True))
        return mounts

    def create(self, box_id: str, episode_id: str, spec: BoxSpec) -> str:
        if spec.image not in self.settings.episode_images:
            raise ValueError("The task image is not approved")
        self._check_helper()
        if (self.settings.docker_runtime != "runsc"
                and not self.settings.episode_allow_unquoted_disk):
            raise InfraUnavailable("Task boxes run under gVisor runsc only")
        try:
            image = self.client.images.get(spec.image)
        except ImageNotFound as exc:
            self.invalidate_images()  # the inventory held it: list again on the next lookup
            raise InfraUnavailable("The task image must be present locally") from exc
        except Exception as exc:
            raise InfraFault("Docker is unavailable") from exc
        options: dict[str, Any] = {}
        if not self.settings.episode_allow_unquoted_disk:
            options["storage_opt"] = {"size": str(spec.disk_bytes)}
        try:
            container = self.client.containers.create(
                image=image.id,
                entrypoint=["sleep", str(int(spec.lifetime_s))],
                command=[],
                name=self._name(box_id),
                hostname=_BOX_HOSTNAME,
                labels={self.LABEL: "true", self.BOX_LABEL: box_id,
                        self.EPISODE_LABEL: episode_id, self.LABEL_NAMESPACE: self.namespace},
                runtime=self.settings.docker_runtime,
                user=spec.user,
                working_dir=spec.workdir,
                environment=dict(spec.env),
                read_only=False,
                mounts=self._etc_mounts(box_id),
                network_mode="none",
                cap_drop=["ALL"],
                cap_add=list(TASK_CAPABILITIES),
                security_opt=["no-new-privileges:true"],
                privileged=False,
                pids_limit=spec.pids,
                mem_limit=spec.memory_bytes,
                memswap_limit=spec.memory_bytes,
                nano_cpus=spec.nano_cpus,
                init=True,
                detach=True,
                log_config=docker.types.LogConfig(type="none"),
                auto_remove=False,
                **options,
            )
        except Exception as exc:
            shutil.rmtree(self._etc_dir(box_id), ignore_errors=True)
            raise InfraFault("The task box could not be created") from exc
        try:
            container.start()
            container.reload()
            if container.status != "running":
                raise InfraFault("The task box failed to start")
            made = self.exec(box_id, [HELPER, "sh", "-c", MAKE_TOOLS_DIR, "mk", TOOLS_DIR],
                             timeout_s=30, output_limit=4096, user="0", workdir="/")
            if made.exit_code != 0 or made.timed_out:
                raise InfraFault("The task box could not prepare its tools directory")
            self._box_users[box_id] = (spec.user, self._resolve_user(box_id, spec.user))  # before agent code
            return container.id
        except Exception as exc:
            try:
                container.remove(force=True)
            finally:
                shutil.rmtree(self._etc_dir(box_id), ignore_errors=True)
            # The box never ran for the agent: whatever failed here is ours.
            if isinstance(exc, InfraFault):
                raise
            raise InfraFault("The task box failed to start") from exc

    def _container(self, box_id: str) -> Any:
        container = self.client.containers.get(self._name(box_id))
        labels = container.labels
        if (labels.get(self.LABEL) != "true" or labels.get(self.BOX_LABEL) != box_id
                or labels.get(self.LABEL_NAMESPACE) != self.namespace):
            raise InfraFault("Container ownership does not match the box")
        return container

    def status(self, box_id: str) -> str:
        try:
            container = self._container(box_id)
            container.reload()
            return container.status
        except NotFound:
            return "deleted"

    def remove(self, box_id: str) -> None:
        try:
            self._container(box_id).remove(force=True)
        except NotFound:
            pass
        self._cpu_last.pop(box_id, None)
        self._box_users.pop(box_id, None)
        self._helper_retries.pop(box_id, None)
        shutil.rmtree(self._etc_dir(box_id), ignore_errors=True)

    def take_helper_retries(self, box_id: str) -> int:
        """How many times a read-only helper was run again in place in `box_id` since the
        last call (conformance reports it; the count is dropped with the box)."""
        return self._helper_retries.pop(box_id, 0)

    def managed(self) -> dict[str, str]:
        """box id -> episode id, for this controller's namespace."""
        found = self.client.containers.list(all=True, filters={"label": [
            self.LABEL + "=true", self.LABEL_NAMESPACE + "=" + self.namespace]})
        return {c.labels[self.BOX_LABEL]: c.labels.get(self.EPISODE_LABEL, "")
                for c in found if self.BOX_LABEL in c.labels}

    # -- fault attribution ---------------------------------------------------

    def pids_exhausted(self, box_id: str) -> bool:
        """True when the box's pids are at their limit (a fork bomb)."""
        try:
            container = self._container(box_id)
            if self.settings.episode_cpu_source == "cgroupfs":
                scope = self._cgroup_scope(container.id)
                current = int((scope / "pids.current").read_text())
                limit = (scope / "pids.max").read_text().strip()
                return limit != "max" and current >= int(limit)
            pids = self.client.api.stats(container.id, stream=False, one_shot=True)["pids_stats"]
            return int(pids["current"]) >= int(pids["limit"])
        except Exception:
            return False

    def _daemon_started_at(self) -> float | None:
        """Epoch seconds the Docker daemon started (pidfile + /proc), or None."""
        try:
            pidfile = getattr(self.settings, "episode_docker_pidfile", "/var/run/docker.pid")
            pid = int(Path(pidfile).read_text())
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            btime = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                         if line.startswith("btime"))
            return btime + int(fields[19]) / os.sysconf("SC_CLK_TCK")
        except Exception:
            return None

    def _daemon_restarted_since(self, started_at: str | None) -> bool:
        match = re.fullmatch(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?Z", started_at or "")
        if not match:
            return False
        box_started = datetime.fromisoformat(
            match.group(1) + (match.group(2) or ".0")[:7] + "+00:00").timestamp()
        daemon = self._daemon_started_at()
        return daemon is not None and daemon > box_started

    @staticmethod
    def _infra_cause(cause: BaseException | None) -> bool:
        if cause is None:
            return False
        if isinstance(cause, NotFound):
            return True
        if isinstance(cause, APIError):
            if "OCI runtime" in str(cause):
                return False  # a start failure inside the box: the agent's state
            status = getattr(cause, "status_code", None)
            return status is not None and status >= 500
        if isinstance(cause, (DockerException, OSError)):
            return isinstance(cause, OSError)  # connection errors, timeouts to the daemon
        _log.error("unexpected %s while classifying a box failure: our bug",
                   type(cause).__name__, exc_info=cause)
        return True

    def _settled(self, container: Any) -> bool | None:
        """After a settle period, True if the box still reads running, False if
        it no longer does, None if Docker cannot be reached."""
        time.sleep(float(getattr(self.settings, "episode_fault_settle_s", 1.0)))
        try:
            container.reload()
        except Exception:
            return None
        return container.status == "running"

    def _memory_oom(self, container_id: str) -> int | None:
        """The cgroup's own `oom` count (limit hit), or None when unreadable."""
        try:
            events = (self._cgroup_scope(container_id) / "memory.events").read_text()
            return next(int(line.split()[1]) for line in events.splitlines()
                        if line.split()[:1] == ["oom"])
        except Exception:
            return None

    def classify_failure(self, box_id: str, what: str, cause: BaseException | None = None,
                         *, start_failed: bool = False) -> RuntimeOperationError:
        """Return (not raise) the fault for a failed box operation; see the
        module docstring for the rule. The box's state is checked first.
        `start_failed`: the exec ended in the runtime's generic start failure
        (also read from `cause`); on a running box below its pids limit that
        still runs a helper, that is a RuntimeExecFailed."""
        try:
            container = self._container(box_id)
            container.reload()
            status = container.status
            state = (container.attrs or {}).get("State", {})
        except InfraFault as exc:
            return exc
        except NotFound:
            return InfraFault(f"{what}: the box no longer exists")
        except Exception as exc:
            if not isinstance(exc, (DockerException, OSError)):
                _log.error("unexpected %s inspecting a box: our bug", type(exc).__name__, exc_info=exc)
            return InfraFault(f"{what}: Docker is unreachable")
        detail = f"the box is {status}"
        if status != "running":
            if state.get("Error"):
                return InfraFault(f"{what}: the runtime failed the box: {state['Error']}")
            if self._daemon_restarted_since(state.get("StartedAt")):
                return InfraFault(f"{what}: the Docker daemon restarted")
            if state.get("OOMKilled"):
                if self._memory_oom(container.id) == 0:
                    return InfraFault(f"{what}: the host ran out of memory, not the box's limit")
                detail += ", out of memory"
        elif self._infra_cause(cause):
            # Before blaming our side, let a box that is dying under us finish.
            settled = self._settled(container)
            if settled is None:
                return InfraFault(f"{what}: Docker is unreachable")
            if settled:
                fault: RuntimeOperationError = InfraFault(f"{what}: the Docker API failed")
                fault.__cause__ = cause
                return fault
            return self.classify_failure(box_id, what, None)
        at_limit = self.pids_exhausted(box_id)
        if at_limit:
            detail += ", at its pids limit"
        start_failed = start_failed or _start_failed_cause(cause)
        if (start_failed and status == "running" and not at_limit
                and not getattr(self._probing, "active", False) and self._probe(box_id)):
            fault = RuntimeExecFailed(f"{what}: the runtime failed to start the command; "
                                      f"{detail}")
        else:
            fault = BoxFault(f"{what}: {detail}")
        fault.__cause__ = cause
        return fault

    def _check_helper_result(self, box_id: str, result: ExecResult, what: str, *,
                             budget: bool = False, opened: bytes | None = None) -> None:
        """A helper that timed out, exited 126/127 (the shell's codes: this is only what sends
        it to `classify_failure`; the runtime's own start failure is 126/128, `_start_failed`)
        or left OCI text in the stream means the box is in a state the agent made. Except a
        timeout when `budget` (the caller's time, not the helper's own bound,
        limited it): that is the caller's step timing out (HelperBudgetTimeout).
        `opened`: a framed helper's open marker (`_start_failed`)."""
        if result.timed_out and budget:
            raise HelperBudgetTimeout(f"{what}: the caller's time ran out")
        # stdout counts only when the helper failed: a read file may say anything.
        oci = b"OCI runtime" in result.stderr or (result.exit_code != 0 and b"OCI runtime" in result.stdout)
        if result.timed_out or result.exit_code in (126, 127) or oci:
            raise self.classify_failure(box_id, what,
                                        start_failed=_start_failed(result, opened))

    def _exec_read_only(self, box_id: str, argv: list[str], what: str, *, fixed: float,
                        timeout_s: float | None, opened: bytes | None = None,
                        **kwargs: Any) -> tuple[ExecResult, bool]:
        """`exec` for a helper that only reads: run again (up to RUNTIME_START_RETRIES
        times) when the runtime failed to start it (`_start_failed`, `opened` its open
        marker if framed). A retry that succeeds is that read, done; a ghost of a failed
        run (if it ever started) only reads. Every run gets the helper's own bound
        `fixed`, cut only by the caller's deadline (`timeout_s` from now), so a retry
        never eats into it. Returns the last result (for the usual checks) and whether
        the caller's deadline, not `fixed`, bounded that run (the `budget` of
        `_check_helper_result`); RuntimeExecFailed when the last run raised it."""
        caller_deadline = None if timeout_s is None else time.monotonic() + timeout_s
        left = timeout_s
        for attempt in range(RUNTIME_START_RETRIES + 1):
            last = attempt == RUNTIME_START_RETRIES
            try:
                result = self.exec(box_id, argv, timeout_s=_bounded(fixed, left), **kwargs)
            except RuntimeExecFailed:
                if last:
                    raise
            else:
                if last or not _start_failed(result, opened):
                    return result, _cut_by_budget(fixed, left)
            if caller_deadline is not None:
                left = caller_deadline - time.monotonic() - RUNTIME_START_PAUSE_S
                if left <= 0:
                    raise self.classify_failure(box_id, f"{what}: the runtime failed to start "
                                                        "it and no time is left to run it again",
                                                start_failed=True)
            _log.warning("box %s: %s: the runtime failed to start the helper (attempt %d); "
                         "running it again", box_id, what, attempt + 1)
            self._helper_retries[box_id] = self._helper_retries.get(box_id, 0) + 1
            time.sleep(RUNTIME_START_PAUSE_S)
        raise AssertionError("unreachable")

    def _resolve_user(self, box_id: str, user: str | None) -> str:
        """One-time `uid:gid` lookup with the helper `id`, in `create`, before
        any agent code ran. Never called again for a box."""
        for _ in range(OUTPUT_ATTEMPTS):
            result = self.exec(box_id, [HELPER, "id"], timeout_s=15, output_limit=1024,
                               user=user, workdir="/")
            found = re.search(rb"uid=(\d+).*gid=(\d+)", result.stdout)
            if found or result.timed_out or result.exit_code != 0:
                break
            _log.warning("box %s: the user lookup's output was lost; asking again", box_id)
        if result.timed_out or result.exit_code != 0 or not found:
            raise InfraFault(f"The box user could not be resolved before the agent ran "
                             f"(exit {result.exit_code}, timed out {result.timed_out}, "
                             f"output {(result.stdout + result.stderr)[-200:]!r})")
        return f"{int(found.group(1))}:{int(found.group(2))}"

    def box_user(self, box_id: str) -> str | None:
        """The box's resolved default `uid:gid`; the episode manager persists it."""
        known = self._box_users.get(box_id)
        return known[1] if known else None

    def restore_user(self, box_id: str, uid_gid: str, spec_user: str | None = None) -> None:
        """Re-seed the resolved user (and the name it was resolved from, the
        task's `user`) after a controller restart."""
        if not re.fullmatch(r"\d+:\d+", uid_gid):
            raise ValueError("uid:gid expected")
        self._box_users[box_id] = (spec_user, uid_gid)

    def _numeric_user(self, box_id: str, user: str | None) -> str:
        """uid[:gid] for a call: numeric users pass through, anything else is
        the box's own default user, resolved once at creation. Never looked
        up inside a box the agent has touched (it could have edited passwd)."""
        if user is not None and re.fullmatch(r"\d+(:\d+)?", user):
            return user
        if box_id not in self._box_users:
            raise InfraFault("The box user was never resolved: the caller must restore_user() it")
        name, resolved = self._box_users[box_id]
        if user is not None and user != name:
            raise ValueError("Only numeric ids or the box's own user may run helpers")
        return resolved

    # -- execution -----------------------------------------------------------

    def _abort(self, exec_id: str | None) -> None:
        """Drop an abandoned exec's connection so no thread or fd lingers."""
        stream = self._connections.pop(exec_id, None) if exec_id else None
        if stream is not None:
            stream.close()

    def _exec_stream(self, exec_id: str, stdin: bytes | None
                     ) -> Iterator[tuple[bytes | None, bytes | None]]:
        if self.exec_transport is None:
            raise InfraUnavailable("Episode boxes need the Docker daemon on a Unix socket")
        abandon = getattr(threading.current_thread(), "abandon", None)

        def registered(sock: Any) -> None:
            # Registered before the handshake, so an abort can close the socket
            # of a start the daemon never answers.
            self._connections[exec_id] = ExecStream(sock)
            if abandon is not None and abandon.is_set():
                self._abort(exec_id)

        try:
            stream = self.exec_transport.start(exec_id, on_connect=registered)
        except BaseException:
            self._abort(exec_id)
            raise
        if exec_id not in self._connections:  # aborted during the handshake
            stream.close()
            return
        self._connections[exec_id] = stream
        try:
            # An abort that ran before the registration above left only this flag.
            if abandon is not None and abandon.is_set():
                return
            if stdin is not None:
                try:
                    stream.send_stdin(stdin)
                except OSError:
                    pass  # the reader quit early (EPIPE): its exit code tells why
            try:
                for kind, chunk in stream.frames():
                    yield ((chunk, None) if kind == STDOUT
                           else (None, chunk) if kind == STDERR else (None, None))
            except (ConnectionResetError, BrokenPipeError):
                pass  # the peer closed with our input unread: the exit code tells why
        finally:
            self._abort(exec_id)

    def exec(self, box_id: str, argv: list[str], *, timeout_s: float, output_limit: int,
             stdin: bytes | None = None, env: dict[str, str] | None = None,
             workdir: str | None = None, user: str | None = None,
             on_timeout: Callable[[], None] | None = None, drain_s: float = 0.0) -> ExecResult:
        """Run argv; on timeout return what was captured and leave the process
        running (the caller decides whether to kill a group or the box). The
        stream is closed so nothing lingers on our side. With `on_timeout`
        (which stops the process), the stream is read for up to `drain_s` more
        after it returns, so output flushed on the way out is kept; an
        exception from it propagates once the stream is closed."""
        if not argv or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("A command and a positive finite timeout are required")
        try:
            container = self._container(box_id)
        except InfraFault:
            raise
        except NotFound as exc:
            raise InfraFault("The box no longer exists") from exc
        except Exception as exc:
            raise InfraFault("Docker is unreachable") from exc
        stdout, stderr = bytearray(), bytearray()
        state: dict[str, Any] = {"truncated": False, "exit_code": -1, "exec_id": None}
        finished = threading.Event()

        def collect() -> None:
            try:
                created = self.client.api.exec_create(
                    container.id, argv, stdout=True, stderr=True, stdin=stdin is not None,
                    tty=False, environment=env or {}, workdir=workdir,
                    **({"user": user} if user else {}))
                state["exec_id"] = created["Id"]
                for out, err in self._exec_stream(created["Id"], stdin):
                    for target, chunk in ((stdout, out), (stderr, err)):
                        if chunk:
                            room = max(0, output_limit - len(target))
                            target.extend(chunk[:room])
                            state["truncated"] |= len(chunk) > room
                while not worker.abandon.is_set():
                    inspection = self.client.api.exec_inspect(created["Id"])
                    if not inspection["Running"]:
                        state["exit_code"] = inspection["ExitCode"]
                        break
                    time.sleep(0.02)
            except Exception as exc:
                state["error"] = exc
            finally:
                finished.set()

        worker = threading.Thread(target=collect, daemon=True)
        worker.abandon = threading.Event()  # type: ignore[attr-defined]
        worker.start()
        if not finished.wait(timeout_s):
            try:
                if on_timeout is not None:
                    on_timeout()
                    finished.wait(drain_s)
            finally:
                worker.abandon.set()  # type: ignore[attr-defined]
                self._abort(state["exec_id"])
            return ExecResult(bytes(stdout), bytes(stderr), -1, True, state["truncated"])
        if "error" in state:
            raise self.classify_failure(box_id, "Box execution failed", state["error"])
        return ExecResult(bytes(stdout), bytes(stderr), state["exit_code"], False, state["truncated"])

    def bash(self, box_id: str, call_id: str, command: str, *, timeout_s: float,
             output_limit: int, env: dict[str, str], workdir: str, user: str | None) -> ExecResult:
        """One `bash` tool call. A timeout kills the call's process group only;
        the box and anything the agent started in earlier calls keep running.
        `BoxFault` when the runner refuses or the group cannot be stopped."""
        if not re.fullmatch(r"[a-f0-9]{32}", call_id):
            raise ValueError("Invalid call identifier")
        numeric = self._numeric_user(box_id, user)
        argv = [HELPER, "sh", "-c", BASH_RUNNER, "reliquary-bash", str(output_limit), call_id,
                command, workdir]

        def kill() -> None:
            for _ in range(5):
                # From `/`, with the helper busybox, as the call's own uid: it can
                # only signal what that uid can, and nothing the agent did to its
                # shell, workdir or userland can stand in the way.
                killed = self.exec(box_id, [HELPER, "sh", "-c", KILL_GROUP, "reliquary-kill",
                                            call_id],
                                   timeout_s=10, output_limit=4096, user=numeric, workdir="/")
                if killed.exit_code == 0 and not killed.timed_out:
                    return
                time.sleep(0.2)
            raise self.classify_failure(box_id, "A timed-out command could not be stopped")

        trailer = re.compile(re.escape(f"{TRAILER} {call_id} ".encode()) + rb"(\d{1,3})\Z")
        room = len(f"{TRAILER} {call_id} 255")
        # The runner caps each stream at output_limit + 1 inside the box; the
        # trailer comes on top of stdout.
        result = self.exec(box_id, argv, timeout_s=timeout_s, output_limit=output_limit + 1 + room,
                           env=env, workdir="/", user=numeric, on_timeout=kill,
                           drain_s=RUNNER_DRAIN_S)
        found = trailer.search(result.stdout)
        stdout = result.stdout[:found.start()] if found else result.stdout
        truncated = (result.truncated or len(stdout) > output_limit
                     or len(result.stderr) > output_limit)
        if not result.timed_out:
            if result.exit_code == 125 and f"{REFUSED_MARKER} {call_id}".encode() in result.stderr:
                raise self.classify_failure(box_id, "The runner refused the call")
            finished = found is not None and int(found.group(1)) == result.exit_code
            # The box may have died under the call (the agent killed everything,
            # ran out of memory, forked to its pids limit). A trailer can be
            # forged, so runtime-looking text or an exit >= 126 is probed too.
            # A box that no longer runs a helper fails here (classified; only a
            # host cause is ours); on a live box the call is the agent's.
            if (not finished or result.exit_code >= 126
                    or _RUNTIME_TEXT.search(stdout + result.stderr)):
                if not self._alive(box_id):
                    raise self.classify_failure(box_id, "The box died during the call")
        # Runtime-looking text never reaches a signed record, whoever printed it.
        return ExecResult(_scrub(stdout)[:output_limit], _scrub(result.stderr)[:output_limit],
                          result.exit_code, result.timed_out, truncated)

    def _probe(self, box_id: str) -> bool:
        self._probing.active = True  # a probe that fails is never probed again
        try:
            probe = self.exec(box_id, [HELPER, "true"], timeout_s=10, output_limit=1024,
                              user="0", workdir="/")
        except RuntimeOperationError:
            return False
        finally:
            self._probing.active = False
        return probe.exit_code == 0 and not probe.timed_out

    def _alive(self, box_id: str) -> bool:
        """The box runs a helper now; only when it does not, the settle period
        is paid (a box dying under us) before one more reading and probe."""
        try:
            container = self._container(box_id)
        except Exception:
            return False
        if self._probe(box_id):
            return True
        if not self._settled(container):
            return False
        return self._probe(box_id)

    def run_program(self, box_id: str, argv: list[str], *, timeout_s: float, output_limit: int,
                    env: dict[str, str], workdir: str, user: str | None) -> ExecResult:
        """An env hook's program (prepare, extract, grade), run as given through
        RUN_PROGRAM with a fresh nonce. Its stdout without the open marker is
        an InfraFault (ours) whatever the exit code, unless the exec could not
        start (OCI text); exit 0 without the close marker too. Never re-run:
        a program may have side effects."""
        if not argv:
            raise ValueError("A command is required")
        nonce = _new_nonce()
        opened, closed_marker = frame_markers(nonce)
        overhead = len(opened) + len(closed_marker)
        result = self.exec(box_id, [HELPER, "sh", "-c", RUN_PROGRAM, "run", nonce, *argv],
                           timeout_s=timeout_s, output_limit=output_limit + overhead, env=env,
                           workdir=workdir, user=user)
        out = result.stdout
        if out.startswith(opened):
            closed = out.endswith(closed_marker) and len(out) >= overhead
            if closed or result.timed_out or result.truncated or result.exit_code != 0:
                end = len(out) - len(closed_marker) if closed else len(out)
                return replace(result, stdout=out[len(opened):end][:output_limit])
        elif result.timed_out:
            return replace(result, stdout=out[:output_limit])
        elif result.exit_code != 0 and b"OCI runtime" in out + result.stderr:
            if _start_failed(result, opened):  # not re-run: it may have side effects
                fault = self.classify_failure(box_id, f"{argv[0]}: the program could not start",
                                              start_failed=True)
                if isinstance(fault, RuntimeExecFailed):
                    raise fault
            return replace(result, stdout=out[:output_limit])
        raise self._output_lost(box_id, f"{argv[0]}: the program's output was lost")

    def _output_lost(self, box_id: str, what: str, *,
                     start_failed: bool = False) -> RuntimeOperationError:
        """A lost output is classified like any failed operation: on a box that
        still runs it is the agent's (our transport no longer loses output, so
        something in the box killed or starved the helper); on a box the host
        or runtime stopped it is ours."""
        return self.classify_failure(box_id, what, start_failed=start_failed)

    def daemon_reachable(self) -> bool:
        """A cheap Docker API round trip: tells a stuck daemon from a slow box."""
        try:
            self.client.ping()
            return True
        except Exception:
            return False

    def _framed_exec(self, box_id: str, script: str, name: str, args: list[str], what: str, *,
                     fixed: float, timeout_s: float | None, output_limit: int,
                     user: str) -> ExecResult:
        """Run a framed helper (FRAMED_SCRIPTS) with a fresh nonce per run and
        return its output without the markers. A run that exited 0 with a
        marker missing lost its output and is run again (the helpers are
        idempotent): each re-run gets the helper's fixed budget, bounded by the
        caller's deadline. A failed or timed-out re-run, or an output still
        lost after OUTPUT_ATTEMPTS runs, is classified (`_output_lost`)."""
        caller_deadline = None if timeout_s is None else time.monotonic() + timeout_s
        for attempt in range(OUTPUT_ATTEMPTS):
            nonce = _new_nonce()
            opened, closed = frame_markers(nonce)
            overhead = len(opened) + len(closed)
            argv = [HELPER, "sh", "-c", script, name, nonce, *args]
            if attempt == 0:
                result, cut = self._exec_read_only(box_id, argv, what, fixed=fixed,
                                                   timeout_s=timeout_s, opened=opened,
                                                   output_limit=output_limit + overhead,
                                                   workdir="/", user=user)
                self._check_helper_result(box_id, result, what, budget=cut, opened=opened)
            else:
                left = fixed if caller_deadline is None else min(
                    fixed, caller_deadline - time.monotonic())
                if left <= 0:
                    raise self._output_lost(box_id, f"{what}: the output was lost, "
                                                    "no time left to run it again")
                result, _ = self._exec_read_only(
                    box_id, argv, what, fixed=fixed, opened=opened,
                    timeout_s=None if caller_deadline is None else left,
                    output_limit=output_limit + overhead, workdir="/", user=user)
                if result.timed_out or result.exit_code != 0:
                    raise self._output_lost(box_id, f"{what}: the output was lost and running "
                                                    f"it again failed (exit {result.exit_code}, "
                                                    f"timed out {result.timed_out})",
                                            start_failed=_start_failed(result, opened))
            out = result.stdout
            if out.startswith(opened) and (out.endswith(closed) or result.truncated):
                end = len(out) - len(closed) if out.endswith(closed) and len(out) >= overhead \
                    else len(out)
                return replace(result, stdout=out[len(opened):end])
            if result.exit_code != 0:
                return replace(result, stdout=b"")  # the exit code says what happened
            _log.warning("box %s: %s: the helper's output was lost (attempt %d); running it again",
                         box_id, what, attempt + 1)
        raise self._output_lost(box_id, f"{what}: the output was lost {OUTPUT_ATTEMPTS} times")

    def stop_processes(self, box_id: str, *, timeout_s: float) -> None:
        """SIGKILL every process the agent left in its box (STOP_PROCESSES),
        as root with the helper busybox, so nothing runs during extract."""
        result = self.exec(box_id, [HELPER, "sh", "-c", STOP_PROCESSES, "stop"],
                           timeout_s=_bounded(30, timeout_s), output_limit=4096, user="0",
                           workdir="/")
        self._check_helper_result(box_id, result, "Stopping the agent's processes",
                                  budget=_cut_by_budget(30, timeout_s))
        if result.exit_code != 0:
            raise self.classify_failure(box_id, "Stopping the agent's processes")

    def file_kind(self, box_id: str, path: str, *, user: str | None = None,
                  timeout_s: float | None = None) -> str:
        """"file", "other" (exists, not a regular file) or "absent"."""
        result, cut = self._exec_read_only(box_id, [HELPER, "sh", "-c", FILE_KIND, "kind", path],
                                           "Inspecting a file", fixed=30, timeout_s=timeout_s,
                                           output_limit=1024, workdir="/",
                                           user=self._numeric_user(box_id, user))
        self._check_helper_result(box_id, result, "Inspecting a file", budget=cut)
        kinds = {0: "file", 3: "other", 4: "absent"}
        if result.exit_code not in kinds:
            raise self.classify_failure(box_id, "Inspecting a file")
        return kinds[result.exit_code]

    def read_file(self, box_id: str, path: str, max_bytes: int, *, user: str | None = None,
                  timeout_s: float | None = None) -> bytes:
        result = self._framed_exec(box_id, READ_FILE, "read", [path, str(max_bytes + 1)],
                                   "Reading a file", fixed=60,
                                   timeout_s=timeout_s, output_limit=max_bytes + 1,
                                   user=self._numeric_user(box_id, user))
        if result.exit_code == 3:
            raise FileNotFoundError(path)
        if result.exit_code == 4:
            raise IsADirectoryError(path)
        if result.exit_code == 5:
            raise NotRegularFile(f"{path} is not a regular file")
        if result.exit_code != 0:
            raise PermissionError(path)
        if len(result.stdout) > max_bytes:
            raise ValueError(f"{path} exceeds {max_bytes} bytes")
        return result.stdout

    def write_file(self, box_id: str, path: str, data: bytes, *, user: str | None = None,
                   timeout_s: float | None = None) -> None:
        result = self.exec(box_id, [HELPER, "sh", "-c", WRITE_FILE, "write", path],
                           timeout_s=_bounded(120, timeout_s), output_limit=4096, stdin=data, workdir="/",
                           user=self._numeric_user(box_id, user))
        self._check_helper_result(box_id, result, "Writing a file",
                                  budget=_cut_by_budget(120, timeout_s))
        if result.exit_code != 0:
            if b"No space left" in result.stderr:
                raise OSError(errno.ENOSPC, "No space left on device", path)
            raise PermissionError(path)

    def make_dirs(self, box_id: str, path: str, *, user: str | None = None,
                  timeout_s: float | None = None) -> None:
        """`mkdir -p` with the helper busybox."""
        result = self.exec(box_id, [HELPER, "sh", "-c", MAKE_DIRS, "mkdirs", path],
                           timeout_s=_bounded(30, timeout_s), output_limit=4096, workdir="/",
                           user=self._numeric_user(box_id, user))
        self._check_helper_result(box_id, result, "Creating a directory",
                                  budget=_cut_by_budget(30, timeout_s))
        if result.exit_code != 0:
            if b"No space left" in result.stderr:
                raise OSError(errno.ENOSPC, "No space left on device", path)
            raise PermissionError(path)

    def resolve_existing(self, box_id: str, path: str, *, user: str | None = None,
                         timeout_s: float | None = None) -> str:
        """Canonical path of the nearest existing ancestor of `path`, symlinks
        followed (a dangling link resolves to its target). OSError if the
        helper cannot resolve it."""
        result = self._framed_exec(box_id, RESOLVE_EXISTING, "resolve", [path],
                                   "Resolving a path", fixed=30, timeout_s=timeout_s,
                                   output_limit=8192, user=self._numeric_user(box_id, user))
        resolved = result.stdout.decode("utf-8", "replace").strip()
        if result.exit_code != 0 or not resolved.startswith("/"):
            raise OSError(errno.ENOENT, "cannot resolve", path)
        return resolved

    def resolve_links(self, box_id: str, paths: list[str], *, user: str | None = None,
                      timeout_s: float | None = None) -> list[str | None]:
        """Per path (a restored symlink): the canonical path its links lead to, or
        None when nothing exists there (dangling, a loop, unreadable as `user`).
        OSError if the helper fails or its output does not match `paths`."""
        if not paths:
            return []
        result = self._framed_exec(box_id, RESOLVE_LINKS, "links", list(paths),
                                   "Resolving restored links", fixed=60, timeout_s=timeout_s,
                                   output_limit=len(paths) * (PATH_MAX + 2),
                                   user=self._numeric_user(box_id, user))
        records = result.stdout.split(b"\0")
        if (result.exit_code != 0 or result.truncated or len(records) != len(paths) + 1
                or records[-1] or not all(r.endswith(b"\n") for r in records[:-1])):
            raise OSError(errno.EIO, "cannot resolve the restored links")
        resolved: list[str | None] = []
        for record in records[:-1]:
            text = record[:-1].decode("utf-8", "surrogateescape")
            resolved.append(text if text.startswith("/") else None)
        return resolved

    def tar_create(self, box_id: str, members: list[str], max_bytes: int, *,
                   user: str | None = None, timeout_s: float | None = None) -> bytes:
        """A tar of `members` (paths relative to `/`) made by the helper busybox,
        symlinks stored, never followed. ValueError past `max_bytes`, OSError
        when tar fails (a missing or unreadable path)."""
        if not members:
            raise ValueError("nothing to archive")
        result = self._framed_exec(box_id, TAR_CREATE, "archive", list(members),
                                   "Archiving the state", fixed=300, timeout_s=timeout_s,
                                   output_limit=max_bytes, user=self._numeric_user(box_id, user))
        if result.truncated:
            raise ValueError(f"the archive exceeds {max_bytes} bytes")
        if result.exit_code != 0:
            raise OSError(errno.EIO, "tar failed: " + result.stderr[-300:].decode("utf-8", "replace"))
        return result.stdout

    def clear_root(self, box_id: str, root: str, *, user: str | None = None,
                   timeout_s: float | None = None) -> None:
        """`rm -rf` a restore root with the helper busybox (rm never follows
        symlinks below it). InfraFault when the root itself is a symlink."""
        result = self.exec(box_id, [HELPER, "sh", "-c", CLEAR_ROOT, "clear", root],
                           timeout_s=_bounded(120, timeout_s), output_limit=4096, workdir="/",
                           user=self._numeric_user(box_id, user))
        self._check_helper_result(box_id, result, "Clearing a restore root",
                                  budget=_cut_by_budget(120, timeout_s))
        if result.exit_code == 3:
            raise InfraFault(f"The restore root {root} is a symlink in the image")
        if result.exit_code != 0:
            raise OSError(errno.EIO, "rm failed: " + result.stderr[-300:].decode("utf-8", "replace"))

    def remove_restored(self, box_id: str, root: str, *, user: str | None = None,
                        timeout_s: float | None = None) -> None:
        """`rm -rf` a root just restored (a link included, never followed) with the
        helper busybox. OSError when rm fails."""
        result = self.exec(box_id, [HELPER, "sh", "-c", REMOVE_RESTORED, "remove", root],
                           timeout_s=_bounded(120, timeout_s), output_limit=4096, workdir="/",
                           user=self._numeric_user(box_id, user))
        self._check_helper_result(box_id, result, "Removing a refused restore",
                                  budget=_cut_by_budget(120, timeout_s))
        if result.exit_code != 0:
            raise OSError(errno.EIO, "rm failed: " + result.stderr[-300:].decode("utf-8", "replace"))

    def tar_extract(self, box_id: str, data: bytes, dest_root: str, *, user: str | None = None,
                    timeout_s: float | None = None) -> None:
        """Extract a tar (already vetted by the caller) into `dest_root` with
        the helper busybox."""
        result = self.exec(box_id, [HELPER, "sh", "-c", TAR_EXTRACT, "restore", dest_root],
                           timeout_s=_bounded(300, timeout_s), output_limit=4096, stdin=data,
                           workdir="/", user=self._numeric_user(box_id, user))
        self._check_helper_result(box_id, result, "Restoring the state",
                                  budget=_cut_by_budget(300, timeout_s))
        if result.exit_code != 0:
            if b"No space left" in result.stderr:
                raise OSError(errno.ENOSPC, "No space left on device", dest_root)
            raise OSError(errno.EIO, "tar failed: " + result.stderr[-300:].decode("utf-8", "replace"))

    # -- accounting ----------------------------------------------------------

    def cpu_ns(self, box_id: str) -> int:
        """Cumulative CPU of the box's cgroup: every process in the sandbox,
        gVisor's Sentry and gofer included. A box that exists but is no longer
        running reports its last known value (`BoxFault` if never measured):
        a dead box is the agent's doing, never an `InfraFault`."""
        try:
            container = self._container(box_id)
            container.reload()
        except InfraFault:
            raise
        except NotFound as exc:
            raise InfraFault("The box no longer exists") from exc
        except Exception as exc:
            raise InfraFault("Docker is unreachable") from exc
        if container.status != "running":
            return self._dead_box_cpu(box_id)
        try:
            if self.settings.episode_cpu_source == "cgroupfs":
                value = self._cgroup_cpu_ns(container.id)
            else:
                stats = self.client.api.stats(container.id, stream=False, one_shot=True)
                value = int(stats["cpu_stats"]["cpu_usage"]["total_usage"])
        except Exception as exc:
            # A failing read may just mean the box died under us: look again.
            try:
                container.reload()
            except Exception as reload_error:
                raise InfraFault("Docker is unreachable") from reload_error
            if container.status != "running":
                return self._dead_box_cpu(box_id)
            settled = self._settled(container)
            if settled is None:
                raise InfraFault("Docker is unreachable") from exc
            if not settled:
                return self._dead_box_cpu(box_id)
            raise InfraFault("Box CPU usage is unavailable") from exc
        self._cpu_last[box_id] = value
        return value

    def _dead_box_cpu(self, box_id: str) -> int:
        if box_id in self._cpu_last:
            return self._cpu_last[box_id]
        raise BoxFault("The box stopped before its CPU was measured")

    def _cgroup_scope(self, container_id: str) -> Path:
        root = Path(self.settings.episode_cgroup_root)
        for candidate in (root / "system.slice" / f"docker-{container_id}.scope",
                          root / "docker" / container_id):
            if candidate.exists():
                return candidate
        raise OSError("no cgroup v2 scope for the box")

    def _cgroup_cpu_ns(self, container_id: str) -> int:
        for line in (self._cgroup_scope(container_id) / "cpu.stat").read_text().splitlines():
            name, _, value = line.partition(" ")
            if name == "usage_usec":
                return int(value) * 1000
        raise OSError("no usage_usec in the box's cpu.stat")
