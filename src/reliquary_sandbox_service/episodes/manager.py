"""Episode lifecycle: open, sequential idempotent calls, finish, abnormal ends.

Never holds a lock across a Docker call. `self.lock` guards only the capacity
count, the insert of the single-use row and the per-episode lock table, so
opens proceed in parallel. Every method here blocks (Docker, env hooks through
`run_step`) and must run in a worker thread, never on the event loop: the
public entry points refuse to run on a running loop.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import stat
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from reliquary_sandbox.attest import (
    CLOCK_SKEW_S,
    GRADING_GRACE_S,
    OPEN_REFUSAL_DOMAIN,
    STATUS_ABORTED,
    STATUS_BUDGET_EXHAUSTED,
    STATUS_EXPIRED,
    STATUS_GRADED,
    Budgets,
    CallBody,
    CanonicalError,
    FinalBody,
    OpenBody,
    RecordError,
    SessionClaims,
    Signer,
    canonical_json,
    chain_hash,
    public_key_from_b64,
    sha256_hex,
    sign_record,
    token_sha256,
    verify_session_token,
)
from reliquary_sandbox.attest.records import STATUS_BOX_FAILED
from reliquary_sandbox.episode_task import SandboxTask
from reliquary_sandbox.observation import MAX_ARGUMENT_BYTES

from ..settings import transcript_reserve_bytes
from ..task_runtime import BoxFault, BoxSpec, InfraFault
from .grader import RETRY_MARGIN_S, Grader, run_step
from .registry import TaskRegistry, TaskResolutionError, task_fingerprint
from .resources import (
    cleanup_reservation,
    episode_reservation,
    grading_pool,
    in_use,
    resources_enabled,
    totals,
)
from .store import (
    CLOSED,
    FINISHING,
    OPENING,
    RUNNING,
    EpisodeStore,
    dump_record,
    transcript_frame_bytes,
)
from .tools import TOOLS_VERSION, run_tool

_log = logging.getLogger(__name__)

_NO_KEY = "0" * 64  # never a sha256 we accept (checked by identity)
PREPARE_SLACK_S = 60
"""`run_step` may return a few seconds past a step's timeout."""
_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
BOX_GRACE_S = 600
"""A box outlives its episode's deadline by this much (at least; more when the
task's extract may take longer, see `_box_grace_s`), then dies on its own
(`sleep` as PID 1), so a dead controller cannot leave boxes running forever."""
RETRY_AFTER_S = "5"
_INT_BOUND = 2**63
OPEN_FAILED = "the task box could not be prepared: token spent; request a new token"
OPEN_REFUSAL_HEADER = "X-Reliquary-Open-Refusal"
"""The signed receipt of a refusal of a fresh token that is not the miner's doing (base64 of its
canonical JSON). The token is spent with it. The miner forwards it in its close; validators then do not
charge the miner. Presenting the same token again answers the same receipt (a lost response loses nothing)."""
REFUSED = "refused"
"""Local final status of a token refused with a signed receipt (a tombstone row: never opened)."""
WINDOW_MARGIN_S = 60
"""Grading must end this long before `expires_at + GRADING_GRACE_S`, the last
time the verifier accepts for the final record."""


SECRET_FILE = "episode-secret"
"""32 random bytes in the state directory (0600), created once: episode keys
are HMAC-SHA256(secret, token_sha256), so a re-sent token gets its key back."""
ABORTED_UNSIGNED = "aborted_unsigned"
"""Local final status of an episode whose record 0 was signed with another
machine key: nothing more is signed for it (a final under the new key would
not verify), it is closed and its slot freed. Rotate keys with no live episode."""
KEY_ROTATED = "the machine key changed: the episode is void and unsigned"


def load_machine_secret(state_dir: Path) -> bytes:
    """The machine secret, created on first use (atomically: written to a
    private temp file, then hard-linked into place, so a crash never leaves a
    short file and two processes agree on one secret)."""
    path = Path(state_dir) / SECRET_FILE
    if not path.exists():
        partial = path.with_name(f".{SECRET_FILE}.{os.getpid()}.{secrets.token_hex(6)}")
        descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, secrets.token_bytes(32))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(partial, path)
        except FileExistsError:
            pass
        finally:
            partial.unlink(missing_ok=True)
    # Same checks as the machine key: one descriptor, no symlink, ours, private.
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise RuntimeError(f"{path} must be a regular file, not a symlink") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"{path} must be a regular file")
        if info.st_uid != os.geteuid():
            raise RuntimeError(f"{path} must belong to the owner of this process")
        if info.st_mode & 0o077:
            raise RuntimeError(f"{path} must be readable by its owner only (0600)")
        secret = os.read(descriptor, 33)
    finally:
        os.close(descriptor)
    if len(secret) != 32:
        raise RuntimeError(f"{path} must hold 32 bytes")
    return secret


def grading_fits_the_window(grading_timeout_s: float) -> bool:
    """Extract plus one grading attempt (and the retry margin) fit the verification
    window even for a `finish` at the token's expiry."""
    return 2 * grading_timeout_s + RETRY_MARGIN_S + WINDOW_MARGIN_S <= GRADING_GRACE_S


def _off_the_event_loop() -> None:
    """Docker calls and env hooks block: running them on the event loop would
    stall every other request. The HTTP layer calls in from a worker thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError("the episode manager blocks; call it from a worker thread, "
                       "never on the event loop")


def _refusal_headers(receipt: str | None) -> dict[str, str]:
    return {} if receipt is None else {OPEN_REFUSAL_HEADER: receipt}


class _Expired(Exception):
    """The session token expired while the box was being prepared."""


class TaskChanged(InfraFault):
    """After a restart the env resolves the episode's task differently (another
    package version, image, users, limits...): ours, the episode is aborted
    with reason `task_changed` rather than graded by an unapproved task."""


TASK_CHANGED = "task_changed"
TRANSCRIPT_BYTES = "transcript_bytes"


class EpisodeManager:
    def __init__(self, settings: Any, boxes: Any, store: Any, signer: Signer,
                 registry: TaskRegistry, clock=time.time):
        self.settings = settings
        self.boxes = boxes
        self.store = EpisodeStore(store)
        self.signer = signer
        self.registry = registry
        self.grader = Grader(settings, boxes, clock=clock)
        self.clock = clock
        self.validator_keys = {key_id: public_key_from_b64(value)
                               for key_id, value in settings.validator_public_keys.items()}
        self.lock = threading.Lock()
        self.episode_locks: dict[str, threading.Lock] = {}
        self._reading: set[str] = set()
        """Episodes whose transcript is being read (one read at a time each)."""
        self._tasks: dict[str, SandboxTask] = {}
        """Resolved tasks of live episodes (resolved again after a restart)."""
        self.secret = load_machine_secret(settings.state_dir)
        self.host_facts: dict[str, Any] = {}
        """The last successful `boxes.preflight()` (runsc and busybox versions),
        reported in the signed capacity report; set by the factory and readiness."""
        self.grading_pool = grading_pool(self.settings)
        self.grader.pool = self.grading_pool          # each grading attempt runs inside it
        if self.grading_pool is not None:
            for episode in self.store.with_cleanup():
                for box_id in episode["cleanup"]:
                    if box_id != episode["box_id"]:
                        self.grader.retain_cleanup(
                            box_id, cleanup_reservation(episode, settings, self.grading_pool.total),
                            episode["claims"]["hotkey"], restore=True)

    # -- helpers -------------------------------------------------------------

    def _now(self) -> int:
        return int(self.clock())

    def _at(self, episode: dict[str, Any]) -> int:
        """A record time that never goes backwards within an episode, even if
        the host clock does (the verifier refuses a decreasing `at`)."""
        at = max(self._now(), int(episode.get("last_at") or 0))
        episode["last_at"] = at
        return at

    def _sign(self, body) -> dict[str, Any]:
        """Every record of every episode is signed with this machine's one key."""
        return sign_record(self.signer, self.settings.machine_id, body)

    def _episode_key(self, digest: str) -> str:
        """The bearer key of the episode opened by the token whose sha256 is `digest`."""
        return hmac.new(self.secret, digest.encode("ascii"), hashlib.sha256).hexdigest()

    def _key_changed(self, episode: dict[str, Any]) -> bool:
        """Record 0 was signed with another machine key than the one loaded now."""
        key_id = episode.get("key_id")
        if key_id is None:
            first = self.store.record(episode["id"], 0)
            key_id = first["key_id"] if first else None
        return key_id is not None and key_id != self.signer.key_id

    def _void_unsigned(self, episode: dict[str, Any]) -> None:
        """Close an episode whose key changed without signing anything: free its
        slot and boxes; the miner sees `aborted_unsigned` and no final."""
        _log.error("episode %s: record 0 was signed with machine key %r, this process "
                   "signs with %r: closed unsigned (rotate keys with no live episode)",
                   episode["id"], episode.get("key_id"), self.signer.key_id)
        episode.update(status=CLOSED, final_status=ABORTED_UNSIGNED, pending=None,
                       closed_at=self.clock())
        self._remove_boxes(episode, [episode["box_id"]])
        self.store.commit(episode)
        self._tasks.pop(episode["id"], None)

    @contextmanager
    def _exclusive(self, episode_id: str):
        """One request at a time per episode: a concurrent one gets 409 busy
        instead of waiting, so an episode's calls are strictly sequential and
        its chain is well defined."""
        with self.lock:  # looked up and taken together: `_prune_locks` cannot interleave
            lock = self.episode_locks.setdefault(episode_id, threading.Lock())
            acquired = lock.acquire(blocking=False)
        if not acquired:
            raise HTTPException(409, "episode is busy")
        try:
            yield
        finally:
            lock.release()

    def _authorized(self, episode_id: str, key: Any) -> dict[str, Any]:
        """The episode, if `key` is its bearer key (compared in constant time)."""
        episode = self.store.get(episode_id) if isinstance(episode_id, str) else None
        # An unknown (or still opening) episode and a wrong key get the same answer, after the
        # same comparison, so the endpoint does not reveal which session ids exist.
        expected = episode["key_sha256"] if episode and episode["status"] != OPENING else _NO_KEY
        raw = key.encode("utf-8", "surrogatepass") if isinstance(key, str) else b""
        supplied = hashlib.sha256(raw).hexdigest()
        if not hmac.compare_digest(supplied, expected) or expected is _NO_KEY:
            raise HTTPException(401, "valid episode key required")
        return episode

    def _task(self, episode: dict[str, Any]) -> SandboxTask:
        """The episode's task: cached since open, resolved again after a restart
        and refused (TaskChanged) unless it is exactly the task opened."""
        task = self._tasks.get(episode["id"])
        if task is None:
            claims = episode["claims"]
            try:
                resolved = self.registry.resolve(claims["env"], claims["split"], claims["index"])
            except TaskResolutionError as exc:
                raise TaskChanged("the env no longer declares this task") from exc
            if task_fingerprint(resolved) != episode.get("task_fingerprint"):
                _log.error("episode %s: the env now declares another task (%s)",
                           episode["id"], resolved.env_package)
                raise TaskChanged("the env now declares another task")
            task = resolved.task
            self._tasks[episode["id"]] = task
        return task

    def _charge_cpu(self, episode: dict[str, Any]) -> int:
        """CPU (ms) the box used since the last charge, background processes
        included. Never negative, so the running total never decreases; the
        sub-millisecond remainder carries over to the next charge."""
        now_ns = self.boxes.cpu_ns(episode["box_id"])
        mark = episode.get("cpu_mark_ns")
        if mark is None:  # unreachable for an opened episode; charge nothing rather than all
            episode["cpu_mark_ns"] = now_ns
            return 0
        cpu_ms = max(0, now_ns - mark) // 1_000_000
        episode["cpu_mark_ns"] = mark + cpu_ms * 1_000_000
        return cpu_ms

    @staticmethod
    def _over_cpu(episode: dict[str, Any]) -> bool:
        return episode["cpu_calls_ms"] > episode["claims"]["budgets"]["cpu_s"] * 1000

    @staticmethod
    def _box_grace_s(task: SandboxTask) -> int:
        """How long the agent's box outlives the deadline: `finish` may arrive
        just before it, and extract then runs on that box for up to the task's
        grading timeout (a box dying under extract would be blamed on the agent)."""
        return max(BOX_GRACE_S, math.ceil(task.grading_timeout_s) + PREPARE_SLACK_S)

    def _close(self, episode: dict[str, Any], status: str, *, reason: str | None,
               outcome: Any = None, state: bytes | None = None) -> dict[str, Any] | None:
        """Sign the final record, persist it with the closed episode, remove the
        agent box (a failed removal is left in `cleanup` for the reaper).
        `outcome` is the grader's GradeOutcome when there was grading; `state`
        the state to return on every `finish` (published tasks only). None
        (closed unsigned) when the machine key changed since record 0."""
        if self._key_changed(episode):
            self._remove_boxes(episode, list((outcome.cleanup if outcome is not None else ())))
            self._void_unsigned(episode)
            return None
        at = self._at(episode)
        if at >= episode["claims"]["expires_at"] + GRADING_GRACE_S and status == STATUS_GRADED:
            # Unreachable when grading honours its deadline; a final past the window
            # would not verify anyway, so it must not pose as an agent verdict.
            _log.error("episode %s: grading ended past the verification window", episode["id"])
            status, reason, state = STATUS_ABORTED, "grading outlasted the verification window", None
        graded = status == STATUS_GRADED and outcome is not None
        cpu_total = episode["cpu_calls_ms"] + (outcome.cpu_ms if outcome is not None else 0)
        body = FinalBody(
            i=episode["next_i"], session_id=episode["id"], status=status,
            reward=float(outcome.reward) if graded else None,
            grading=outcome.facts if graded else None,
            state_sha256=outcome.state_sha256 if graded else None,
            cpu_total_ms=cpu_total, reason=reason, at=at, prev=episode["head"])
        record = self._sign(body)
        episode.update(status=CLOSED, final_status=status, head=chain_hash(body),
                       next_i=episode["next_i"] + 1, pending=None, closed_at=self.clock())
        boxes = [episode["box_id"], *(outcome.cleanup if outcome is not None else ())]
        self._remove_boxes(episode, boxes)
        self.store.commit(episode, record, state)
        self._tasks.pop(episode["id"], None)
        return record

    def _final(self, episode_id: str) -> dict[str, Any] | None:
        last = self.store.last_record(episode_id)
        return last if last is not None and last["body"]["kind"] == "final" else None

    def _over_transcript(self, episode: dict[str, Any]) -> bool:
        """Whether one more worst-case call and the final might not fit the cap any more
        (`transcript_bytes` counts the transcript as validators receive it)."""
        reserve = transcript_reserve_bytes(self.settings.episode_output_cap_bytes)
        return (episode.get("transcript_bytes", 0) + reserve
                > self.settings.episode_max_transcript_bytes)

    def _closed_error(self, episode_id: str, message: str, status_code: int = 409):
        return HTTPException(status_code, {"message": message, "final": self._final(episode_id)})

    def _nano_cpus(self, task: SandboxTask) -> int:
        """The task's CPUs, never more than the machine gives a box (`episode_nano_cpus`)."""
        machine = self.settings.episode_nano_cpus
        if task.limits.cpus is None:
            return machine
        # Clamped before rounding: a huge request overflows to inf, which round() refuses.
        return max(100_000_000, round(min(machine, task.limits.cpus * 1_000_000_000)))

    def _spec(self, task: SandboxTask, budgets: Budgets, lifetime_s: int) -> BoxSpec:
        return BoxSpec(image=task.image, workdir=task.workdir, env=dict(task.env), user=task.user,
                       memory_bytes=budgets.memory_bytes, nano_cpus=self._nano_cpus(task),
                       pids=budgets.pids, disk_bytes=budgets.disk_bytes, lifetime_s=lifetime_s)

    def _remove_boxes(self, episode: dict[str, Any], box_ids: list[str]) -> None:
        """Remove boxes now; the ones that could not be removed go to `cleanup` (the reaper)."""
        pending = set(episode.get("cleanup", ()))
        for box_id in box_ids:
            try:
                self.boxes.remove(box_id)
            except Exception:
                _log.warning("could not remove box %s", box_id, exc_info=True)
                pending.add(box_id)
            else:
                pending.discard(box_id)
                self.grader.release_cleanup(box_id)
        episode["cleanup"] = sorted(pending)

    # -- open ----------------------------------------------------------------

    def open(self, token: Any) -> dict[str, Any]:
        """Open the token's episode. Idempotent per token: presenting it again
        returns record 0 and the same key while the episode runs (the key is
        HMAC(machine secret, token_sha256), never stored), 409 busy while it
        opens, 409 with the signed final once closed."""
        _off_the_event_loop()
        claims = self._admissible(token)
        digest = token_sha256(token)
        existing = self.store.get(claims.session_id)
        if existing is not None:
            return self._reopened(existing, digest)
        now = self._now()
        window = getattr(self.settings, "episode_present_window_s", None)
        if window is not None and now > int(claims.issued_at) + int(window):
            # A token held unpresented past its present window never opens:
            # unsigned and unspent, like a late token, so it cannot squat a place later.
            raise HTTPException(401, "session token presented after its present window")
        try:
            resolved, unresolvable = self.registry.resolve(claims.env, claims.split, claims.index), None
        except TaskResolutionError as exc:
            resolved, unresolvable = None, str(exc)
        prepare_s = math.ceil(resolved.task.prepare_timeout_s) if resolved is not None else 0
        if now + prepare_s + PREPARE_SLACK_S >= claims.expires_at:
            # No time left to prepare a box. Unsigned and unspent, and checked before any
            # signed refusal: holding a token until its last seconds is the miner's doing and buys nothing.
            raise HTTPException(401, "session token too close to its expiry to prepare a box")
        # From here every refusal is of a fresh, correctly addressed token and not the miner's doing:
        # the token is spent and the refusal signed.
        problem = self._claims_problem(claims)
        if problem is not None:
            raise self._refuse_signed(claims, digest, token, 422, problem, "task_refused")
        if resolved is None:
            raise self._refuse_signed(claims, digest, token, 422, unresolvable, "task_refused")
        task = resolved.task
        if task.image != claims.image:
            raise self._refuse_signed(claims, digest, token, 422,
                                      "the token's image is not this task's image", "task_refused")
        if task.image not in self.settings.episode_images:
            raise self._refuse_signed(claims, digest, token, 422, "image not approved on this machine",
                                      "task_refused")
        if not grading_fits_the_window(task.grading_timeout_s):
            raise self._refuse_signed(
                claims, digest, token, 422,
                "the task's grading timeout does not fit the verification window", "task_refused")
        short = task.limits.shortfall(claims.budgets)
        if short:
            raise self._refuse_signed(
                claims, digest, token, 422,
                f"session budgets are below the task's needs: {', '.join(short)}", "task_refused")
        if not self.boxes.image_present(task.image):   # a set lookup in the image inventory
            raise self._refuse_signed(claims, digest, token, 409, "image not present on this machine",
                                      "task_refused")
        accounted = resources_enabled(self.settings)
        if (accounted and task.limits.cpus is not None
                and task.limits.cpus * 1_000_000_000 > self.settings.episode_nano_cpus):
            # Never a smaller box than the task declares (its tests would time out as the agent's fault).
            raise self._refuse_signed(
                claims, digest, token, 422,
                "the task needs more CPUs per box than this machine gives", "task_refused")
        reserve = episode_reservation(self._nano_cpus(task), claims.budgets.memory_bytes)
        if self.grading_pool is not None and not reserve.within(self.grading_pool.total):
            raise self._refuse_signed(
                claims, digest, token, 422,
                "the task needs more CPU or memory than this machine's grading pool has", "task_refused")
        if accounted and not reserve.within(totals(self.settings)):
            raise self._refuse_signed(
                claims, digest, token, 422,
                "the task needs more CPU or memory than this machine's episodes have", "task_refused")
        key = self._episode_key(digest)
        # The box must outlive any deadline this open can set (see `_start`).
        latest_deadline = min(claims.expires_at,
                              now + math.ceil(task.prepare_timeout_s) + PREPARE_SLACK_S
                              + claims.budgets.wall_s)
        episode = {
            "id": claims.session_id, "token": dict(token), "token_sha256": digest,
            "claims": claims.to_dict(), "status": OPENING,
            "key_sha256": hashlib.sha256(key.encode()).hexdigest(),
            "box_id": claims.session_id, "opened_at": now, "deadline": latest_deadline,
            "next_i": 0,
            "head": None, "calls": 0, "last": None, "cpu_calls_ms": 0, "cpu_mark_ns": None,
            "pending": None, "last_at": now, "publish_state": task.publish_state,
            "env_package": resolved.env_package, "box_user": None, "box_user_name": task.user,
            "task_fingerprint": task_fingerprint(resolved), "transcript_bytes": 0,
            "reserve": reserve.to_dict(),
            # Past this an OPENING row whose lock is free has no opener (see `reap`).
            "open_by": now + math.ceil(task.prepare_timeout_s) + PREPARE_SLACK_S,
        }
        with self.lock:
            active = self.store.reserving(include_grading=self.grading_pool is None)
            if len(active) >= self.settings.episode_capacity:
                raise self._refuse_signed(claims, digest, token, 503, "no free episode slot", "no_free_slot",
                                          {"Retry-After": RETRY_AFTER_S})
            if accounted and not (in_use(active, self.settings) + reserve).within(totals(self.settings)):
                raise self._refuse_signed(claims, digest, token, 503, "no free resources",
                                          "no_free_resources", {"Retry-After": RETRY_AFTER_S})
            inserted = self.store.insert(episode)
            if inserted:  # held until the open ends, taken with the insert: no gap
                opening = self.episode_locks.setdefault(episode["id"], threading.Lock())
                opening.acquire(blocking=False)
        if not inserted:  # the same token (or session id) raced this open
            return self._reopened(self.store.get(claims.session_id), digest)
        # From here the token is spent: any failure burns it (the validator reissues)
        # and must not leave the row OPENING, which would hold a slot forever.
        try:
            try:
                record = self._start(episode, task, claims, resolved.env_package,
                                     latest_deadline - now + self._box_grace_s(task))
            except Exception as exc:
                _log.warning("episode %s could not be opened: %s", episode["id"], exc,
                             exc_info=not isinstance(exc, (InfraFault, _Expired)))
                # A preparation that outlasted the token is not signed (a late token).
                # The token is already spent (the open's insert); the receipt is kept on the row so a
                # resend of the token answers the same one.
                receipt = None
                if not isinstance(exc, _Expired):
                    try:
                        receipt = self._open_refusal(claims.session_id, digest, "open_failed")
                    except Exception:
                        _log.exception("episode %s: the failed open could not be signed", episode["id"])
                self._open_failed(episode, receipt)
                raise HTTPException(503, OPEN_FAILED, headers=_refusal_headers(receipt)) from None
            self._tasks[episode["id"]] = task
        finally:
            opening.release()
        return {"episode_id": episode["id"], "episode_key": key, "record": record}

    def _reopened(self, existing: dict[str, Any] | None, digest: str) -> dict[str, Any]:
        """A token presented again. Only the very token that opened the episode
        learns anything; its key is derived again, never read back."""
        if existing is None or not hmac.compare_digest(existing["token_sha256"], digest):
            raise HTTPException(409, "session token already used")
        key = self._episode_key(digest)
        if existing["status"] == OPENING:
            raise HTTPException(409, "episode is busy: it is still opening; retry")
        if existing["status"] == CLOSED:
            error = self._closed_error(existing["id"], "session token already used")
            if existing.get("open_refusal"):
                # The very token a receipt was signed for gets the same receipt again (its first answer
                # may have been lost); nothing new is signed, and the token stays spent.
                error.headers = _refusal_headers(existing["open_refusal"])
            raise error
        if not hmac.compare_digest(hashlib.sha256(key.encode()).hexdigest(),
                                   existing["key_sha256"]):
            raise HTTPException(409, "session token already used")
        return {"episode_id": existing["id"], "episode_key": key,
                "record": self.store.record(existing["id"], 0)}

    def _start(self, episode: dict[str, Any], task: SandboxTask, claims: SessionClaims,
               env_package: str, lifetime_s: int) -> dict[str, Any]:
        """Prepare the box and sign record 0. The agent's wall time starts when
        preparation ends: `deadline = min(prepared + wall_s, expires_at)`."""
        setup_output = self._prepare(episode, task, claims.budgets, lifetime_s)
        prepared = self._now()
        if prepared >= claims.expires_at:
            raise _Expired("preparation outlasted the session token")
        # Capped by the box lifetime chosen before preparation (only reachable
        # when `prepare` overran its own timeout).
        episode["deadline"] = min(prepared + claims.budgets.wall_s, claims.expires_at,
                                  episode["deadline"])
        # Record 0 carries the time preparation ended: the deadline's own reading.
        at = max(prepared, int(episode.get("last_at") or 0))
        episode["last_at"] = at
        body = OpenBody(i=0, session_id=claims.session_id, machine_id=self.settings.machine_id,
                        token_sha256=episode["token_sha256"], image=claims.image, env=claims.env,
                        env_package=env_package, tools_version=TOOLS_VERSION,
                        tools=tuple(sorted(set(task.tools))),
                        setup_output_sha256=sha256_hex(setup_output.encode("utf-8", "replace")),
                        budgets=claims.budgets, at=at)
        record = self._sign(body)
        episode.update(status=RUNNING, next_i=1, head=chain_hash(body), key_id=record["key_id"],
                       transcript_bytes=transcript_frame_bytes(episode["token"])
                       + len(dump_record(record).encode("utf-8")))
        self.store.commit(episode, record)
        return record

    def _open_failed(self, episode: dict[str, Any], receipt: str | None = None) -> None:
        """Close a failed open: no record 0, token burnt, box removed, slot freed. `receipt` is its
        signed open refusal (the header value), kept for a resend of the token."""
        episode.update(status=CLOSED, final_status="open_failed", closed_at=self.clock(),
                       next_i=0, head=None)
        if receipt is not None:
            episode["open_refusal"] = receipt
        self._remove_boxes(episode, [episode["box_id"]])
        try:
            self.store.commit(episode)
        except Exception:
            _log.exception("episode %s: could not record the failed open", episode["id"])

    def _admissible(self, token: Any) -> SessionClaims:
        """The token's claims, if this machine may serve them now. Cheap: no Docker, no env."""
        try:
            claims = verify_session_token(token, self.validator_keys)
        except RecordError:
            raise HTTPException(401, "invalid session token") from None
        now = self._now()
        if claims.machine_id != self.settings.machine_id:
            raise HTTPException(403, "session token is for another machine")
        # A validator clock up to CLOCK_SKEW_S ahead of ours is tolerated.
        if not claims.issued_at - CLOCK_SKEW_S <= now < claims.expires_at:
            raise HTTPException(401, "session token expired or not yet valid")
        if not _SESSION_ID.fullmatch(claims.session_id):
            raise HTTPException(422, "invalid session id")
        return claims

    # -- signed open refusals ------------------------------------------------

    def _open_refusal(self, session_id: str, token_sha256: str, reason: str) -> str:
        """A receipt, signed with the machine key, that this machine refused (or failed) this very token
        for `reason`: the value of the `OPEN_REFUSAL_HEADER` (base64 of the receipt's canonical JSON)."""
        document = {"kind": "open_refusal", "machine_id": self.settings.machine_id,
                    "key_id": self.signer.key_id, "session_id": session_id, "token_sha256": token_sha256,
                    "reason": reason, "at": self._now()}
        receipt = {"document": document, "signature": self.signer.sign(OPEN_REFUSAL_DOMAIN, document)}
        return base64.b64encode(canonical_json(receipt)).decode("ascii")

    def _spend(self, claims: SessionClaims, digest: str, token: Any, receipt: str) -> bool:
        """Spend a refused token: a closed tombstone row under its session id, so neither
        this token nor its session id ever opens later (`_reopened` answers 409, with `receipt` again
        for this very token). False when the id or token was used meanwhile (a concurrent open of
        it): then nothing may be signed, its box may be running."""
        now = self._now()
        return self.store.insert({
            "id": claims.session_id, "token": dict(token), "token_sha256": digest,
            "claims": claims.to_dict(), "status": CLOSED, "final_status": REFUSED,
            "key_sha256": _NO_KEY, "opened_at": now, "closed_at": now, "deadline": now,
            "next_i": 0, "head": None, "box_id": claims.session_id, "cleanup": [],
            "open_refusal": receipt})

    def _refuse_signed(self, claims: SessionClaims, digest: str, token: Any, status: int, detail: Any,
                       reason: str, headers: dict[str, str] | None = None) -> HTTPException:
        """A refusal of a fresh, correctly addressed token that is not the miner's doing: the token is
        spent and the refusal signed. When the token cannot be spent (another open of the same token or
        session id got there first), the refusal is unsigned: retry, the token's own row answers."""
        receipt = self._open_refusal(claims.session_id, digest, reason)
        if not self._spend(claims, digest, token, receipt):
            return HTTPException(409, "episode is busy: it is still opening; retry")
        return HTTPException(status, detail, headers={**(headers or {}), **_refusal_headers(receipt)})

    def _claims_problem(self, claims: SessionClaims) -> str | None:
        """What this machine cannot serve in the validator's claims, regardless of the task."""
        # The single-use row is pruned `episode_retention_days` after the close, and only
        # once the token expired; a token valid longer than that is refused outright.
        if claims.expires_at - claims.issued_at > self.settings.episode_retention_days * 86400:
            return "session token validity exceeds this machine's retention"
        budgets, settings = claims.budgets, self.settings
        if (budgets.memory_bytes > settings.episode_max_memory_bytes
                or budgets.pids > settings.episode_max_pids
                or budgets.disk_bytes > settings.episode_max_disk_bytes
                or budgets.wall_s > settings.episode_max_wall_s
                or budgets.max_calls > settings.episode_max_calls
                or budgets.cpu_s > settings.episode_max_cpu_s
                or budgets.per_call_timeout_s > settings.episode_max_call_timeout_s):
            return "session budgets exceed this machine's limits"
        return None

    def _prepare(self, episode: dict[str, Any], task: SandboxTask, budgets: Budgets,
                 lifetime_s: int) -> str:
        """Create the agent's box, persist its resolved user, run the env's `prepare`."""
        box_id = episode["box_id"]
        self.boxes.create(box_id, episode["id"], self._spec(task, budgets, lifetime_s))
        uid_gid = self.boxes.box_user(box_id)
        if uid_gid is None:
            raise InfraFault("the box user was not resolved at creation")
        # Persisted before any agent code runs; restore_user() after a restart.
        episode["box_user"] = uid_gid
        self.store.commit(episode)
        runtime = self.grader.runtime(box_id, task, task.prepare_timeout_s)
        setup_output = run_step(lambda: task.prepare(runtime), task.prepare_timeout_s)
        if not isinstance(setup_output, str):
            raise TypeError("prepare must return its output as text")
        # The CPU the agent is charged for starts after our preparation.
        episode["cpu_mark_ns"] = self.boxes.cpu_ns(box_id)
        return setup_output

    # -- calls ---------------------------------------------------------------

    @staticmethod
    def _follows(last: list[int] | None, turn: int, k: int) -> bool:
        """The next call is the next `k` in the same turn, or `k = 0` in a later turn."""
        if turn < 0 or k < 0:
            return False
        if last is None:
            return k == 0
        if turn == last[0]:
            return k == last[1] + 1
        return turn > last[0] and k == 0

    def call(self, episode_id: str, key: str, turn: int, k: int, tool: str,
             arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one tool call and return its signed record (plus the final record
        when this call ended the episode). Re-sending a recorded `(turn, k)`
        returns the stored record without running anything.

        Arguments must have a canonical JSON form (no lone surrogates, integers
        in [-2**63, 2**63), finite floats) or the call is refused with 422 and
        nothing is recorded. The miner runtime must refuse the same
        arguments the same way before sending them, or its transcript diverges
        from what the model produced."""
        _off_the_event_loop()
        self._authorized(episode_id, key)
        if not isinstance(arguments, dict) or not isinstance(tool, str):
            raise HTTPException(422, "a tool name and an arguments object are required")
        if (type(turn) is not int or type(k) is not int
                or not 0 <= turn < _INT_BOUND or not 0 <= k < _INT_BOUND):
            raise HTTPException(422, "turn and k must be integers in [0, 2**63)")
        try:
            canonical_json(tool)
            size = len(canonical_json(arguments))
        except CanonicalError:
            raise HTTPException(422, "arguments have no canonical JSON form") from None
        if size > MAX_ARGUMENT_BYTES:
            raise HTTPException(413, "arguments exceed the configured limit")
        with self._exclusive(episode_id):
            episode = self.store.get(episode_id)
            stored = self.store.call_record(episode_id, turn, k)
            if stored is not None:
                body = stored["body"]
                if body["tool"] != tool or canonical_json(body["arguments"]) != canonical_json(arguments):
                    raise HTTPException(409, "this (turn, k) was recorded with a different call")
                final = self._final(episode_id) if episode["status"] == CLOSED else None
                return {"record": stored, "final": final}
            if episode["status"] != RUNNING:
                raise self._closed_error(episode_id, "episode is closed")
            if self._key_changed(episode):
                self._void_unsigned(episode)
                raise self._closed_error(episode_id, KEY_ROTATED)
            if episode.get("pending") is not None:
                # A call ran (or may have) without its record being kept: what the box
                # now holds is unknown to the transcript, so nothing more may run.
                _log.error("episode %s: call %s executed but was never recorded",
                           episode_id, episode["pending"])
                self._close(episode, STATUS_ABORTED, reason="a call was not recorded")
                raise self._closed_error(episode_id, "a previous call was not recorded", 503)
            if not self._follows(episode["last"], turn, k):
                raise HTTPException(
                    409, "calls must be sequential: next k in this turn, or k=0 in a later turn")
            return self._run_call(episode, turn, k, tool, arguments)

    def _run_call(self, episode: dict[str, Any], turn: int, k: int, tool: str,
                  arguments: dict[str, Any]) -> dict[str, Any]:
        episode_id = episode["id"]
        budgets = Budgets.from_dict(episode["claims"]["budgets"])
        now = self.clock()
        if now >= episode["deadline"]:
            self._close(episode, STATUS_EXPIRED, reason="deadline passed")
            raise self._closed_error(episode_id, "episode deadline passed")
        if self._over_cpu(episode):  # a crash came between a call's commit and this close
            self._close(episode, STATUS_BUDGET_EXHAUSTED, reason="cpu_s")
            raise self._closed_error(episode_id, "cpu budget exhausted")
        if self._over_transcript(episode):  # likewise
            self._close(episode, STATUS_BUDGET_EXHAUSTED, reason=TRANSCRIPT_BYTES)
            raise self._closed_error(episode_id, "transcript budget exhausted")
        if episode["calls"] >= budgets.max_calls:
            self._close(episode, STATUS_BUDGET_EXHAUSTED, reason="max_calls")
            raise self._closed_error(episode_id, "call budget exhausted")
        timeout = max(0.001, min(budgets.per_call_timeout_s, episode["deadline"] - now))
        episode["pending"] = {"turn": turn, "k": k}
        self.store.commit(episode)
        started = time.monotonic()
        try:
            task = self._task(episode)
            outcome = run_tool(self.boxes, episode["box_id"], task, tool, arguments,
                               timeout_s=timeout, output_cap=self.settings.episode_output_cap_bytes,
                               edit_max_bytes=self.settings.episode_edit_max_bytes)
            cpu_ms = self._charge_cpu(episode)
        except BoxFault as exc:  # before InfraFault: both are RuntimeOperationErrors
            _log.info("episode %s: the agent's box failed: %s", episode_id, exc)
            self._close(episode, STATUS_BOX_FAILED, reason="the agent's box is unusable")
            raise self._closed_error(episode_id, "the agent's box is unusable") from None
        except Exception as exc:
            # InfraFault is ours; anything else (a plain OSError, a ValueError)
            # escaping a tool is our bug. Either way the episode is void.
            if isinstance(exc, InfraFault):
                _log.warning("episode %s: infrastructure failure: %s", episode_id, exc)
            else:
                _log.exception("episode %s: unexpected error running a tool", episode_id)
            reason = TASK_CHANGED if isinstance(exc, TaskChanged) else "sandbox failure"
            self._close(episode, STATUS_ABORTED, reason=reason)
            raise self._closed_error(episode_id, reason, 503) from None
        try:
            record = self._record_call(episode, turn, k, tool, arguments, outcome, cpu_ms, started)
        except Exception:
            # The call ran but its record was not kept: void the episode from the
            # last persisted state (its chain head is the last kept record).
            _log.exception("episode %s: could not record a call", episode_id)
            persisted = self.store.get(episode_id) or episode
            self._close(persisted, STATUS_ABORTED, reason="a call was not recorded")
            raise self._closed_error(episode_id, "the call could not be recorded", 503) from None
        final = None
        if self._over_cpu(episode):
            final = self._close(episode, STATUS_BUDGET_EXHAUSTED, reason="cpu_s")
        elif self._over_transcript(episode):
            final = self._close(episode, STATUS_BUDGET_EXHAUSTED, reason=TRANSCRIPT_BYTES)
        elif self.clock() >= episode["deadline"]:
            final = self._close(episode, STATUS_EXPIRED, reason="deadline passed")
        return {"record": record, "final": final}

    def _record_call(self, episode: dict[str, Any], turn: int, k: int, tool: str,
                     arguments: dict[str, Any], outcome: Any, cpu_ms: int,
                     started: float) -> dict[str, Any]:
        body = CallBody(i=episode["next_i"], session_id=episode["id"], turn=turn, k=k, tool=tool,
                        arguments=arguments, output=outcome.output, exit_code=outcome.exit_code,
                        truncated=outcome.truncated, timed_out=outcome.timed_out, cpu_ms=cpu_ms,
                        wall_ms=int((time.monotonic() - started) * 1000), at=self._at(episode),
                        prev=episode["head"])
        record = self._sign(body)
        size = len(dump_record(record).encode("utf-8")) + 1  # and its separating comma
        updated = {**episode, "next_i": episode["next_i"] + 1, "head": chain_hash(body),
                   "calls": episode["calls"] + 1, "last": [turn, k],
                   "cpu_calls_ms": episode["cpu_calls_ms"] + cpu_ms, "pending": None,
                   "transcript_bytes": episode.get("transcript_bytes", 0) + size}
        self.store.commit(updated, record)
        episode.update(updated)
        return record

    # -- finish --------------------------------------------------------------

    def finish(self, episode_id: str, key: str) -> dict[str, Any]:
        """Grade the episode and return its signed final record, plus the state
        (base64) when the task publishes it. Idempotent: once closed, every
        `finish` returns the same final record and state. `finish` must arrive
        before the deadline; grading may then outlast it (the reaper leaves a
        FINISHING episode alone) but ends before `expires_at + GRADING_GRACE_S
        - WINDOW_MARGIN_S`, so the final verifies.

        Miner runtime: the first `finish` after the deadline raises
        409 carrying the signed `expired` final; later ones return it with 200.
        Treat both as the episode's end. CPU the box burnt since the last call
        (background jobs) is charged here; over `cpu_s` ends `budget_exhausted`."""
        _off_the_event_loop()
        self._authorized(episode_id, key)
        with self._exclusive(episode_id):
            episode = self.store.get(episode_id)
            if episode["status"] == CLOSED:
                return self._finished(episode)
            if episode["status"] != RUNNING:  # FINISHING here means a crash we did not settle
                raise HTTPException(409, "episode is busy")
            if self._key_changed(episode):
                self._void_unsigned(episode)
                raise self._closed_error(episode_id, KEY_ROTATED)
            if episode.get("pending") is not None:
                _log.error("episode %s: call %s executed but was never recorded",
                           episode_id, episode["pending"])
                self._close(episode, STATUS_ABORTED, reason="a call was not recorded")
                return self._finished(episode)
            if self.clock() >= episode["deadline"]:
                self._close(episode, STATUS_EXPIRED, reason="deadline passed")
                raise self._closed_error(episode_id, "episode deadline passed")
            try:
                self._task(episode)
            except TaskChanged:
                self._close(episode, STATUS_ABORTED, reason=TASK_CHANGED)
                return self._finished(episode)
            try:
                episode["cpu_calls_ms"] += self._charge_cpu(episode)
            except BoxFault:
                self._close(episode, STATUS_BOX_FAILED, reason="the agent's box is unusable")
                return self._finished(episode)
            except Exception:
                _log.exception("episode %s: the box CPU could not be read at finish", episode_id)
                self._close(episode, STATUS_ABORTED, reason="sandbox failure")
                return self._finished(episode)
            if self._over_cpu(episode):
                self._close(episode, STATUS_BUDGET_EXHAUSTED, reason="cpu_s")
                return self._finished(episode)
            episode["status"] = FINISHING
            self.store.commit(episode)
            try:
                self._grade(episode)
            except Exception:
                # Our own bug (the grader never raises): void, never an agent verdict.
                _log.exception("episode %s: unexpected error while finishing", episode_id)
                persisted = self.store.get(episode_id) or episode
                if persisted["status"] != CLOSED:
                    self._close(persisted, STATUS_ABORTED, reason="sandbox failure")
            return self._finished(self.store.get(episode_id))

    _OUTCOME_STATUSES = frozenset({STATUS_GRADED, STATUS_BOX_FAILED, STATUS_ABORTED})

    def _grade(self, episode: dict[str, Any]) -> None:
        claims = episode["claims"]
        task = self._task(episode)
        budgets = Budgets.from_dict(claims["budgets"])
        spec = self._spec(task, budgets, math.ceil(task.grading_timeout_s) + BOX_GRACE_S)
        deadline = claims["expires_at"] + GRADING_GRACE_S - WINDOW_MARGIN_S
        outcome = self.grader.grade(episode["id"], episode["box_id"], task, spec,
                                    env=claims["env"], deadline=deadline, owner=claims["hotkey"])
        status = outcome.status
        if status not in self._OUTCOME_STATUSES:
            _log.error("episode %s: the grader returned status %r", episode["id"], status)
            status = STATUS_ABORTED
        reason = None if status == STATUS_GRADED else (outcome.reason or "grading failed")[:200]
        state = outcome.state if (status == STATUS_GRADED and episode.get("publish_state")) else None
        self._close(episode, status, reason=reason, outcome=outcome, state=state)

    def _finished(self, episode: dict[str, Any]) -> dict[str, Any]:
        final = self._final(episode["id"])
        if final is None:  # closed without a final: a failed open, or a key change
            message = (KEY_ROTATED if episode.get("final_status") == ABORTED_UNSIGNED
                       else "episode never opened")
            raise HTTPException(409, {"message": message, "final": None})
        state = self.store.state(episode["id"])
        return {"record": final,
                "state": base64.b64encode(state).decode("ascii") if state is not None else None}

    # -- reading -------------------------------------------------------------

    def transcript_json(self, episode_id: str, key: str) -> bytes:
        """The transcript as JSON bytes, built in this (worker) thread from the
        stored record texts (never decoded and re-encoded), so a large one
        costs the event loop nothing. A snapshot: the records committed with
        the episode row read first. One read at a time per episode (409 busy);
        the store lock is held per page only, never for the whole read. The
        HTTP route runs it on its own two-thread pool."""
        episode = self._authorized(episode_id, key)
        with self.lock:
            if episode_id in self._reading:
                raise HTTPException(409, "the transcript is already being read")
            self._reading.add(episode_id)
        try:
            head = json.dumps({"status": episode["status"],
                               "final_status": episode.get("final_status")}, ensure_ascii=False)
            parts = [head[:-1], ', "transcript": {"token": ',
                     json.dumps(episode["token"], ensure_ascii=False), ', "records": [',
                     ", ".join(self.store.record_texts(episode_id, episode["next_i"])), "]}}"]
            return "".join(parts).encode("utf-8")
        finally:
            with self.lock:
                self._reading.discard(episode_id)

    def transcript(self, episode_id: str, key: str) -> dict[str, Any]:
        """`transcript_json`, decoded (in-process callers)."""
        return json.loads(self.transcript_json(episode_id, key))

    def active_count(self) -> int:
        return len(self.store.reserving(include_grading=self.grading_pool is None))

    # -- background ----------------------------------------------------------

    def reap(self) -> None:
        """The periodic sweep: close RUNNING episodes past their deadline (so
        their slot frees without a restart), settle FINISHING and OPENING
        episodes nobody is working on, retry failed box removals, remove boxes
        no live episode owns, prune old closed episodes and their locks. A
        FINISHING episode is never expired: `finish` arrived in time."""
        _off_the_event_loop()
        now = self.clock()
        failures = 0
        for episode in self.store.active():
            if episode["status"] == RUNNING and now >= episode["deadline"]:
                failures += not self._guarded(self._sweep, episode["id"])
            elif episode["status"] == FINISHING:
                failures += not self._guarded(self._settle_orphaned_finish, episode["id"])
            elif episode["status"] == OPENING and now >= self._open_by(episode):
                failures += not self._guarded(self._settle_orphaned_open, episode["id"])
        failures += not self._guarded(self._retry_cleanup)
        failures += not self._guarded(self._remove_orphans)
        failures += not self._guarded(
            self.store.prune, now, self.settings.episode_retention_days * 86400)
        failures += not self._guarded(self._prune_locks)
        if failures:
            raise RuntimeError("episode cleanup incomplete")

    def reconcile(self) -> None:
        """At startup, before serving: settle what the previous process left.
        Run it via `asyncio.to_thread` and to completion before the
        routes serve and before the reaper loop starts.
        OPENING ⇒ open failed (no record, slot freed); FINISHING ⇒ aborted
        (grading interrupted, ours); RUNNING with a `pending` call ⇒ aborted;
        RUNNING past its deadline ⇒ expired; RUNNING without its box ⇒ aborted;
        otherwise the box user is restored and the CPU reading re-seeded before
        any call, and the episode goes on. Then leftover boxes are removed."""
        _off_the_event_loop()
        for episode in self.store.active():
            try:
                self._recover(episode)
            except Exception:
                _log.exception("episode %s: recovery failed", episode["id"])
                current = self.store.get(episode["id"])
                if current is not None and current["status"] != CLOSED:
                    self._guarded(self._close, current, STATUS_ABORTED,
                                  reason="controller restart recovery failed")
        self._guarded(self._retry_cleanup)
        # An untracked box that cannot be removed has no reservation to restore:
        # startup must stop rather than serve with unknown resource use.
        self._remove_orphans()

    @staticmethod
    def _guarded(step, *args, **kwargs) -> bool:
        """Run one cleanup step; a failure is logged and does not stop the others."""
        try:
            step(*args, **kwargs)
        except Exception:
            _log.exception("episode cleanup step %s failed", getattr(step, "__name__", step))
            return False
        return True

    def _sweep(self, episode_id: str) -> None:
        try:
            with self._exclusive(episode_id):
                episode = self.store.get(episode_id)
                if (episode is None or episode["status"] != RUNNING
                        or self.clock() < episode["deadline"]):
                    return
                if episode.get("pending") is not None:
                    _log.error("episode %s: call %s executed but was never recorded",
                               episode_id, episode["pending"])
                    self._close(episode, STATUS_ABORTED, reason="a call was not recorded")
                else:
                    self._close(episode, STATUS_EXPIRED, reason="deadline passed")
        except HTTPException:
            pass  # busy: a call clips itself to the deadline; a finish arrived in time

    def _settle_orphaned_finish(self, episode_id: str) -> None:
        """A FINISHING episode whose lock is free has nobody grading it: its
        `finish` failed to close it (the store failed). Close it aborted."""
        try:
            with self._exclusive(episode_id):
                episode = self.store.get(episode_id)
                if episode is not None and episode["status"] == FINISHING:
                    _log.error("episode %s: finish left it unsettled", episode_id)
                    self._close(episode, STATUS_ABORTED, reason="grading interrupted")
        except HTTPException:
            pass  # its finish is grading it now

    @staticmethod
    def _open_by(episode: dict[str, Any]) -> float:
        return episode.get("open_by", episode["opened_at"] + BOX_GRACE_S)

    def _settle_orphaned_open(self, episode_id: str) -> None:
        """An OPENING episode past its open time whose lock is free has no
        opener (it holds the lock until the open ends): settle it as a failed
        open (no record, token burnt, box removed, slot freed)."""
        try:
            with self._exclusive(episode_id):
                episode = self.store.get(episode_id)
                if (episode is not None and episode["status"] == OPENING
                        and self.clock() >= self._open_by(episode)):
                    _log.error("episode %s: its open never ended", episode_id)
                    self._open_failed(episode)
        except HTTPException:
            pass  # its opener is still preparing it

    def _recover(self, episode: dict[str, Any]) -> None:
        episode_id, status = episode["id"], episode["status"]
        if status == OPENING:
            _log.warning("episode %s: the controller restarted while it was opening", episode_id)
            self._open_failed(episode)
            return
        if self._key_changed(episode):
            self._void_unsigned(episode)
            return
        if status == FINISHING:
            self._close(episode, STATUS_ABORTED, reason="controller restarted during grading")
            return
        if episode.get("pending") is not None:
            self._close(episode, STATUS_ABORTED, reason="controller restarted during a call")
            return
        if self.clock() >= episode["deadline"]:
            self._close(episode, STATUS_EXPIRED, reason="deadline passed")
            return
        if self._over_cpu(episode):  # a crash came between a call's commit and its close
            self._close(episode, STATUS_BUDGET_EXHAUSTED, reason="cpu_s")
            return
        if self._over_transcript(episode):
            self._close(episode, STATUS_BUDGET_EXHAUSTED, reason=TRANSCRIPT_BYTES)
            return
        box_id = episode["box_id"]
        if self.boxes.status(box_id) != "running":
            # Kept `aborted` even for an exited box: after a restart we cannot tell
            # the agent killing it from a host or daemon restart stopping it. An
            # agent cannot cause the controller restart this needs; validators cap
            # aborted episodes per hotkey, which bounds what is left.
            self._close(episode, STATUS_ABORTED, reason="task box lost")
            return
        uid_gid = episode.get("box_user")
        if not uid_gid:  # an InfraFault of ours: tools cannot run as an unknown user
            self._close(episode, STATUS_ABORTED, reason="the box user was not persisted")
            return
        self.boxes.restore_user(box_id, uid_gid, episode.get("box_user_name"))
        try:
            # Re-seeds the runtime's last known CPU reading (what a box that dies
            # later reports); the episode's own mark was persisted with it.
            now_ns = self.boxes.cpu_ns(box_id)
        except BoxFault:
            self._close(episode, STATUS_BOX_FAILED, reason="the agent's box is unusable")
            return
        except InfraFault:
            self._close(episode, STATUS_ABORTED, reason="task box lost")
            return
        if episode.get("cpu_mark_ns") is None:
            episode["cpu_mark_ns"] = now_ns
            self.store.commit(episode)

    def _retry_cleanup(self) -> None:
        for episode in self.store.with_cleanup():
            try:
                with self._exclusive(episode["id"]):
                    current = self.store.get(episode["id"])
                    if current is None or not current.get("cleanup"):
                        continue
                    leftover, current["cleanup"] = list(current["cleanup"]), []
                    self._remove_boxes(current, leftover)
                    self.store.commit(current)
            except HTTPException:
                continue  # busy; next sweep

    def _remove_orphans(self) -> None:
        """Remove managed boxes (agent or grading) whose episode is not live. The
        boxes are listed before the live episodes are read: a box is created
        only after its episode's row, so a box listed here whose episode is
        not live is never one an open or a finish is still using."""
        managed = self.boxes.managed()
        live = {episode["id"] for episode in self.store.active()}
        failed = False
        for box_id, episode_id in managed.items():
            if episode_id not in live:
                try:
                    with self._exclusive(episode_id):
                        episode = self.store.get(episode_id)
                        if episode is not None:
                            if episode["status"] != CLOSED:
                                continue
                            # A crash may precede the final cleanup save. Journal the
                            # leftover before removing it, so a failed removal (or
                            # another crash) cannot lose its resource reservation.
                            episode["cleanup"] = sorted(set(episode.get("cleanup", ())) | {box_id})
                            self.store.commit(episode)
                            if self.grading_pool is not None and box_id != episode["box_id"]:
                                self.grader.retain_cleanup(
                                    box_id, cleanup_reservation(
                                        episode, self.settings, self.grading_pool.total),
                                    episode["claims"]["hotkey"], restore=True)
                            self._remove_boxes(episode, [box_id])
                            self.store.commit(episode)
                        else:
                            self.boxes.remove(box_id)
                            self.grader.release_cleanup(box_id)
                except HTTPException:
                    continue  # an in-flight request owns this row; retry next sweep
                except Exception:
                    failed = True
                    _log.warning("could not remove leftover box %s", box_id, exc_info=True)
        if failed:
            raise RuntimeError("orphaned box cleanup incomplete")

    def _prune_locks(self) -> None:
        """Forget the locks of episodes that are no longer live. Read under
        `self.lock`, which also guards inserts, so no live episode is missed;
        a lock someone holds is kept."""
        with self.lock:
            live = {episode["id"] for episode in self.store.active()}
            for episode_id, lock in list(self.episode_locks.items()):
                if episode_id not in live and lock.acquire(blocking=False):
                    del self.episode_locks[episode_id]
                    lock.release()
