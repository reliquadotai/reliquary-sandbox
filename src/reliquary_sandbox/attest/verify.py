"""Pure transcript verification: a function of (transcript, directory, token
verifier, expectations). No clock, no network, no state; anyone can rerun it.

`ok` means only that the transcript is internally consistent, was signed by the
listed machine key and matches the token the validator issued. It does NOT
mean the episode is payable. The caller must still (i) check the calls, arguments
and observations against the TOPLOC-proven tokens and the final state hash
against the submitted final_diff, and
(ii) record the session id atomically with acceptance, so one session pays
once (`Expected.seen_session_ids` is only a snapshot).

Time: the machine key is looked up at record 0's `at`, which the validator's
token window bounds; later records must share record 0's `key_id`, have
non-decreasing `at` values, and stay within the token window plus a grading
grace. A signer therefore cannot backdate records to use a retired key.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .canonical import digest
from .records import (
    KIND_CALL,
    KIND_FINAL,
    KIND_OPEN,
    STATUS_GRADED,
    CallBody,
    FinalBody,
    OpenBody,
    RecordError,
    SessionClaims,
    chain_hash,
    envelope,
    parse_body,
    record_signature_valid,
    token_sha256,
    verify_session_token,
)
from .signing import public_key_from_b64

CLOCK_SKEW_S = 30
GRADING_GRACE_S = 1800


class Reason(str, Enum):
    MALFORMED = "malformed"
    TOKEN_INVALID = "token_invalid"
    TOKEN_HASH_MISMATCH = "token_hash_mismatch"
    OPENED_OUTSIDE_VALIDITY = "opened_outside_validity"
    MACHINE_MISMATCH = "machine_mismatch"
    UNKNOWN_KEY = "unknown_key"
    BAD_SIGNATURE = "bad_signature"
    INDEX_GAP = "index_gap"
    PREV_MISMATCH = "prev_mismatch"
    SESSION_MISMATCH = "session_mismatch"
    IMAGE_MISMATCH = "image_mismatch"
    KIND_ORDER = "kind_order"
    CALL_ORDER = "call_order"
    NO_FINAL = "no_final"
    NOT_GRADED = "not_graded"
    HOTKEY_MISMATCH = "hotkey_mismatch"
    TASK_MISMATCH = "task_mismatch"
    CHECKPOINT_MISMATCH = "checkpoint_mismatch"
    SESSION_REUSED = "session_reused"
    KEY_CHANGED = "key_changed"
    TIME_OUT_OF_RANGE = "time_out_of_range"
    TIME_NOT_MONOTONIC = "time_not_monotonic"
    TOO_MANY_RECORDS = "too_many_records"
    OPEN_MISMATCH = "open_mismatch"
    REWARD_OUT_OF_RANGE = "reward_out_of_range"
    TOOL_NOT_OFFERED = "tool_not_offered"
    """A call record names a tool outside record 0's `tools` (a miner's router refuses
    such calls with `not_offered` and never sends them)."""


class MachineDirectory(Protocol):
    def public_key(self, machine_id: str, key_id: str, at: int) -> Ed25519PublicKey | None:
        """The key the directory lists for this machine and key id, valid at `at`."""


class TokenVerifier(Protocol):
    def __call__(self, token: Any) -> SessionClaims:
        """Return the claims of a token this validator issued; raise RecordError otherwise."""


@dataclass(frozen=True)
class MachineKey:
    machine_id: str
    key_id: str
    public_key_b64: str
    valid_from: int
    valid_until: int | None = None


class StaticDirectory:
    def __init__(self, keys: Iterable[MachineKey]):
        self._keys = {(key.machine_id, key.key_id): key for key in keys}

    def public_key(self, machine_id: str, key_id: str, at: int) -> Ed25519PublicKey | None:
        key = self._keys.get((machine_id, key_id))
        if key is None or at < key.valid_from or (key.valid_until is not None and at >= key.valid_until):
            return None
        return public_key_from_b64(key.public_key_b64)


class Ed25519TokenVerifier:
    def __init__(self, public_keys_b64: Mapping[str, str]):
        self._keys = {key_id: public_key_from_b64(value) for key_id, value in public_keys_b64.items()}

    def __call__(self, token: Any) -> SessionClaims:
        return verify_session_token(token, self._keys)


@dataclass(frozen=True)
class Expected:
    """What the submission claims. Every binding is required: a verifier that
    can silently skip a comparison would accept any validly signed episode."""

    hotkey: str
    engagement: str
    env: str
    split: str
    index: int
    checkpoint: str
    seen_session_ids: Collection[str] = ()
    require_graded: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.seen_session_ids, (str, bytes)):
            raise TypeError("seen_session_ids must be a collection of session ids, not a string")
        object.__setattr__(self, "seen_session_ids", frozenset(self.seen_session_ids))


@dataclass(frozen=True)
class VerificationResult:
    reasons: tuple[Reason, ...]
    claims: SessionClaims | None = None
    open: OpenBody | None = None
    calls: tuple[CallBody, ...] = field(default_factory=tuple)
    final: FinalBody | None = None

    @property
    def ok(self) -> bool:
        return not self.reasons

    @property
    def tools(self) -> tuple[str, ...] | None:
        """Record 0's offered tools (pass them as `plan_call(offered=...)`); None when
        record 0 could not be read."""
        return self.open.tools if self.open is not None else None


def _call_follows(previous: CallBody | None, current: CallBody) -> bool:
    if current.turn < 0 or current.k < 0:
        return False
    if previous is None:
        return current.k == 0
    if current.turn == previous.turn:
        return current.k == previous.k + 1
    return current.turn > previous.turn and current.k == 0


def verify_transcript(
    transcript: Any,
    directory: MachineDirectory,
    token_verifier: TokenVerifier,
    expected: Expected,
    *,
    grading_grace_s: int = GRADING_GRACE_S,
) -> VerificationResult:
    """Never raises on JSON-decodable input; a bad transcript yields reasons."""
    try:
        return _verify(transcript, directory, token_verifier, expected, grading_grace_s)
    except (ValueError, TypeError, RecursionError, OverflowError, AttributeError, KeyError):
        return VerificationResult((Reason.MALFORMED,))


def _verify(transcript, directory, token_verifier, expected, grading_grace_s):
    if (
        not isinstance(transcript, Mapping)
        or set(transcript) != {"token", "records"}
        or not isinstance(transcript["records"], list)
        or not transcript["records"]
    ):
        return VerificationResult((Reason.MALFORMED,))
    try:
        claims = token_verifier(transcript["token"])
    except (RecordError, ValueError, TypeError):
        return VerificationResult((Reason.TOKEN_INVALID,))
    raw_records = transcript["records"]
    if len(raw_records) > claims.budgets.max_calls + 2:
        return VerificationResult((Reason.TOO_MANY_RECORDS,), claims)

    # Parse and canonicalise everything first; hashes and signatures are always
    # over the raw received dict, never a re-serialised parse.
    parsed: list[tuple[Mapping[str, Any], Any, str]] = []
    for raw in raw_records:
        try:
            record = envelope(raw)
            body = parse_body(record["body"])
            parsed.append((record, body, chain_hash(record["body"])))
            digest(dict(record))
        except (RecordError, ValueError, TypeError, RecursionError):
            return VerificationResult((Reason.MALFORMED,), claims)

    reasons: list[Reason] = []

    def flag(reason: Reason) -> None:
        if reason not in reasons:
            reasons.append(reason)

    first_record, first_body, _ = parsed[0]
    if first_body.kind != KIND_OPEN:
        return VerificationResult((Reason.MALFORMED,), claims)
    opened: OpenBody = first_body
    key_id = first_record["key_id"]
    key = directory.public_key(first_record["machine_id"], key_id, opened.at)
    if key is None:
        flag(Reason.UNKNOWN_KEY)

    low, high = claims.issued_at - CLOCK_SKEW_S, claims.expires_at + grading_grace_s
    calls: list[CallBody] = []
    final: FinalBody | None = None
    previous_hash: str | None = None
    previous_at = 0
    for position, (record, body, body_hash) in enumerate(parsed):
        if record["machine_id"] != claims.machine_id:
            flag(Reason.MACHINE_MISMATCH)
        if record["key_id"] != key_id:
            flag(Reason.KEY_CHANGED)
        elif key is not None and not record_signature_valid(key, record):
            flag(Reason.BAD_SIGNATURE)
        if body.i != position:
            flag(Reason.INDEX_GAP)
        if body.session_id != claims.session_id:
            flag(Reason.SESSION_MISMATCH)
        if position and body.prev != previous_hash:
            flag(Reason.PREV_MISMATCH)
        previous_hash = body_hash
        if not low <= body.at <= high:
            flag(Reason.TIME_OUT_OF_RANGE)
        if body.at < previous_at:
            flag(Reason.TIME_NOT_MONOTONIC)
        previous_at = max(previous_at, body.at)
        if final is not None:
            flag(Reason.KIND_ORDER)
        if position == 0:
            if body.token_sha256 != token_sha256(transcript["token"]):
                flag(Reason.TOKEN_HASH_MISMATCH)
            if not low <= body.at < claims.expires_at:
                flag(Reason.OPENED_OUTSIDE_VALIDITY)
            if body.image != claims.image:
                flag(Reason.IMAGE_MISMATCH)
            if body.machine_id != claims.machine_id:
                flag(Reason.MACHINE_MISMATCH)
            if body.env != claims.env or body.budgets != claims.budgets:
                flag(Reason.OPEN_MISMATCH)
        elif body.kind == KIND_CALL:
            if not _call_follows(calls[-1] if calls else None, body):
                flag(Reason.CALL_ORDER)
            if body.tool not in opened.tools:
                flag(Reason.TOOL_NOT_OFFERED)
            calls.append(body)
        elif body.kind == KIND_FINAL:
            final = body
            if body.reward is not None and not 0.0 <= body.reward <= 1.0:
                flag(Reason.REWARD_OUT_OF_RANGE)
        else:
            flag(Reason.KIND_ORDER)

    if len(calls) > claims.budgets.max_calls:
        flag(Reason.TOO_MANY_RECORDS)
    if final is None:
        flag(Reason.NO_FINAL)
    elif expected.require_graded and final.status != STATUS_GRADED:
        flag(Reason.NOT_GRADED)
    if claims.hotkey != expected.hotkey:
        flag(Reason.HOTKEY_MISMATCH)
    for name in ("engagement", "env", "split", "index"):
        if getattr(claims, name) != getattr(expected, name):
            flag(Reason.TASK_MISMATCH)
    if claims.checkpoint != expected.checkpoint:
        flag(Reason.CHECKPOINT_MISMATCH)
    if claims.session_id in expected.seen_session_ids:
        flag(Reason.SESSION_REUSED)
    return VerificationResult(tuple(reasons), claims, opened, tuple(calls), final)


class HeartbeatReason(str, Enum):
    """Why `heartbeat_valid` refused a capacity report."""

    MALFORMED = "malformed"
    STALE = "stale"
    FROM_THE_FUTURE = "from_the_future"
    UNKNOWN_KEY = "unknown_key"
    BAD_SIGNATURE = "bad_signature"


@dataclass(frozen=True)
class HeartbeatCheck:
    """Outcome of `heartbeat_valid`: `reason` is None when the report is valid.
    Truthy exactly when valid."""

    reason: HeartbeatReason | None = None

    @property
    def ok(self) -> bool:
        return self.reason is None

    def __bool__(self) -> bool:
        return self.ok


_HEARTBEAT_COUNTS = ("capacity", "active", "free")


HEARTBEAT_CAPS = frozenset({
    "max_calls", "max_cpu_s", "max_call_timeout_s", "max_wall_s", "max_memory_bytes",
    "max_pids", "max_disk_bytes", "max_argument_bytes", "output_cap_bytes",
    "max_token_validity_s", "max_transcript_bytes",
})
"""The caps a capacity report must carry, each a non-negative integer.
`max_transcript_bytes` bounds the signed transcript as compact UTF-8 JSON
(ensure_ascii=False), final record included."""


def heartbeat_valid(report: Any, directory: MachineDirectory, *, now: int, max_age_s: int,
                    clock_skew_s: int = CLOCK_SKEW_S) -> HeartbeatCheck:
    """Whether a capacity report is fresh, self-consistent and was signed, under
    the heartbeat domain, by the key the directory lists for that machine at the
    report's `at`. Pure: the caller supplies its own clock (`now`) and the maximum
    age it accepts; a report up to `clock_skew_s` ahead of `now` is tolerated.
    Never raises: a malformed report, or a directory that fails, is a refusal.

    Validator: the report's `public_base_url` is signed, but the address
    to send miners to comes from the machine directory, never from the sender of
    a report; a URL in a report that differs from the directory's is a refusal."""
    from .canonical import CanonicalError, canonical_json
    from .signing import HEARTBEAT_DOMAIN, signature_valid

    malformed = HeartbeatCheck(HeartbeatReason.MALFORMED)

    def integer(value: Any) -> bool:
        return type(value) is int

    if not isinstance(report, Mapping) or set(report) != {"document", "signature"}:
        return malformed
    document, signature = report["document"], report["signature"]
    if not isinstance(document, Mapping) or not isinstance(signature, str):
        return malformed
    machine_id, key_id, at = document.get("machine_id"), document.get("key_id"), document.get("at")
    if not isinstance(machine_id, str) or not isinstance(key_id, str) or not integer(at):
        return malformed
    capacity, active, free = (document.get(name) for name in _HEARTBEAT_COUNTS)
    if not all(integer(value) and value >= 0 for value in (capacity, active, free)):
        return malformed
    # Active may exceed capacity when an operator lowered it while episodes
    # were live; those episodes run to completion and nothing is free.
    if free != max(0, capacity - active):
        return malformed
    caps = document.get("caps")
    if (not isinstance(caps, Mapping) or not HEARTBEAT_CAPS <= set(caps)
            or not all(integer(caps[name]) and caps[name] >= 0 for name in HEARTBEAT_CAPS)):
        return malformed
    try:
        canonical_json(dict(document))
    except (CanonicalError, TypeError, ValueError, RecursionError):
        return malformed
    if at > now + clock_skew_s:
        return HeartbeatCheck(HeartbeatReason.FROM_THE_FUTURE)
    if now - at > max_age_s:
        return HeartbeatCheck(HeartbeatReason.STALE)
    try:
        key = directory.public_key(machine_id, key_id, at)
    except Exception:
        key = None
    if key is None:
        return HeartbeatCheck(HeartbeatReason.UNKNOWN_KEY)
    try:
        valid = signature_valid(key, HEARTBEAT_DOMAIN, dict(document), signature)
    except Exception:
        valid = False
    return HeartbeatCheck(None if valid else HeartbeatReason.BAD_SIGNATURE)


OPEN_REFUSAL_REASONS = ("open_failed", "no_free_slot", "no_free_resources", "task_refused")
"""Why a gateway refused (or failed) a fresh token, for a reason that is not the miner's."""
_OPEN_REFUSAL_FIELDS = frozenset({"kind", "machine_id", "key_id", "session_id", "token_sha256", "reason", "at"})


def open_refusal_valid(receipt: Any, directory: MachineDirectory, *, machine_id: str, session_id: str,
                       token_sha256: str, max_age_s: int = 7 * 86_400, now: int | None = None,
                       clock_skew_s: int = CLOCK_SKEW_S) -> bool:
    """Whether ``receipt`` is a gateway's signed refusal (or failure) of exactly this session's token on
    exactly this machine, for a reason that is not the miner's (a failed preparation, no free slot or
    resources, claims this machine cannot serve). The gateway spends the token when it signs one, so the
    token can never open a box afterwards. Pure, never raises: anything malformed is False."""
    import time as _time

    from .signing import OPEN_REFUSAL_DOMAIN, signature_valid

    try:
        if not isinstance(receipt, Mapping) or set(receipt) != {"document", "signature"}:
            return False
        document, signature = receipt["document"], receipt["signature"]
        if not isinstance(document, Mapping) or set(document) != _OPEN_REFUSAL_FIELDS:
            return False
        if not isinstance(signature, str) or document["kind"] != "open_refusal":
            return False
        if document["reason"] not in OPEN_REFUSAL_REASONS or type(document["at"]) is not int:
            return False
        if not isinstance(document["key_id"], str):
            return False
        if (document["machine_id"], document["session_id"], document["token_sha256"]) != (
                machine_id, session_id, token_sha256):
            return False
        current = int(_time.time()) if now is None else int(now)
        if document["at"] > current + clock_skew_s or current - document["at"] > max_age_s:
            return False
        key = directory.public_key(machine_id, document["key_id"], document["at"])
        if key is None:
            return False
        return signature_valid(key, OPEN_REFUSAL_DOMAIN, dict(document), signature)
    except Exception:
        return False
