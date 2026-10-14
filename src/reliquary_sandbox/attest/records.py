"""Session token claims and the three signed record kinds.

A transcript is `{"token": <signed session token>, "records": [record 0,
call records..., final record]}`. Every record is an envelope
`{"body", "machine_id", "key_id", "signature"}`; the signature covers
RECORD_DOMAIN + canonical({"body", "machine_id", "key_id"}). The chain hash of
a record is `h(i) = sha256(canonical(body_i))`, and `body_i["prev"] = h(i-1)`
(`None` for record 0, which binds the token through `token_sha256`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

from .canonical import digest
from .signing import RECORD_DOMAIN, TOKEN_DOMAIN, Signer, signature_valid

KIND_OPEN = "open"
KIND_CALL = "call"
KIND_FINAL = "final"
STATUS_GRADED = "graded"
STATUS_EXPIRED = "expired"
STATUS_BUDGET_EXHAUSTED = "budget_exhausted"
STATUS_ABORTED = "aborted"
# The agent's own doing left the box unusable (it was killed, filled or wedged):
# not paid, engagement consumed, and never voided like `aborted` (our fault).
STATUS_BOX_FAILED = "box_failed"
FINAL_STATUSES = frozenset({STATUS_GRADED, STATUS_EXPIRED, STATUS_BUDGET_EXHAUSTED,
                            STATUS_ABORTED, STATUS_BOX_FAILED})
TOKEN_VERSION = 1
TOOLS = frozenset({"bash", "edit"})
"""The tools a record 0 may offer (`reliquary-tools/1`)."""
_INT_BOUND = 2**63
_ENVELOPE_KEYS = frozenset({"body", "machine_id", "key_id", "signature"})


class RecordError(ValueError):
    """A token or record does not have the documented shape."""


def _field(data: Mapping[str, Any], name: str, kind: type | tuple[type, ...], *, nullable=False):
    if name not in data:
        raise RecordError(f"missing field {name!r}")
    value = data[name]
    if value is None and nullable:
        return None
    # bool is an int subclass; never accept it where a number is meant.
    if isinstance(value, bool) and bool not in (kind if isinstance(kind, tuple) else (kind,)):
        raise RecordError(f"field {name!r} has the wrong type")
    if not isinstance(value, kind):
        raise RecordError(f"field {name!r} has the wrong type")
    return value


def _uint(data: Mapping[str, Any], name: str) -> int:
    value = _field(data, name, int)
    if not 0 <= value < _INT_BOUND:
        raise RecordError(f"field {name!r} is outside the allowed integer range")
    return value


def _exit_code(data: Mapping[str, Any]) -> int:
    value = _field(data, "exit_code", int)
    if not -_INT_BOUND <= value < _INT_BOUND:
        raise RecordError("field 'exit_code' is outside the allowed integer range")
    return value


def _exact_keys(data: Any, keys: set[str] | frozenset[str], what: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping) or set(data) != set(keys):
        raise RecordError(f"{what} has unexpected or missing fields")
    return data


@dataclass(frozen=True)
class Budgets:
    max_calls: int
    per_call_timeout_s: int
    cpu_s: int
    wall_s: int
    memory_bytes: int
    pids: int
    disk_bytes: int

    @classmethod
    def from_dict(cls, data: Any) -> Budgets:
        names = {f.name for f in fields(cls)}
        data = _exact_keys(data, names, "budgets")
        values = {name: _uint(data, name) for name in names}
        if any(value <= 0 for value in values.values()):
            raise RecordError("budgets must be positive")
        return cls(**values)


@dataclass(frozen=True)
class SessionClaims:
    session_id: str
    hotkey: str
    engagement: str
    env: str
    split: str
    index: int
    image: str
    checkpoint: str
    machine_id: str
    issued_at: int
    expires_at: int
    budgets: Budgets

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> SessionClaims:
        names = {f.name for f in fields(cls)}
        data = _exact_keys(data, names, "session claims")
        values: dict[str, Any] = {
            name: _field(data, name, str)
            for name in ("session_id", "hotkey", "engagement", "env", "split", "image",
                         "checkpoint", "machine_id")
        }
        for name in ("index", "issued_at", "expires_at"):
            values[name] = _uint(data, name)
        values["budgets"] = Budgets.from_dict(data["budgets"])
        if not values["session_id"] or not values["machine_id"]:
            raise RecordError("session_id and machine_id are required")
        if "@sha256:" not in values["image"]:
            raise RecordError("the image must be pinned by digest")
        if values["index"] < 0 or values["expires_at"] <= values["issued_at"]:
            raise RecordError("invalid index or validity window")
        return cls(**values)


def issue_session_token(claims: SessionClaims, signer: Signer) -> dict[str, Any]:
    """What the validator hands the miner. Also used by tests and tools."""
    payload = {"v": TOKEN_VERSION, "claims": claims.to_dict(), "key_id": signer.key_id}
    return {**payload, "signature": signer.sign(TOKEN_DOMAIN, payload)}


def token_sha256(token: Mapping[str, Any]) -> str:
    return digest(dict(token))


def verify_session_token(token: Any, public_keys: Mapping[str, Any]) -> SessionClaims:
    """Check shape and validator signature; `public_keys` maps key_id to an
    Ed25519PublicKey. Expiry is the caller's decision (it depends on *when*)."""
    token = _exact_keys(token, {"v", "claims", "key_id", "signature"}, "session token")
    if type(token["v"]) is not int or token["v"] != TOKEN_VERSION:
        raise RecordError("unsupported token version")
    key = public_keys.get(token["key_id"]) if isinstance(token["key_id"], str) else None
    if key is None:
        raise RecordError("token signed by an unknown validator key")
    payload = {"v": token["v"], "claims": token["claims"], "key_id": token["key_id"]}
    if not isinstance(token["signature"], str) or not signature_valid(
        key, TOKEN_DOMAIN, payload, token["signature"]
    ):
        raise RecordError("token signature is invalid")
    return SessionClaims.from_dict(token["claims"])


def _check_tools(value: Any) -> tuple[str, ...]:
    if (not isinstance(value, (list, tuple)) or not value
            or not all(isinstance(tool, str) for tool in value)
            or list(value) != sorted(set(value)) or not set(value) <= TOOLS):
        raise RecordError(f"tools must be a sorted, unique, non-empty subset of {sorted(TOOLS)}")
    return tuple(value)


@dataclass(frozen=True)
class OpenBody:
    i: int
    session_id: str
    machine_id: str
    token_sha256: str
    image: str
    env: str
    env_package: str
    tools_version: str
    tools: tuple[str, ...]
    """The task's offered tools: sorted, unique, non-empty, a subset of TOOLS. Miners and
    validators pass them to `observation.plan_call(offered=...)`."""
    setup_output_sha256: str
    budgets: Budgets
    at: int
    prev: None = None
    kind: str = KIND_OPEN

    def __post_init__(self) -> None:
        _check_tools(self.tools)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tools"] = list(self.tools)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> OpenBody:
        data = _exact_keys(data, {f.name for f in fields(cls)}, "open record")
        if data["kind"] != KIND_OPEN or data["prev"] is not None:
            raise RecordError("not an open record")
        return cls(
            i=_uint(data, "i"),
            session_id=_field(data, "session_id", str),
            machine_id=_field(data, "machine_id", str),
            token_sha256=_field(data, "token_sha256", str),
            image=_field(data, "image", str),
            env=_field(data, "env", str),
            env_package=_field(data, "env_package", str),
            tools_version=_field(data, "tools_version", str),
            tools=_check_tools(data["tools"]),
            setup_output_sha256=_field(data, "setup_output_sha256", str),
            budgets=Budgets.from_dict(data["budgets"]),
            at=_uint(data, "at"),
        )


@dataclass(frozen=True)
class CallBody:
    i: int
    session_id: str
    turn: int
    k: int
    tool: str
    arguments: dict[str, Any]
    output: str
    exit_code: int
    truncated: bool
    timed_out: bool
    cpu_ms: int
    wall_ms: int
    at: int
    prev: str
    kind: str = KIND_CALL

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> CallBody:
        data = _exact_keys(data, {f.name for f in fields(cls)}, "call record")
        if data["kind"] != KIND_CALL:
            raise RecordError("not a call record")
        return cls(
            i=_uint(data, "i"),
            session_id=_field(data, "session_id", str),
            turn=_uint(data, "turn"),
            k=_uint(data, "k"),
            tool=_field(data, "tool", str),
            arguments=dict(_field(data, "arguments", dict)),
            output=_field(data, "output", str),
            exit_code=_exit_code(data),
            truncated=_field(data, "truncated", bool),
            timed_out=_field(data, "timed_out", bool),
            cpu_ms=_uint(data, "cpu_ms"),
            wall_ms=_uint(data, "wall_ms"),
            at=_uint(data, "at"),
            prev=_field(data, "prev", str),
        )


@dataclass(frozen=True)
class FinalBody:
    i: int
    session_id: str
    status: str
    reward: float | None
    grading: dict[str, Any] | None
    state_sha256: str | None
    cpu_total_ms: int
    reason: str | None
    at: int
    prev: str
    kind: str = KIND_FINAL

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> FinalBody:
        data = _exact_keys(data, {f.name for f in fields(cls)}, "final record")
        if data["kind"] != KIND_FINAL or data["status"] not in FINAL_STATUSES:
            raise RecordError("not a final record")
        reward = data["reward"]
        # Strictly a float: an int reward would re-serialise differently (1 vs
        # 1.0) and so hash differently once parsed. bool and int are refused.
        if reward is not None and type(reward) is not float:
            raise RecordError("reward must be a float or null")
        graded = data["status"] == STATUS_GRADED
        if graded != (reward is not None) or (reward is not None and not math.isfinite(reward)):
            raise RecordError("a reward is present exactly when the status is graded")
        return cls(
            i=_uint(data, "i"),
            session_id=_field(data, "session_id", str),
            status=data["status"],
            reward=reward,
            grading=_field(data, "grading", dict, nullable=True),
            state_sha256=_field(data, "state_sha256", str, nullable=True),
            cpu_total_ms=_uint(data, "cpu_total_ms"),
            reason=_field(data, "reason", str, nullable=True),
            at=_uint(data, "at"),
            prev=_field(data, "prev", str),
        )


Body = OpenBody | CallBody | FinalBody


def parse_body(data: Any) -> Body:
    kind = data.get("kind") if isinstance(data, Mapping) else None
    parser = {KIND_OPEN: OpenBody, KIND_CALL: CallBody, KIND_FINAL: FinalBody}.get(kind)
    if parser is None:
        raise RecordError("unknown record kind")
    return parser.from_dict(data)


def chain_hash(body: Body | Mapping[str, Any]) -> str:
    """`h(i)`. Chain hashes and signatures are always computed over the raw
    received dict, never over a re-serialised parse of it; a parsed body is
    accepted here only because parsing is lossless (strict types, no coercion).
    """
    return digest(body.to_dict() if hasattr(body, "to_dict") else dict(body))


def sign_record(signer: Signer, machine_id: str, body: Body) -> dict[str, Any]:
    payload = {"body": body.to_dict(), "machine_id": machine_id, "key_id": signer.key_id}
    return {**payload, "signature": signer.sign(RECORD_DOMAIN, payload)}


def envelope(record: Any) -> Mapping[str, Any]:
    record = _exact_keys(record, _ENVELOPE_KEYS, "record envelope")
    for name in ("machine_id", "key_id", "signature"):
        _field(record, name, str)
    return record


def record_signature_valid(public_key: Any, record: Mapping[str, Any]) -> bool:
    """False, never an exception, on a malformed record."""
    try:
        payload = {
            "body": record["body"],
            "machine_id": record["machine_id"],
            "key_id": record["key_id"],
        }
        signature = record["signature"]
    except (KeyError, TypeError):
        return False
    return isinstance(signature, str) and signature_valid(
        public_key, RECORD_DOMAIN, payload, signature
    )
