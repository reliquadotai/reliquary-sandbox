"""Signed episode transcripts: record format, signing and pure verification.

Depends only on the standard library and `cryptography`, so a validator can
verify transcripts without the sandbox service's dependencies.
"""

from .canonical import CanonicalError, canonical_json, digest, sha256_hex
from .records import (
    FINAL_STATUSES,
    KIND_CALL,
    KIND_FINAL,
    KIND_OPEN,
    STATUS_ABORTED,
    STATUS_BOX_FAILED,
    STATUS_BUDGET_EXHAUSTED,
    STATUS_EXPIRED,
    STATUS_GRADED,
    Budgets,
    CallBody,
    FinalBody,
    OpenBody,
    RecordError,
    SessionClaims,
    chain_hash,
    issue_session_token,
    parse_body,
    sign_record,
    token_sha256,
    verify_session_token,
)
from .signing import (
    HEARTBEAT_DOMAIN,
    OPEN_REFUSAL_DOMAIN,
    RECORD_DOMAIN,
    TOKEN_DOMAIN,
    Signer,
    generate_private_key_pem,
    load_private_key,
    public_key_b64,
    public_key_from_b64,
    signature_valid,
)
from .verify import (
    CLOCK_SKEW_S,
    GRADING_GRACE_S,
    OPEN_REFUSAL_REASONS,
    Ed25519TokenVerifier,
    Expected,
    HeartbeatCheck,
    HeartbeatReason,
    MachineDirectory,
    MachineKey,
    Reason,
    StaticDirectory,
    TokenVerifier,
    VerificationResult,
    heartbeat_valid,
    open_refusal_valid,
    verify_transcript,
)

__all__ = [
    "CLOCK_SKEW_S", "GRADING_GRACE_S", "FINAL_STATUSES", "HEARTBEAT_DOMAIN", "KIND_CALL", "KIND_FINAL",
    "KIND_OPEN", "OPEN_REFUSAL_DOMAIN", "OPEN_REFUSAL_REASONS", "RECORD_DOMAIN", "STATUS_ABORTED",
    "STATUS_BOX_FAILED", "STATUS_BUDGET_EXHAUSTED",
    "STATUS_EXPIRED", "STATUS_GRADED", "TOKEN_DOMAIN", "Budgets", "CallBody",
    "CanonicalError", "Ed25519TokenVerifier", "Expected", "FinalBody", "HeartbeatCheck",
    "HeartbeatReason", "MachineDirectory",
    "MachineKey", "OpenBody", "Reason", "RecordError", "SessionClaims", "Signer",
    "StaticDirectory", "TokenVerifier", "VerificationResult", "canonical_json", "chain_hash",
    "digest", "generate_private_key_pem", "heartbeat_valid", "issue_session_token", "load_private_key",
    "open_refusal_valid",
    "parse_body", "public_key_b64", "public_key_from_b64", "sha256_hex", "sign_record",
    "signature_valid", "token_sha256", "verify_session_token", "verify_transcript",
]
