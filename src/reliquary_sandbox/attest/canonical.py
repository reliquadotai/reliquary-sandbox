"""Canonical JSON: the only byte form that is ever hashed or signed.

Keys sorted, no whitespace, UTF-8, no NaN or infinity. Floats use Python's
shortest round-trip repr, so a verifier that parses a record and
re-canonicalizes it gets the same bytes back. Integers are bounded to the
signed 64-bit range so other languages can verify the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

_MAX_DEPTH = 64
_INT_BOUND = 2**63


class CanonicalError(ValueError):
    """The value has no canonical form."""


def _check(value: Any, depth: int) -> None:
    if depth > _MAX_DEPTH:
        raise CanonicalError("value is nested too deeply")
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        if not -_INT_BOUND <= value < _INT_BOUND:
            raise CanonicalError("integer outside the signed 64-bit range")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalError("non-finite float")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _check(item, depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalError("object keys must be strings")
            _check(item, depth + 1)
        return
    raise CanonicalError(f"unsupported type {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    _check(value, 0)
    text = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as exc:  # lone surrogates
        raise CanonicalError("string is not valid Unicode") from exc


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    """`sha256(canonical(value))`, hex: the hash used for chaining and token binding."""
    return sha256_hex(canonical_json(value))
