"""Ed25519 keys and domain-separated signatures over canonical JSON."""

from __future__ import annotations

import base64
import binascii
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .canonical import canonical_json

TOKEN_DOMAIN = b"reliquary-sandbox/session-token/v1\x00"
RECORD_DOMAIN = b"reliquary-sandbox/record/v1\x00"
HEARTBEAT_DOMAIN = b"reliquary-sandbox/heartbeat/v1\x00"
OPEN_REFUSAL_DOMAIN = b"reliquary-sandbox/open-refusal/v1\x00"


def _b64decode(text: str) -> bytes:
    if not isinstance(text, str):
        raise ValueError("base64 text required")
    try:
        raw = base64.b64decode(text.encode("ascii"), validate=True)
    except (binascii.Error, UnicodeEncodeError) as exc:
        raise ValueError("invalid base64") from exc
    # Reject non-canonical encodings (non-zero trailing bits): otherwise two
    # different texts decode to one value and token hashes become malleable.
    if base64.b64encode(raw).decode("ascii") != text:
        raise ValueError("non-canonical base64")
    return raw


def public_key_b64(key: Ed25519PublicKey) -> str:
    raw = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode("ascii")


def public_key_from_b64(text: str) -> Ed25519PublicKey:
    raw = _b64decode(text)
    if len(raw) != 32:
        raise ValueError("an Ed25519 public key is 32 bytes")
    return Ed25519PublicKey.from_public_bytes(raw)


def generate_private_key_pem() -> bytes:
    return Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def load_private_key(path: Path) -> Ed25519PrivateKey:
    """Load a PKCS#8 PEM Ed25519 key that only its owner can read.

    The file is opened once (no symlinks) and checked on the descriptor, so the
    checked file is the file read.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise ValueError(f"cannot open the signing key without following links: {exc}") from exc
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise ValueError("the signing key must be a regular file with mode 0600 or stricter")
        if info.st_uid != os.geteuid():
            raise ValueError("the signing key must be owned by the current user")
        data = handle.read()
    key = serialization.load_pem_private_key(data, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("the signing key must be Ed25519")
    return key


@dataclass(frozen=True)
class Signer:
    key_id: str
    private_key: Ed25519PrivateKey

    @property
    def public_key_b64(self) -> str:
        return public_key_b64(self.private_key.public_key())

    def sign(self, domain: bytes, payload: Any) -> str:
        signature = self.private_key.sign(domain + canonical_json(payload))
        return base64.b64encode(signature).decode("ascii")


def signature_valid(
    public_key: Ed25519PublicKey, domain: bytes, payload: Any, signature_b64: str
) -> bool:
    try:
        signature = _b64decode(signature_b64)
        public_key.verify(signature, domain + canonical_json(payload))
    except (InvalidSignature, ValueError):
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    """`python -m reliquary_sandbox.attest.signing generate PATH` writes a 0600 key
    and prints its public half (base64), for the machine directory."""
    import sys

    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or args[0] != "generate":
        print("usage: python -m reliquary_sandbox.attest.signing generate PATH", file=sys.stderr)
        return 2
    path = Path(args[1])
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        print(f"{path} already exists; refusing to overwrite a signing key", file=sys.stderr)
        return 2
    with os.fdopen(descriptor, "wb") as output:
        output.write(generate_private_key_pem())
    print(public_key_b64(load_private_key(path).public_key()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
