import ast
import os
from pathlib import Path

import pytest

from reliquary_sandbox.attest.canonical import CanonicalError, canonical_json, digest
from reliquary_sandbox.attest.signing import (
    RECORD_DOMAIN,
    TOKEN_DOMAIN,
    Signer,
    generate_private_key_pem,
    load_private_key,
    main,
    public_key_from_b64,
    signature_valid,
)

ATTEST_DIR = Path(__file__).resolve().parents[1] / "src" / "reliquary_sandbox" / "attest"


def test_canonical_form_is_sorted_compact_utf8():
    assert canonical_json({"b": 1, "a": [True, None, "é"], "c": 0.5}) == (
        '{"a":[true,null,"é"],"b":1,"c":0.5}'.encode()
    )


def test_canonical_form_round_trips_through_a_parser():
    import json

    value = {"reward": 0.1 + 0.2, "n": -3, "s": "line\nbreak "}
    assert canonical_json(json.loads(canonical_json(value))) == canonical_json(value)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), {1: "x"}, b"bytes", 2**63, {"s": "\ud800"}, object()],
)
def test_values_without_a_canonical_form_are_refused(value):
    with pytest.raises(CanonicalError):
        canonical_json(value)


def test_digest_is_sha256_of_canonical_bytes():
    import hashlib

    assert digest({"a": 1}) == hashlib.sha256(b'{"a":1}').hexdigest()


def write_key(tmp_path, mode=0o600):
    path = tmp_path / "machine.pem"
    path.write_bytes(generate_private_key_pem())
    path.chmod(mode)
    return path


def test_signatures_are_domain_separated_and_tamper_evident(tmp_path):
    signer = Signer("m1-k1", load_private_key(write_key(tmp_path)))
    public = public_key_from_b64(signer.public_key_b64)
    signature = signer.sign(RECORD_DOMAIN, {"output": "ok"})
    assert signature_valid(public, RECORD_DOMAIN, {"output": "ok"}, signature)
    assert not signature_valid(public, RECORD_DOMAIN, {"output": "forged"}, signature)
    assert not signature_valid(public, TOKEN_DOMAIN, {"output": "ok"}, signature)
    assert not signature_valid(public, RECORD_DOMAIN, {"output": "ok"}, "not base64!")


def test_private_key_must_not_be_group_or_world_readable(tmp_path):
    with pytest.raises(ValueError):
        load_private_key(write_key(tmp_path, 0o640))


def test_generate_command_writes_private_key_and_prints_public_half(tmp_path, capsys):
    path = tmp_path / "new.pem"
    assert main(["generate", str(path)]) == 0
    assert os.stat(path).st_mode & 0o777 == 0o600
    printed = capsys.readouterr().out.strip()
    assert public_key_from_b64(printed)


def test_attest_imports_only_stdlib_and_cryptography():
    import sys

    allowed = set(sys.stdlib_module_names) | {"cryptography", "reliquary_sandbox"}
    sources = list(ATTEST_DIR.glob("*.py"))
    assert sources, "the import scan found no files"
    for source in sources:
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module]
            else:
                continue
            for name in names:
                assert name.split(".")[0] in allowed, f"{source.name} imports {name}"


def test_non_canonical_base64_is_refused(tmp_path):
    signer = Signer("m1-k1", load_private_key(write_key(tmp_path)))
    public = public_key_from_b64(signer.public_key_b64)
    signature = signer.sign(RECORD_DOMAIN, {"a": 1})
    assert signature_valid(public, RECORD_DOMAIN, {"a": 1}, signature)
    # 64-byte signature -> 88 chars ending "=="; the last data char carries 4
    # padding bits. Changing them keeps the decoded bytes but not the text.
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    stem = signature[:-3]
    last = signature[-3]
    mutated = stem + alphabet[alphabet.index(last) ^ 1] + "=="
    assert mutated != signature
    assert not signature_valid(public, RECORD_DOMAIN, {"a": 1}, mutated)
    with pytest.raises(ValueError):
        public_key_from_b64(_noncanonical_key(signer))


def _noncanonical_key(signer):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    text = signer.public_key_b64  # 32 bytes -> 44 chars ending "="
    return text[:-2] + alphabet[alphabet.index(text[-2]) ^ 1] + "="


def test_symlinked_key_file_is_refused(tmp_path):
    real = write_key(tmp_path)
    link = tmp_path / "link.pem"
    link.symlink_to(real)
    with pytest.raises(ValueError):
        load_private_key(link)


def test_generate_on_an_existing_path_is_a_usage_error(tmp_path, capsys):
    path = tmp_path / "exists.pem"
    path.write_text("x")
    assert main(["generate", str(path)]) == 2
    assert "exists" in capsys.readouterr().err
    assert path.read_text() == "x"
