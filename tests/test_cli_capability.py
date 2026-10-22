"""The source-owned CPU mediation identity is fixed package data, not caller input."""

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("verifiers.v1")

from reliquary_sandbox.attest.canonical import canonical_json
from reliquary_sandbox_verifiers import cli
from reliquary_sandbox_verifiers.cli_runner import capability_sha256

SEALED_SHA256 = "748cb39764e33d9bb35842bfc441a64c53c631822026da53ac7e1a9cf075757b"


def test_packaged_cpu_identity_matches_the_sealed_qualification():
    capability = cli.qualified_capability("signed-claude-code-cli-v1")
    assert capability_sha256(capability) == SEALED_SHA256
    assert capability.qualification_sha256 == "cfc9390456771d98e98ed1e9cf62a6783c0af90f2149805b27193c4e0b906c4d"
    assert len(capability.artifacts) == 145 and len(capability.model_tools) == 2
    assert capability.date_recipe == cli.CLI_DATE_RECIPE
    assert capability.runtime_image == "sha256:4a3eb2b0fc0e3047677b708ce1e6f297f32e4c145076fdd059b9da0e4963fe9c"
    with pytest.raises(TypeError):
        cli._QUALIFIED["caller-supplied"] = capability
    tools = capability.model_tools
    tools.clear()
    assert len(capability.model_tools) == 2


@pytest.mark.parametrize("protocol", ["signed-codex-cli-v1", "unknown", "signed-bash-v2"])
def test_unqualified_protocols_stay_unavailable(protocol):
    with pytest.raises(cli.CLIUnavailable, match="no approved actual-CLI qualification"):
        cli.qualified_capability(protocol)


def test_modified_packaged_bytes_fail_before_construction(monkeypatch):
    monkeypatch.setattr(Path, "read_bytes", lambda path: b"{}")
    with pytest.raises(cli.CLIUnavailable, match="could not be verified"):
        cli._load_qualified_capability()


@pytest.mark.parametrize("change", [
    {"versions": []}, {"date_template_sha256": "f" * 64}, {"max_turns": 49},
    {"schema": "caller-schema"}, {"extra": True}, {"date_recipe": None},
    {"qualification_sha256": "0" * 64},
])
def test_source_updates_still_require_the_constructor_and_derived_identity(monkeypatch, change):
    path = Path(cli.__file__).with_name("signed_cli_capability.json")
    identity = json.loads(path.read_bytes())
    identity.update(change)
    raw = canonical_json(identity)
    # Simulate a source edit of both data and its pin; derived contracts still apply.
    monkeypatch.setattr(cli, "_CAPABILITY_RESOURCE_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(Path, "read_bytes", lambda path: raw)
    with pytest.raises(cli.CLIUnavailable, match="could not be verified"):
        cli._load_qualified_capability()
