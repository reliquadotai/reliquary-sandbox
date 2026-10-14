"""Shared builders for signed-episode tests: keys, tokens, transcripts."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from reliquary_sandbox.attest.records import (
    Budgets,
    CallBody,
    FinalBody,
    OpenBody,
    SessionClaims,
    chain_hash,
    issue_session_token,
    sign_record,
    token_sha256,
)
from reliquary_sandbox.attest.signing import Signer, generate_private_key_pem, load_private_key

IMAGE = "registry.example/task@sha256:" + "a" * 64
MACHINE = "machine-1"


def make_signer(directory: Path, name: str, key_id: str) -> tuple[Signer, Path]:
    path = directory / f"{name}.pem"
    path.write_bytes(generate_private_key_pem())
    path.chmod(0o600)
    return Signer(key_id, load_private_key(path)), path


def budgets(**overrides) -> Budgets:
    values = dict(max_calls=8, per_call_timeout_s=30, cpu_s=600, wall_s=3600,
                  memory_bytes=2 * 1024**3, pids=256, disk_bytes=1024**3)
    values.update(overrides)
    return Budgets(**values)


def claims(session_id="s-1", now=None, **overrides) -> SessionClaims:
    now = int(time.time()) if now is None else now
    values = dict(session_id=session_id, hotkey="5Hotkey", engagement="corpus:job-1:7",
                  env="fake-env", split="train", index=7, image=IMAGE, checkpoint="ck-1",
                  machine_id=MACHINE, issued_at=now - 5, expires_at=now + 3600,
                  budgets=budgets())
    values.update(overrides)
    return SessionClaims(**values)


def transcript(validator: Signer, machine: Signer, session_claims: SessionClaims,
               outputs=("hello\n",), status="graded", reward=1.0,
               tools=("bash", "edit")) -> dict:
    token = issue_session_token(session_claims, validator)
    at = session_claims.issued_at + 10
    bodies = [OpenBody(i=0, session_id=session_claims.session_id, machine_id=session_claims.machine_id,
                       token_sha256=token_sha256(token), image=session_claims.image,
                       env=session_claims.env, env_package="fake-env==1.0",
                       tools_version="reliquary-tools/1", tools=tools,
                       setup_output_sha256="0" * 64,
                       budgets=session_claims.budgets, at=at)]
    for k, output in enumerate(outputs):
        bodies.append(CallBody(i=len(bodies), session_id=session_claims.session_id, turn=0, k=k,
                               tool="bash", arguments={"command": f"echo {k}"}, output=output,
                               exit_code=0, truncated=False, timed_out=False, cpu_ms=5,
                               wall_ms=40, at=at + 1, prev=chain_hash(bodies[-1])))
    bodies.append(FinalBody(i=len(bodies), session_id=session_claims.session_id, status=status,
                            reward=reward if status == "graded" else None,
                            grading={"tests": "ok"} if status == "graded" else None,
                            state_sha256="1" * 64 if status == "graded" else None,
                            cpu_total_ms=99, reason=None if status == "graded" else status,
                            at=at + 2, prev=chain_hash(bodies[-1])))
    return {"token": token, "records": [sign_record(machine, session_claims.machine_id, b) for b in bodies]}


def directory_for(machine: Signer, valid_from=0, valid_until=None):
    from reliquary_sandbox.attest.verify import MachineKey, StaticDirectory

    return StaticDirectory([MachineKey(MACHINE, machine.key_id, machine.public_key_b64,
                                       valid_from, valid_until)])


def episode_settings(tmp_path, **overrides):
    """Settings for an episode host plus the validator signer that issues its tokens."""
    from reliquary_sandbox_service.settings import Settings

    validator = make_signer(tmp_path, "validator", "v-1")[0]
    _, key_path = make_signer(tmp_path, "machine", "m-1")
    helper = tmp_path / "busybox"
    helper.write_bytes(b"not really busybox")
    values = dict(
        tokens={"evals": "e" * 32}, state_dir=tmp_path / "state", reaper_interval_seconds=3600,
        episodes_enabled=True, machine_id=MACHINE, machine_key_file=key_path,
        machine_key_id="m-1", validator_public_keys={"v-1": validator.public_key_b64},
        episode_images=[IMAGE], episode_envs={"fake-env": "fake_episode_env:sandbox_task"},
        episode_allow_unquoted_disk=True, episode_helper_busybox=helper,
        episode_helper_busybox_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
    )
    values.update(overrides)
    return Settings(**values), validator
