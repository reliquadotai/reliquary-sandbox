"""Validated controller configuration."""

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from reliquary_sandbox.episode_task import IMAGE_DIGEST, VERIFIERS_ENTRY, VERIFIERS_ENTRY_PREFIX
from reliquary_sandbox.observation import MAX_ARGUMENT_BYTES

ENTRY_POINT = r"[A-Za-z_][\w.]*:[A-Za-z_]\w*"
"""An `episode_envs` value naming a `sandbox_task(split, index)` factory."""
MAX_TRANSCRIPT_BYTES = 8 * 1024**2
"""A signed transcript, as validators receive it (compact UTF-8 JSON, ensure_ascii=False:
`{"token", "records"}`), never exceeds this: Catalyst's wire cap for a submission."""
RECORD_OVERHEAD_BYTES = 8 * 1024
"""A record's keys, numbers, ids, short texts and signature, generously."""
FINAL_FACTS_BYTES = 64 * 1024
"""The grader's canonical facts cap (grader.MAX_FACTS_BYTES)."""


def transcript_reserve_bytes(output_cap_bytes: int) -> int:
    """The most one more call record plus the final record can add to a transcript: a
    bash output of `output_cap_bytes` raw bytes serialises to at most 6x (a control byte
    becomes `\\u00XX`; an invalid byte, U+FFFD, is 3); an edit output repeats its path
    (at most the argument limit); the arguments are at most MAX_ARGUMENT_BYTES canonical,
    which is how they serialise here too."""
    output = max(6 * output_cap_bytes, MAX_ARGUMENT_BYTES + 1024)
    call = output + MAX_ARGUMENT_BYTES + RECORD_OVERHEAD_BYTES
    return call + FINAL_FACTS_BYTES + RECORD_OVERHEAD_BYTES


def _http_url(value: str, name: str, allow_path: bool) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{name} must be an HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment or (
        parsed.path.strip("/") and not allow_path
    ):
        raise ValueError(f"{name} must be an origin without credentials or path")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(f"non-loopback {name} must use HTTPS")
    return value.rstrip("/")


MAX_PRESENT_WINDOW_S = 120
"""The validator's PRESENT_WINDOW_S: it stops holding a token's place after this long."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RELIQUARY_SANDBOX_", extra="ignore")

    tokens: dict[str, str]
    state_dir: Path = Path("data")
    public_base_url: str = "http://127.0.0.1:8000"
    docker_host: str | None = None
    docker_runtime: str = "runsc"
    require_gvisor: bool = True
    allowed_images: list[str] = Field(default_factory=lambda: ["reliquary-sandbox-guest:0.1.0"])
    default_image: str = "reliquary-sandbox-guest:0.1.0"
    memory_limit: str = "1g"
    nano_cpus: int = Field(default=1_000_000_000, ge=1)
    pids_limit: int = Field(default=128, ge=16, le=4096)
    workspace_size_bytes: int = Field(default=5 * 1024**3, ge=1024**2)
    tmp_size_bytes: int = Field(default=64 * 1024**2, ge=1024**2)
    max_file_bytes: int = Field(default=16 * 1024**2, ge=1024)
    max_snapshot_bytes: int = Field(default=16 * 1024**2, ge=1024)
    max_snapshot_files: int = Field(default=1000, ge=1)
    max_output_bytes: int = Field(default=1024**2, ge=1024)
    default_uid: int = Field(default=1000, ge=1)
    docker_connect_timeout: int = Field(default=10, ge=1, le=60)
    max_sandboxes_per_tenant: int = Field(default=16, ge=1)
    max_cpu_per_tenant: float = Field(default=16, gt=0, allow_inf_nan=False)
    max_memory_gb_per_tenant: float = Field(default=16, gt=0, allow_inf_nan=False)
    max_timeout_minutes: int = Field(default=120, ge=1)
    max_command_seconds: int = Field(default=300, ge=1)
    reaper_interval_seconds: float = Field(default=5, ge=0.1)
    gateway_token_seconds: int = Field(default=900, ge=60, le=3600)
    max_checkpoints_per_tenant: int = Field(default=64, ge=1)
    retention_days: int = Field(default=7, ge=1, le=365)
    max_terminal_records_per_tenant: int = Field(default=1000, ge=1)

    # Signed episodes. Off by default; see docs/deployment.md, "Episode hosts".
    episodes_enabled: bool = False
    machine_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    machine_key_file: Path | None = None
    machine_key_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    validator_public_keys: dict[str, str] = Field(default_factory=dict)
    episode_images: list[str] = Field(default_factory=list)
    episode_envs: dict[str, str] = Field(default_factory=dict)
    # Per verifiers env (`verifiers:` entries only): VerifiersEnvOptions as JSON.
    episode_env_options: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Serve a verifiers env that is not an installed distribution (tests only).
    episode_allow_unpackaged_verifiers: bool = False
    episode_capacity: int = Field(default=4, ge=1, le=512)
    episode_nano_cpus: int = Field(default=2_000_000_000, ge=100_000_000)
    # CPU (thousandths of a CPU) and memory all live episodes' agent boxes may reserve: each
    # episode reserves its box's CPUs and its memory budget. Both or neither; unset = slots only. The
    # reservable CPU is the total x episode_cpu_overcommit (measured at qualification: agent boxes idle
    # while the miner runs its model). Grading boxes run in the grading pool (episode_grading_*).
    episode_cpu_millis_total: int | None = Field(default=None, ge=100)
    episode_memory_bytes_total: int | None = Field(default=None, ge=64 * 1024**2)
    episode_cpu_overcommit: float = Field(default=1.0, ge=1.0, le=4.0)
    # The grading pool: what grading boxes may use at once. Both or neither; required
    # when the totals above are set.
    episode_grading_cpu_millis_total: int | None = Field(default=None, ge=100)
    episode_grading_memory_bytes_total: int | None = Field(default=None, ge=64 * 1024**2)
    # A session token presented later than its `issued_at` plus this many seconds is refused 401, unsigned
    # and unspent (an unpresented token never squats a place later). RL machines: 120, the
    # validator's PRESENT_WINDOW_S; corpus machines: unset.
    episode_present_window_s: int | None = Field(default=None, ge=10)
    episode_max_memory_bytes: int = Field(default=8 * 1024**3, ge=64 * 1024**2)
    episode_max_pids: int = Field(default=1024, ge=16, le=65536)
    episode_max_disk_bytes: int = Field(default=10 * 1024**3, ge=64 * 1024**2)
    episode_max_wall_s: int = Field(default=4 * 3600, ge=60)
    episode_max_calls: int = Field(default=512, ge=1, le=100_000)
    episode_max_cpu_s: int = Field(default=3600, ge=1)
    episode_max_call_timeout_s: int = Field(default=600, ge=1, le=24 * 3600)
    episode_output_cap_bytes: int = Field(default=65536, ge=1024, le=16 * 1024**2)
    # The argument size limit is not a setting: it is reliquary_sandbox.observation's
    # MAX_ARGUMENT_BYTES, shared with the miners' and validators' renderer.
    episode_edit_max_bytes: int = Field(default=4 * 1024**2, ge=1024, le=64 * 1024**2)
    episode_grader_output_bytes: int = Field(default=32 * 1024**2, ge=1024)
    # Bytes of the episode's transcript as compact UTF-8 JSON (`{"token", "records"}`,
    # ensure_ascii=False), at most MAX_TRANSCRIPT_BYTES. A call is always recorded; when
    # what is left could not hold one more worst-case call and the final
    # (`transcript_reserve_bytes`), the episode ends budget_exhausted, so the signed
    # transcript, final included, never passes this.
    episode_max_transcript_bytes: int = Field(default=MAX_TRANSCRIPT_BYTES, ge=256 * 1024,
                                              le=MAX_TRANSCRIPT_BYTES)
    episode_grading_attempts: int = Field(default=2, ge=1, le=5)
    # A grading box's create + first CPU read; past it the attempt is ours. At most
    # the grader's RETRY_MARGIN_S (120), so a grade step keeps its full timeout.
    episode_grading_create_timeout_s: float = Field(default=90, gt=0, le=120)
    episode_retention_days: int = Field(default=7, ge=1, le=365)
    episode_cpu_source: Literal["docker", "cgroupfs"] = "docker"
    episode_cgroup_root: Path = Path("/sys/fs/cgroup")
    episode_allow_unquoted_disk: bool = False
    # A static busybox on the host, bind-mounted read-only into every box: all
    # plumbing (runner, kill, file I/O) runs from it, never from the image.
    episode_helper_busybox: Path | None = None
    episode_helper_busybox_sha256: str | None = None
    episode_docker_pidfile: Path = Path("/var/run/docker.pid")
    episode_fault_settle_s: float = Field(default=1.0, ge=0, le=30)
    episode_runsc_version: str = "release-20260928.0"
    # The verifiers bridge's scoring processes (docs/deployment.md, "The scoring users").
    # Slot i runs as uid:gid scoring_uid_base + i; scoring_slots must be at least the
    # number of episodes grading at once (episode_capacity; refused otherwise when a
    # verifiers entry is configured).
    scoring_uid_base: int = Field(default=200000, ge=1000)
    scoring_slots: int = Field(default=64, ge=1, le=4096)
    scoring_tmp_root: Path | None = None
    scoring_wrapper: Path = Path("/usr/local/libexec/reliquary-scoring-python")
    heartbeat_url: str | None = None
    heartbeat_interval_seconds: float = Field(default=5, ge=0.5, le=300)
    # How often the episode host checks (preflight without the quota probe) run;
    # `/readyz` serves the last verdict.
    episode_host_check_seconds: float = Field(default=30, ge=0.01, le=3600)

    @field_validator("tokens")
    @classmethod
    def validate_tokens(cls, value):
        if not value or len(set(value.values())) != len(value):
            raise ValueError("unique tenant API keys are required")
        for tenant, token in value.items():
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", tenant):
                raise ValueError("tenant names must contain letters, numbers, underscores or hyphens")
            if len(token) < 32 or any(c.isspace() for c in token):
                raise ValueError("API keys must contain at least 32 non-whitespace characters")
        return value

    @field_validator("public_base_url")
    @classmethod
    def validate_url(cls, value):
        return _http_url(value, "public_base_url", allow_path=False)

    @field_validator("heartbeat_url")
    @classmethod
    def validate_heartbeat_url(cls, value):
        return None if value is None else _http_url(value, "heartbeat_url", allow_path=True)

    @model_validator(mode="after")
    def validate_runtime(self):
        if self.require_gvisor and self.docker_runtime != "runsc":
            raise ValueError("gVisor requires the runsc runtime")
        if self.default_image not in self.allowed_images:
            raise ValueError("default_image must be allowlisted")
        return self

    @model_validator(mode="after")
    def validate_transcript_room(self):
        if (self.episode_max_transcript_bytes - transcript_reserve_bytes(
                self.episode_output_cap_bytes) < 128 * 1024):
            raise ValueError("episode_max_transcript_bytes must exceed "
                             "transcript_reserve_bytes(episode_output_cap_bytes) by 128 KiB "
                             "(lower episode_output_cap_bytes)")
        return self

    @model_validator(mode="after")
    def validate_episodes(self):
        if not self.episodes_enabled:
            return self
        if (self.episode_cpu_millis_total is None) != (self.episode_memory_bytes_total is None):
            raise ValueError("episode_cpu_millis_total and episode_memory_bytes_total go together")
        if (self.episode_grading_cpu_millis_total is None) != (self.episode_grading_memory_bytes_total is None):
            raise ValueError("episode_grading_cpu_millis_total and episode_grading_memory_bytes_total go together")
        if self.episode_cpu_millis_total is not None and self.episode_grading_cpu_millis_total is None:
            raise ValueError("a machine that reserves episode resources needs a grading pool "
                             "(episode_grading_cpu_millis_total, episode_grading_memory_bytes_total)")
        if self.episode_cpu_millis_total is not None and not (
                self.episode_present_window_s is not None and self.episode_present_window_s <= MAX_PRESENT_WINDOW_S):
            raise ValueError("a machine that reserves episode resources needs episode_present_window_s "
                             f"(10 to {MAX_PRESENT_WINDOW_S}): the validator releases an unpresented token's "
                             "place after that long")
        from reliquary_sandbox.attest import public_key_from_b64

        if not (self.machine_id and self.machine_key_file and self.machine_key_id):
            raise ValueError("episodes require machine_id, machine_key_file and machine_key_id")
        if not self.validator_public_keys:
            raise ValueError("episodes require at least one validator public key")
        for key in self.validator_public_keys.values():
            public_key_from_b64(key)
        if not self.episode_images or any(
            not IMAGE_DIGEST.fullmatch(image) for image in self.episode_images
        ):
            raise ValueError("episode images must be listed and pinned by digest (repo@sha256:...)")
        if not self.machine_key_file.is_file():
            raise ValueError("machine_key_file does not exist")
        if not self.episode_helper_busybox or not self.episode_helper_busybox.is_file():
            raise ValueError("episode_helper_busybox must point to an existing static busybox")
        if not re.fullmatch(r"[0-9a-f]{64}", self.episode_helper_busybox_sha256 or ""):
            raise ValueError("episode_helper_busybox_sha256 must pin the busybox (64 hex digits)")
        if not all(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", env) for env in self.episode_envs):
            raise ValueError("episode_envs names must contain letters, numbers, underscores or hyphens")
        if not self.episode_envs or not all(
            VERIFIERS_ENTRY.fullmatch(entry) if entry.startswith(VERIFIERS_ENTRY_PREFIX)
            else re.fullmatch(ENTRY_POINT, entry)
            for entry in self.episode_envs.values()
        ):
            raise ValueError("episode_envs maps env names to 'module:callable' entry points "
                             "or 'verifiers:<taskset-id>==<version>'")
        for env in self.episode_env_options:
            if not VERIFIERS_ENTRY.fullmatch(self.episode_envs.get(env, "")):
                raise ValueError(f"episode_env_options[{env!r}] needs a verifiers entry in "
                                 "episode_envs")
        if (any(entry.startswith(VERIFIERS_ENTRY_PREFIX) for entry in self.episode_envs.values())
                and self.scoring_slots < self.episode_capacity):
            raise ValueError(f"scoring_slots ({self.scoring_slots}) must be at least "
                             f"episode_capacity ({self.episode_capacity}): every episode may "
                             "grade at once, each in a scoring process of its own slot")
        return self
