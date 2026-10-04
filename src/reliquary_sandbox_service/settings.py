"""Validated controller configuration."""

import re
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
        from urllib.parse import urlparse

        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("public_base_url must be an HTTP(S) origin")
        if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path.strip("/"):
            raise ValueError("public_base_url must be an origin without credentials or path")
        if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("non-loopback public URLs must use HTTPS")
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_runtime(self):
        if self.require_gvisor and self.docker_runtime != "runsc":
            raise ValueError("gVisor requires the runsc runtime")
        if self.default_image not in self.allowed_images:
            raise ValueError("default_image must be allowlisted")
        return self
