"""Reliquary Sandboxes SDK.

A standalone SDK for managing remote code execution environments (sandboxes).
Includes HTTP client, authentication, and full sandbox lifecycle management.

The SDK names load lazily (PEP 562): importing a submodule such as
`reliquary_sandbox.attest`, `.episode_task`, `.observation` or `.episode_client` loads
none of the SDK, and so no protobuf (whose generated command_session module would
otherwise be registered next to other libraries' copies).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the names, for type checkers and linters; loaded lazily below
    from .core import (
        APIClient,
        APIError,
        APITimeoutError,
        AsyncAPIClient,
        Config,
        PaymentRequiredError,
        UnauthorizedError,
    )
    from .exceptions import (
        BatchStatusUnsupportedError,
        CommandTimeoutError,
        DownloadTimeoutError,
        SandboxFileNotFoundError,
        SandboxFileTooLargeError,
        SandboxImagePullError,
        SandboxNotRunningError,
        SandboxOOMError,
        SandboxTimeoutError,
        UploadTimeoutError,
    )
    from .images import AsyncImageClient, ImageClient
    from .models import (
        AdvancedConfigs,
        BackgroundJob,
        BackgroundJobStatus,
        BackgroundJobStatusSnapshot,
        BatchCheckpointResponse,
        BatchSandboxStatusResponse,
        BuildImageRequest,
        BuildImageResponse,
        BulkBuildImageResponse,
        BulkDeleteSandboxRequest,
        BulkDeleteSandboxResponse,
        CheckpointLookupError,
        CommandResponse,
        CreateSandboxRequest,
        EgressPolicyStatus,
        FileUploadResponse,
        ImageArtifactType,
        ImageBuildStatus,
        ImageCoordinateState,
        ImageListItem,
        ImageListResponse,
        ImageMutationError,
        ImageOwner,
        ImageOwnerType,
        ImageUpdateItem,
        ImageUpdatePatch,
        ImageUpdateResult,
        ImageUpdateSource,
        ImageVisibility,
        PersonalImageOwner,
        PlatformImageOwner,
        ReadFileResponse,
        Sandbox,
        SandboxCheckpoint,
        SandboxEgressPolicy,
        SandboxListResponse,
        SandboxStatus,
        SandboxStatusLookupError,
        SandboxStatusSnapshot,
        SourceImageBuildResult,
        SSHSession,
        StartCommand,
        TeamImageOwner,
        UpdateImagesRequest,
        UpdateImagesResponse,
    )
    from .process import AsyncSandboxProcess
    from .sandbox import AsyncSandboxClient, SandboxClient

__version__ = "0.1.0"

_LAZY = {
    "APIClient": ".core",
    "APIError": ".core",
    "APITimeoutError": ".core",
    "AsyncAPIClient": ".core",
    "Config": ".core",
    "PaymentRequiredError": ".core",
    "UnauthorizedError": ".core",
    "BatchStatusUnsupportedError": ".exceptions",
    "CommandTimeoutError": ".exceptions",
    "DownloadTimeoutError": ".exceptions",
    "SandboxFileNotFoundError": ".exceptions",
    "SandboxFileTooLargeError": ".exceptions",
    "SandboxImagePullError": ".exceptions",
    "SandboxNotRunningError": ".exceptions",
    "SandboxOOMError": ".exceptions",
    "SandboxTimeoutError": ".exceptions",
    "UploadTimeoutError": ".exceptions",
    "AsyncImageClient": ".images",
    "ImageClient": ".images",
    "AdvancedConfigs": ".models",
    "BackgroundJob": ".models",
    "BackgroundJobStatus": ".models",
    "BackgroundJobStatusSnapshot": ".models",
    "BatchCheckpointResponse": ".models",
    "BatchSandboxStatusResponse": ".models",
    "BuildImageRequest": ".models",
    "BuildImageResponse": ".models",
    "BulkBuildImageResponse": ".models",
    "BulkDeleteSandboxRequest": ".models",
    "BulkDeleteSandboxResponse": ".models",
    "CheckpointLookupError": ".models",
    "CommandResponse": ".models",
    "CreateSandboxRequest": ".models",
    "EgressPolicyStatus": ".models",
    "FileUploadResponse": ".models",
    "ImageArtifactType": ".models",
    "ImageBuildStatus": ".models",
    "ImageCoordinateState": ".models",
    "ImageListItem": ".models",
    "ImageListResponse": ".models",
    "ImageMutationError": ".models",
    "ImageOwner": ".models",
    "ImageOwnerType": ".models",
    "ImageUpdateItem": ".models",
    "ImageUpdatePatch": ".models",
    "ImageUpdateResult": ".models",
    "ImageUpdateSource": ".models",
    "ImageVisibility": ".models",
    "PersonalImageOwner": ".models",
    "PlatformImageOwner": ".models",
    "ReadFileResponse": ".models",
    "Sandbox": ".models",
    "SandboxCheckpoint": ".models",
    "SandboxEgressPolicy": ".models",
    "SandboxListResponse": ".models",
    "SandboxStatus": ".models",
    "SandboxStatusLookupError": ".models",
    "SandboxStatusSnapshot": ".models",
    "SourceImageBuildResult": ".models",
    "SSHSession": ".models",
    "StartCommand": ".models",
    "TeamImageOwner": ".models",
    "UpdateImagesRequest": ".models",
    "UpdateImagesResponse": ".models",
    "AsyncSandboxProcess": ".process",
    "AsyncSandboxClient": ".sandbox",
    "SandboxClient": ".sandbox",
}
# Deprecated alias for backward compatibility
_ALIASES = {"TimeoutError": "APITimeoutError"}


def __getattr__(name: str) -> Any:
    target = _ALIASES.get(name, name)
    module = _LAZY.get(target)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), target)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY, *_ALIASES})


__all__ = [
    # Core HTTP Client & Config
    "APIClient",
    "AsyncAPIClient",
    "Config",
    # Sandbox Clients
    "SandboxClient",
    "AsyncSandboxClient",
    "AsyncSandboxProcess",
    "ImageClient",
    "AsyncImageClient",
    # Models
    "Sandbox",
    "SandboxCheckpoint",
    "SandboxEgressPolicy",
    "SandboxStatus",
    "SandboxListResponse",
    "CreateSandboxRequest",
    "StartCommand",
    "CommandResponse",
    "FileUploadResponse",
    "ReadFileResponse",
    "BulkDeleteSandboxRequest",
    "BulkDeleteSandboxResponse",
    "EgressPolicyStatus",
    "AdvancedConfigs",
    "BackgroundJob",
    "BackgroundJobStatus",
    "BackgroundJobStatusSnapshot",
    "BatchCheckpointResponse",
    "CheckpointLookupError",
    "BatchSandboxStatusResponse",
    "SandboxStatusSnapshot",
    "SandboxStatusLookupError",
    "SSHSession",
    "BuildImageRequest",
    "BuildImageResponse",
    "BulkBuildImageResponse",
    "SourceImageBuildResult",
    "ImageVisibility",
    "ImageBuildStatus",
    "ImageArtifactType",
    "ImageOwnerType",
    "ImageListItem",
    "ImageListResponse",
    "ImageOwner",
    "PersonalImageOwner",
    "TeamImageOwner",
    "PlatformImageOwner",
    "ImageUpdateSource",
    "ImageUpdatePatch",
    "ImageUpdateItem",
    "UpdateImagesRequest",
    "UpdateImagesResponse",
    "ImageUpdateResult",
    "ImageCoordinateState",
    "ImageMutationError",
    # Exceptions
    "APIError",
    "BatchStatusUnsupportedError",
    "UnauthorizedError",
    "PaymentRequiredError",
    "SandboxFileNotFoundError",
    "SandboxFileTooLargeError",
    "APITimeoutError",
    "TimeoutError",  # Deprecated alias
    "SandboxOOMError",
    "SandboxTimeoutError",
    "SandboxImagePullError",
    "SandboxNotRunningError",
    "CommandTimeoutError",
    "UploadTimeoutError",
    "DownloadTimeoutError",
]
