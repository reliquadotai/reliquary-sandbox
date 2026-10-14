"""The image lock file of a verifiers env package (env norm, rule 7).

`<the taskset's top-level package>/sandbox-images.lock.json`:

    {"version": 1, "images": {"<tag as in TaskData.image>": "<repo>@sha256:<64 hex>"}}

The bridge serves a task only when its `TaskData.image` is a key of the lock (or is itself
one of the lock's digests): an image the env did not pin is refused at resolve.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, ModuleType

from reliquary_sandbox.episode_task import IMAGE_DIGEST

__all__ = ["LOCK_FILE", "LOCK_VERSION", "ImageLock", "ImageLockError"]

LOCK_FILE = "sandbox-images.lock.json"
LOCK_VERSION = 1


class ImageLockError(ValueError):
    """The lock file is missing or malformed, or an image is not pinned by it."""


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in pairs:
        if key in out:
            raise ImageLockError(f"{LOCK_FILE}: duplicate key {key!r}")
        out[key] = value
    return out


@dataclass(frozen=True)
class ImageLock:
    images: Mapping[str, str]
    """tag -> `repo@sha256:<digest>`."""

    @classmethod
    def parse(cls, data: bytes) -> ImageLock:
        try:
            document = json.loads(data, object_pairs_hook=_no_duplicate_keys)
        except ImageLockError:
            raise
        except (UnicodeDecodeError, ValueError) as exc:
            raise ImageLockError(f"{LOCK_FILE} is not JSON") from exc
        if not isinstance(document, dict) or set(document) != {"version", "images"}:
            raise ImageLockError(f'{LOCK_FILE} must be {{"version": 1, "images": {{...}}}}')
        if document["version"] != LOCK_VERSION:
            raise ImageLockError(f"{LOCK_FILE} version must be {LOCK_VERSION}")
        images = document["images"]
        if not isinstance(images, dict) or not images:
            raise ImageLockError(f"{LOCK_FILE} must pin at least one image")
        for tag, digest in images.items():
            if not isinstance(tag, str) or not tag or any(c.isspace() for c in tag):
                raise ImageLockError(f"{LOCK_FILE}: invalid image tag {tag!r}")
            if not isinstance(digest, str) or not IMAGE_DIGEST.fullmatch(digest):
                raise ImageLockError(f"{LOCK_FILE}: {tag!r} must map to repo@sha256:<64 hex>")
            if IMAGE_DIGEST.fullmatch(tag) and digest != tag:
                raise ImageLockError(f"{LOCK_FILE}: digest key {tag!r} must map to itself")
        return cls(MappingProxyType(dict(images)))

    @classmethod
    def of_module(cls, module: ModuleType) -> ImageLock:
        """The lock shipped in `module`'s package (the taskset's top-level package)."""
        package = module.__name__.split(".", 1)[0]
        try:
            data = (importlib.resources.files(package) / LOCK_FILE).read_bytes()
        except (FileNotFoundError, NotADirectoryError, TypeError, ModuleNotFoundError) as exc:
            raise ImageLockError(f"{package} ships no {LOCK_FILE}") from exc
        return cls.parse(data)

    def sha256(self) -> str:
        """sha256 of the lock's canonical JSON (sorted keys, no spaces): the digest a gateway
        publishes per bridged env and a validator recomputes from its own install."""
        document = {"version": LOCK_VERSION, "images": dict(sorted(self.images.items()))}
        return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=True).encode("ascii")).hexdigest()

    def digest(self, image: str) -> str:
        """The pinned `repo@sha256:...` of `image` (a locked tag, or a locked digest)."""
        if image in self.images:
            return self.images[image]
        if image in self.images.values():
            return image
        raise ImageLockError(f"image {image!r} is not pinned in {LOCK_FILE}")

    def digests(self) -> list[str]:
        return sorted(set(self.images.values()))
