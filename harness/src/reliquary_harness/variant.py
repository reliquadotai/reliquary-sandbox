"""Which surface variant of a harness an episode plays, drawn from the validator's seed.

A spec's `variants.axes` is a list of finite axes (`{"name": ..., "values": [...]}`); a
variant is one value per axis, numbered in mixed radix over the axes in order (the first
axis varies slowest). The index is a pure function of the validator's seed, the lease or
engagement, the task index and the harness digest, through sha256 (never `hash()`), so
every machine and Python version draws the same variant; the miner never chooses it.
A spec without axes has one variant, index 0, which is the spec itself.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .spec import HarnessSpec


@dataclass(frozen=True)
class Variant:
    index: int
    values: Mapping[str, Any]


def variant_count(spec: HarnessSpec) -> int:
    count = 1
    for axis in spec.variants.get("axes", []):
        values = axis["values"]
        if not values:
            raise ValueError(f"axis {axis['name']!r} has no value")
        count *= len(values)
    return count


def variant_index(spec: HarnessSpec, *, seed: bytes | str, engagement: str, task_index: int) -> int:
    """`H(seed ‖ engagement ‖ task_index ‖ digest) mod |variants|`, each part length-prefixed."""
    count = variant_count(spec)
    if count == 1:
        return 0
    seed_bytes = seed.encode() if isinstance(seed, str) else bytes(seed)
    digest = hashlib.sha256()
    for part in (seed_bytes, engagement.encode(), str(int(task_index)).encode(), spec.digest.encode()):
        digest.update(len(part).to_bytes(8, "big") + part)
    return int.from_bytes(digest.digest(), "big") % count


def variant(spec: HarnessSpec, index: int) -> Variant:
    count = variant_count(spec)
    if not 0 <= int(index) < count:
        raise ValueError(f"variant {index} is outside 0..{count - 1} of {spec.key}")
    values: dict[str, Any] = {}
    rest = int(index)
    for axis in reversed(spec.variants.get("axes", [])):
        rest, at = divmod(rest, len(axis["values"]))
        values[axis["name"]] = axis["values"][at]
    return Variant(int(index), dict(reversed(list(values.items()))))


__all__ = ["Variant", "variant", "variant_count", "variant_index"]
