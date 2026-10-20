"""CPU and memory a machine's episodes reserve (capacity in CPU and memory units).

An episode reserves, from the moment its open is accepted until its agent box is removed, that box: the box's
CPUs as the machine sizes it (`TaskLimits.cpus`, never above `episode_nano_cpus` on an accounting
machine: a bigger task is refused) and the token's memory budget. The grading box does not draw from
these totals: it runs in the machine's grading pool (`GradingPool`). The reservable CPU is
`episode_cpu_millis_total` x `episode_cpu_overcommit` (the agent box idles while the miner runs its
model; the factor is fixed at qualification); memory is never overcommitted. An open whose reservation
does not fit what is left is refused 503 (retried); one that could never fit is refused 422. With both
totals unset nothing is accounted (slots only, and today's CPU clamp). A failed removal retains the
box's reservation even after the episode closes; the reaper releases it only after cleanup succeeds."""

from __future__ import annotations

import itertools
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

BOXES_PER_EPISODE = 1


@dataclass(frozen=True)
class Reservation:
    cpu_millis: int
    memory_bytes: int

    def __add__(self, other: Reservation) -> Reservation:
        return Reservation(self.cpu_millis + other.cpu_millis, self.memory_bytes + other.memory_bytes)

    def within(self, limit: Reservation) -> bool:
        return self.cpu_millis <= limit.cpu_millis and self.memory_bytes <= limit.memory_bytes

    def to_dict(self) -> dict[str, int]:
        return {"cpu_millis": self.cpu_millis, "memory_bytes": self.memory_bytes}

    @classmethod
    def from_dict(cls, value: Any) -> Reservation | None:
        if not isinstance(value, Mapping):
            return None
        cpu, memory = value.get("cpu_millis"), value.get("memory_bytes")
        if type(cpu) is not int or type(memory) is not int or cpu < 0 or memory < 0:
            return None
        return cls(cpu, memory)


NOTHING = Reservation(0, 0)


def episode_reservation(box_nano_cpus: int, memory_bytes: int) -> Reservation:
    """What one episode reserves: its agent box's CPUs (thousandths of a CPU) and memory."""
    return Reservation(BOXES_PER_EPISODE * (int(box_nano_cpus) // 1_000_000),
                       BOXES_PER_EPISODE * int(memory_bytes))


def resources_enabled(settings: Any) -> bool:
    return (getattr(settings, "episode_cpu_millis_total", None) is not None
            and getattr(settings, "episode_memory_bytes_total", None) is not None)


def overcommit_permille(settings: Any) -> int:
    return int(round(float(getattr(settings, "episode_cpu_overcommit", 1.0)) * 1000))


def totals(settings: Any) -> Reservation:
    """The reservable totals: the CPU total times the overcommit (rounded down), the memory total."""
    cpu = int(settings.episode_cpu_millis_total) * overcommit_permille(settings) // 1000
    return Reservation(cpu, int(settings.episode_memory_bytes_total))


def row_reservation(episode: Mapping[str, Any], settings: Any) -> Reservation:
    """A live row's reservation; a row opened before reservations were stored is charged the
    machine's largest box and its token's memory budget (never less than it may use)."""
    stored = Reservation.from_dict(episode.get("reserve"))
    if stored is not None:
        return stored
    budgets = (episode.get("claims") or {}).get("budgets") or {}
    memory = budgets.get("memory_bytes")
    if type(memory) is not int or memory <= 0:
        memory = settings.episode_max_memory_bytes
    return episode_reservation(settings.episode_nano_cpus, memory)


def cleanup_reservation(episode: Mapping[str, Any], settings: Any,
                        budget: Reservation) -> Reservation:
    """An existing box's share, reserving the whole CPU budget for legacy rows.

    Without a stored reservation its historical CPU limit is unknown; the current
    box maximum may have been lowered. Admit no extra CPU work until it is removed.
    The token still states its exact memory limit.
    """
    need = row_reservation(episode, settings)
    if Reservation.from_dict(episode.get("reserve")) is None:
        need = Reservation(max(need.cpu_millis, budget.cpu_millis), need.memory_bytes)
    return need


def in_use(rows: Iterable[Mapping[str, Any]], settings: Any) -> Reservation:
    used = NOTHING
    for row in rows:
        need = (cleanup_reservation(row, settings, totals(settings))
                if row.get("status") == "CLOSED" and resources_enabled(settings)
                else row_reservation(row, settings))
        used = used + need
    return used


def resources_report(settings: Any, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The capacity report's `resources` block, or None when nothing is accounted. `sessions`: the ids of
    the live rows `used_*` counts (a validator stops holding its own copy of a reservation once listed)."""
    if not resources_enabled(settings):
        return None
    rows = list(rows)
    total, used = totals(settings), in_use(rows, settings)
    return {"cpu_millis": total.cpu_millis, "memory_bytes": total.memory_bytes,
            "used_cpu_millis": used.cpu_millis, "used_memory_bytes": used.memory_bytes,
            "free_cpu_millis": max(0, total.cpu_millis - used.cpu_millis),
            "free_memory_bytes": max(0, total.memory_bytes - used.memory_bytes),
            "max_box_cpu_millis": int(settings.episode_nano_cpus) // 1_000_000,
            "boxes_per_episode": BOXES_PER_EPISODE,
            "cpu_overcommit_permille": overcommit_permille(settings),
            "present_window_s": int(settings.episode_present_window_s),
            "sessions": sorted(str(row["id"]) for row in rows if row.get("id") is not None)}


GRADING_POOL_FULL = "the grading pool stayed full"
"""A grading the pool could not start in time while other miners' gradings held it: the machine's doing
(validators count it toward no abort cap and free its engagement)."""
GRADING_POOL_OWN = "the grading pool stayed full with this miner's own gradings"
"""The same, while this hotkey's own gradings held more than half the pool: an ordinary machine abort (neutral
for the yield, but counted toward the miner's daily abort cap: a burst of hanging gradings is the miner's)."""


def _minus(a: Reservation, b: Reservation) -> Reservation:
    return Reservation(max(0, a.cpu_millis - b.cpu_millis), max(0, a.memory_bytes - b.memory_bytes))


class GradingPool:
    """What grading boxes may use at once. Waiters are served one at a time in
    the order (the pool share their hotkey holds now, arrival), and only the first in that order may take room:
    one hotkey's burst queues behind the other hotkeys' gradings, and a large box is never starved by smaller
    ones (head of line). A box larger than the whole pool is refused at once. Thread-safe (grading runs on
    worker threads; the condition is never held while a box runs)."""

    def __init__(self, total: Reservation) -> None:
        self.total = total
        self._used = NOTHING
        self._held: dict[str, Reservation] = {}
        self._queue: list[tuple[int, str, Reservation]] = []
        self._arrivals = itertools.count()
        self._room = threading.Condition()
        self._waits = self._timeouts = 0

    def _first(self) -> tuple[int, str, Reservation]:
        return min(self._queue, key=lambda waiter: (self._held.get(waiter[1], NOTHING).cpu_millis, waiter[0]))

    def _own(self, owner: str) -> bool:
        held = self._held.get(owner, NOTHING)
        return (held.cpu_millis * 2 > self.total.cpu_millis              # more than half
                or held.memory_bytes * 2 > self.total.memory_bytes)

    def acquire(self, need: Reservation, timeout_s: float | None, owner: str = "") -> str | None:
        """None once ``need`` is taken; else why not (``GRADING_POOL_OWN`` when ``owner``'s own gradings held more
        than half the pool, ``GRADING_POOL_FULL`` otherwise). ``timeout_s`` None waits without a bound."""
        owner = str(owner)
        with self._room:
            if not need.within(self.total):
                self._timeouts += 1
                return GRADING_POOL_FULL
            me = (next(self._arrivals), owner, need)
            self._queue.append(me)

            def ready() -> bool:
                return self._first() is me and (self._used + need).within(self.total)

            try:
                if not ready():
                    self._waits += 1
                ok = self._room.wait_for(ready, None if timeout_s is None else max(0.0, float(timeout_s)))
            finally:
                self._queue.remove(me)
                self._room.notify_all()                 # the next waiter may now be the first
            if not ok:
                self._timeouts += 1
                return GRADING_POOL_OWN if self._own(owner) else GRADING_POOL_FULL
            self._used = self._used + need
            self._held[owner] = self._held.get(owner, NOTHING) + need
            return None

    def release(self, need: Reservation, owner: str = "") -> None:
        owner = str(owner)
        with self._room:
            self._used = _minus(self._used, need)
            held = _minus(self._held.get(owner, NOTHING), need)
            if held == NOTHING:
                self._held.pop(owner, None)
            else:
                self._held[owner] = held
            self._room.notify_all()

    def restore(self, need: Reservation, owner: str = "") -> None:
        """Account for a persisted box left by the previous controller.

        It already exists, so retain its full share even if the operator reduced the
        configured pool in the meantime. New work waits until cleanup frees it.
        """
        owner = str(owner)
        with self._room:
            self._used = self._used + need
            self._held[owner] = self._held.get(owner, NOTHING) + need

    def snapshot(self) -> dict[str, int]:
        with self._room:
            return {"cpu_millis": self.total.cpu_millis, "memory_bytes": self.total.memory_bytes,
                    "used_cpu_millis": self._used.cpu_millis, "used_memory_bytes": self._used.memory_bytes,
                    "waiting": len(self._queue), "waits": self._waits, "timeouts": self._timeouts}


def grading_reservation(spec: Any) -> Reservation:
    """A grading box's share of the pool: the box's CPUs and memory (its `BoxSpec`)."""
    return Reservation(int(spec.nano_cpus) // 1_000_000, int(spec.memory_bytes))


def grading_pool(settings: Any) -> GradingPool | None:
    cpu = getattr(settings, "episode_grading_cpu_millis_total", None)
    memory = getattr(settings, "episode_grading_memory_bytes_total", None)
    if cpu is None or memory is None:
        return None
    return GradingPool(Reservation(int(cpu), int(memory)))
