"""Episode state and signed records, in the controller's SQLite database."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from typing import Any

from ..store import Store

OPENING = "OPENING"
RUNNING = "RUNNING"
FINISHING = "FINISHING"
CLOSED = "CLOSED"
ACTIVE = frozenset({OPENING, RUNNING, FINISHING})


RECORD_PAGE = 32
"""Records read per store-lock hold when a transcript is paged out."""


def dump_record(record: dict[str, Any]) -> str:
    """A record as stored: compact JSON with non-ASCII kept as UTF-8, the form validators
    receive, so the stored bytes are what the transcript cap counts."""
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def dump_token(token: dict[str, Any]) -> str:
    return json.dumps(token, ensure_ascii=False, separators=(",", ":"))


def transcript_frame_bytes(token: dict[str, Any]) -> int:
    """`{"token":<token>,"records":[]}` as compact UTF-8 JSON: the transcript without
    records. Each record adds its bytes plus one separating comma (none for the first)."""
    return len(('{"token":' + dump_token(token) + ',"records":[]}').encode("utf-8"))


class EpisodeStore:
    def __init__(self, store: Store):
        self.lock = store.lock
        self.db = store.db
        with self.lock, self.db:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY, token_sha256 TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL, expires_at REAL NOT NULL, payload TEXT NOT NULL,
                    has_cleanup INTEGER NOT NULL DEFAULT 0, closed_at REAL
                );
                CREATE INDEX IF NOT EXISTS episode_status ON episodes(status);
                CREATE INDEX IF NOT EXISTS episode_cleanup ON episodes(has_cleanup, closed_at);
                CREATE TABLE IF NOT EXISTS episode_records (
                    episode_id TEXT NOT NULL, i INTEGER NOT NULL, turn INTEGER, k INTEGER,
                    record TEXT NOT NULL,
                    PRIMARY KEY (episode_id, i), UNIQUE (episode_id, turn, k)
                );
                CREATE TABLE IF NOT EXISTS episode_states (
                    episode_id TEXT PRIMARY KEY, state BLOB NOT NULL
                );
            """)

    @staticmethod
    def _closed_at(episode: dict[str, Any]) -> float | None:
        if episode["status"] != CLOSED:
            return None
        return float(episode.get("closed_at") or time.time())

    def insert(self, episode: dict[str, Any]) -> bool:
        """False when the session id or token was already used (single use)."""
        try:
            with self.lock, self.db:
                self.db.execute(
                    "INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (episode["id"], episode["token_sha256"], episode["status"],
                     episode["claims"]["expires_at"], json.dumps(episode),
                     int(bool(episode.get("cleanup"))), self._closed_at(episode)),
                )
        except sqlite3.IntegrityError as error:
            if "UNIQUE" in str(error) or "PRIMARY KEY" in str(error):
                return False
            raise
        return True

    def commit(self, episode: dict[str, Any], record: dict[str, Any] | None = None,
               state: bytes | None = None) -> None:
        """Persist the episode and, atomically with it, its next signed record and
        the state published at finish (kept apart: the row is read on every request)."""
        with self.lock, self.db:
            cursor = self.db.execute(
                "UPDATE episodes SET status=?, payload=?, has_cleanup=?, "
                "closed_at=COALESCE(closed_at, ?) WHERE id=?",
                (episode["status"], json.dumps(episode), int(bool(episode.get("cleanup"))),
                 self._closed_at(episode), episode["id"]))
            if cursor.rowcount != 1:
                raise KeyError(f"unknown episode {episode['id']!r}")
            if record is not None:
                body = record["body"]
                self.db.execute(
                    "INSERT INTO episode_records VALUES (?, ?, ?, ?, ?)",
                    (episode["id"], body["i"], body.get("turn"), body.get("k"), dump_record(record)),
                )
            if state is not None:
                self.db.execute("INSERT OR REPLACE INTO episode_states VALUES (?, ?)",
                                (episode["id"], bytes(state)))

    def get(self, episode_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.db.execute("SELECT payload FROM episodes WHERE id=?", (episode_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def active(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.db.execute(
                "SELECT payload FROM episodes WHERE status != ?", (CLOSED,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def with_cleanup(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.db.execute(
                "SELECT payload FROM episodes WHERE status = ? AND has_cleanup = 1", (CLOSED,)
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def records(self, episode_id: str) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.db.execute(
                "SELECT record FROM episode_records WHERE episode_id=? ORDER BY i", (episode_id,)
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def last_record(self, episode_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.db.execute(
                "SELECT record FROM episode_records WHERE episode_id=? ORDER BY i DESC LIMIT 1",
                (episode_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def _record_page(self, episode_id: str, start: int, below: int) -> list[tuple[int, str]]:
        with self.lock:
            return self.db.execute(
                "SELECT i, record FROM episode_records WHERE episode_id=? AND i>=? AND i<? "
                "ORDER BY i LIMIT ?", (episode_id, start, below, RECORD_PAGE)).fetchall()

    def record_texts(self, episode_id: str, below: int) -> Iterator[str]:
        """The stored JSON of records `0 .. below-1`, a page per lock hold, so a
        long transcript never holds the store (shared by every episode)."""
        start = 0
        while start < below:
            rows = self._record_page(episode_id, start, below)
            if not rows:
                return
            for _, text in rows:
                yield text
            start = rows[-1][0] + 1

    def record(self, episode_id: str, i: int) -> dict[str, Any] | None:
        with self.lock:
            row = self.db.execute(
                "SELECT record FROM episode_records WHERE episode_id=? AND i=?", (episode_id, i)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def call_record(self, episode_id: str, turn: int, k: int) -> dict[str, Any] | None:
        with self.lock:
            row = self.db.execute(
                "SELECT record FROM episode_records WHERE episode_id=? AND turn=? AND k=?",
                (episode_id, turn, k),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def state(self, episode_id: str) -> bytes | None:
        with self.lock:
            row = self.db.execute(
                "SELECT state FROM episode_states WHERE episode_id=?", (episode_id,)).fetchone()
        return bytes(row[0]) if row else None

    def prune(self, now: float, retention_s: float) -> int:
        """Drop closed episodes (with their records) closed more than `retention_s` ago whose
        token has expired, and closed episodes that never recorded anything once their token
        expired. A row is never dropped while its token could still be presented (that would
        make the token reusable). Episodes with box cleanup pending are kept."""
        with self.lock, self.db:
            deleted = [row[0] for row in self.db.execute(
                "SELECT id FROM episodes WHERE status=? AND has_cleanup=0 AND ("
                "(closed_at < ? AND expires_at < ?) OR (expires_at < ? AND NOT EXISTS "
                "(SELECT 1 FROM episode_records WHERE episode_id=episodes.id)))",
                (CLOSED, now - retention_s, now, now))]
            self.db.executemany("DELETE FROM episode_records WHERE episode_id=?", ((i,) for i in deleted))
            self.db.executemany("DELETE FROM episode_states WHERE episode_id=?",
                                ((i,) for i in deleted))
            self.db.executemany("DELETE FROM episodes WHERE id=?", ((i,) for i in deleted))
            return len(deleted)
